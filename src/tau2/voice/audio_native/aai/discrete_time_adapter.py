"""Discrete-time adapter for the AAI (AssemblyAI voice-agent host) provider.

Mirrors the AssemblyAI adapter's lifecycle/tick structure, with one key
difference: the aai host speaks PCM16 over its WebSocket at rates IT chooses
and announces in the handshake (the agent's SDK defaults), while the adapter's
external interface stays tau2 telephony mu-law/8k. Audio conversion therefore
happens at this boundary:

- Send path: mu-law 8k (external) -> PCM16 -> resample to the host's input
  rate -> provider.
- Receive path: PCM16 at the host's output rate -> resample to 8kHz -> mu-law.

aai events don't carry a reply/turn id, so a synthetic running turn id
(``turn-N``) is maintained locally and incremented on ``reply_done``.
"""

import asyncio
import os
from typing import Any, List, Optional

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
)
from tau2.data_model.audio import (
    TELEPHONY_AUDIO_FORMAT,
    AudioData,
    AudioEncoding,
    AudioFormat,
)
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.aai.events import (
    AAIAgentCaptionEvent,
    AAIAgentTranscriptEvent,
    AAIAudioChunkEvent,
    AAIAudioDoneEvent,
    AAICancelledEvent,
    AAIErrorEvent,
    AAIReplyDoneEvent,
    AAISpeechStartedEvent,
    AAISpeechStoppedEvent,
    AAITimeoutEvent,
    AAIToolCallEvent,
    AAIUserTranscriptEvent,
)
from tau2.voice.audio_native.aai.provider import AAIVoiceAgentProvider
from tau2.voice.audio_native.aai.transcript_pacing import PacedUtteranceTranscript
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.tick_result import TickResult
from tau2.voice.utils.audio_preprocessing import (
    convert_to_pcm16,
    convert_to_ulaw,
    resample_audio,
)

# Experiment switch for the oracle client-side duck above.
DUCK_ON_USER_SPEECH = os.environ.get("TAU2_CLIENT_DUCK") == "1"
# Mean |amplitude| over a tick's PCM16 that counts as the caller speaking. Above
# the injected background noise of `--speech-complexity regular` and well below
# speech; logged per run so a threshold that ducks on every tick is visible
# rather than silently flattering the result.
USER_SPEECH_RMS_THRESHOLD = 400.0


# Whether to send `playback_progress` to the AAI host. On by default; set
# AAI_PLAYBACK_REPORT=0 to run the open-loop arm of an A/B. The frame is the
# host's only closed-loop view of playback — without it the host assumes every
# chunk it forwards starts playing on arrival at 1.0x, while this harness plays
# bytes_per_tick per tick and buffers the rest.
PLAYBACK_REPORT_ENABLED = os.environ.get("AAI_PLAYBACK_REPORT", "1").lower() not in (
    "0",
    "false",
    "no",
    "off",
)


class DiscreteTimeAAIAdapter(DiscreteTimeAdapter):
    """Discrete-time adapter for the aai voice-agent host provider."""

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = True,
        reasoning_effort: Optional[str] = None,
        provider: Optional[AAIVoiceAgentProvider] = None,
        system_prompt: str = "",
        tools: tuple = (),
    ):
        if reasoning_effort is not None:
            raise ValueError(
                f"aai provider does not support reasoning_effort "
                f"(got '{reasoning_effort}')"
            )
        # External interface stays tau2 telephony mu-law/8k (base default);
        # PCM16 conversion happens at the WebSocket boundary, not here.
        super().__init__(tick_duration_ms, send_audio_instant=send_audio_instant)
        # Set on connect from the rates the host announces; see
        # _apply_negotiated_rates.
        self._send_sample_rate: Optional[int] = None
        self._receive_audio_format: Optional[AudioFormat] = None
        self._chunk_size = 0
        self.system_prompt = system_prompt
        self.tools = tools
        self._provider = provider
        self._owns_provider = provider is None
        self._bg_loop = BackgroundAsyncLoop()
        # Oracle client-duck state. Held (not discarded) agent audio, plus a
        # duck-rate counter so a threshold that fires on every tick is visible.
        self._ducked_agent_audio: list = []
        self._duck_ticks = 0
        self._total_ticks = 0
        self._connected = False
        self._turn_index = 0
        # Turn ids interrupted by a barge-in (AAISpeechStartedEvent); their
        # reply_done is expected to carry zero audio/transcript, so the
        # loud-failure guard in _process_event skips them.
        self._interrupted_turn_ids: set[str] = set()
        # Turn ids whose text came from the live caption. Their committed
        # transcript is NOT allowed to replace it: the commit drops the
        # dead-air fillers the caller heard, so it no longer matches the audio.
        self._captioned_turn_ids: set[str] = set()

    @property
    def provider(self) -> AAIVoiceAgentProvider:
        if self._provider is None:
            self._provider = AAIVoiceAgentProvider(
                system_prompt=self.system_prompt,
                tools=self.tools,
            )
        return self._provider

    @property
    def is_connected(self) -> bool:
        return self._connected and self._bg_loop.is_running

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()
        self.system_prompt = system_prompt
        self.tools = tuple(tools)
        self._bg_loop.start()
        try:
            self._bg_loop.run_coroutine(
                self.provider.connect(),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._apply_negotiated_rates()
            self._connected = True
            logger.info(
                f"DiscreteTimeAAIAdapter connected "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(f"DiscreteTimeAAIAdapter failed to connect: {e}")
            self._bg_loop.stop()
            raise RuntimeError(f"Failed to connect to aai host: {e}") from e

    def disconnect(self) -> None:
        if not self._connected:
            return
        if self._bg_loop.is_running:
            try:
                self._bg_loop.run_coroutine(
                    self._async_disconnect(),
                    timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
                )
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
        self._bg_loop.stop()
        self._connected = False
        self._tick_count = 0
        self._ducked_agent_audio: list = []
        self._duck_ticks = 0
        self._total_ticks = 0
        self._cumulative_user_audio_ms = 0
        self.clear_buffers()
        if self._total_ticks:
            logger.info(
                f"client duck: {self._duck_ticks}/{self._total_ticks} ticks had user "
                f"speech ({100 * self._duck_ticks / self._total_ticks:.0f}%)"
            )
        logger.info("DiscreteTimeAAIAdapter disconnected")

    async def _async_disconnect(self) -> None:
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        if not self.is_connected:
            raise RuntimeError("Not connected to aai host. Call connect() first.")
        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1
        try:
            return self._bg_loop.run_coroutine(
                self._async_run_tick(user_audio, tick_number),
                timeout=self.tick_duration_ms / 1000
                + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
            )
        except Exception as e:
            logger.error(f"Error in run_tick (tick={tick_number}): {e}")
            raise

    async def _flush_pending_tool_results(self) -> None:
        # aai's tool_result auto-fires the next reply; no follow-up message.
        for (
            call_id,
            result_str,
            _request_response,
            _is_error,
        ) in self._pending_tool_results:
            await self.provider.send_tool_result(call_id, result_str)
        self._pending_tool_results.clear()

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        pcm16_send = self._convert_ulaw_to_pcm16(user_audio, self._send_sample_rate)

        # Close the host's playback loop before anything else this tick. What
        # is already in `result.agent_audio_chunks` at this point is precisely
        # the carry-over the base adapter prepended in step 3 — audio the host
        # sent us that we have not played yet. Without this the host cannot
        # observe that we consume slower than real time; see
        # `send_playback_progress` for what it costs when it cannot.
        backlog_bytes = sum(len(c) for c, _ in result.agent_audio_chunks)
        if backlog_bytes and PLAYBACK_REPORT_ENABLED:
            await self.provider.send_playback_progress(
                backlog_bytes / self.audio_format.bytes_per_second * 1000
            )

        async def receive_events():
            elapsed = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed)
            return await self.provider.receive_events_for_duration(remaining)

        _, events = await asyncio.gather(
            self._send_audio_chunked(
                pcm16_send, self.provider.send_audio, self._chunk_size
            ),
            receive_events(),
        )
        for event in events:
            self._process_event(result, event)

        # ── ORACLE CLIENT-SIDE DUCK (experiment) ────────────────────────────
        # Measures the CEILING of a client-side VAD, by giving this client the
        # one thing it does not have: instant local knowledge that the caller
        # started talking, and instant control of its own playout.
        #
        # Why a ceiling is worth measuring before building anything: the
        # server-side path cannot go faster than ~470ms (its first STT partial
        # is a model floor) plus whatever audio is already in this client's
        # buffer. A client that detects locally pays neither. If holding playout
        # the instant the caller speaks does NOT reduce case-(a) truncations,
        # no real VAD will either.
        #
        # Ducking, not discarding: held audio is played later, so a false
        # positive costs a pause rather than the rest of the reply — the same
        # reversibility argument the host-side duck rests on.
        #
        # NOT leaderboard-comparable. This changes the harness's own playout
        # semantics; the number it produces is an upper bound on what our agent
        # would gain from a capable client, not a tau2 score.
        if DUCK_ON_USER_SPEECH:
            if self._tick_has_user_speech(pcm16_send):
                self._ducked_agent_audio.extend(result.agent_audio_chunks)
                result.agent_audio_chunks.clear()
            elif self._ducked_agent_audio:
                result.agent_audio_chunks[:0] = self._ducked_agent_audio
                self._ducked_agent_audio = []

    def _apply_negotiated_rates(self) -> None:
        """Size the send path and receive format from the host's handshake."""
        send_rate = self.provider.input_sample_rate
        receive_rate = self.provider.tts_sample_rate
        if send_rate is None or receive_rate is None:
            raise RuntimeError("aai provider connected without negotiated rates")
        self._send_sample_rate = send_rate
        self._receive_audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=receive_rate, channels=1
        )
        self._chunk_size = int(send_rate * 2 * self._voip_interval_ms / 1000)

    @staticmethod
    def _convert_ulaw_to_pcm16(ulaw_8k: bytes, sample_rate: int) -> bytes:
        """Convert mu-law 8k user audio to PCM16 at the host's input rate."""
        audio = AudioData(data=ulaw_8k, format=TELEPHONY_AUDIO_FORMAT)
        pcm16 = convert_to_pcm16(audio)
        resampled = resample_audio(pcm16, sample_rate)
        return resampled.data

    @staticmethod
    def _convert_pcm16_to_ulaw_8k(pcm16: bytes, pcm_format: AudioFormat) -> bytes:
        """Convert PCM16 agent audio (from the aai host) to mu-law 8k."""
        audio = AudioData(data=pcm16, format=pcm_format)
        resampled = resample_audio(audio, 8000)
        ulaw = convert_to_ulaw(resampled)
        return ulaw.data

    def _tick_has_user_speech(self, pcm16: bytes) -> bool:
        """Oracle stand-in for a client-side VAD: is the caller talking now?"""
        if not pcm16:
            return False
        import array

        samples = array.array("h")
        samples.frombytes(pcm16[: len(pcm16) - (len(pcm16) % 2)])
        if not samples:
            return False
        level = sum(abs(v) for v in samples) / len(samples)
        speaking = level > USER_SPEECH_RMS_THRESHOLD
        self._duck_ticks += int(speaking)
        self._total_ticks += 1
        return speaking

    def _process_event(self, result: TickResult, event: Any) -> None:
        result.events.append(event)

        if isinstance(event, AAIAgentCaptionEvent):
            item_id = self._ensure_current_item_id()
            # The caption is what the caller hears; full-replacement snapshots,
            # so overwrite. It also arrives as text reaches TTS rather than at
            # the end of the reply, which is what lets the caller read along.
            self._utterance_transcript(item_id).transcript_received = event.text
            self._captioned_turn_ids.add(item_id)

        elif isinstance(event, AAIAgentTranscriptEvent):
            item_id = self._ensure_current_item_id()
            if item_id not in self._captioned_turn_ids:
                # aai sends the full transcript each time (not deltas); overwrite.
                self._utterance_transcript(item_id).transcript_received = event.text

        elif isinstance(event, AAIAudioChunkEvent):
            item_id = self._ensure_current_item_id()
            ulaw_bytes = self._convert_pcm16_to_ulaw_8k(
                event.pcm16, self._receive_audio_format
            )
            if result.skip_item_id is not None and item_id == result.skip_item_id:
                result.truncated_audio_bytes += len(ulaw_bytes)
                return
            if ulaw_bytes:
                result.agent_audio_chunks.append((ulaw_bytes, item_id))
            self._utterance_transcript(item_id).add_audio(len(ulaw_bytes))

        elif isinstance(event, AAIToolCallEvent):
            result.tool_calls.append(
                ToolCall(
                    id=event.tool_call_id,
                    name=event.tool_name,
                    arguments=event.args,
                )
            )
            logger.debug(f"Tool call detected: {event.tool_name}({event.tool_call_id})")

        elif isinstance(event, AAISpeechStartedEvent):
            # `speech_started` means "the agent yielded": the reply audio this
            # client still holds is for a floor the host gave up, so it is
            # flushed — `was_truncated` is what makes `buffer_excess_audio`
            # discard the backlog instead of carrying it to the next tick.
            #
            # It does NOT end the reply. The host holds this edge back while
            # its audio is on the line and releases it on a real barge-in only
            # together with `cancelled` — so a bare `speech_started` arrives
            # when no reply audio is playing, often while a reply is still
            # being generated (a caller's "hello?" during a tool chain).
            # Marking the current reply skipped here, as this adapter used to,
            # threw that whole upcoming reply away even though the host never
            # cancelled it. `cancelled` below is what ends a reply.
            logger.debug("Speech started - agent yields the floor")
            result.vad_events.append("speech_started")
            result.was_truncated = True
            # Held audio dies here too. A duck is "not yet", but `speech_started`
            # means the host really yielded and abandoned that reply — so
            # replaying what we held would put a dead turn back on the line
            # SECONDS late, out of context. Without this the experiment would
            # have measured stale speech resurfacing rather than ducking.
            if self._ducked_agent_audio:
                result.truncated_audio_bytes += sum(
                    len(c[0]) for c in self._ducked_agent_audio
                )
                self._ducked_agent_audio = []

        elif isinstance(event, AAICancelledEvent):
            # The host abandoned the in-flight reply (a barge-in). It is
            # terminal for that reply: no `reply_done` follows. So discard any
            # of its audio still arriving, and start the NEXT reply on a fresh
            # id — without the advance, the next reply inherited the skipped
            # id and every byte of it was discarded as "truncated", however
            # normally it finished (measured: 108s of one call unheard).
            logger.debug("Reply cancelled - interruption")
            result.was_truncated = True
            if self._current_item_id is not None:
                result.skip_item_id = self._current_item_id
                self._interrupted_turn_ids.add(self._current_item_id)
                self._turn_index += 1
                self._current_item_id = None

        elif isinstance(event, AAISpeechStoppedEvent):
            result.vad_events.append("speech_stopped")

        elif isinstance(event, AAIUserTranscriptEvent):
            # The turn the host's model was handed. Without it a trajectory
            # cannot show that a write tool followed a bare "Uh-huh." — only
            # the task log's debug lines could.
            result.provider_user_transcripts.append(event.text)

        elif isinstance(event, AAIReplyDoneEvent):
            item_id = self._current_item_id
            self._mark_audio_complete(item_id)
            was_interrupted = item_id in self._interrupted_turn_ids
            ut = self._utterance_transcripts.get(item_id) if item_id else None
            if ut is not None and not was_interrupted:
                if ut.audio_bytes_received == 0:
                    logger.warning(f"Reply {item_id} completed with no audio")
                if ut.transcript_received == "":
                    logger.warning(
                        f"Reply {item_id} completed with no transcript — "
                        "possible event schema mismatch"
                    )
            self._interrupted_turn_ids.discard(item_id)
            self._turn_index += 1
            self._current_item_id = None

        elif isinstance(event, AAIAudioDoneEvent):
            logger.debug("Audio playback done")
            self._mark_audio_complete(self._current_item_id)

        elif isinstance(event, AAIErrorEvent):
            logger.error(f"aai host error: {event.code} {event.message}")

        elif isinstance(event, AAITimeoutEvent):
            pass

        else:
            logger.debug(f"Event {type(event).__name__} received")

    def _utterance_transcript(self, item_id: str) -> PacedUtteranceTranscript:
        """The paced transcript for ``item_id``, created on first use."""
        return self._utterance_transcripts.setdefault(
            item_id,
            PacedUtteranceTranscript(
                item_id=item_id, bytes_per_second=self.audio_format.bytes_per_second
            ),
        )

    def _mark_audio_complete(self, item_id: Optional[str]) -> None:
        """No more audio is coming for ``item_id``: pace against its real total.

        Never for a CUT reply. Its received audio is the truncated part, not the
        utterance, so pacing against it reveals the unspoken rest of the text at
        once — the caller read "…couldn't locate the account. Please s" in the
        tick the audio stopped at "ZIP 2". A cut reply keeps the estimate for
        its whole text, which reveals nothing past what was played.
        """
        if item_id is None or item_id in self._interrupted_turn_ids:
            return
        ut = self._utterance_transcripts.get(item_id)
        if isinstance(ut, PacedUtteranceTranscript):
            ut.audio_complete = True

    def _ensure_current_item_id(self) -> str:
        """Return the synthetic turn id for the in-progress turn, creating it
        if this is the first audio/transcript event of the turn."""
        if self._current_item_id is None:
            self._current_item_id = f"turn-{self._turn_index}"
        return self._current_item_id
