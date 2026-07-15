"""Discrete-time adapter for the AssemblyAI Voice Agent API.

Native G.711 μ-law (audio/pcmu) — no audio conversion. Mirrors the xAI adapter;
differences: agent transcript arrives as one full transcript.agent (not deltas),
and tool.result auto-fires the next reply (no reply.create follow-up).
"""

import asyncio
import base64
from typing import Any, List, Optional

from loguru import logger

from tau2.config import (
    DEFAULT_ASSEMBLYAI_VOICE,
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
)
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.assemblyai.events import (
    AAIAgentTranscriptEvent,
    AAIReplyAudioEvent,
    AAIReplyDoneEvent,
    AAIReplyStartedEvent,
    AAISessionErrorEvent,
    AAISpeechStartedEvent,
    AAISpeechStoppedEvent,
    AAITimeoutEvent,
    AAIToolCallEvent,
    AAIUserTranscriptEvent,
)
from tau2.voice.audio_native.assemblyai.provider import (
    AssemblyAIAudioFormat,
    AssemblyAIVADConfig,
    AssemblyAIVoiceAgentProvider,
)
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript


class DiscreteTimeAssemblyAIAdapter(DiscreteTimeAdapter):
    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = True,
        reasoning_effort: Optional[str] = None,
        provider: Optional[AssemblyAIVoiceAgentProvider] = None,
        voice: str = DEFAULT_ASSEMBLYAI_VOICE,
    ):
        if reasoning_effort is not None:
            raise ValueError(
                f"AssemblyAI provider does not support reasoning_effort "
                f"(got '{reasoning_effort}')"
            )
        super().__init__(tick_duration_ms, send_audio_instant=send_audio_instant)
        self._chunk_size = int(
            self.audio_format.bytes_per_second * self._voip_interval_ms / 1000
        )
        self.voice = voice
        self._provider = provider
        self._owns_provider = provider is None
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False

    @property
    def provider(self) -> AssemblyAIVoiceAgentProvider:
        if self._provider is None:
            self._provider = AssemblyAIVoiceAgentProvider(
                voice=self.voice, audio_format=AssemblyAIAudioFormat.PCMU
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
        if vad_config is None:
            vad_config = AssemblyAIVADConfig()
        self._bg_loop.start()
        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools, vad_config),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                f"DiscreteTimeAssemblyAIAdapter connected "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(f"DiscreteTimeAssemblyAIAdapter failed to connect: {e}")
            self._bg_loop.stop()
            raise RuntimeError(f"Failed to connect to AssemblyAI API: {e}") from e

    async def _async_connect(self, system_prompt, tools, vad_config) -> None:
        await self.provider.connect()
        await self.provider.configure_session(
            system_prompt=system_prompt,
            tools=tools,
            vad_config=vad_config,
            voice=self.voice,
        )

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
        self._cumulative_user_audio_ms = 0
        self.clear_buffers()
        logger.info("DiscreteTimeAssemblyAIAdapter disconnected")

    async def _async_disconnect(self) -> None:
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        if not self.is_connected:
            raise RuntimeError("Not connected to AssemblyAI API. Call connect() first.")
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
        # AssemblyAI tool.result auto-fires the next reply; no reply.create.
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
        async def receive_events():
            elapsed = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed)
            return await self.provider.receive_events_for_duration(remaining)

        _, events = await asyncio.gather(
            self._send_audio_chunked(
                user_audio, self.provider.send_audio, self._chunk_size
            ),
            receive_events(),
        )
        for event in events:
            self._process_event(result, event)

    def _process_event(self, result: TickResult, event: Any) -> None:
        result.events.append(event)

        if isinstance(event, AAIReplyStartedEvent):
            # A fresh reply means any prior barge-in is resolved. Clear the skip
            # target so this reply is heard — reply.audio frames carry no
            # reply_id and fall back to _current_item_id, so a stale skip target
            # would otherwise swallow the entire new reply (see test 3 mute bug).
            result.skip_item_id = None
            if event.reply_id:
                self._current_item_id = event.reply_id
                self._utterance_transcripts.setdefault(
                    event.reply_id, UtteranceTranscript(item_id=event.reply_id)
                )

        elif isinstance(event, AAIReplyAudioEvent):
            item_id = event.reply_id or self._current_item_id
            if result.skip_item_id is not None and item_id == result.skip_item_id:
                audio_bytes = base64.b64decode(event.audio) if event.audio else b""
                result.truncated_audio_bytes += len(audio_bytes)
                return
            audio_bytes = base64.b64decode(event.audio) if event.audio else b""
            if audio_bytes:
                result.agent_audio_chunks.append((audio_bytes, item_id))
            if item_id:
                self._current_item_id = item_id
                ut = self._utterance_transcripts.setdefault(
                    item_id, UtteranceTranscript(item_id=item_id)
                )
                ut.add_audio(len(audio_bytes))

        elif isinstance(event, AAIAgentTranscriptEvent):
            item_id = event.reply_id or self._current_item_id
            if item_id and event.text:
                ut = self._utterance_transcripts.setdefault(
                    item_id, UtteranceTranscript(item_id=item_id)
                )
                # Full text arrives once (not deltas); overwrite to avoid dupes.
                ut.transcript_received = event.text

        elif isinstance(event, AAISpeechStartedEvent):
            logger.debug("Speech started - interruption detected")
            result.vad_events.append("speech_started")
            if self._buffered_agent_audio:
                buffered = sum(len(c[0]) for c in self._buffered_agent_audio)
                result.truncated_audio_bytes += buffered
                self._buffered_agent_audio.clear()
            result.was_truncated = True
            result.skip_item_id = self._current_item_id

        elif isinstance(event, AAISpeechStoppedEvent):
            result.vad_events.append("speech_stopped")
            # User finished talking: whatever the agent says next is its reply to
            # this turn and must be heard. Clear the barge-in skip target — a
            # null-reply_id audio frame falls back to the stale interrupted item,
            # so leaving skip set here mutes the agent for the rest of the call.
            result.skip_item_id = None

        elif isinstance(event, AAIToolCallEvent):
            result.tool_calls.append(
                ToolCall(id=event.call_id, name=event.name, arguments=event.arguments)
            )
            logger.debug(f"Tool call detected: {event.name}({event.call_id})")

        elif isinstance(event, AAIReplyDoneEvent):
            if event.status == "interrupted":
                self._pending_tool_results.clear()
                logger.debug(f"Reply done (status={event.status})")
                return
            reply_id = event.reply_id or self._current_item_id
            ut = self._utterance_transcripts.get(reply_id) if reply_id else None
            if ut is not None:
                if ut.audio_bytes_received == 0:
                    logger.warning(f"Reply {reply_id} completed with no audio")
                if ut.transcript_received == "":
                    logger.warning(
                        f"Reply {reply_id} completed with no transcript — "
                        "possible event schema mismatch"
                    )
            logger.debug(f"Reply done (status={event.status})")

        elif isinstance(event, AAIUserTranscriptEvent):
            logger.debug(f"Input transcription: {event.text}")

        elif isinstance(event, AAISessionErrorEvent):
            logger.error(f"AssemblyAI session error: {event.code} {event.message}")

        elif isinstance(event, AAITimeoutEvent):
            pass

        else:
            logger.debug(f"Event {type(event).__name__} received")
