"""Tests for the AAI discrete-time adapter (PCM16 <-> mu-law conversion).

Unlike AssemblyAI's native G.711 mu-law transport, the aai voice-agent host
speaks PCM16 over its WebSocket at the rates it announces in its handshake. The
adapter's external interface stays tau2 telephony mu-law/8k, so conversion
happens at this boundary: send path (mu-law 8k -> PCM16 at the host's input
rate) and receive path (PCM16 at the host's output rate -> mu-law 8k).
"""

import asyncio
import logging

import pytest
from loguru import logger

from tau2.voice.audio_native.aai.discrete_time_adapter import DiscreteTimeAAIAdapter
from tau2.voice.audio_native.aai.events import (
    AAIAgentTranscriptEvent,
    AAIAudioChunkEvent,
    AAICancelledEvent,
    AAIReplyDoneEvent,
    AAISpeechStartedEvent,
    AAISpeechStoppedEvent,
    AAIToolCallEvent,
)
from tau2.voice.audio_native.tick_result import TickResult


class _PropagateHandler(logging.Handler):
    """Forward loguru records into stdlib logging so pytest's ``caplog``
    fixture (which only listens on stdlib logging) can capture them."""

    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


@pytest.fixture
def caplog(caplog: pytest.LogCaptureFixture):
    handler_id = logger.add(_PropagateHandler(), format="{message}")
    yield caplog
    logger.remove(handler_id)


def _adapter(
    input_rate: int = 16000, output_rate: int = 24000
) -> DiscreteTimeAAIAdapter:
    """An adapter as it stands after a handshake announcing these rates."""
    a = DiscreteTimeAAIAdapter(tick_duration_ms=200, send_audio_instant=True)
    a.provider.input_sample_rate = input_rate
    a.provider.tts_sample_rate = output_rate
    a._apply_negotiated_rates()
    return a


def _result() -> TickResult:
    return TickResult(
        tick_number=1,
        audio_sent_bytes=0,
        audio_sent_duration_ms=0.0,
        bytes_per_tick=1600,
        bytes_per_second=8000,
    )


def test_reasoning_effort_rejected() -> None:
    with pytest.raises(ValueError):
        DiscreteTimeAAIAdapter(tick_duration_ms=200, reasoning_effort="high")


def test_audio_chunk_converted_from_pcm16_24k_to_ulaw_8k() -> None:
    a = _adapter()
    r = _result()
    # ~20ms of PCM16 audio at 24kHz (480 samples, 2 bytes/sample).
    pcm16_24k = (b"\x10\x00") * 480
    a._process_event(r, AAIAudioChunkEvent(pcm16=pcm16_24k))

    assert r.agent_audio_bytes > 0
    assert len(r.agent_audio_chunks) == 1
    ulaw_bytes, item_id = r.agent_audio_chunks[0]
    # mu-law is 1 byte/sample; 24k->8k resample should yield ~1/3 the samples.
    assert len(ulaw_bytes) == pytest.approx(160, abs=5)
    assert item_id == a._current_item_id


def test_agent_transcript_overwrites_not_appends() -> None:
    a = _adapter()
    r = _result()
    a._process_event(r, AAIAgentTranscriptEvent(text="first text"))
    a._process_event(r, AAIAgentTranscriptEvent(text="second text"))

    turn_id = a._current_item_id
    assert turn_id is not None
    assert a._utterance_transcripts[turn_id].transcript_received == "second text"


def test_tool_call_recorded_with_id_name_arguments() -> None:
    a = _adapter()
    r = _result()
    a._process_event(
        r,
        AAIToolCallEvent(tool_call_id="c-1", tool_name="lookup", args={"x": 1}),
    )

    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].id == "c-1"
    assert r.tool_calls[0].name == "lookup"
    assert r.tool_calls[0].arguments == {"x": 1}


def test_speech_started_flushes_backlog_but_does_not_skip_the_reply() -> None:
    """`speech_started` flushes what the client holds (`was_truncated` makes
    `buffer_excess_audio` discard the backlog) but does not end the reply:
    the host releases it bare only when no reply audio is on the line, often
    while a reply is still being generated."""
    a = _adapter()
    r = _result()
    a._process_event(r, AAIAgentTranscriptEvent(text="let me check that"))

    a._process_event(r, AAISpeechStartedEvent())

    assert r.was_truncated is True
    assert "speech_started" in r.vad_events
    assert r.skip_item_id is None


def test_reply_audio_after_a_bare_speech_started_is_played() -> None:
    """The regression: a caller's "hello?" while the reply was still being
    generated used to mark that reply skipped, discarding all of it."""
    a = _adapter()
    r = _result()
    a._process_event(r, AAIAgentTranscriptEvent(text="let me check that"))
    a._process_event(r, AAISpeechStartedEvent())

    r2 = _result()
    r2.skip_item_id = r.skip_item_id  # carried over, as the base adapter does
    a._process_event(r2, AAIAudioChunkEvent(pcm16=(b"\x10\x00") * 480))

    assert r2.agent_audio_chunks != []
    assert r2.truncated_audio_bytes == 0


def test_cancelled_skips_the_reply_and_starts_the_next_on_a_fresh_id() -> None:
    """`cancelled` is terminal (no reply_done follows): stragglers of the
    cancelled reply are discarded, and the next reply must NOT inherit its id —
    it did, and every byte of it was then discarded as truncated."""
    a = _adapter()
    r = _result()
    a._process_event(r, AAIAgentTranscriptEvent(text="your order is"))
    cancelled_id = a._current_item_id
    a._process_event(r, AAISpeechStartedEvent())
    a._process_event(r, AAICancelledEvent())

    assert r.skip_item_id == cancelled_id
    assert cancelled_id in a._interrupted_turn_ids
    assert a._current_item_id is None

    # The next reply's audio plays.
    r2 = _result()
    r2.skip_item_id = r.skip_item_id
    a._process_event(r2, AAIAgentTranscriptEvent(text="sure, go ahead"))
    a._process_event(r2, AAIAudioChunkEvent(pcm16=(b"\x10\x00") * 480))
    assert a._current_item_id != cancelled_id
    assert r2.agent_audio_chunks != []
    assert r2.truncated_audio_bytes == 0


def test_speech_stopped_adds_vad_event() -> None:
    a = _adapter()
    r = _result()
    a._process_event(r, AAISpeechStoppedEvent())

    assert "speech_stopped" in r.vad_events


def test_barge_in_skip_discards_subsequent_audio_chunk() -> None:
    a = _adapter()
    r = _result()
    a._process_event(r, AAIAgentTranscriptEvent(text="hi"))
    current_item_id = a._current_item_id
    r.skip_item_id = current_item_id

    pcm16_24k = (b"\x10\x00") * 480
    a._process_event(r, AAIAudioChunkEvent(pcm16=pcm16_24k))

    assert r.agent_audio_chunks == []
    assert r.truncated_audio_bytes > 0


def test_reply_done_on_non_interrupted_empty_turn_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A genuinely empty, non-interrupted turn is a real schema-drift
    signal and should still trigger the loud-failure warning."""
    a = _adapter()
    r = _result()
    # Starts the turn (via transcript-less ensure) with zero audio and no
    # transcript ever recorded for it.
    item_id = a._ensure_current_item_id()
    from tau2.voice.audio_native.tick_result import UtteranceTranscript

    a._utterance_transcripts[item_id] = UtteranceTranscript(item_id=item_id)

    with caplog.at_level("WARNING"):
        a._process_event(r, AAIReplyDoneEvent())

    assert any("no audio" in msg for msg in caplog.messages)
    assert any("no transcript" in msg for msg in caplog.messages)


def test_reply_done_on_interrupted_turn_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """On a normal barge-in, the interrupted turn legitimately has zero
    audio (chunks were discarded via skip_item_id); the loud-failure guard
    must not fire a spurious warning for it."""
    a = _adapter()
    r = _result()
    item_id = a._ensure_current_item_id()

    # A cancelled barge-in marks the turn interrupted, and a late reply_done
    # (should a host send one) must not warn about its missing audio.
    a._process_event(r, AAICancelledEvent())
    assert item_id in a._interrupted_turn_ids
    a._current_item_id = item_id
    caplog.clear()

    with caplog.at_level("WARNING"):
        a._process_event(r, AAIReplyDoneEvent())

    assert caplog.messages == []
    # Turn state should still advance normally and the id should be
    # discarded to avoid unbounded growth of the interrupted-turn set.
    assert a._current_item_id is None
    assert item_id not in a._interrupted_turn_ids


def test_flush_pending_tool_results_sends_tool_result_only() -> None:
    """_flush_pending_tool_results should forward queued tool results to
    the provider's send_tool_result and send nothing else."""

    class _FakeProvider:
        def __init__(self) -> None:
            self.tool_results: list[tuple[str, str]] = []
            self.other_calls: list[str] = []

        async def send_tool_result(self, call_id: str, result: str) -> None:
            self.tool_results.append((call_id, result))

    fake_provider = _FakeProvider()
    a = DiscreteTimeAAIAdapter(tick_duration_ms=200, provider=fake_provider)
    a.send_tool_result("call-1", "42")

    asyncio.run(a._flush_pending_tool_results())

    assert fake_provider.tool_results == [("call-1", "42")]
    assert fake_provider.other_calls == []
    assert a._pending_tool_results == []


@pytest.mark.parametrize("rate,factor", [(16000, 4), (24000, 6)])
def test_convert_ulaw_8k_silence_to_pcm16_scales_with_host_rate(
    rate: int, factor: int
) -> None:
    """mu-law is 1 byte/sample at 8kHz; PCM16 is 2 bytes/sample, so the byte
    count grows by 2 * rate / 8000 — whatever rate the host announced (16k for
    a pipeline agent, 24k for S2S)."""
    ulaw_silence_8k = b"\xff" * 160  # 20ms of mu-law silence at 8kHz.

    pcm16_send = DiscreteTimeAAIAdapter._convert_ulaw_to_pcm16(ulaw_silence_8k, rate)

    assert len(pcm16_send) == pytest.approx(160 * factor, abs=8)


def test_negotiated_rates_size_the_send_chunk() -> None:
    """The VoIP chunk is sized from the host's input rate, not a constant."""
    assert _adapter(input_rate=16000)._chunk_size * 3 == (
        _adapter(input_rate=24000)._chunk_size * 2
    )


def test_apply_negotiated_rates_requires_a_handshake() -> None:
    a = DiscreteTimeAAIAdapter(tick_duration_ms=200)
    with pytest.raises(RuntimeError):
        a._apply_negotiated_rates()
