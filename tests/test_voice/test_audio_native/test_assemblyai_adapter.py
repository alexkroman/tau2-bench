import asyncio
import base64

from tau2.voice.audio_native.assemblyai.discrete_time_adapter import (
    DiscreteTimeAssemblyAIAdapter,
)
from tau2.voice.audio_native.assemblyai.events import (
    AAIAgentTranscriptEvent,
    AAIReplyAudioEvent,
    AAIReplyDoneEvent,
    AAIReplyStartedEvent,
    AAISpeechStartedEvent,
    AAISpeechStoppedEvent,
    AAIToolCallEvent,
)
from tau2.voice.audio_native.tick_result import TickResult


def _adapter():
    return DiscreteTimeAssemblyAIAdapter(tick_duration_ms=200, send_audio_instant=True)


def _result():
    return TickResult(
        tick_number=1,
        audio_sent_bytes=0,
        audio_sent_duration_ms=0.0,
        bytes_per_tick=1600,
        bytes_per_second=8000,
    )


def test_reasoning_effort_rejected():
    import pytest

    with pytest.raises(ValueError):
        DiscreteTimeAssemblyAIAdapter(tick_duration_ms=200, reasoning_effort="high")


def test_reply_audio_appended_and_transcript_set():
    a = _adapter()
    r = _result()
    a._process_event(r, AAIReplyStartedEvent(reply_id="r-1"))
    a._process_event(
        r,
        AAIReplyAudioEvent(
            audio=base64.b64encode(b"\xff" * 40).decode(), reply_id="r-1"
        ),
    )
    a._process_event(r, AAIAgentTranscriptEvent(text="hi there", reply_id="r-1"))
    assert r.agent_audio_bytes == 40
    assert a._utterance_transcripts["r-1"].transcript_received == "hi there"


def test_tool_call_recorded_with_arguments():
    a = _adapter()
    r = _result()
    a._process_event(
        r, AAIToolCallEvent(call_id="c-1", name="lookup", arguments={"x": 1})
    )
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].id == "c-1"
    assert r.tool_calls[0].name == "lookup"
    assert r.tool_calls[0].arguments == {"x": 1}


def test_barge_in_marks_truncation():
    a = _adapter()
    r = _result()
    a._process_event(r, AAIReplyStartedEvent(reply_id="r-1"))
    a._process_event(r, AAISpeechStartedEvent())
    assert r.was_truncated is True
    assert "speech_started" in r.vad_events
    assert r.skip_item_id == "r-1"


def test_reply_done_interrupted_discards_pending_tools():
    a = _adapter()
    r = _result()
    a.send_tool_result("c-1", "{}")
    a._process_event(r, AAIReplyDoneEvent(reply_id="r-1", status="interrupted"))
    assert a._pending_tool_results == []


def test_agent_transcript_overwrites_not_appends():
    a = _adapter()
    r = _result()
    a._process_event(r, AAIReplyStartedEvent(reply_id="r-1"))
    a._process_event(r, AAIAgentTranscriptEvent(text="first text", reply_id="r-1"))
    a._process_event(r, AAIAgentTranscriptEvent(text="second text", reply_id="r-1"))
    # transcript.agent carries the full text each time (not deltas); the
    # handler must overwrite, not concatenate.
    assert a._utterance_transcripts["r-1"].transcript_received == "second text"


def test_skip_item_id_discards_subsequent_audio():
    a = _adapter()
    r1 = _result()
    a._process_event(r1, AAIReplyStartedEvent(reply_id="r-1"))
    a._process_event(r1, AAISpeechStartedEvent())
    assert r1.skip_item_id == "r-1"

    # A fresh tick whose skip_item_id carries over the truncated item.
    r2 = _result()
    r2.skip_item_id = "r-1"
    audio_bytes = b"\xab" * 24
    a._process_event(
        r2,
        AAIReplyAudioEvent(
            audio=base64.b64encode(audio_bytes).decode(), reply_id="r-1"
        ),
    )
    assert r2.agent_audio_chunks == []
    assert r2.truncated_audio_bytes == len(audio_bytes)


def test_speech_stopped_clears_skip_so_next_reply_is_heard():
    """Regression (aai_retail_quick task 3): after a barge-in, once the user
    stops speaking the agent's next reply must be heard.

    ``reply.audio`` frames carry ``reply_id=None`` on the wire, so item_id
    falls back to the stale ``_current_item_id`` (the interrupted reply). If the
    barge-in ``skip_item_id`` is never cleared, every subsequent chunk matches it
    and the agent is muted for the rest of the session — 148,900 bytes of reply
    audio were silently discarded and the agent never spoke again.
    """
    a = _adapter()
    r = _result()
    # Greeting reply r-1 is playing; _current_item_id becomes r-1.
    a._process_event(r, AAIReplyStartedEvent(reply_id="r-1"))
    a._process_event(
        r, AAIReplyAudioEvent(audio=base64.b64encode(b"\xff" * 20).decode(), reply_id="r-1")
    )
    # User barges in, then stops talking.
    a._process_event(r, AAISpeechStartedEvent())
    assert r.skip_item_id == "r-1"
    a._process_event(r, AAISpeechStoppedEvent())
    assert r.skip_item_id is None

    # Agent's real response arrives with a null reply_id (falls back to r-1).
    # It must play, not be truncated.
    audio = b"\xcd" * 16
    a._process_event(
        r, AAIReplyAudioEvent(audio=base64.b64encode(audio).decode(), reply_id=None)
    )
    # 20 greeting bytes + 16 new reply bytes; nothing truncated.
    assert r.agent_audio_bytes == 36
    assert r.truncated_audio_bytes == 0


def test_new_reply_started_clears_stale_skip():
    """A fresh reply.started means the barge-in is resolved: the stale skip
    target must be cleared so a following null-reply_id audio chunk (which falls
    back to the new current item) is not wrongly discarded."""
    a = _adapter()
    r = _result()
    a._process_event(r, AAIReplyStartedEvent(reply_id="r-1"))
    a._process_event(r, AAISpeechStartedEvent())
    assert r.skip_item_id == "r-1"
    a._process_event(r, AAIReplyStartedEvent(reply_id="r-2"))
    assert r.skip_item_id is None


class FakeAssemblyAIProvider:
    """Fake provider for testing tool result flushing."""

    def __init__(self):
        self.send_tool_result_calls = []

    async def send_tool_result(self, call_id: str, result: str) -> None:
        """Record tool result calls."""
        self.send_tool_result_calls.append((call_id, result))


def test_flush_sends_tool_result_without_reply_create():
    fake_provider = FakeAssemblyAIProvider()
    a = DiscreteTimeAssemblyAIAdapter(
        tick_duration_ms=200, send_audio_instant=True, provider=fake_provider
    )
    a.send_tool_result("c-1", "{}")
    assert len(a._pending_tool_results) == 1

    # Run the async flush
    asyncio.run(a._flush_pending_tool_results())

    # Assert the fake provider received exactly one call
    assert len(fake_provider.send_tool_result_calls) == 1
    assert fake_provider.send_tool_result_calls[0] == ("c-1", "{}")
    # Assert pending results were cleared
    assert a._pending_tool_results == []
