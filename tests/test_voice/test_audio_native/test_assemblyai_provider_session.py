"""Tests for AssemblyAIVoiceAgentProvider.configure_session handshake.

The Voice Agent API does not emit session.ready unprompted: the client sends
session.update, and the server replies with session.updated (config echoed)
then session.ready (agent live). configure_session waits for session.ready,
treating session.updated as intermediate.

Covers: (a) a completed handshake returns and records session_id; (b) an early
non-handshake frame is buffered (not dropped) and surfaced by the next
receive_events_for_duration(); (c) a stalled connection with no response raises
a clear error rather than hanging.
"""

import asyncio
import json

import pytest
from websockets.protocol import State

from tau2.voice.audio_native.assemblyai.provider import (
    AssemblyAIVADConfig,
    AssemblyAIVoiceAgentProvider,
)


class FakeWebSocket:
    """Minimal async fake matching the websockets client surface we rely on."""

    def __init__(self, frames):
        self.state = State.OPEN
        self._frames = list(frames)
        self.sent = []

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        if not self._frames:
            # Simulate a stalled connection: nothing to hand back before the
            # caller's own per-frame timeout should fire.
            await asyncio.sleep(10)
        return self._frames.pop(0)


def _provider():
    return AssemblyAIVoiceAgentProvider(api_key="test-key")


def test_configure_session_completes_on_session_ready():
    provider = _provider()
    updated = json.dumps({"type": "session.updated", "config": {"id": "sess_abc"}})
    ready = json.dumps({"type": "session.ready", "session_id": "sess_abc"})
    provider.ws = FakeWebSocket(frames=[updated, ready])

    asyncio.run(
        provider.configure_session(
            system_prompt="hi", tools=[], vad_config=AssemblyAIVADConfig()
        )
    )

    # A session.update was sent, and session_id was captured from the handshake.
    assert len(provider.ws.sent) == 1
    assert json.loads(provider.ws.sent[0])["type"] == "session.update"
    assert provider.session_id == "sess_abc"
    assert provider._buffered_events == []


def test_configure_session_buffers_early_frame_for_next_receive():
    provider = _provider()
    early_frame = json.dumps(
        {"type": "transcript.agent", "text": "hello", "reply_id": "r-1"}
    )
    updated = json.dumps({"type": "session.updated", "config": {"id": "s1"}})
    ready = json.dumps({"type": "session.ready", "session_id": "s1"})
    # An early conversation frame arrives before the handshake completes.
    provider.ws = FakeWebSocket(frames=[early_frame, updated, ready])

    asyncio.run(
        provider.configure_session(
            system_prompt="hi", tools=[], vad_config=AssemblyAIVADConfig()
        )
    )

    # The early transcript.agent frame was parsed and buffered, not dropped.
    assert len(provider._buffered_events) == 1
    assert provider._buffered_events[0].type == "transcript.agent"

    # The next receive_events_for_duration() surfaces it first, then clears.
    provider.ws = FakeWebSocket(frames=[])
    events = asyncio.run(provider.receive_events_for_duration(0.02))
    assert len(events) == 1
    assert events[0].type == "transcript.agent"
    assert provider._buffered_events == []


def test_configure_session_raises_when_no_response(monkeypatch):
    # Shrink the timeout so the test doesn't actually wait 5s.
    monkeypatch.setattr(
        "tau2.voice.audio_native.assemblyai.provider.SESSION_UPDATE_FRAME_TIMEOUT",
        0.05,
    )
    provider = _provider()
    provider.ws = FakeWebSocket(frames=[])  # server never responds

    # No session.updated/ready arrives -> clear error, not a silent proceed
    # (which previously surfaced as an empty asyncio.TimeoutError).
    with pytest.raises(RuntimeError, match="did not initialize"):
        asyncio.run(
            provider.configure_session(
                system_prompt="hi", tools=[], vad_config=AssemblyAIVADConfig()
            )
        )


def test_receive_events_for_duration_returns_and_clears_preseeded_buffer():
    from tau2.voice.audio_native.assemblyai.events import AAIUnknownEvent

    provider = _provider()
    seeded = AAIUnknownEvent(type="seeded")
    provider._buffered_events = [seeded]
    provider.ws = FakeWebSocket(frames=[])

    events = asyncio.run(provider.receive_events_for_duration(0.02))

    assert events[0] is seeded
    assert provider._buffered_events == []
