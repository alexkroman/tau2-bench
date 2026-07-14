"""Tests for AssemblyAIVoiceAgentProvider.configure_session hardening.

Covers: (a) a stalled connection times out per-frame and configure_session
proceeds without raising, and (b) an early non-session frame received while
awaiting session.updated is buffered and surfaced by the next
receive_events_for_duration() call, instead of being silently dropped.
"""

import asyncio
import json

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
            # caller's own timeout should fire.
            await asyncio.sleep(10)
        return self._frames.pop(0)


def _provider():
    return AssemblyAIVoiceAgentProvider(api_key="test-key")


def test_configure_session_timeout_proceeds_without_raising(monkeypatch):
    # Shrink the timeout so the test doesn't actually wait 5s.
    monkeypatch.setattr(
        "tau2.voice.audio_native.assemblyai.provider.SESSION_UPDATE_FRAME_TIMEOUT",
        0.05,
    )
    provider = _provider()
    provider.ws = FakeWebSocket(frames=[])

    # Should NOT raise, just log a warning and return.
    asyncio.run(
        provider.configure_session(
            system_prompt="hi", tools=[], vad_config=AssemblyAIVADConfig()
        )
    )
    assert provider._buffered_events == []


def test_configure_session_buffers_early_frame_for_next_receive():
    provider = _provider()
    early_frame = json.dumps(
        {"type": "transcript.agent", "text": "hello", "reply_id": "r-1"}
    )
    updated_frame = json.dumps({"type": "session.updated", "session": {}})
    provider.ws = FakeWebSocket(frames=[early_frame, updated_frame])

    asyncio.run(
        provider.configure_session(
            system_prompt="hi", tools=[], vad_config=AssemblyAIVADConfig()
        )
    )

    # The early transcript.agent frame was parsed and buffered rather than
    # discarded, since session.updated hadn't arrived yet.
    assert len(provider._buffered_events) == 1
    assert provider._buffered_events[0].type == "transcript.agent"

    # The next receive_events_for_duration() call should surface it first,
    # then clear the buffer.
    provider.ws = FakeWebSocket(frames=[])
    events = asyncio.run(provider.receive_events_for_duration(0.02))
    assert len(events) == 1
    assert events[0].type == "transcript.agent"
    assert provider._buffered_events == []


def test_receive_events_for_duration_returns_and_clears_preseeded_buffer():
    from tau2.voice.audio_native.assemblyai.events import AAIUnknownEvent

    provider = _provider()
    seeded = AAIUnknownEvent(type="seeded")
    provider._buffered_events = [seeded]
    provider.ws = FakeWebSocket(frames=[])

    events = asyncio.run(provider.receive_events_for_duration(0.02))

    assert events[0] is seeded
    assert provider._buffered_events == []
