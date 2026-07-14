# AssemblyAI Voice Agent audio-native provider — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `assemblyai` provider to tau2-bench voice full-duplex mode that benchmarks AssemblyAI's managed Voice Agent API end-to-end.

**Architecture:** A new native provider under `src/tau2/voice/audio_native/assemblyai/`, subclassing `DiscreteTimeAdapter` and modeled on the existing `xai` provider. Like xAI it speaks G.711 μ-law 8 kHz (`audio/pcmu`) natively, so no audio conversion is needed. Model is endpoint-determined (AssemblyAI's managed LLM).

**Tech Stack:** Python 3.12+, `websockets`, Pydantic v2, `loguru`, pytest, `uv`.

## Global Constraints

- Python `>=3.12, <3.14`; run everything via `uv run`.
- Follow the audio-native template method (`src/tau2/voice/audio_native/AGENTS.md`): implement `_execute_tick()` + `_flush_pending_tool_results()` only; never a custom `_async_run_tick()`.
- Import constants from `src/tau2/config.py` directly; never re-declare them.
- WebSocket URL `wss://agents.assemblyai.com/v1/ws`; auth header `Authorization: Bearer $ASSEMBLYAI_API_KEY` (Bearer prefix — unlike AssemblyAI's REST API).
- Audio encoding `audio/pcmu` (G.711 μ-law, 8 kHz) on both input and output.
- Tool `tool.call` argument dict field is `arguments` (never `args`).
- `tool.result` auto-fires the next reply — never follow it with a `reply.create`/`response.create`.
- `greeting` is omitted (it bypasses the LLM straight to TTS and would corrupt the benchmark).
- Model is endpoint-determined; `reasoning_effort` is unsupported (must be `None`).
- Reference spec: `docs/superpowers/specs/2026-07-14-assemblyai-voice-agent-provider-design.md`.

---

### Task 1: Event models (`events.py`)

**Files:**
- Create: `src/tau2/voice/audio_native/assemblyai/__init__.py` (empty for now)
- Create: `src/tau2/voice/audio_native/assemblyai/events.py`
- Test: `tests/test_voice/test_audio_native/test_assemblyai_events.py`

**Interfaces:**
- Produces: `parse_assemblyai_event(data: dict) -> BaseAAIEvent`; event classes `AAISessionReadyEvent`, `AAISessionUpdatedEvent`, `AAISpeechStartedEvent`, `AAISpeechStoppedEvent`, `AAIUserTranscriptDeltaEvent`, `AAIUserTranscriptEvent`, `AAIReplyStartedEvent`, `AAIReplyAudioEvent` (field `audio: str`, `reply_id: Optional[str]`), `AAIAgentTranscriptEvent` (fields `text: str`, `interrupted: bool`, `reply_id: Optional[str]`), `AAIReplyDoneEvent` (fields `reply_id: Optional[str]`, `status: Optional[str]`), `AAIToolCallEvent` (fields `call_id: str`, `name: str`, `arguments: dict`), `AAISessionErrorEvent` (fields `code`, `message`, `param`, `timestamp`), `AAITimeoutEvent`, `AAIUnknownEvent`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_voice/test_audio_native/test_assemblyai_events.py
from tau2.voice.audio_native.assemblyai.events import (
    AAIAgentTranscriptEvent,
    AAIReplyAudioEvent,
    AAISessionReadyEvent,
    AAIToolCallEvent,
    AAIUnknownEvent,
    parse_assemblyai_event,
)


def test_parse_session_ready():
    ev = parse_assemblyai_event({"type": "session.ready", "session_id": "s-123"})
    assert isinstance(ev, AAISessionReadyEvent)
    assert ev.session_id == "s-123"


def test_parse_reply_audio_keeps_base64():
    ev = parse_assemblyai_event(
        {"type": "reply.audio", "audio": "QUJD", "reply_id": "r-1"}
    )
    assert isinstance(ev, AAIReplyAudioEvent)
    assert ev.audio == "QUJD"
    assert ev.reply_id == "r-1"


def test_parse_agent_transcript_interrupted():
    ev = parse_assemblyai_event(
        {"type": "transcript.agent", "text": "hello", "interrupted": True, "reply_id": "r-1"}
    )
    assert isinstance(ev, AAIAgentTranscriptEvent)
    assert ev.text == "hello"
    assert ev.interrupted is True


def test_parse_tool_call_uses_arguments():
    ev = parse_assemblyai_event(
        {
            "type": "tool.call",
            "call_id": "c-1",
            "name": "lookup_order",
            "arguments": {"order_id": "ORD-1"},
        }
    )
    assert isinstance(ev, AAIToolCallEvent)
    assert ev.call_id == "c-1"
    assert ev.name == "lookup_order"
    assert ev.arguments == {"order_id": "ORD-1"}


def test_parse_unknown_event():
    ev = parse_assemblyai_event({"type": "does.not.exist", "foo": 1})
    assert isinstance(ev, AAIUnknownEvent)
    assert ev.raw == {"type": "does.not.exist", "foo": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_voice/test_audio_native/test_assemblyai_events.py -v`
Expected: FAIL with `ModuleNotFoundError: ...assemblyai.events`

- [ ] **Step 3: Create the package marker**

```python
# src/tau2/voice/audio_native/assemblyai/__init__.py
"""AssemblyAI Voice Agent API integration for audio-native voice processing."""
```

- [ ] **Step 4: Write `events.py`**

```python
# src/tau2/voice/audio_native/assemblyai/events.py
"""Pydantic models for AssemblyAI Voice Agent API server events.

Reference: https://www.assemblyai.com/docs/voice-agents/voice-agent-api

Field names for some events (reply.audio audio field, transcript delta text
field) are modeled from the docs and verified against the live API in the
standalone smoke test. Models ignore unknown fields so parsing never fails.
"""

from typing import Any, Dict, Literal, Optional, Union

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class BaseAAIEvent(BaseModel):
    """Base class for all AssemblyAI Voice Agent events."""

    model_config = ConfigDict(extra="ignore")
    type: str


class AAISessionReadyEvent(BaseAAIEvent):
    type: Literal["session.ready"] = "session.ready"
    session_id: Optional[str] = None


class AAISessionUpdatedEvent(BaseAAIEvent):
    type: Literal["session.updated"] = "session.updated"
    session: Optional[Dict[str, Any]] = None


class AAISpeechStartedEvent(BaseAAIEvent):
    type: Literal["input.speech.started"] = "input.speech.started"


class AAISpeechStoppedEvent(BaseAAIEvent):
    type: Literal["input.speech.stopped"] = "input.speech.stopped"


class AAIUserTranscriptDeltaEvent(BaseAAIEvent):
    type: Literal["transcript.user.delta"] = "transcript.user.delta"
    text: str = ""
    item_id: Optional[str] = None


class AAIUserTranscriptEvent(BaseAAIEvent):
    type: Literal["transcript.user"] = "transcript.user"
    text: str = ""
    item_id: Optional[str] = None


class AAIReplyStartedEvent(BaseAAIEvent):
    type: Literal["reply.started"] = "reply.started"
    reply_id: Optional[str] = None


class AAIReplyAudioEvent(BaseAAIEvent):
    type: Literal["reply.audio"] = "reply.audio"
    audio: str = Field(default="", exclude=True)  # base64, in output encoding
    reply_id: Optional[str] = None


class AAIAgentTranscriptEvent(BaseAAIEvent):
    type: Literal["transcript.agent"] = "transcript.agent"
    text: str = ""
    interrupted: bool = False
    reply_id: Optional[str] = None


class AAIReplyDoneEvent(BaseAAIEvent):
    type: Literal["reply.done"] = "reply.done"
    reply_id: Optional[str] = None
    status: Optional[str] = None  # "interrupted" on barge-in


class AAIToolCallEvent(BaseAAIEvent):
    type: Literal["tool.call"] = "tool.call"
    call_id: str = ""
    name: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)  # NOT "args"


class AAISessionErrorEvent(BaseAAIEvent):
    type: Literal["session.error"] = "session.error"
    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None
    timestamp: Optional[Any] = None


class AAITimeoutEvent(BaseAAIEvent):
    type: Literal["timeout"] = "timeout"


class AAIUnknownEvent(BaseAAIEvent):
    type: str = "unknown"
    raw: Optional[Dict[str, Any]] = None


AAIEvent = Union[
    AAISessionReadyEvent,
    AAISessionUpdatedEvent,
    AAISpeechStartedEvent,
    AAISpeechStoppedEvent,
    AAIUserTranscriptDeltaEvent,
    AAIUserTranscriptEvent,
    AAIReplyStartedEvent,
    AAIReplyAudioEvent,
    AAIAgentTranscriptEvent,
    AAIReplyDoneEvent,
    AAIToolCallEvent,
    AAISessionErrorEvent,
    AAITimeoutEvent,
    AAIUnknownEvent,
]

_EVENT_TYPE_MAP: Dict[str, type[BaseAAIEvent]] = {
    "session.ready": AAISessionReadyEvent,
    "session.updated": AAISessionUpdatedEvent,
    "input.speech.started": AAISpeechStartedEvent,
    "input.speech.stopped": AAISpeechStoppedEvent,
    "transcript.user.delta": AAIUserTranscriptDeltaEvent,
    "transcript.user": AAIUserTranscriptEvent,
    "reply.started": AAIReplyStartedEvent,
    "reply.audio": AAIReplyAudioEvent,
    "transcript.agent": AAIAgentTranscriptEvent,
    "reply.done": AAIReplyDoneEvent,
    "tool.call": AAIToolCallEvent,
    "session.error": AAISessionErrorEvent,
}


def parse_assemblyai_event(data: Dict[str, Any]) -> AAIEvent:
    """Parse a raw AssemblyAI WebSocket message into a typed event."""
    event_type = data.get("type", "unknown")

    log_data = data.copy()
    if event_type == "reply.audio" and "audio" in log_data:
        log_data["audio"] = f"<{len(log_data.get('audio', ''))} base64 chars>"
    logger.debug(f"AssemblyAI event: {event_type} - {log_data}")

    event_class = _EVENT_TYPE_MAP.get(event_type)
    if event_class:
        try:
            return event_class(**data)
        except Exception as e:
            logger.warning(f"Failed to parse AssemblyAI event {event_type}: {e}")
            return AAIUnknownEvent(type=event_type, raw=data)
    logger.debug(f"Unknown AssemblyAI event type: {event_type}")
    return AAIUnknownEvent(type=event_type, raw=data)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_voice/test_audio_native/test_assemblyai_events.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add src/tau2/voice/audio_native/assemblyai/__init__.py \
        src/tau2/voice/audio_native/assemblyai/events.py \
        tests/test_voice/test_audio_native/test_assemblyai_events.py
git commit -m "feat(voice): AssemblyAI Voice Agent event models"
```

---

### Task 2: Provider client (`provider.py`)

**Files:**
- Create: `src/tau2/voice/audio_native/assemblyai/provider.py`
- Test: `tests/test_voice/test_audio_native/test_assemblyai_provider.py`

**Interfaces:**
- Consumes: `parse_assemblyai_event`, `AAITimeoutEvent` from Task 1; `Tool` from `tau2.environment.tool`.
- Produces:
  - `AssemblyAIAudioFormat` (`Enum`: `PCMU="audio/pcmu"`, `PCMA="audio/pcma"`, `PCM="audio/pcm"`).
  - `AssemblyAIVADConfig(BaseModel)` with `vad_threshold: float = 0.5`, `min_silence: int = 1000`, `max_silence: int = 3000`, `interrupt_response: bool = True`.
  - `AssemblyAIVoiceAgentProvider` with async `connect()`, `configure_session(system_prompt, tools, vad_config, voice)`, `send_audio(bytes)`, `send_tool_result(call_id, result)`, `receive_events()`, `receive_events_for_duration(seconds)`, sync property `is_connected`, and pure helpers `_build_session_payload(...)` and `_format_tools_for_api(tools)`.

- [ ] **Step 1: Write the failing test** (pure builders only — no network)

```python
# tests/test_voice/test_audio_native/test_assemblyai_provider.py
from tau2.environment.tool import Tool
from tau2.voice.audio_native.assemblyai.provider import (
    AssemblyAIAudioFormat,
    AssemblyAIVADConfig,
    AssemblyAIVoiceAgentProvider,
)


def _provider():
    return AssemblyAIVoiceAgentProvider(api_key="test-key")


def test_session_payload_sets_pcmu_and_voice_and_prompt():
    p = _provider()
    payload = p._build_session_payload(
        system_prompt="You are helpful.",
        tools=[],
        vad_config=AssemblyAIVADConfig(),
        voice="ivy",
    )
    assert payload["type"] == "session.update"
    s = payload["session"]
    assert s["system_prompt"] == "You are helpful."
    assert s["input"]["format"]["encoding"] == "audio/pcmu"
    assert s["output"]["format"]["encoding"] == "audio/pcmu"
    assert s["output"]["voice"] == "ivy"
    assert "greeting" not in s  # greeting deliberately omitted
    assert s["input"]["turn_detection"]["min_silence"] == 1000
    assert s["input"]["turn_detection"]["max_silence"] == 3000


def test_tools_use_flat_format():
    def lookup(order_id: str) -> str:
        """Look up an order.

        Args:
            order_id: The order id.
        """
        return order_id

    tools = [Tool(func=lookup)]
    p = _provider()
    formatted = p._format_tools_for_api(tools)
    assert formatted[0]["type"] == "function"
    assert formatted[0]["name"] == "lookup"
    assert "parameters" in formatted[0]
    assert "function" not in formatted[0]  # flat, not nested


def test_audio_format_default_is_pcmu():
    assert AssemblyAIVoiceAgentProvider(api_key="k").audio_format == AssemblyAIAudioFormat.PCMU


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    import pytest

    with pytest.raises(ValueError):
        AssemblyAIVoiceAgentProvider()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_voice/test_audio_native/test_assemblyai_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: ...assemblyai.provider`

- [ ] **Step 3: Write `provider.py`**

```python
# src/tau2/voice/audio_native/assemblyai/provider.py
"""AssemblyAI Voice Agent API provider (WebSocket, full-duplex).

Single WebSocket handling speech-in -> managed LLM -> speech-out. Native
G.711 μ-law (audio/pcmu) support means no conversion against tau2 telephony.

Reference: https://www.assemblyai.com/docs/voice-agents/voice-agent-api
"""

import asyncio
import base64
import json
import os
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional

import websockets
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from tau2.config import (
    DEFAULT_ASSEMBLYAI_VOICE,
    DEFAULT_ASSEMBLYAI_VOICE_AGENT_URL,
)
from tau2.environment.tool import Tool
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.assemblyai.events import (
    AAIEvent,
    AAISessionErrorEvent,
    AAITimeoutEvent,
    AAIUnknownEvent,
    parse_assemblyai_event,
)

load_dotenv()


class AssemblyAIAudioFormat(str, Enum):
    PCMU = "audio/pcmu"  # G.711 μ-law 8kHz (telephony, no conversion)
    PCMA = "audio/pcma"  # G.711 A-law 8kHz
    PCM = "audio/pcm"    # PCM16 24kHz


class AssemblyAIVADConfig(BaseModel):
    """session.input.turn_detection config (Voice Agent field names)."""

    vad_threshold: float = 0.5
    min_silence: int = 1000
    max_silence: int = 3000
    interrupt_response: bool = True


class AssemblyAIVoiceAgentProvider:
    BASE_URL = DEFAULT_ASSEMBLYAI_VOICE_AGENT_URL
    DEFAULT_VOICE = DEFAULT_ASSEMBLYAI_VOICE

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: Optional[str] = None,
        audio_format: AssemblyAIAudioFormat = AssemblyAIAudioFormat.PCMU,
    ):
        self.api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "AssemblyAI API key not provided. Set ASSEMBLYAI_API_KEY env var."
            )
        self.voice = voice or self.DEFAULT_VOICE
        self.audio_format = audio_format
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.session_id: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    @websocket_retry
    async def connect(self) -> None:
        if self.is_connected:
            return
        headers = {
            "Authorization": f"Bearer {self.api_key}",  # Bearer for Voice Agent API
            "Content-Type": "application/json",
        }
        logger.info(f"AssemblyAI Voice Agent: Connecting to {self.BASE_URL}")
        self.ws = await websockets.connect(self.BASE_URL, additional_headers=headers)

        # Wait for session.ready (may be preceded by other frames).
        for _ in range(50):
            data = json.loads(await self.ws.recv())
            if data.get("type") == "session.ready":
                self.session_id = data.get("session_id")
                logger.info(
                    f"AssemblyAI Voice Agent: session.ready (session_id={self.session_id})"
                )
                return
            if data.get("type") == "session.error":
                raise RuntimeError(f"Connection failed: {data}")
        raise RuntimeError("Did not receive session.ready")

    async def disconnect(self) -> None:
        if self.ws:
            logger.info("AssemblyAI Voice Agent: Disconnecting")
            await self.ws.close()
            self.ws = None

    def _build_audio_config(self) -> Dict:
        enc = {"encoding": self.audio_format.value}
        return {"input": {"format": dict(enc)}, "output": {"format": dict(enc)}}

    def _format_tools_for_api(self, tools: List[Tool]) -> List[Dict]:
        formatted = []
        for tool in tools:
            schema = tool.openai_schema
            formatted.append(
                {
                    "type": "function",
                    "name": schema["function"]["name"],
                    "description": schema["function"]["description"],
                    "parameters": schema["function"]["parameters"],
                }
            )
        return formatted

    def _build_session_payload(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: AssemblyAIVADConfig,
        voice: str,
    ) -> Dict:
        audio = self._build_audio_config()
        return {
            "type": "session.update",
            "session": {
                "system_prompt": system_prompt,
                "input": {
                    "format": audio["input"]["format"],
                    "turn_detection": {
                        "vad_threshold": vad_config.vad_threshold,
                        "min_silence": vad_config.min_silence,
                        "max_silence": vad_config.max_silence,
                        "interrupt_response": vad_config.interrupt_response,
                    },
                },
                "output": {"voice": voice, "format": audio["output"]["format"]},
                "tools": self._format_tools_for_api(tools),
            },
        }

    async def configure_session(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: AssemblyAIVADConfig,
        voice: Optional[str] = None,
    ) -> None:
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        payload = self._build_session_payload(
            system_prompt, tools, vad_config, voice or self.voice
        )
        logger.debug(f"AssemblyAI session config: {json.dumps(payload)}")
        await self.ws.send(json.dumps(payload))

        # Wait for session.updated (or error). Bounded to avoid hanging.
        for _ in range(50):
            data = json.loads(await self.ws.recv())
            t = data.get("type")
            if t == "session.updated":
                logger.info("AssemblyAI Voice Agent: session configured")
                return
            if t == "session.error":
                raise RuntimeError(f"Session configuration failed: {data}")
        logger.warning("No session.updated received; continuing")

    async def send_audio(self, audio_data: bytes) -> None:
        if not self.is_connected:
            raise RuntimeError("Not connected to API")
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        await self.ws.send(json.dumps({"type": "input.audio", "audio": audio_b64}))

    async def send_tool_result(self, call_id: str, result: str) -> None:
        """Send a tool result. Auto-fires the next reply — do NOT send reply.create."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")
        await self.ws.send(
            json.dumps({"type": "tool.result", "call_id": call_id, "result": result})
        )

    async def receive_events(self) -> AsyncGenerator[AAIEvent, None]:
        if not self.is_connected:
            raise RuntimeError("Not connected to API")
        while self.is_connected:
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                yield parse_assemblyai_event(json.loads(raw))
            except asyncio.TimeoutError:
                yield AAITimeoutEvent(type="timeout")
            except (websockets.ConnectionClosed, websockets.ConnectionClosedError) as e:
                logger.error(
                    f"AssemblyAI Voice Agent: connection closed "
                    f"(code={e.code}, reason='{e.reason or 'no reason'}')"
                )
                raise RuntimeError(
                    f"WebSocket connection closed unexpectedly (code={e.code})"
                ) from e
            except Exception as e:
                logger.error(f"AssemblyAI Voice Agent: error receiving event: {e}")
                yield AAIUnknownEvent(type="error", raw={"error": str(e)})

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[AAIEvent]:
        events: List[AAIEvent] = []
        end_time = asyncio.get_event_loop().time() + duration_seconds
        async for event in self.receive_events():
            if not isinstance(event, AAITimeoutEvent):
                events.append(event)
            if asyncio.get_event_loop().time() >= end_time:
                break
        return events
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_voice/test_audio_native/test_assemblyai_provider.py -v`
Expected: FAIL — `ImportError: cannot import name 'DEFAULT_ASSEMBLYAI_VOICE'`. This is expected; the config constants land in Task 4. Temporarily hardcode the two constants inline to confirm the builders, OR jump to Task 4 Step 3 (add config constants) first, then return. **Recommended: add the two config constants now** (they are also required by Task 4):

```python
# src/tau2/config.py — add under the provider config section (near the XAI block, ~line 169)
# =============================================================================
# ASSEMBLYAI PROVIDER (overridable voice, fixed API constants)
# =============================================================================
DEFAULT_ASSEMBLYAI_VOICE_AGENT_URL = "wss://agents.assemblyai.com/v1/ws"  # fixed
DEFAULT_ASSEMBLYAI_VOICE = "ivy"  # overridable
DEFAULT_ASSEMBLYAI_MODEL = "managed"  # fixed, determined by endpoint
```

Re-run: `uv run pytest tests/test_voice/test_audio_native/test_assemblyai_provider.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/tau2/voice/audio_native/assemblyai/provider.py \
        src/tau2/config.py \
        tests/test_voice/test_audio_native/test_assemblyai_provider.py
git commit -m "feat(voice): AssemblyAI Voice Agent provider client"
```

---

### Task 3: Discrete-time adapter (`discrete_time_adapter.py`)

**Files:**
- Create: `src/tau2/voice/audio_native/assemblyai/discrete_time_adapter.py`
- Test: `tests/test_voice/test_audio_native/test_assemblyai_adapter.py`

**Interfaces:**
- Consumes: `DiscreteTimeAdapter`, `TickResult`, `UtteranceTranscript`, `BackgroundAsyncLoop`, all Task 1 events, and Task 2 provider types.
- Produces: `DiscreteTimeAssemblyAIAdapter(tick_duration_ms, send_audio_instant=True, reasoning_effort=None, voice="ivy", provider=None)` with public `_process_event(result, event)` and the base-class overrides.

- [ ] **Step 1: Write the failing test** (drives `_process_event` with synthetic events — no network)

```python
# tests/test_voice/test_audio_native/test_assemblyai_adapter.py
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
    a._process_event(r, AAIReplyAudioEvent(audio=base64.b64encode(b"\xff" * 40).decode(), reply_id="r-1"))
    a._process_event(r, AAIAgentTranscriptEvent(text="hi there", reply_id="r-1"))
    assert r.agent_audio_bytes == 40
    assert a._utterance_transcripts["r-1"].transcript_received == "hi there"


def test_tool_call_recorded_with_arguments():
    a = _adapter()
    r = _result()
    a._process_event(r, AAIToolCallEvent(call_id="c-1", name="lookup", arguments={"x": 1}))
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


def test_reply_done_interrupted_discards_pending_tools():
    a = _adapter()
    r = _result()
    a.send_tool_result("c-1", "{}")
    a._process_event(r, AAIReplyDoneEvent(reply_id="r-1", status="interrupted"))
    assert a._pending_tool_results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_voice/test_audio_native/test_assemblyai_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: ...assemblyai.discrete_time_adapter`

- [ ] **Step 3: Write `discrete_time_adapter.py`**

```python
# src/tau2/voice/audio_native/assemblyai/discrete_time_adapter.py
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
        voice: str = "ivy",
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
        for call_id, result_str, _request_response, _is_error in self._pending_tool_results:
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

        elif isinstance(event, AAIToolCallEvent):
            result.tool_calls.append(
                ToolCall(id=event.call_id, name=event.name, arguments=event.arguments)
            )
            logger.debug(f"Tool call detected: {event.name}({event.call_id})")

        elif isinstance(event, AAIReplyDoneEvent):
            if event.status == "interrupted":
                self._pending_tool_results.clear()
            logger.debug(f"Reply done (status={event.status})")

        elif isinstance(event, AAIUserTranscriptEvent):
            logger.debug(f"Input transcription: {event.text}")

        elif isinstance(event, AAISessionErrorEvent):
            logger.error(f"AssemblyAI session error: {event.code} {event.message}")

        elif isinstance(event, AAITimeoutEvent):
            pass

        else:
            logger.debug(f"Event {type(event).__name__} received")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_voice/test_audio_native/test_assemblyai_adapter.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/tau2/voice/audio_native/assemblyai/discrete_time_adapter.py \
        tests/test_voice/test_audio_native/test_assemblyai_adapter.py
git commit -m "feat(voice): AssemblyAI discrete-time adapter"
```

---

### Task 4: Registration (exports, config registry, factory, CLI, config model)

**Files:**
- Modify: `src/tau2/voice/audio_native/assemblyai/__init__.py`
- Modify: `src/tau2/config.py` (registry dicts) — constants block already added in Task 2
- Modify: `src/tau2/voice/audio_native/adapter.py:456` (factory) and endpoint-determined tuple
- Modify: `src/tau2/data_model/simulation.py:72` (`AudioNativeConfig.provider` Literal + docstring)
- Modify: `src/tau2/cli.py:242` (`--audio-native-provider` choices)
- Test: `tests/test_voice/test_audio_native/test_assemblyai_registration.py`

**Interfaces:**
- Consumes: `create_adapter` from `tau2.voice.audio_native.adapter`; `AudioNativeConfig` from `tau2.data_model.simulation`; `DiscreteTimeAssemblyAIAdapter` from Task 3.
- Produces: `create_adapter(provider="assemblyai", ...)` returns a `DiscreteTimeAssemblyAIAdapter`; `AudioNativeConfig(provider="assemblyai")` validates.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_voice/test_audio_native/test_assemblyai_registration.py
from tau2.config import (
    AUDIO_NATIVE_PROVIDER_TYPES,
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_AUDIO_NATIVE_REASONING_EFFORT,
)
from tau2.data_model.simulation import AudioNativeConfig
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.assemblyai.discrete_time_adapter import (
    DiscreteTimeAssemblyAIAdapter,
)


def test_registry_entries_present():
    assert DEFAULT_AUDIO_NATIVE_MODELS["assemblyai"] == "managed"
    assert DEFAULT_AUDIO_NATIVE_REASONING_EFFORT["assemblyai"] is None
    assert AUDIO_NATIVE_PROVIDER_TYPES["assemblyai"] == "audio_native"


def test_config_accepts_provider():
    cfg = AudioNativeConfig(provider="assemblyai")
    assert cfg.provider == "assemblyai"


def test_factory_builds_adapter_without_connecting(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key")
    adapter, model = create_adapter(
        provider="assemblyai", tick_duration_ms=200, model=None
    )
    assert isinstance(adapter, DiscreteTimeAssemblyAIAdapter)
    assert model == "managed"
    assert adapter.is_connected is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_voice/test_audio_native/test_assemblyai_registration.py -v`
Expected: FAIL — `KeyError: 'assemblyai'` (registry) / `ValidationError` (Literal).

- [ ] **Step 3: Add registry entries in `src/tau2/config.py`**

Add `"assemblyai"` to each dict (import `DEFAULT_ASSEMBLYAI_MODEL` is already defined in the constants block from Task 2):

```python
DEFAULT_AUDIO_NATIVE_MODELS = {
    "openai": DEFAULT_OPENAI_REALTIME_MODEL,
    "gemini": DEFAULT_GEMINI_MODEL,
    "xai": DEFAULT_XAI_MODEL,
    "nova": DEFAULT_NOVA_MODEL,
    "qwen": DEFAULT_QWEN_MODEL,
    "assemblyai": DEFAULT_ASSEMBLYAI_MODEL,
    "livekit": "dummy",
}

DEFAULT_AUDIO_NATIVE_REASONING_EFFORT: dict[str, str | None] = {
    "openai": None,
    "gemini": "high",
    "xai": None,
    "nova": None,
    "qwen": None,
    "assemblyai": None,
    "livekit": None,
}

AUDIO_NATIVE_PROVIDER_TYPES = {
    "openai": "audio_native",
    "gemini": "audio_native",
    "xai": "audio_native",
    "nova": "audio_native",
    "qwen": "audio_native",
    "assemblyai": "audio_native",
    "livekit": "cascaded",
}
```

- [ ] **Step 4: Register in the factory `src/tau2/voice/audio_native/adapter.py`**

Add `"assemblyai"` to the endpoint-determined tuple:

```python
_PROVIDERS_WITH_ENDPOINT_DETERMINED_MODEL = ("xai", "assemblyai")
```

Add a branch in `create_adapter()` immediately before the `else: raise ValueError` (after the `elif provider == "livekit":` block):

```python
    elif provider == "assemblyai":
        from tau2.voice.audio_native.assemblyai.discrete_time_adapter import (
            DiscreteTimeAssemblyAIAdapter,
        )

        adapter = DiscreteTimeAssemblyAIAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=send_audio_instant,
            reasoning_effort=reasoning_effort,
        )
```

- [ ] **Step 5: Extend the `AudioNativeConfig.provider` Literal in `src/tau2/data_model/simulation.py`**

```python
    provider: Literal[
        "openai", "gemini", "xai", "nova", "qwen", "assemblyai", "livekit"
    ] = Field(
        default=DEFAULT_AUDIO_NATIVE_PROVIDER,
        description="Audio native API provider: 'openai' (OpenAI Realtime), 'gemini' (Gemini Live), 'xai' (xAI Grok Voice Agent), 'nova' (Amazon Nova Sonic), 'qwen' (Alibaba Qwen Omni), 'assemblyai' (AssemblyAI Voice Agent, managed model), or 'livekit' (LiveKit cascaded STT→LLM→TTS)",
    )
```

- [ ] **Step 6: Add the CLI choice in `src/tau2/cli.py:242`**

```python
        choices=["openai", "gemini", "xai", "assemblyai", "livekit"],
```

- [ ] **Step 7: Fill in `__init__.py` exports**

```python
# src/tau2/voice/audio_native/assemblyai/__init__.py
"""AssemblyAI Voice Agent API integration for audio-native voice processing.

Native G.711 μ-law (audio/pcmu) — no audio conversion. Managed model
(endpoint-determined). Reference:
https://www.assemblyai.com/docs/voice-agents/voice-agent-api
"""

from tau2.voice.audio_native.assemblyai.discrete_time_adapter import (
    DiscreteTimeAssemblyAIAdapter,
)
from tau2.voice.audio_native.assemblyai.events import (
    AAIAgentTranscriptEvent,
    AAIReplyAudioEvent,
    AAIReplyDoneEvent,
    AAIReplyStartedEvent,
    AAISessionErrorEvent,
    AAISessionReadyEvent,
    AAISessionUpdatedEvent,
    AAISpeechStartedEvent,
    AAISpeechStoppedEvent,
    AAITimeoutEvent,
    AAIToolCallEvent,
    AAIUnknownEvent,
    AAIUserTranscriptDeltaEvent,
    AAIUserTranscriptEvent,
    parse_assemblyai_event,
)
from tau2.voice.audio_native.assemblyai.provider import (
    AssemblyAIAudioFormat,
    AssemblyAIVADConfig,
    AssemblyAIVoiceAgentProvider,
)

__all__ = [
    "AAIAgentTranscriptEvent",
    "AAIReplyAudioEvent",
    "AAIReplyDoneEvent",
    "AAIReplyStartedEvent",
    "AAISessionErrorEvent",
    "AAISessionReadyEvent",
    "AAISessionUpdatedEvent",
    "AAISpeechStartedEvent",
    "AAISpeechStoppedEvent",
    "AAITimeoutEvent",
    "AAIToolCallEvent",
    "AAIUnknownEvent",
    "AAIUserTranscriptDeltaEvent",
    "AAIUserTranscriptEvent",
    "parse_assemblyai_event",
    "AssemblyAIAudioFormat",
    "AssemblyAIVADConfig",
    "AssemblyAIVoiceAgentProvider",
    "DiscreteTimeAssemblyAIAdapter",
]
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_voice/test_audio_native/test_assemblyai_registration.py -v`
Expected: PASS (3 passed)

- [ ] **Step 9: Run the full new-test set + lint**

Run:
```bash
uv run pytest tests/test_voice/test_audio_native/test_assemblyai_events.py \
              tests/test_voice/test_audio_native/test_assemblyai_provider.py \
              tests/test_voice/test_audio_native/test_assemblyai_adapter.py \
              tests/test_voice/test_audio_native/test_assemblyai_registration.py -v
uv run ruff check src/tau2/voice/audio_native/assemblyai/
```
Expected: all PASS; ruff clean.

- [ ] **Step 10: Commit**

```bash
git add src/tau2/config.py src/tau2/voice/audio_native/adapter.py \
        src/tau2/data_model/simulation.py src/tau2/cli.py \
        src/tau2/voice/audio_native/assemblyai/__init__.py \
        tests/test_voice/test_audio_native/test_assemblyai_registration.py
git commit -m "feat(voice): register assemblyai audio-native provider"
```

---

### Task 5: Standalone smoke test + docs/env

**Files:**
- Create: `src/tau2/voice/audio_native/assemblyai/test_provider_standalone.py`
- Modify: `.env.example`
- Modify: `src/tau2/voice/README.md` (Providers table + Environment Variables table)
- Modify: `src/tau2/voice/audio_native/README.md` (Supported Providers table)

**Interfaces:**
- Consumes: `AssemblyAIVoiceAgentProvider`, `AssemblyAIVADConfig`, `parse_assemblyai_event` from Tasks 1–2. Loads μ-law test audio `tests/test_voice/test_audio_native/testdata/hello.ulaw`.

- [ ] **Step 1: Write the standalone smoke script**

This is a live connectivity check (not pytest — needs network + `ASSEMBLYAI_API_KEY`). It verifies `session.ready`, sends μ-law speech + silence, and confirms events flow.

```python
# src/tau2/voice/audio_native/assemblyai/test_provider_standalone.py
#!/usr/bin/env python3
"""Standalone connectivity test for the AssemblyAI Voice Agent provider.

Requires ASSEMBLYAI_API_KEY and network access. Exits 0 on success, 1 on failure.

Usage:
    uv run src/tau2/voice/audio_native/assemblyai/test_provider_standalone.py
"""

import asyncio
import os
import signal
import sys

sys.path.insert(0, "src")

AUDIO_PATH = "tests/test_voice/test_audio_native/testdata/hello.ulaw"


async def main() -> int:
    if not os.environ.get("ASSEMBLYAI_API_KEY"):
        print("SKIP: ASSEMBLYAI_API_KEY not set")
        return 0

    from tau2.voice.audio_native.assemblyai.events import (
        AAISessionReadyEvent,
        AAITimeoutEvent,
    )
    from tau2.voice.audio_native.assemblyai.provider import (
        AssemblyAIVADConfig,
        AssemblyAIVoiceAgentProvider,
    )

    provider = AssemblyAIVoiceAgentProvider()
    received = []
    try:
        print("1. Connecting...")
        await provider.connect()
        assert provider.session_id, "no session_id after connect"
        print(f"   session_id={provider.session_id}")

        print("2. Configuring session...")
        await provider.configure_session(
            system_prompt="You are a helpful assistant. Keep replies to one short sentence.",
            tools=[],
            vad_config=AssemblyAIVADConfig(),
        )

        print("3. Sending μ-law speech + 1s silence...")
        with open(AUDIO_PATH, "rb") as f:
            speech = f.read()
        await provider.send_audio(speech)
        await provider.send_audio(b"\xff" * 8000)  # 1s μ-law silence @ 8kHz

        print("4. Collecting events (up to 20s)...")
        got_ready = False
        for _ in range(200):
            events = await provider.receive_events_for_duration(0.1)
            for ev in events:
                if not isinstance(ev, AAITimeoutEvent):
                    received.append(ev)
                if isinstance(ev, AAISessionReadyEvent):
                    got_ready = True
            if len(received) >= 3:
                break

        types = sorted({type(e).__name__ for e in received})
        print(f"   received {len(received)} events: {types}")
        if received:
            print("✅ SUCCESS")
            return 0
        print("❌ FAILED — no events received")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        await provider.disconnect()


if __name__ == "__main__":
    def _timeout(signum, frame):
        print("\n❌ SCRIPT TIMEOUT (60s)")
        sys.exit(1)

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(60)
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 2: Run the smoke test**

Run: `uv run src/tau2/voice/audio_native/assemblyai/test_provider_standalone.py`
Expected (with `ASSEMBLYAI_API_KEY` set): connects, prints a `session_id`, prints received event types (expect at least `AAIReplyStartedEvent` / `AAIReplyAudioEvent` / `AAIAgentTranscriptEvent`), then `✅ SUCCESS`. Without the key: `SKIP` and exit 0.

**Verify open items here** (from the spec): (a) the agent produces a first turn with no greeting; (b) `reply.audio` carries a usable `reply_id` (else audio falls back to `_current_item_id`); (c) `session.updated` is actually emitted. If any assumption is wrong, fix the affected event model / provider builder and re-run this script and the Task 1–3 unit tests before committing.

- [ ] **Step 3: Update `.env.example`**

Add after the `OPENAI_API_KEY` line:

```
ASSEMBLYAI_API_KEY=<your_key_here>
```

- [ ] **Step 4: Update `src/tau2/voice/README.md`**

Add a row to the Providers table:

```
| AssemblyAI Voice Agent | `--audio-native-provider assemblyai` | `ASSEMBLYAI_API_KEY` |
```

Add a row to the Environment Variables table:

```
| `ASSEMBLYAI_API_KEY` | AssemblyAI Voice Agent provider |
```

- [ ] **Step 5: Update `src/tau2/voice/audio_native/README.md`**

Add a row to the Supported Providers table:

```
| **assemblyai** | Native audio | AssemblyAI Voice Agent API | managed (endpoint-determined) |
```

- [ ] **Step 6: Commit**

```bash
git add src/tau2/voice/audio_native/assemblyai/test_provider_standalone.py \
        .env.example src/tau2/voice/README.md src/tau2/voice/audio_native/README.md
git commit -m "feat(voice): AssemblyAI provider smoke test + docs"
```

---

### Task 6: End-to-end verification (manual)

**Files:** none (verification only).

- [ ] **Step 1: Confirm the CLI recognizes the provider**

Run: `uv run tau2 run --help | grep -A2 audio-native-provider`
Expected: help text lists `assemblyai` among the choices.

- [ ] **Step 2: Run a single control-complexity task end-to-end**

Prerequisites: `uv sync --extra voice`, `brew install portaudio ffmpeg`, and your own ElevenLabs personas in `.env` (`TAU2_VOICE_ID_*`, per `docs/voice-personas.md`) plus `ELEVENLABS_API_KEY` and `ASSEMBLYAI_API_KEY`.

Run:
```bash
uv run tau2 run --domain airline --audio-native \
  --audio-native-provider assemblyai \
  --num-tasks 1 --speech-complexity control --verbose-logs
```
Expected: the run completes and writes results under `data/simulations/<run_name>/`, including `artifacts/.../audio/both.wav`. Inspect the transcript/audio to confirm the agent responded and any tool calls resolved. Report the outcome (pass/fail with the actual reward + any errors) — do not claim success without the run output.

---

## Self-Review

**Spec coverage:**
- Managed model, endpoint-determined → Task 4 (endpoint tuple + `DEFAULT_ASSEMBLYAI_MODEL="managed"`). ✓
- Working adapter + smoke test scope → Tasks 1–5; end-to-end run → Task 6. ✓
- New files (`provider.py`, `events.py`, `discrete_time_adapter.py`, `__init__.py`, standalone test) → Tasks 1,2,3,4,5. ✓
- Auth Bearer, `input.audio`, μ-law both directions, greeting omitted, first-update immutables → Task 2 `provider.py`. ✓
- Event mapping table (reply.audio/transcript.agent/tool.call/barge-in/reply.done) → Task 3 `_process_event`. ✓
- `tool.result` auto-fires, no reply.create → Task 3 `_flush_pending_tool_results`. ✓
- Full-text transcript (not deltas) → Task 3 `AAIAgentTranscriptEvent` handling. ✓
- Registration (config dicts, factory, Literal, CLI) → Task 4. ✓
- Env + docs → Task 5. ✓
- Open items verified during build → Task 5 Step 2. ✓

**Placeholder scan:** No TBD/TODO; every code step contains full code; no "handle errors appropriately" hand-waves. ✓

**Type consistency:** Provider `AssemblyAIVoiceAgentProvider`, adapter `DiscreteTimeAssemblyAIAdapter`, format `AssemblyAIAudioFormat`, VAD `AssemblyAIVADConfig`, event classes `AAI*`, and `parse_assemblyai_event` are used identically across Tasks 1–5. `send_tool_result(call_id, result)` (2 args) matches the call in `_flush_pending_tool_results`. `_build_session_payload`/`_format_tools_for_api` names match their tests. ✓
