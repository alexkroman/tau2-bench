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
    AAITimeoutEvent,
    AAIUnknownEvent,
    parse_assemblyai_event,
)

load_dotenv()

# No existing tau2.config timeout fits a per-frame wait during session
# configuration; kept local since it's specific to this provider's handshake.
SESSION_UPDATE_FRAME_TIMEOUT = 5.0  # seconds


class AssemblyAIAudioFormat(str, Enum):
    PCMU = "audio/pcmu"  # G.711 μ-law 8kHz (telephony, no conversion)
    PCMA = "audio/pcma"  # G.711 A-law 8kHz
    PCM = "audio/pcm"  # PCM16 24kHz


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
        # Non-session frames seen while waiting for session.updated in
        # configure_session(); surfaced by the next receive_events_for_duration().
        self._buffered_events: List[AAIEvent] = []

    @property
    def is_connected(self) -> bool:
        """Check whether the WebSocket connection is currently open."""
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    @websocket_retry
    async def connect(self) -> None:
        """Open the WebSocket connection.

        The Voice Agent API does NOT emit ``session.ready`` unprompted — the
        client must send a ``session.update`` first (done in
        ``configure_session``), and the server then replies with
        ``session.updated`` followed by ``session.ready``. So connect only opens
        the socket; the handshake completes in ``configure_session``.
        """
        if self.is_connected:
            return
        headers = {
            "Authorization": f"Bearer {self.api_key}",  # Bearer for Voice Agent API
            "Content-Type": "application/json",
        }
        logger.info(f"AssemblyAI Voice Agent: Connecting to {self.BASE_URL}")
        self.ws = await websockets.connect(self.BASE_URL, additional_headers=headers)

    async def disconnect(self) -> None:
        """Close the WebSocket connection, if open."""
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
        """Send session.update and complete the handshake by waiting for session.ready.

        The server replies with ``session.updated`` (config echoed) and then
        ``session.ready`` (agent live). We wait for ``session.ready`` — the
        definitive "agent initialized" signal — treating ``session.updated`` as
        an intermediate frame. Each recv is individually bounded so a stalled
        connection can't hang forever. Non-handshake frames are buffered so they
        aren't dropped; the next receive_events_for_duration() surfaces them.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        payload = self._build_session_payload(
            system_prompt, tools, vad_config, voice or self.voice
        )
        logger.debug(f"AssemblyAI session config: {json.dumps(payload)}")
        await self.ws.send(json.dumps(payload))

        saw_updated = False
        while True:
            try:
                raw = await asyncio.wait_for(
                    self.ws.recv(), timeout=SESSION_UPDATE_FRAME_TIMEOUT
                )
            except asyncio.TimeoutError:
                if saw_updated:
                    logger.warning(
                        "AssemblyAI Voice Agent: session.updated received but no "
                        f"session.ready within {SESSION_UPDATE_FRAME_TIMEOUT}s; "
                        "proceeding — session is configured"
                    )
                    return
                raise RuntimeError(
                    "No response to session.update within "
                    f"{SESSION_UPDATE_FRAME_TIMEOUT}s (no session.updated/ready). "
                    "The agent did not initialize."
                )
            data = json.loads(raw)
            t = data.get("type")
            if t == "session.ready":
                self.session_id = data.get("session_id") or self.session_id
                logger.info(
                    f"AssemblyAI Voice Agent: session.ready "
                    f"(session_id={self.session_id})"
                )
                return
            if t == "session.updated":
                saw_updated = True
                cfg = data.get("config") or {}
                self.session_id = cfg.get("id") or self.session_id
                logger.info("AssemblyAI Voice Agent: session configured")
                continue
            if t == "session.error":
                raise RuntimeError(f"Session configuration failed: {data}")
            logger.debug(
                f"AssemblyAI Voice Agent: buffering early frame (type={t}) "
                "received during handshake"
            )
            self._buffered_events.append(parse_assemblyai_event(data))

    async def send_audio(self, audio_data: bytes) -> None:
        """Base64-encode and send a chunk of user audio to the API."""
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
        """Yield parsed events from the WebSocket as they arrive (with periodic timeouts)."""
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
        """Collect events for a fixed duration, surfacing any buffered frames first."""
        events: List[AAIEvent] = []
        if self._buffered_events:
            events.extend(self._buffered_events)
            self._buffered_events = []
        end_time = asyncio.get_event_loop().time() + duration_seconds
        async for event in self.receive_events():
            if not isinstance(event, AAITimeoutEvent):
                events.append(event)
            if asyncio.get_event_loop().time() >= end_time:
                break
        return events
