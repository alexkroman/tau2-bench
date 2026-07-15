"""Pydantic models for AssemblyAI Voice Agent API server events.

Reference: https://www.assemblyai.com/docs/voice-agents/voice-agent-api

Field names are verified against recorded live-API messages: ``reply.audio``
delivers its base64 payload as ``data`` (not ``audio``), ``tool.call``
delivers its arguments as ``args`` (the protocol has also used ``arguments``;
both are accepted, ``arguments`` wins), and ``session.updated`` echoes the
applied config as ``config``. Models tolerate unknown fields via
extra="ignore" so unexpected frames never crash the receive loop.
"""

from typing import Any, Dict, Literal, Optional, Union

from loguru import logger
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class BaseAAIEvent(BaseModel):
    """Base class for all AssemblyAI Voice Agent events."""

    model_config = ConfigDict(extra="ignore")
    type: str


class AAISessionReadyEvent(BaseAAIEvent):
    type: Literal["session.ready"] = "session.ready"
    session_id: Optional[str] = None


class AAISessionUpdatedEvent(BaseAAIEvent):
    type: Literal["session.updated"] = "session.updated"
    config: Optional[Dict[str, Any]] = None


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
    # Live API sends the base64 payload as `data`; `audio` kept for compat.
    audio: str = Field(
        default="",
        exclude=True,
        validation_alias=AliasChoices("data", "audio"),
    )
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
    # The wire format has used both names; prefer `arguments`, accept `args`.
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("arguments", "args"),
    )


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
    if event_type == "reply.audio":
        for audio_key in ("data", "audio"):
            if audio_key in log_data:
                log_data[audio_key] = (
                    f"<{len(log_data.get(audio_key, ''))} base64 chars>"
                )
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
