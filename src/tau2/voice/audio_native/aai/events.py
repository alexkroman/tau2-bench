"""Pydantic models for AAI voice agent events.

AAI server sends camelCase JSON messages that are parsed into typed event models.
All events inherit from BaseAAIEvent with configurable extra field handling.
"""

from typing import Any, Literal, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class BaseAAIEvent(BaseModel):
    """Base class for all AAI events."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    type: str


# =============================================================================
# Configuration Events
# =============================================================================


class AAIConfigEvent(BaseAAIEvent):
    """Server configuration acknowledgment (`session.configured`).

    Carries the audio rates the host settled on — the ones this client must
    send and expect, whatever it asked for.
    """

    type: Literal["config"] = "config"
    audio_format: Optional[str] = Field(default=None, alias="audioFormat")
    sample_rate: Optional[int] = Field(default=None, alias="sampleRate")
    tts_sample_rate: Optional[int] = Field(default=None, alias="ttsSampleRate")


# =============================================================================
# Speech Detection Events
# =============================================================================


class AAISpeechStartedEvent(BaseAAIEvent):
    """User speech detection started."""

    type: Literal["speech_started"] = "speech_started"


class AAISpeechStoppedEvent(BaseAAIEvent):
    """User speech detection stopped."""

    type: Literal["speech_stopped"] = "speech_stopped"


# =============================================================================
# Transcript Events
# =============================================================================


class AAIUserTranscriptEvent(BaseAAIEvent):
    """User speech transcript from the voice agent."""

    type: Literal["user_transcript"] = "user_transcript"
    text: str
    turn_order: Optional[int] = Field(default=None, alias="turnOrder")


class AAIAgentTranscriptEvent(BaseAAIEvent):
    """Agent response transcript."""

    type: Literal["agent_transcript"] = "agent_transcript"
    text: str


# =============================================================================
# Tool Events
# =============================================================================


class AAIToolCallEvent(BaseAAIEvent):
    """Agent is requesting to call a tool."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    args: dict = Field(default_factory=dict)


class AAIToolCallDoneEvent(BaseAAIEvent):
    """Tool execution has completed."""

    type: Literal["tool_call_done"] = "tool_call_done"
    tool_call_id: str = Field(alias="toolCallId")
    result: str = ""


# =============================================================================
# Response Events
# =============================================================================


class AAIReplyDoneEvent(BaseAAIEvent):
    """Agent response is complete."""

    type: Literal["reply_done"] = "reply_done"


class AAIAudioDoneEvent(BaseAAIEvent):
    """Audio output is complete."""

    type: Literal["audio_done"] = "audio_done"


# =============================================================================
# Session Control Events
# =============================================================================


class AAICancelledEvent(BaseAAIEvent):
    """The in-flight reply was abandoned (barge-in). Terminal: no reply_done follows."""

    type: Literal["cancelled"] = "cancelled"


class AAIResetEvent(BaseAAIEvent):
    """Session was reset."""

    type: Literal["reset"] = "reset"


class AAIIdleTimeoutEvent(BaseAAIEvent):
    """Session idle timeout occurred."""

    type: Literal["idle_timeout"] = "idle_timeout"


# =============================================================================
# Error and Custom Events
# =============================================================================


class AAIErrorEvent(BaseAAIEvent):
    """Error from the AAI server."""

    type: Literal["error"] = "error"
    code: Optional[str] = None
    message: Optional[str] = None


class AAICustomEvent(BaseAAIEvent):
    """Custom application-defined event."""

    type: Literal["custom_event"] = "custom_event"
    event: str
    data: Optional[Any] = None


# =============================================================================
# Internal/Helper Events
# =============================================================================


class AAISpeechSuspectedEvent(BaseAAIEvent):
    """The host MIGHT have heard the caller start; it ducked rather than yielded."""

    type: Literal["speech_suspected"] = "speech_suspected"


class AAISpeechDismissedEvent(BaseAAIEvent):
    """A duck taken back: the host resumed the reply it had paused."""

    type: Literal["speech_dismissed"] = "speech_dismissed"


class AAITimeoutEvent(BaseAAIEvent):
    """Timeout waiting for events (used internally, not from server)."""

    type: Literal["timeout"] = "timeout"


class AAIUnknownEvent(BaseAAIEvent):
    """Unknown or unrecognized event type."""

    type: str
    raw: Optional[dict] = None


class AAIAudioChunkEvent(BaseAAIEvent):
    """Audio frame chunk (constructed directly by provider, not parsed).

    pcm16 field is excluded from serialization due to binary size.
    """

    type: Literal["audio_chunk"] = "audio_chunk"
    pcm16: bytes = Field(default=b"", exclude=True)


# =============================================================================
# Event Type Mapping
# =============================================================================

_EVENT_TYPE_MAP: dict[str, type[BaseAAIEvent]] = {
    "config": AAIConfigEvent,
    "speech_started": AAISpeechStartedEvent,
    "speech_stopped": AAISpeechStoppedEvent,
    "user_transcript": AAIUserTranscriptEvent,
    "agent_transcript": AAIAgentTranscriptEvent,
    "tool_call": AAIToolCallEvent,
    "tool_call_done": AAIToolCallDoneEvent,
    "reply_done": AAIReplyDoneEvent,
    "audio_done": AAIAudioDoneEvent,
    "cancelled": AAICancelledEvent,
    "reset": AAIResetEvent,
    "idle_timeout": AAIIdleTimeoutEvent,
    "error": AAIErrorEvent,
    "custom_event": AAICustomEvent,
    "timeout": AAITimeoutEvent,
    # Logged and otherwise ignored: this harness does not act on a duck (it has
    # no pause/resume for its playout buffer), so the server-side pacer pause is
    # the whole mechanism under test and these are here for OBSERVABILITY.
    "speech_suspected": AAISpeechSuspectedEvent,
    "speech_dismissed": AAISpeechDismissedEvent,
}


# =============================================================================
# Wire-name translation (AAI protocol vocabulary, 2026-09)
# =============================================================================

# The AAI SDK renamed every session event from a flat name to `noun.verb-past`
# ("every event is `noun.verb-past`, and that is a decision about API" in
# packages/aai/src/sdk/protocol-events.ts). The FIELDS did not change — `text`,
# `toolCallId`, `toolName`, `args`, `result`, `code`, `message` all still
# arrive under those names — so the whole drift is the discriminator, and the
# models above stay as they are.
#
# Translating at parse time rather than renaming the models keeps the flat
# names as this integration's internal vocabulary: the discrete-time adapter,
# the tick-level trajectory format, the interaction metrics, and the per-run
# scripts in scripts/ all key off them, and scripts/latency.py in particular
# scrapes `AAI event: <name>` out of the task log with a `[a-z_]+` pattern that
# a dotted name would not match. One dict here beats five call-site renames and
# a silent regex miss.
#
# Not translating this cost a whole run: every task failed with "aai host did
# not send a config frame within 10.0s" while the host log said the session was
# ready, because the ack now arrives as `session.configured`.
_WIRE_ALIASES: dict[str, str] = {
    # The reversible duck pair. Mapped to `custom_event`-shaped no-ops purely so
    # they are LOGGED: an unknown type returns early from parse_aai_event before
    # the log line, so a duck that fired was indistinguishable from one that did
    # not — which is exactly what made the first duck run unverifiable.
    "speech.suspected": "speech_suspected",
    "speech.dismissed": "speech_dismissed",
    "session.configured": "config",
    "speech.started": "speech_started",
    "speech.stopped": "speech_stopped",
    "user-transcript.committed": "user_transcript",
    "agent-transcript.committed": "agent_transcript",
    "tool.called": "tool_call",
    "tool.completed": "tool_call_done",
    "reply.completed": "reply_done",
    "audio.completed": "audio_done",
    "reply.cancelled": "cancelled",
    "session.reset": "reset",
    "session.timed-out": "idle_timeout",
    "error.reported": "error",
    "custom.emitted": "custom_event",
}

# The two INTERIM streams are dropped deliberately, and this is the one place
# the choice is visible. `agent-transcript.updated` / `user-transcript.updated`
# carry the partial text for the turn in progress; the flat `agent_transcript`
# this integration was written against was the COMMITTED text, so mapping the
# updated stream to it as well would double every agent utterance in the
# caller-heard concatenation and move the response-latency measurement onto the
# first partial rather than the settled sentence. They fall through to
# AAIUnknownEvent, which the adapter ignores.
_IGNORED_WIRE_EVENTS: frozenset[str] = frozenset(
    {"agent-transcript.updated", "user-transcript.updated"}
)


# =============================================================================
# Event Parsing
# =============================================================================


def parse_aai_event(data: dict) -> BaseAAIEvent:
    """Parse raw AAI event data into a typed event model.

    Args:
        data: Raw event dictionary from AAI server.

    Returns:
        Typed event instance. Unknown types return AAIUnknownEvent.
        Parse failures return AAIUnknownEvent with warning logged.
    """
    wire_type = data.get("type", "unknown")
    event_type = _WIRE_ALIASES.get(wire_type, wire_type)
    if event_type != wire_type:
        # The models are keyed by the flat name and validate `type` as a
        # Literal, so the payload has to carry the translated name too.
        data = {**data, "type": event_type}
    event_class = _EVENT_TYPE_MAP.get(event_type)

    if event_class is None:
        return AAIUnknownEvent(type=event_type, raw=data)

    # Log event, redacting large binary/base64 fields
    log_data = _prepare_log_data(data)
    logger.debug(f"AAI event: {event_type} - {log_data}")

    try:
        return event_class.model_validate(data)
    except Exception as e:
        logger.warning(f"Failed to parse AAI event {event_type}: {e}")
        return AAIUnknownEvent(type=event_type, raw=data)


def _prepare_log_data(data: dict) -> dict:
    """Prepare event data for logging, eliding large fields like base64.

    Args:
        data: Raw event data.

    Returns:
        Copy of data with large fields redacted for logging.
    """
    log_data = data.copy()
    # Redact any large base64 or binary fields if present
    for key in list(log_data.keys()):
        if key in ("audio", "data", "content") and isinstance(log_data[key], str):
            if len(log_data[key]) > 100:
                log_data[key] = f"<{len(log_data[key])} chars>"
    return log_data
