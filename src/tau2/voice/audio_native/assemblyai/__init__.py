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
