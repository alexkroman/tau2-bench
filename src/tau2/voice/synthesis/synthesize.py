"""Core voice synthesis (TTS) functions."""

import os
import threading

from dotenv import load_dotenv

from tau2.config import DEFAULT_TTS_MAX_CONCURRENCY
from tau2.data_model.audio import AudioData
from tau2.data_model.voice import ProviderConfig
from tau2.utils.retry import tts_retry
from tau2.voice.utils.assemblyai_utils import tts_assemblyai
from tau2.voice.utils.elevenlabs_utils import tts_elevenlabs

load_dotenv()


def _tts_concurrency_limit() -> int:
    """Process-wide TTS concurrency ceiling; see DEFAULT_TTS_MAX_CONCURRENCY."""
    raw = os.environ.get("TAU2_TTS_MAX_CONCURRENCY")
    if raw is None:
        return DEFAULT_TTS_MAX_CONCURRENCY
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(
            f"TAU2_TTS_MAX_CONCURRENCY must be an integer, got {raw!r}"
        ) from e
    if value < 1:
        raise ValueError(f"TAU2_TTS_MAX_CONCURRENCY must be >= 1, got {value}")
    return value


# Bounds provider calls in flight across EVERY synthesis path in the process —
# the streaming user simulator, the legacy half-duplex simulator, the CLI, and
# OutOfTurnSpeechGenerator's thread pool. This module is the one choke point all
# four pass through, which is why the limit lives here rather than at each fan-out
# site: capping a single thread pool would leave the other three unbounded.
#
# Read once at import so every caller shares one semaphore.
_TTS_SEMAPHORE = threading.BoundedSemaphore(_tts_concurrency_limit())


# The caller's voice, by provider name. A TABLE rather than a branch so that
# adding one is an entry — and so the error above can NAME what it knows.
#
# **The two are not acoustically interchangeable**, and a run is comparable
# only to runs using the same one: ElevenLabs v3 renders `[cough]`/`[pause]`
# as audio and AssemblyAI has no tag vocabulary, so those markers are stripped
# there (`assemblyai_utils`). Switching providers re-bases the baseline.
SYNTHESIS_PROVIDERS = {
    "elevenlabs": tts_elevenlabs,
    "assemblyai": tts_assemblyai,
}


@tts_retry
def synthesize_voice(
    text: str,
    provider: str,
    provider_config: ProviderConfig,
) -> AudioData:
    """Synthesize voice from text using the specified configuration.

    Serialized against DEFAULT_TTS_MAX_CONCURRENCY: providers cap concurrent
    requests per subscription and answer 429 over it, which costs a caller
    utterance rather than just time.
    """
    synth = SYNTHESIS_PROVIDERS.get(provider)
    if synth is None:
        # Raised before acquiring: an unsupported provider is a config error,
        # not work, and must not queue behind in-flight synthesis.
        raise ValueError(
            f"Unsupported synthesis provider: {provider} "
            f"(known: {', '.join(sorted(SYNTHESIS_PROVIDERS))})"
        )

    # Acquired INSIDE the retried function, so a backing-off attempt releases
    # its slot while it sleeps. Holding across the retry would let a few
    # rate-limited calls idle the whole allowance and stall every other caller.
    with _TTS_SEMAPHORE:
        audio_data = synth(text=text, config=provider_config)

    if not audio_data.format.is_pcm16:
        raise ValueError(
            f"TTS must output PCM_S16LE, got {audio_data.format.encoding}. "
            "Configure the provider to use PCM output format."
        )

    return audio_data
