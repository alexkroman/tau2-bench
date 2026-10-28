"""Caller-voice synthesis via AssemblyAI streaming TTS.

A second synthesis provider beside `elevenlabs_utils`, added because the
ElevenLabs quota is a hard stop: it drained mid-run and turned 54 of 114 sims
into `infrastructure_error`, with `'code': 'quota_exceeded'` and 0 credits
remaining. A benchmark whose caller loses its voice half way through produces a
run that looks like data and is not.

## The protocol, and why it is a WebSocket

AssemblyAI exposes no HTTP text-to-speech route (`/v1/tts`, `/v2/tts` and the
streaming host's own `/v1/tts` all answer 404 — probed). Synthesis is
`wss://streaming-tts.assemblyai.com/v1/ws/`, so a whole utterance is collected
by draining frames rather than read from one response body:

- the API key goes in `Authorization` **raw**, not as `Bearer <key>`;
- `voice` and `sample_rate` are QUERY parameters, which is what fixes the
  output format — there is no format negotiation in-band;
- the client sends `{"type":"Generate","text":...}` then `{"type":"Flush"}`;
- the server may open with `Begin`, then sends `{"type":"Audio","audio":<base64
  PCM16 LE>}` frames, and ends the utterance with `FlushDone` (or an `Audio`
  frame flagged `is_final`);
- `{"type":"Terminate"}` closes cleanly; `Error` frames carry `error_code`.

Measured against production: one 54-character utterance returned 49 `Audio`
frames totalling 122,880 bytes — 3.84s of 16 kHz PCM16 — with frame kinds
`Begin`, `Audio`, `FlushDone`.

## What it does NOT do, stated because the caller relies on it

ElevenLabs v3 renders `[cough]`, `[sneeze]`, `[sniffle]` and `[pause]` as real
audio, and the voice personas use them to make a simulated caller sound human.
This service has no audio-tag vocabulary, so those markers would be SPOKEN as
words — "bracket cough bracket" — which is worse than dropping them. They are
stripped here and `[pause]` becomes an ellipsis, the same substitution
`elevenlabs_utils` already makes for its non-v3 models.

So this provider is not a drop-in acoustic equivalent, and a run using it is
**not comparable to a run using ElevenLabs**. It is comparable to other
AssemblyAI-voiced runs. See `SYNTHESIS_PROVIDERS` in `synthesize.py`.
"""

import asyncio
import base64
import json
import os
import re
from copy import deepcopy

import websockets
from loguru import logger

from tau2.data_model.audio import AudioData
from tau2.data_model.voice import AssemblyAITTSConfig

ASSEMBLYAI_TTS_HOST = "streaming-tts.assemblyai.com"

# Audio tags this service cannot render. Stripped rather than passed through:
# the alternative is the voice reading the word "cough" aloud.
_AUDIO_TAG_PATTERN = re.compile(
    r"\[(cough|sneeze|sniffle|laughs?|sighs?)\]", re.IGNORECASE
)
_PAUSE_TAG_PATTERN = re.compile(r"\[pause\]", re.IGNORECASE)

# A whole utterance is 2-6 seconds of audio arriving as ~50 frames. The deadline
# bounds one recv, not the utterance, so a slow first frame does not truncate a
# long one.
_FRAME_TIMEOUT_S = 30.0


def strip_unsupported_tags(text: str) -> str:
    """Remove audio tags and turn `[pause]` into an ellipsis — see module doc."""
    text = _PAUSE_TAG_PATTERN.sub("...", text)
    text = _AUDIO_TAG_PATTERN.sub("", text)
    # Collapse the double spaces a removed tag leaves behind.
    return re.sub(r"\s{2,}", " ", text).strip()


async def _synthesize(text: str, voice: str, sample_rate: int, api_key: str) -> bytes:
    url = f"wss://{ASSEMBLYAI_TTS_HOST}/v1/ws/?voice={voice}&sample_rate={sample_rate}"
    chunks: list[bytes] = []
    async with websockets.connect(
        url, additional_headers={"Authorization": api_key}
    ) as ws:
        await ws.send(json.dumps({"type": "Generate", "text": text}))
        await ws.send(json.dumps({"type": "Flush"}))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=_FRAME_TIMEOUT_S)
            if not isinstance(raw, str):
                continue
            frame = json.loads(raw)
            kind = frame.get("type")
            if kind == "Audio":
                chunks.append(base64.b64decode(frame.get("audio") or ""))
                if frame.get("is_final"):
                    break
            elif kind == "FlushDone":
                break
            elif kind == "Error":
                # Raised, never logged-and-returned: an empty utterance reads
                # downstream as the caller saying nothing, which scores as a
                # silent agent failure rather than the API fault it is.
                raise RuntimeError(
                    f"AssemblyAI TTS error {frame.get('error_code')}: {frame.get('error')}"
                )
        # Best-effort: the audio is already in hand, so a failed close must not
        # lose an utterance.
        try:
            await ws.send(json.dumps({"type": "Terminate"}))
        except Exception:  # noqa: BLE001
            pass
    return b"".join(chunks)


def tts_assemblyai(text: str, config: AssemblyAITTSConfig) -> AudioData:
    """Text to speech using AssemblyAI streaming TTS.

    Args:
        text: The text to synthesize
        config: The configuration to use
    Returns:
        AudioData carrying 16-bit little-endian PCM at the configured rate
    """
    api_key = config.api_key or os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise ValueError("ASSEMBLYAI_API_KEY not found in config or environment")

    spoken = strip_unsupported_tags(text)
    if not spoken:
        raise ValueError(
            f"AssemblyAI TTS: nothing left to speak after tag removal: {text!r}"
        )

    preview = spoken[:50] + "..." if len(spoken) > 50 else spoken
    logger.debug(
        f"AssemblyAI TTS: synthesizing '{preview}' "
        f"(voice={config.voice}, sample_rate={config.sample_rate})"
    )

    audio_bytes = asyncio.run(
        _synthesize(spoken, config.voice, config.sample_rate, api_key)
    )

    if len(audio_bytes) == 0:
        raise ValueError(f"AssemblyAI TTS returned empty audio for text: {spoken!r}")

    logger.debug(f"AssemblyAI TTS: received {len(audio_bytes)} bytes of audio")
    return AudioData(data=audio_bytes, format=deepcopy(config.output_audio_format))
