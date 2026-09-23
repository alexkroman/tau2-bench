#!/usr/bin/env python3
"""Intra-utterance silence gaps in the caller channel of finished tau2 runs.

    uv run python scripts/caller_gaps.py <run> [<run> ...]

This is the OFFLINE half of endpointing tuning: it needs nothing but the wavs
already on disk, and it answers one question — how much silence appears INSIDE
a caller utterance, which is the silence an endpointer must tolerate without
splitting the turn.

`both.wav` is stereo, so the caller is already on its own channel, and
`user_labels.txt` carries the harness's own utterance boundaries. So utterance
starts and ends are ground truth here; only the structure *inside* an utterance
is estimated, with a 20ms energy VAD referenced to each utterance's own peak.

## What this can and cannot decide

A gap wider than `min_turn_silence` is necessary but NOT sufficient for a turn
to split: on `universal-3-5-pro` that silence only triggers an end-of-turn
CHECK, and a turn whose transcript reads incomplete ("Y, U, S, U,") stays open
and emits a partial instead. Only `max_turn_silence` force-ends regardless of
content. So the cut rates printed below are UPPER BOUNDS — use them to pick
which thresholds are worth replaying, not as predicted truncation rates. The
real rate comes from `scripts/endpoint_sweep.py`, which asks the service.

Reward is a further step removed and is not derivable from audio at all: a
split turn changes what the agent says, which changes what the caller says
next, and the trajectory diverges from there.
"""

from __future__ import annotations

import re
import sys
import wave
from pathlib import Path

import numpy as np

RUNS = Path("data/simulations")
FRAME_MS = 20
# -24dB from the utterance's own peak. Referenced to the peak rather than a
# fixed level because the caller's TTS output is not level-normalised across
# personas; FLOOR keeps a near-silent utterance from making its own noise
# floor look like speech.
REL_THRESHOLD = 0.06
FLOOR = 40
# Tags the ElevenLabs personas use. Tracked separately because v3 renders
# `[pause]` and `[sniffle]` as real dead air, which puts those utterances in a
# different population from ordinary speech — see the split this prints.
TAG_RE = re.compile(r"\[[a-z_ ]+\]", re.IGNORECASE)
# A sentence end followed by more speech is the case semantic endpointing
# cannot protect: the transcript so far reads complete, so the check fires.
MID_SENTENCE_RE = re.compile(r"[.!?]\s+\S")


def read_stereo(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return frames.reshape(-1, w.getnchannels()), w.getframerate()


def read_labels(path: Path) -> list[tuple[float, float, str]]:
    """Audacity-style label track: start<TAB>end<TAB>text."""
    out = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            out.append(
                (float(parts[0]), float(parts[1]), parts[2] if len(parts) > 2 else "")
            )
        except ValueError:
            continue
    return out


def frame_rms(sig: np.ndarray, rate: int, t0: float, t1: float) -> np.ndarray:
    step = int(rate * FRAME_MS / 1000)
    a, b = int(t0 * rate), min(int(t1 * rate), len(sig))
    n = (b - a) // step
    if n <= 0:
        return np.empty(0)
    block = sig[a : a + n * step].astype(np.float64).reshape(n, step)
    return np.sqrt((block**2).mean(axis=1))


def caller_channel(stereo: np.ndarray, rate: int, user, agent) -> int:
    """Whichever channel is loud during caller labels and quiet during agent ones.

    Derived per sim rather than hardcoded: channel order is not something the
    recording states, and guessing it wrong silently measures the agent's
    pauses instead of the caller's.
    """
    best, best_ratio = 0, -1.0
    for ci in range(stereo.shape[1]):
        sig = stereo[:, ci]
        u = sum(frame_rms(sig, rate, s, e).sum() for s, e, _ in user[:4]) + 1
        a = sum(frame_rms(sig, rate, s, e).sum() for s, e, _ in agent[:4]) + 1
        if u / a > best_ratio:
            best, best_ratio = ci, u / a
    return best


def internal_gaps(sig: np.ndarray, rate: int, t0: float, t1: float) -> list[int]:
    """Silence runs strictly inside the utterance, in ms.

    A run touching either edge is dropped: leading silence is label slack and
    trailing silence is the end of the turn, neither of which an endpointer
    has to tolerate.
    """
    e = frame_rms(sig, rate, t0, t1)
    if e.size == 0:
        return []
    quiet = e < max(e.max() * REL_THRESHOLD, FLOOR)
    gaps, run = [], 0
    for i, q in enumerate(quiet):
        if q:
            run += 1
            continue
        if run and i - run > 0:  # a run at index 0 is leading silence
            gaps.append(run * FRAME_MS)
        run = 0
    return gaps  # a trailing run is deliberately never appended


def collect(run_dirs: list[str]) -> list[dict]:
    out = []
    for name in run_dirs:
        root = Path(name) if Path(name).exists() else RUNS / name
        for audio in sorted(root.glob("artifacts/*/*/audio")):
            wav = audio / "both.wav"
            ulab, alab = audio / "user_labels.txt", audio / "assistant_labels.txt"
            if not (wav.exists() and ulab.exists() and alab.exists()):
                continue
            user, agent = read_labels(ulab), read_labels(alab)
            if not user or not agent:
                continue
            try:
                stereo, rate = read_stereo(wav)
            except (ValueError, wave.Error, EOFError) as exc:
                print(f"  skipped {wav}: {exc}", file=sys.stderr)
                continue
            if stereo.ndim != 2 or stereo.shape[1] < 2:
                continue
            sig = stereo[:, caller_channel(stereo, rate, user, agent)]
            for start, end, text in user:
                # Below ~0.6s an utterance is "Okay" / "Yes" and has no room
                # for an internal pause; including them only dilutes the tail.
                if end - start < 0.6:
                    continue
                gaps = internal_gaps(sig, rate, start, end)
                out.append(
                    {
                        "sim": str(audio.parent),
                        "dur": end - start,
                        "max_gap": max(gaps) if gaps else 0,
                        "gaps": gaps,
                        "tagged": bool(TAG_RE.search(text)),
                        "mid_sentence": bool(
                            MID_SENTENCE_RE.search(TAG_RE.sub(" ", text))
                        ),
                        "text": text,
                    }
                )
    return out


def describe(title: str, utts: list[dict], thresholds: list[int]) -> None:
    if not utts:
        return
    mx = np.array(sorted(u["max_gap"] for u in utts))
    pcts = " ".join(f"p{p}={int(np.percentile(mx, p))}" for p in (50, 90, 95, 99))
    print(f"\n{title}: n={len(mx)}  {pcts} max={mx[-1]}")
    print(
        "  upper-bound split rate:  "
        + "  ".join(f"{t}ms {100 * (mx >= t).mean():.0f}%" for t in thresholds)
    )


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    thresholds = [400, 600, 800, 1000, 1200, 1600, 2000, 2560, 3500]
    utts = collect(argv)
    if not utts:
        print("no sims with stereo audio and label tracks found", file=sys.stderr)
        return 1

    sims = len({u["sim"] for u in utts})
    print(
        f"sims={sims}  utterances>=0.6s={len(utts)}  internal gaps={sum(len(u['gaps']) for u in utts)}"
    )
    describe("all utterances", utts, thresholds)
    describe("WITH [tag] markers", [u for u in utts if u["tagged"]], thresholds)
    describe("no tags", [u for u in utts if not u["tagged"]], thresholds)
    # The population semantic endpointing cannot save: a complete sentence,
    # then more speech. Everything else relies on the transcript reading
    # incomplete at the moment the check fires.
    describe(
        "sentence end mid-utterance", [u for u in utts if u["mid_sentence"]], thresholds
    )

    print("\nwidest internal gaps:")
    for u in sorted(utts, key=lambda u: -u["max_gap"])[:12]:
        flags = ("tag " if u["tagged"] else "") + ("sent" if u["mid_sentence"] else "")
        print(
            f"  {u['max_gap']:>5}ms  dur={u['dur']:>5.1f}s  {flags:<8} {u['text'][:64]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
