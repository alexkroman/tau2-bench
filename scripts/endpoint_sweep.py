#!/usr/bin/env python3
"""Sweep endpointing thresholds by replaying recorded caller audio to STT.

    # what it would cost and which arms would run
    uv run python scripts/endpoint_sweep.py retail-default-341 --dry-run

    # the real sweep
    uv run python scripts/endpoint_sweep.py retail-default-341 \
        --arms 1600:3500 1000:3500 600:2560 --label retail341 --concurrency 6

    # recompute the table from events already on disk, spending nothing
    uv run python scripts/endpoint_sweep.py --report data/endpoint_sweep/retail341

`min_turn_silence` and `max_turn_silence` are CONNECTION parameters on
Universal Streaming v3, so one arm is one WebSocket and arms are independent.
That is what makes this affordable: the sweep needs the speech model only, not
the LLM, not TTS, not tau2's user simulator, and not a reward. It measures the
thing the thresholds actually control — where the service puts turn boundaries
in known audio — and nothing downstream of it.

## Why replay instead of running the harness

`AAI_MIN_TURN_SILENCE` on the voice-agent host changes endpointing, the LLM's
inputs, its tool calls, the caller's next line and the trajectory, all at once.
A reward delta from a graded run cannot be attributed to endpointing, and at 25
retail tasks with binary reward it usually cannot be detected either. Here the
audio is FIXED across arms, so a difference in turn boundaries is caused by the
thresholds and by nothing else.

The trade this is meant to price:

- too low  -> one caller utterance becomes several turns. The agent answers half
              a sentence, or acts on a truncated order id.
- too high -> dead air after the caller has finished. `min_turn_silence` is
              paid on EVERY turn, before the end-of-turn check even runs.

`scripts/latency.py` cannot see the second cost — its `user_transcript` clock
starts after endpointing has already fired — so `close_lag_ms` below is the
missing half of the caller-perceived gap.

## What it still does not tell you

Reward. Splitting a turn changes the conversation from that point on, and no
amount of replay reconstructs the divergent trajectory. Use this to get from a
grid to one candidate; spend the graded run on that candidate alone.

Billing note: streaming is billed per session wall-clock, and endpointing is
silence-duration sensitive, so audio MUST be paced in real time. An arm
therefore costs about as much as the audio lasts. `--dry-run` prints the
estimate first.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import difflib
import json
import os
import re
import statistics
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import websockets
from dotenv import load_dotenv

RUNS = Path("data/simulations")
OUT_ROOT = Path("data/endpoint_sweep")
WS_HOST = "wss://streaming.assemblyai.com/v3/ws"
DEFAULT_MODEL = "universal-3-5-pro"
# Settings held FIXED across every arm, so that a difference between arms is
# caused by the endpointing thresholds and by nothing else.
#
# Voice Focus is in here because without it this corpus is unmeasurable, not as
# a preference. Measured on one retail sim with 10 caller utterances, replayed
# whole: a bare connection reports 2 of them, and `min_turn_silence` /
# `max_turn_silence` / `mode` make no difference to that number — 1600:3500,
# the service defaults and `mode=balanced` all return exactly 2. Adding
# `voice_focus=near-field` takes the same audio to 8, and batch transcription
# of the same bytes reaches 9, which is what proves the audio was never the
# problem. The caller's channel carries a persona noise bed plus telephony
# noise that buries short backchannels ("Okay", "No backlight.") until voice
# isolation removes it.
#
# So a sweep without this would compare arms that all score the same and read
# as "endpointing does not matter here". It would be measuring VAD misses.
#
# The DEFAULT is 0.9 because that is what the voice-agent host actually runs,
# so an arm that says nothing about isolation reproduces the runs already on
# disk. It is the SDK's override of a documented service default of 0.7, and on
# that same sim 0.7 scored 9 of 10 against 0.9's 8 — which is why it is a
# sweepable field on `Arm` rather than a constant. See `Arm.params`.
FRAME_MS = 50
TAG_RE = re.compile(r"\[[a-z_ ]+\]", re.IGNORECASE)
# Finals come back formatted ("W2378156"), while the caller's script spells the
# same id out ("W, two, three, seven"). Comparing them raw reports a recall
# collapse that is purely a rendering difference, so number words are folded to
# digits before anything is compared.
NUM_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


@dataclass(frozen=True)
class Arm:
    """One connection's worth of settings: endpointing, plus the isolation it ran under.

    Voice Focus lives on the arm rather than in `BASELINE_PARAMS` because it is
    not only a fixed precondition, it is also a dimension worth sweeping — and
    because the two dimensions may INTERACT. Isolation changes which frames
    count as speech, so it changes where silence starts, which is the input
    `min_turn_silence` measures against. A best `min` found at one threshold is
    not automatically the best `min` at another.

    The design rule still holds within a sweep: vary one field across arms and
    hold the rest. Crossing both dimensions in one run is a factorial, not an
    A/B, and costs the product of the two.
    """

    min_ms: int
    max_ms: int
    # None = the service's own behaviour, which is to omit the parameter
    # entirely. That is a real arm, not a disabled one: on the smoke sim a
    # connection without it reported 2 of 10 caller utterances.
    vf_threshold: float | None = 0.9
    vf_mode: str = "near-field"

    @property
    def name(self) -> str:
        vf = "off" if self.vf_threshold is None else f"{self.vf_threshold:g}"
        return f"min{self.min_ms}-max{self.max_ms}-vf{vf}"

    @property
    def params(self) -> dict:
        p = {"min_turn_silence": self.min_ms, "max_turn_silence": self.max_ms}
        if self.vf_threshold is not None:
            p |= {
                "voice_focus": self.vf_mode,
                "voice_focus_threshold": self.vf_threshold,
            }
        return p


@dataclass
class Sim:
    audio: Path
    task: str
    gold: list[tuple[float, float, str]]

    @property
    def name(self) -> str:
        return f"{self.task}_{self.audio.parent.name[:12]}"


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #


def read_labels(path: Path) -> list[tuple[float, float, str]]:
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


def caller_pcm(wav: Path, gold, target_rate: int | None) -> tuple[bytes, int]:
    """The caller's channel of `both.wav`, mono PCM16 at `target_rate`.

    The recording is stereo, so the caller needs no separation — only
    identification. The channel is chosen by energy against the label tracks
    rather than by index, because nothing in the file states the channel order
    and picking wrong would measure the agent's pauses instead.

    Resampling is off by default. The harness records, and runs, at 8kHz ulaw
    (see `audio_format` on every tick of a sim), so the wav is already at full
    pipeline fidelity and upsampling only adds interpolation artefacts —
    measured, linear-upsampled 16kHz reported FEWER utterances than the native
    8kHz it came from. When it is enabled it is at least identical across arms,
    so it cannot move an arm-to-arm comparison.
    """
    with wave.open(str(wav)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM")
        rate, chans = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(
            -1, chans
        )

    if chans < 2:
        raise ValueError("expected stereo (caller + agent)")

    def energy(sig, spans):
        total = 0.0
        for s, e in spans[:4]:
            a, b = int(s * rate), min(int(e * rate), len(sig))
            if b > a:
                total += float(np.abs(sig[a:b].astype(np.int64)).sum())
        return total + 1.0

    user_spans = [(s, e) for s, e, _ in gold]
    agent_spans = user_spans  # replaced below when an agent track exists
    alab = wav.parent / "assistant_labels.txt"
    if alab.exists():
        agent_spans = [(s, e) for s, e, _ in read_labels(alab)]
    ratios = [
        energy(data[:, c], user_spans) / energy(data[:, c], agent_spans)
        for c in range(chans)
    ]
    sig = data[:, int(np.argmax(ratios))]

    if target_rate and target_rate != rate:
        n = int(round(len(sig) * target_rate / rate))
        src = np.linspace(0.0, len(sig) - 1, n)
        sig = np.interp(src, np.arange(len(sig)), sig).astype("<i2")
        rate = target_rate
    return sig.astype("<i2").tobytes(), rate


def discover(run_names: list[str]) -> list[Sim]:
    sims = []
    for name in run_names:
        root = Path(name) if Path(name).exists() else RUNS / name
        for audio in sorted(root.glob("artifacts/*/*/audio")):
            wav, ulab = audio / "both.wav", audio / "user_labels.txt"
            if not (wav.exists() and ulab.exists()):
                continue
            gold = [(s, e, t) for s, e, t in read_labels(ulab) if t.strip()]
            if gold:
                sims.append(Sim(audio=audio, task=audio.parts[-3], gold=gold))
    return sims


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #


async def replay(sim: Sim, arm: Arm, pcm: bytes, rate: int, args, out: Path) -> dict:
    """Stream one sim's caller audio through one arm, logging every message.

    Raw messages are written verbatim to `events.jsonl` and metrics are derived
    in a second pass, so a change of mind about how to score a split does not
    mean paying for the audio again.
    """
    params = {"speech_model": args.model, "sample_rate": rate} | arm.params
    for extra in args.param:
        k, _, v = extra.partition("=")
        params[k] = v
    url = f"{WS_HOST}?{urlencode(params)}"

    step = int(rate * 2 * FRAME_MS / 1000)  # 2 bytes/sample
    events: list[dict] = []
    sent = 0
    began = time.monotonic()

    def audio_ms() -> float:
        return sent / (rate * 2) * 1000

    async with websockets.connect(
        url, additional_headers={"Authorization": args.api_key}, ping_timeout=None
    ) as ws:

        async def drain() -> None:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                # `at_audio_ms` is the position in the audio when this arrived.
                # Audio time, not wall clock, is the unit that compares across
                # arms: it is immune to scheduler jitter on our side, though it
                # does still include one network hop.
                msg["at_audio_ms"] = round(audio_ms(), 1)
                msg["at_wall_s"] = round(time.monotonic() - began, 3)
                events.append(msg)

        reader = asyncio.create_task(drain())
        # Paced at 1.0x on a monotonic deadline. Endpointing is a function of
        # silence DURATION, so sending faster than real time does not merely
        # speed the sweep up, it changes the answer.
        deadline = time.monotonic()
        for off in range(0, len(pcm), step):
            await ws.send(pcm[off : off + step])
            sent = min(off + step, len(pcm))
            deadline += FRAME_MS / 1000
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))

        # Terminate finalises the open turn; without the drain window its final
        # `Turn` is lost and the last caller utterance reads as never heard.
        await ws.send(json.dumps({"type": "Terminate"}))
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(reader), timeout=args.drain_s)
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader

    out.mkdir(parents=True, exist_ok=True)
    with (out / "events.jsonl").open("w") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    (out / "gold.json").write_text(json.dumps(sim.gold))
    summary = score(sim.gold, events, arm)
    summary |= {"sim": str(sim.audio.parent), "task": sim.task, "arm": arm.name}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def normalize(text: str) -> list[str]:
    text = TAG_RE.sub(" ", text).replace("###STOP###", " ").lower()
    toks = []
    for tok in re.split(r"[^a-z0-9]+", text):
        if tok:
            toks.append(NUM_WORDS.get(tok, tok))
    return toks


def score(gold: list[tuple[float, float, str]], events: list[dict], arm: Arm) -> dict:
    """Attribute each final to a gold utterance, then price splits and lag.

    A final is attributed to the last gold utterance that had STARTED when the
    final arrived. Two finals landing in one utterance's window is the split
    this sweep exists to count; zero finals is an utterance the service never
    reported at all.
    """
    finals = [e for e in events if e.get("type") == "Turn" and e.get("end_of_turn")]
    starts = [s for s, _, _ in gold]

    per_utt = []
    for i, (start, end, text) in enumerate(gold):
        lo = start * 1000
        hi = starts[i + 1] * 1000 if i + 1 < len(starts) else float("inf")
        mine = [f for f in finals if lo <= f["at_audio_ms"] < hi]
        heard = " ".join(f.get("transcript", "") for f in mine)
        gold_toks, heard_toks = normalize(text), normalize(heard)
        hpool = list(heard_toks)
        hit = 0
        for t in gold_toks:
            if t in hpool:
                hpool.remove(t)
                hit += 1
        per_utt.append(
            {
                "gold": text,
                "heard": heard,
                "finals": len(mine),
                # Silence the caller sat through after finishing, before the
                # turn closed. This is the cost `latency.py` cannot see.
                "close_lag_ms": round(mine[-1]["at_audio_ms"] - end * 1000, 1)
                if mine
                else None,
                "recall": round(hit / len(gold_toks), 3) if gold_toks else None,
                "similarity": round(
                    difflib.SequenceMatcher(
                        None, "".join(gold_toks), "".join(heard_toks)
                    ).ratio(),
                    3,
                ),
                "eot_confidence": [f.get("end_of_turn_confidence") for f in mine],
            }
        )

    lags = [u["close_lag_ms"] for u in per_utt if u["close_lag_ms"] is not None]
    recalls = [u["recall"] for u in per_utt if u["recall"] is not None]
    # A final arriving within a frame of max_turn_silence after the caller
    # stopped is the acoustic fallback firing, not the semantic check. Counted
    # separately because only this class is content-blind.
    forced = sum(1 for lag in lags if abs(lag - arm.max_ms) <= FRAME_MS)
    return {
        "min_turn_silence": arm.min_ms,
        "max_turn_silence": arm.max_ms,
        "voice_focus": arm.vf_mode if arm.vf_threshold is not None else None,
        "voice_focus_threshold": arm.vf_threshold,
        "gold_utterances": len(gold),
        "finals": len(finals),
        "split_utterances": sum(1 for u in per_utt if u["finals"] > 1),
        "missed_utterances": sum(1 for u in per_utt if u["finals"] == 0),
        "forced_ends": forced,
        "close_lag_ms_p50": round(statistics.median(lags), 1) if lags else None,
        "close_lag_ms_p90": round(float(np.percentile(lags, 90)), 1) if lags else None,
        "recall_mean": round(statistics.mean(recalls), 3) if recalls else None,
        "errors": [e for e in events if e.get("type") == "Error"],
        "utterances": per_utt,
    }


def report(root: Path) -> int:
    """Aggregate every summary.json under an output root, arm by arm."""
    by_arm: dict[str, list[dict]] = {}
    for path in sorted(root.glob("*/*/summary.json")):
        s = json.loads(path.read_text())
        by_arm.setdefault(s["arm"], []).append(s)
    if not by_arm:
        print(f"no summaries under {root}", file=sys.stderr)
        return 1

    print(f"\n{root}")
    head = f"{'arm':<18} {'sims':>5} {'utts':>5} {'split':>7} {'missed':>7} {'forced':>7} {'lag p50':>8} {'lag p90':>8} {'recall':>7}"
    print(head)
    print("-" * len(head))
    for arm, rows in sorted(
        by_arm.items(),
        key=lambda kv: (
            kv[1][0]["min_turn_silence"],
            kv[1][0]["max_turn_silence"],
            kv[1][0].get("voice_focus_threshold") or 0,
        ),
    ):
        utts = sum(r["gold_utterances"] for r in rows)
        lags = [
            u["close_lag_ms"]
            for r in rows
            for u in r["utterances"]
            if u["close_lag_ms"] is not None
        ]
        recs = [
            u["recall"]
            for r in rows
            for u in r["utterances"]
            if u["recall"] is not None
        ]
        print(
            f"{arm:<18} {len(rows):>5} {utts:>5} "
            f"{sum(r['split_utterances'] for r in rows) / max(utts, 1) * 100:>6.1f}% "
            f"{sum(r['missed_utterances'] for r in rows) / max(utts, 1) * 100:>6.1f}% "
            f"{sum(r['forced_ends'] for r in rows) / max(utts, 1) * 100:>6.1f}% "
            f"{statistics.median(lags) if lags else float('nan'):>7.0f}ms "
            f"{np.percentile(lags, 90) if lags else float('nan'):>7.0f}ms "
            f"{statistics.mean(recs) if recs else float('nan'):>7.3f}"
        )
    print("\nsplit/missed/forced are percentages of gold caller utterances.")
    print(
        "lag = silence after the caller stopped before the turn closed, in audio time."
    )
    return 0


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def parse_arms(specs: list[str]) -> list[Arm]:
    """`min:max` at the incumbent isolation, or `min:max@<threshold|off>`."""
    arms = []
    for spec in specs:
        endpointing, _, vf = spec.partition("@")
        lo, _, hi = endpointing.partition(":")
        if vf == "":
            arms.append(Arm(int(lo), int(hi)))
        elif vf.lower() == "off":
            arms.append(Arm(int(lo), int(hi), vf_threshold=None))
        else:
            arms.append(Arm(int(lo), int(hi), vf_threshold=float(vf)))
    return arms


async def run_sweep(sims: list[Sim], arms: list[Arm], args) -> None:
    root = OUT_ROOT / args.label
    gate = asyncio.Semaphore(args.concurrency)
    # The documented streaming limit is on NEW sessions per minute, not on
    # concurrent ones, and exceeding it closes the socket with 1008. Spacing
    # opens is cheaper than discovering that mid-sweep.
    opened = asyncio.Lock()
    last_open = [0.0]
    done = [0]
    total = len(sims) * len(arms)

    async def one(sim: Sim, arm: Arm) -> dict | None:
        out = root / arm.name / sim.name
        if (out / "summary.json").exists() and not args.force:
            done[0] += 1
            return json.loads((out / "summary.json").read_text())
        async with gate:
            async with opened:
                wait = last_open[0] + 60.0 / args.sessions_per_min - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                last_open[0] = time.monotonic()
            try:
                pcm, rate = caller_pcm(
                    sim.audio / "both.wav", sim.gold, args.sample_rate
                )
            except (ValueError, wave.Error, EOFError) as exc:
                print(f"  skip {sim.name}: {exc}", file=sys.stderr)
                return None
            try:
                res = await replay(sim, arm, pcm, rate, args, out)
            except Exception as exc:  # noqa: BLE001 - one bad session must not end the sweep
                print(
                    f"  FAIL {arm.name} {sim.name}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return None
            done[0] += 1
            print(
                f"  [{done[0]}/{total}] {arm.name} {sim.name}: "
                f"{res['finals']} finals / {res['gold_utterances']} utts, "
                f"split {res['split_utterances']}, lag p50 {res['close_lag_ms_p50']}ms",
                flush=True,
            )
            return res

    # Sim-major, not arm-major: every arm advances together, so a partial
    # sweep can still be read as a paired comparison on the sims finished so
    # far. Arm-major order finishes arm 1 completely before arm 2 starts,
    # which makes an interrupted run worthless for comparison.
    await asyncio.gather(*(one(s, a) for s in sims for a in arms))
    report(root)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "runs", nargs="*", help="tau2 run names or paths under data/simulations"
    )
    p.add_argument(
        "--arms",
        nargs="+",
        default=["1600:3500", "1000:3500", "600:2560"],
        help="min:max[@vf] in ms, where vf is a voice_focus_threshold or 'off' "
        "(default 0.9, what the host runs). Keep 1600:3500 as the reference arm.",
    )
    p.add_argument(
        "--label", default="sweep", help="output subdirectory under data/endpoint_sweep"
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--sample-rate",
        type=int,
        default=0,
        help="resample the caller channel to this rate; 0 (default) sends the wav's own "
        "8kHz, which is the rate the harness itself runs at",
    )
    p.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="K=V",
        help="extra connection parameter, repeatable (e.g. --param mode=balanced)",
    )
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--sessions-per-min", type=float, default=30.0)
    p.add_argument(
        "--drain-s",
        type=float,
        default=10.0,
        help="seconds to wait for finals after Terminate",
    )
    p.add_argument("--limit", type=int, default=0, help="use only the first N sims")
    p.add_argument(
        "--force",
        action="store_true",
        help="re-replay sims that already have a summary",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the cost estimate, then stop",
    )
    p.add_argument(
        "--report", metavar="DIR", help="re-aggregate an existing output root and exit"
    )
    args = p.parse_args(argv)

    if args.report:
        return report(Path(args.report))
    if not args.runs:
        p.print_help()
        return 2

    load_dotenv()  # same key source as the rest of tau2 (see utils/utils.py)
    args.api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    arms = parse_arms(args.arms)
    sims = discover(args.runs)
    if args.limit:
        sims = sims[: args.limit]
    if not sims:
        print("no sims with stereo audio and caller labels found", file=sys.stderr)
        return 1

    audio_s = sum(g[-1][1] for s in sims for g in [s.gold])
    print(f"{len(sims)} sims x {len(arms)} arms = {len(sims) * len(arms)} sessions")
    print(f"arms: {', '.join(a.name for a in arms)}")
    print(
        f"paced at 1.0x: ~{audio_s * len(arms) / 60:.0f} session-minutes total, "
        f"~{audio_s * len(arms) / 60 / max(args.concurrency, 1):.0f} min wall clock "
        f"at concurrency {args.concurrency}"
    )
    if args.dry_run:
        return 0
    if not args.api_key:
        print("ASSEMBLYAI_API_KEY is not set", file=sys.stderr)
        return 1

    asyncio.run(run_sweep(sims, arms, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
