#!/usr/bin/env python3
"""Is preemptive generation paying for itself? A within-run natural experiment.

    uv run python scripts/speculation.py <run> [<run> ...]

Pairs the agent's own debug log (which says when a speculation started, was
adopted, and was poisoned) against the wire log (which says when the caller
finished speaking and when the first agent word came back), then compares
time-to-first-word on ADOPTED turns against turns that ran cold.

Two numbers decide the feature, and the second is the one its own design doc
never had:

- the head start adoption actually buys, and
- what a POISONED adoption costs, since a tool call arriving after adoption
  throws the whole generation away and reissues the request.

Confounded on purpose and worth saying out loud: adoption requires a
high-confidence interim, so adopted turns are the CLEANLY-ENDED utterances.
Some of any gap is the utterance being easy, not the speculation helping. This
bounds the effect; only an A/B with the flag flipped settles it.
"""

from __future__ import annotations

import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

RUNS = Path("data/simulations")

SERVER_RE = re.compile(
    r"^(\d{4}-\d\d-\d\dT[\d:.]+Z) Pipeline (speculation \w+|turn committed)"
)
SID_RE = re.compile(r"sid: '([0-9a-f-]+)'")
HEAD_RE = re.compile(r"headStartMs: (\d+)")
WIRE_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+).*AAI event: ([a-z_]+)")


def _server(path: Path) -> list[tuple[datetime, str, str]]:
    """(time, kind, blob) for each speculation/turn line, blob = the following braces."""
    out: list[tuple[datetime, str, str]] = []
    lines = path.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        m = SERVER_RE.match(line)
        if not m:
            continue
        at = datetime.strptime(m[1], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
        out.append((at, m[2], " ".join(lines[i : i + 5])))
    return out


def _wire_turns(log: Path) -> list[tuple[datetime, float | None]]:
    """(user_transcript time, seconds to the next agent_transcript) per caller turn."""
    evs: list[tuple[datetime, str]] = []
    with log.open(errors="replace") as fh:
        for line in fh:
            m = WIRE_RE.match(line)
            if m:
                evs.append(
                    (
                        # The wire log stamps LOCAL time while the server log
                        # stamps UTC; astimezone() on a naive value attaches the
                        # local zone, which is what makes the two comparable.
                        datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S.%f").astimezone(),
                        m[2],
                    )
                )
    out: list[tuple[datetime, float | None]] = []
    pending: datetime | None = None
    for at, kind in evs:
        if kind == "user_transcript":
            if pending is not None:
                out.append((pending, None))
            pending = at
        elif kind == "agent_transcript" and pending is not None:
            out.append((pending, (at - pending).total_seconds()))
            pending = None
    if pending is not None:
        out.append((pending, None))
    return out


def _fmt(xs: list[float]) -> str:
    if not xs:
        return "n=0"
    s = sorted(xs)
    p90 = s[min(len(s) - 1, int(0.9 * len(s)))]
    return f"n={len(s):3} p50 {statistics.median(s):5.2f}s p90 {p90:5.2f}s mean {statistics.mean(s):5.2f}s"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for run in sys.argv[1:]:
        server = RUNS / f"{run}.server.log"
        print(f"\n## {run}")
        if not server.exists():
            print(f"  no server log ({server}) — rerun via run-bench.sh to capture one")
            continue

        events = _server(server)
        started = [e for e in events if e[1] == "speculation started"]
        adopted = [e for e in events if e[1] == "speculation adopted"]
        poisoned = [e for e in events if e[1] == "speculation poisoned"]
        heads = [int(m[1]) for _, _, blob in adopted if (m := HEAD_RE.search(blob))]

        print(
            f"  started {len(started)}  adopted {len(adopted)}  poisoned-after-adopt {len(poisoned)}"
        )
        if heads:
            print(f"  claimed head start: {_fmt([h / 1000 for h in heads])}")
        if adopted:
            print(f"  poison rate among adoptions: {len(poisoned) / len(adopted):.0%}")

        # Cost of each poison: how long the discarded generation had been running.
        costs = []
        for pt, _, _ in poisoned:
            prior = [(pt - at).total_seconds() for at, _, _ in adopted if at <= pt]
            if prior:
                costs.append(min(prior))
        if costs:
            print(f"  discarded generation per poison: {_fmt(costs)}")

        # Wire side: adopted turns vs cold turns, by nearest user_transcript.
        adopt_times = [at for at, _, _ in adopted]
        warm: list[float] = []
        cold: list[float] = []
        for log in sorted((RUNS / run / "artifacts").glob("*/*/task.log")):
            for at, secs in _wire_turns(log):
                if secs is None:
                    continue
                # An adoption is logged within a second of the final it matched.
                near = any(abs((a - at).total_seconds()) < 1.0 for a in adopt_times)
                (warm if near else cold).append(secs)
        print(f"  first word, ADOPTED turns: {_fmt(warm)}")
        print(f"  first word, cold turns:    {_fmt(cold)}")
        if warm and cold:
            print(
                f"  observed gap (cold p50 - adopted p50): "
                f"{statistics.median(cold) - statistics.median(warm):+.2f}s "
                f"— confounded, see the module docstring"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
