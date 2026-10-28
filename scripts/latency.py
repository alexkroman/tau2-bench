#!/usr/bin/env python3
"""Per-turn response latency for a tau2 audio-native run, from the wire log.

    uv run python scripts/latency.py <run> [<run> ...]

For each `user_transcript` (the moment the host has the caller's finished
words) it reports the wait until the next thing the caller could perceive —
the first `agent_transcript` of the reply — and, separately, the wait until
the first `tool_call`, so a turn that went silent because the model was
chaining tools reads differently from one that was simply slow to think.

Turns with no reply at all before the next caller turn are counted as DROPPED:
on a phone call that is an unanswered question, and it is the failure the
reward never names.
"""

from __future__ import annotations

import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

RUNS = Path("data/simulations")
EVENT_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+).*AAI event: ([a-z_]+)")


def events(log: Path) -> list[tuple[datetime, str]]:
    out = []
    with log.open(errors="replace") as fh:
        for line in fh:
            m = EVENT_RE.match(line)
            if m:
                out.append((datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S.%f"), m[2]))
    return out


def turns(evs: list[tuple[datetime, str]]) -> list[dict]:
    """One record per caller turn: wait to first word, to first tool, dropped?"""
    out = []
    pending: dict | None = None
    for at, kind in evs:
        if kind == "user_transcript":
            if pending:
                out.append(pending)
            pending = {"at": at, "word": None, "tool": None}
        elif pending:
            if kind == "agent_transcript" and pending["word"] is None:
                pending["word"] = (at - pending["at"]).total_seconds()
            elif kind == "tool_call" and pending["tool"] is None:
                pending["tool"] = (at - pending["at"]).total_seconds()
    if pending:
        out.append(pending)
    return out


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for run in sys.argv[1:]:
        print(f"\n## {run}")
        allw: list[float] = []
        dropped = tot = 0
        for log in sorted((RUNS / run / "artifacts").glob("*/*/task.log")):
            task = log.parts[-3]
            recs = turns(events(log))
            waits = [r["word"] for r in recs if r["word"] is not None]
            drop = [r for r in recs if r["word"] is None]
            allw += waits
            dropped += len(drop)
            tot += len(recs)
            slow = sum(1 for w in waits if w > 5)
            print(
                f"  {task:8} turns {len(recs):3}  dropped {len(drop):2}  "
                f"first-word p50 {statistics.median(waits) if waits else float('nan'):5.1f}s "
                f"p90 {pct(waits, 0.9):5.1f}s  max {max(waits, default=float('nan')):5.1f}s  >5s: {slow}"
            )
        print(
            f"  TOTAL    turns {tot:3}  dropped {dropped:2} ({dropped / max(tot, 1):.0%})  "
            f"first-word p50 {statistics.median(allw) if allw else float('nan'):5.1f}s "
            f"p90 {pct(allw, 0.9):5.1f}s  mean {statistics.mean(allw) if allw else float('nan'):5.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
