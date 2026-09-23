#!/usr/bin/env python3
"""WHY was agent audio destroyed? Split the loss by cause.

    uv run python scripts/destroyed_why.py <run> [<run> ...]

tau2 discards its whole buffered agent audio on `speech_started`. Some of that
is correct — the caller really took the floor and the agent should stop. The
rest is not: the host released the speaking edge while it still held the floor,
and the client threw away speech the caller had not heard.

The two are told apart by whether the host ALSO cancelled the reply.
`AAICancelledEvent` is the host saying "I am aborting this reply"; a
`speech_started` with no cancel beside it is the host saying "the caller spoke,
but I am NOT stopping" — and the client stops anyway. Only the second class is
a defect, and it is the one worth engineering against.

Correlation is by TICK BLOCK, not by timestamp. The discard line is an
indented continuation with no timestamp of its own, and stamping it with the
nearest preceding one attributes it to the tick-completion log rather than to
the wire event — which silently classified every genuine barge-in as a tail
cut. The tick summary already lists the events that landed in that tick, which
is the correlation the log hands you for free.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

RUNS = Path("data/simulations")
TICK_RE = re.compile(r"^Tick (\d+)$")
DISCARD_RE = re.compile(r"INTERRUPTED: (\d+) bytes \((\d+)ms\)")
EVENT_RE = re.compile(r"^\s+- (AAI\w+Event): (\d+)")


def classify(log: Path) -> list[tuple[str, int]]:
    """(cause, ms) for every lossy discard, attributed via its tick block."""
    out: list[tuple[str, int]] = []
    events: dict[str, int] = {}
    in_tick = False
    for line in log.read_text(errors="replace").splitlines():
        if TICK_RE.match(line.strip()) and line.startswith("Tick "):
            in_tick, events = True, {}
            continue
        if in_tick:
            m = EVENT_RE.match(line)
            if m:
                events[m[1]] = int(m[2])
                continue
            d = DISCARD_RE.search(line)
            if d:
                ms = int(d[2])
                if ms:
                    out.append(
                        (
                            "BARGE_IN"
                            if events.get("AAICancelledEvent")
                            else "UNGATED",
                            ms,
                        )
                    )
                in_tick = False
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for run in sys.argv[1:]:
        buckets: dict[str, list[int]] = {"BARGE_IN": [], "UNGATED": []}
        for log in sorted((RUNS / run / "artifacts").glob("*/*/task.log")):
            for kind, ms in classify(log):
                buckets[kind].append(ms)
        total = sum(sum(v) for v in buckets.values()) or 1
        print(f"\n## {run}   {total / 1000:.1f}s destroyed")
        for kind, label in (
            ("BARGE_IN", "caller really interrupted — correct, not a defect"),
            ("UNGATED", "host did NOT cancel; client stopped anyway — LOSS"),
        ):
            v = buckets[kind]
            print(
                f"  {kind:9} {len(v):3} events  {sum(v) / 1000:6.1f}s  "
                f"{sum(v) / total:4.0%}  — {label}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
