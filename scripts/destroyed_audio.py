#!/usr/bin/env python3
"""How much agent speech the client threw away, per run.

    uv run python scripts/destroyed_audio.py <run> [<run> ...]

tau2's audio-native adapter discards its whole buffered agent audio on
`speech_started` and logs the amount (`⚡ INTERRUPTED: N bytes (Nms)`). That
line is the only direct measure of speech the caller was denied, and it is
independent of reward — a run can score well while the caller heard half of it.

Zero-byte discards are counted separately: they are edges that arrived with an
empty buffer and cost nothing, so folding them into the event count makes a run
look worse the more harmless edges it had.
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

RUNS = Path("data/simulations")
DISCARD_RE = re.compile(r"INTERRUPTED: (\d+) bytes \((\d+)ms\)")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rows = []
    for run in sys.argv[1:]:
        events: list[int] = []
        for log in sorted((RUNS / run / "artifacts").glob("*/*/task.log")):
            for m in DISCARD_RE.finditer(log.read_text(errors="replace")):
                events.append(int(m[2]))
        lost = [ms for ms in events if ms > 0]
        rows.append(
            (
                run,
                len(events),
                len(lost),
                sum(lost) / 1000,
                statistics.median(lost) / 1000 if lost else 0.0,
                max(lost, default=0) / 1000,
            )
        )

    print(f"{'run':34}{'edges':>7}{'lossy':>7}{'total':>9}{'p50':>8}{'max':>8}")
    for run, n, nz, total, p50, mx in rows:
        print(f"{run:34}{n:>7}{nz:>7}{total:>8.1f}s{p50:>7.1f}s{mx:>7.1f}s")

    if len(rows) == 2:
        a, b = rows
        if a[3]:
            print(f"\n{b[0]} vs {a[0]}: {(b[3] - a[3]) / a[3]:+.0%} destroyed audio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
