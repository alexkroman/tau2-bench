#!/usr/bin/env python3
"""Print one tau2 audio-native simulation as a readable conversation.

    uv run python scripts/transcript.py <run> [<task_id> ...]

Ticks are the wire, not a transcript: each carries an `agent_chunk` and a
`user_chunk` that are usually silence, and speech arrives split across dozens
of consecutive chunks. This collapses consecutive same-speaker text into one
line, interleaves tool calls and results at the tick where they landed, and
stamps every line with seconds since the call started — so a gap in the
conversation reads as a gap on the page.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

RUNS = Path("data/simulations")
MAX_RESULT_CHARS = 400


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _chunk_text(chunk: dict | None) -> str:
    if not chunk:
        return ""
        # content is None on pure-audio chunks
    return (chunk.get("content") or "").strip()


def render(sim: dict) -> str:
    ticks = sim["ticks"]
    if not ticks:
        return "(no ticks)"
    t0 = _ts(ticks[0]["timestamp"])
    out: list[str] = []
    pending: tuple[str, float, list[str]] | None = None  # speaker, t, parts

    def flush() -> None:
        nonlocal pending
        if pending:
            who, at, parts = pending
            text = " ".join(parts).strip()
            if text:
                out.append(f"[{at:7.1f}s] {who}: {text}")
        pending = None

    for tick in ticks:
        at = (_ts(tick["timestamp"]) - t0).total_seconds()
        for who, key in (("AGENT", "agent_chunk"), ("USER ", "user_chunk")):
            text = _chunk_text(tick.get(key))
            if not text:
                continue
            if pending and pending[0] == who:
                pending[2].append(text)
            else:
                flush()
                pending = (who, at, [text])
        for call in tick.get("agent_tool_calls") or []:
            flush()
            args = json.dumps(call.get("arguments"), separators=(",", ":"))
            out.append(f"[{at:7.1f}s]   -> {call.get('name')}({args})")
        for res in tick.get("agent_tool_results") or []:
            flush()
            body = str(res.get("content"))
            if len(body) > MAX_RESULT_CHARS:
                body = f"{body[:MAX_RESULT_CHARS]}... [{len(body)} chars total]"
            err = " ERROR" if res.get("error") else ""
            out.append(f"[{at:7.1f}s]   <-{err} {body}")
    flush()
    return "\n".join(out)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run = RUNS / sys.argv[1]
    wanted = set(sys.argv[2:])
    results = json.loads((run / "results.json").read_text())
    for row in sorted(results["simulation_index"], key=lambda r: int(r["task_id"])):
        if wanted and row["task_id"] not in wanted:
            continue
        sim = json.loads((run / "simulations" / f"{row['id']}.json").read_text())
        info = sim.get("reward_info") or {}
        print(f"\n{'=' * 78}")
        print(
            f"task {row['task_id']}  reward {row['reward']}  "
            f"{row['termination_reason']}  {row['duration']:.0f}s  "
            f"breakdown {info.get('reward_breakdown')}"
        )
        print("=" * 78)
        print(render(sim))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
