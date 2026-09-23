# Endpointing and voice-isolation thresholds, measured by replay

Two sweeps over `retail-default-341` (29 sims, 25 retail tasks, 384 caller
utterances), replaying each sim's recorded caller channel to Universal
Streaming v3 and comparing where the service put turn boundaries.

    uv run python scripts/caller_gaps.py retail-default-341        # offline shortlist
    uv run python scripts/endpoint_sweep.py retail-default-341 \
        --arms 1600:3500 1000:3500 600:3500 300:3500 1600:2560 --label retail341
    uv run python scripts/endpoint_sweep.py retail-default-341 \
        --arms 1600:3500@off 1600:3500@0.7 1600:3500@0.8 1600:3500@0.9 --label retail341-vf
    uv run python scripts/endpoint_sweep.py retail-default-341 \
        --arms 1600:3500 1000:2560 --label retail341-combo

No LLM, no TTS, no user simulator, no reward. The audio is FIXED across arms,
so a difference in turn boundaries is caused by the connection parameters and
by nothing else — which is exactly what a graded run cannot establish, since
`AAI_MIN_TURN_SILENCE` moves endpointing, the model's inputs, its tool calls
and the caller's next line all at once.

`split` / `missed` / `forced` are percentages of gold caller utterances.
`lag` is the silence after the caller stopped before the turn closed, in audio
time — the half of caller-perceived latency that `scripts/latency.py`
structurally cannot see, because its `user_transcript` clock starts after
endpointing has already fired.

## The noise floor comes first

All three sweeps included an identical `1600:3500 @ 0.9` arm, run in different
wall-clock windows. Paired on the same 29 sims:

| window | split | missed | forced | lag p50 | lag p90 | recall |
| --- | --- | --- | --- | --- | --- | --- |
| endpointing sweep | 3.6% | 14.1% | 1.0% | 2200ms | 4000ms | 0.689 |
| voice-focus sweep | 4.2% | 14.1% | 1.0% | 2100ms | 3950ms | 0.684 |
| combo sweep | 3.6% | 14.1% | 2.6% | 2200ms | 4000ms | 0.692 |
| **range = noise floor** | **0.5pp** | **0.0pp** | **1.6pp** | **100ms** | **50ms** | **0.007** |

That is the run-to-run floor, and every delta below is quoted against it.
`RESULTS-u35-prod-vs-u36-sandbox.md:137` lists cluster load as a suspected
confounder; this sizes it, and for most metrics it is small.

**`forced_ends` is the exception and should not be interpreted.** The same
settings produced 1.0%, 1.0% and 2.6% — a 1.6pp spread, which is wider than
any forced-end difference this document measures between arms. Two windows
suggested a floor of 0.0pp; the third showed that estimate was an artifact of
having only two. Treat `forced_ends` as diagnostic colour, not evidence, until
someone runs enough replicates to bound it properly. Everything else held
tight across all three.

## Endpointing

| arm | split | missed | forced | lag p50 | lag p90 | recall |
| --- | --- | --- | --- | --- | --- | --- |
| `min1600 max3500` (incumbent) | 3.6% | 14.1% | 1.0% | 2200ms | 4000ms | 0.689 |
| `min1600 max2560` | 4.7% | 13.8% | 2.3% | 2100ms | **3300ms** | 0.688 |
| `min1000 max3500` | 7.3% | 13.5% | 0.5% | **1600ms** | 3950ms | 0.691 |
| `min600 max3500` | 11.2% | 13.0% | 0.5% | 1200ms | 3950ms | 0.687 |
| `min300 max3500` | 14.6% | 13.0% | 1.0% | 850ms | 4035ms | 0.689 |

**`min_turn_silence` has a knee at 1000.** Marginal rate along the ramp:

    1600 -> 1000:  -600ms p50 for +3.7pp split  = 162ms per point
    1000 ->  600:  -400ms p50 for +3.9pp split  = 103ms per point
     600 ->  300:  -350ms p50 for +3.4pp split  = 103ms per point

The first step buys 60% more latency per point of truncation than anything
below it, and below 1000 the rate is flat. So 1000 is where the ramp stops
paying, not a midpoint someone liked.

**`max_turn_silence` is the knob that moves the tail, and no amount of `min`
does.** 3500 -> 2560 takes 700ms off p90 (14x the replicate floor) for +1.1pp
split (2x the floor). Every `min` reduction moved p90 by <=50ms — at or below
the floor — and `min300` made it worse. `min` owns the median; `max` owns the
tail. Since an agent reads as unresponsive on its worst turns rather than its
median one, `max` is the cheaper win.

The +1.1pp split cost says most of what that ceiling cut off were turns that
had genuinely finished. An earlier draft of this document read the arm's
`forced_ends` (1.0% -> 2.3%) as the mechanism visibly confirming itself; the
three-window replicate above shows 2.3% is inside the noise on that metric, so
the claim is withdrawn. The split and lag numbers carry the result on their
own.

Recall is flat across all five arms (0.687-0.691, a 0.004 spread against a
0.005 floor). Endpointing relocates boundaries; it does not destroy words.
`missed` is flat at 13-14% and so does not confound the arms, but see below.

## Voice isolation

| arm | split | missed | forced | lag p50 | lag p90 | recall |
| --- | --- | --- | --- | --- | --- | --- |
| `off` | 1.6% | **62.5%** | 1.0% | 2325ms | 4700ms | 0.290 |
| `0.7` | 5.5% | 18.5% | 2.6% | 2600ms | 4650ms | 0.639 |
| `0.8` | 3.4% | 14.1% | 1.6% | 2350ms | 4100ms | 0.686 |
| `0.9` (what the host runs) | 4.2% | 14.1% | 1.0% | 2100ms | 3950ms | 0.684 |

**Voice Focus is load-bearing, not a preference.** Without it the service never
reports 62.5% of caller turns, and endpointing parameters make no difference
to that number: on the smoke sim, `1600:3500`, the service defaults and
`mode=balanced` all returned exactly 2 finals of 10 utterances. A sweep run
without isolation would have produced a clean symmetric table in which every
arm tied, and it would have been measuring VAD misses. Batch transcription of
the same bytes reached 9 of 10, which is what proves the audio was never the
problem: the caller channel carries a persona noise bed plus telephony noise
that buries short backchannels ("Okay", "No backlight.") until isolation
removes it.

Beware `off`'s 1.6% split — the best split rate in the table, from the arm that
misses two thirds of all turns. Split rate is only meaningful conditional on
having heard the turn, and no single-metric ranking of this table is valid.

**The ramp plateaus at 0.8, and 0.9 is correct.** 0.8 -> 0.9 moves `missed` by
0.0pp and recall by 0.002, both inside the floor; 0.9 is modestly better on
latency (p50 250ms, p90 150ms, 2-3x the floor). 0.7 is really worse: `missed`
+4.4pp, recall -0.045 (9x the floor).

This closes the question `agent.ts.snapshot:280` raises. The SDK's 0.9 is an
override of a documented service default of 0.7, and it had never been
measured; measured, it is right for this corpus. An earlier single-sim probe
suggested the opposite (0.7 scoring 9 of 10 against 0.9's 8) — that was noise
at n=1, and it does not survive n=29.

## The two knobs compose

`min1000 max2560` was run against a `1600:3500` reference in the SAME window,
both arms fresh, since the point was a difference of a few points against a
floor measured at 0.5pp:

| arm | split | missed | forced | lag p50 | lag p90 | recall |
| --- | --- | --- | --- | --- | --- | --- |
| `min1600 max3500` (reference) | 3.6% | 14.1% | 2.6% | 2200ms | 4000ms | 0.692 |
| `min1000 max2560` | 8.3% | 13.5% | 1.3% | **1600ms** | **3100ms** | 0.686 |
| **delta** | **+4.7pp** | -0.5pp | (floor) | **-600ms** | **-900ms** | -0.006 |

Predicted additively from the single-knob arms: p50 1600ms, p90 3300ms, split
8.4%. Measured: p50 1600ms and split 8.3% land exactly, and p90 comes in 200ms
BETTER than additive — 4x the p90 floor, so a real if modest super-additive
gain. Recall moves 0.006 against a 0.007 floor, which is to say not at all.

So the interaction worry does not materialise on this corpus: lowering the
force-end ceiling does not blunt the median gain from a lower `min`, and the
two tail mechanisms slightly reinforce. **`min1000 max2560` is the arm to take
into a graded run** — 600ms off the median wait and 900ms off the tail, for
4.7pp more split turns, with word recall unchanged.

## What this does not say

**Reward.** A split turn changes what the agent answers, which changes the
caller's next line, and the trajectory diverges from there. No amount of
replay reconstructs that. These numbers rank candidates; they do not price
them.

**Whether 4.7pp more splits is worth 600/900ms is a reward question, not a
replay one.** That is the trade this whole document sets up and cannot settle.

**Interaction with isolation is untested.** Isolation decides which frames
count as speech, and therefore when silence starts — the input `min` measures
against. The best `min` at 0.9 is not automatically the best `min` at another
threshold. Both sweeps here held the other dimension fixed.

**The 13-14% `missed` floor is a separate problem.** It is constant across
every arm, so it confounds nothing, but it is one in seven caller utterances
that the STT never reports at the best available settings — mostly very short
backchannels, plus one utterance the caller's TTS never rendered. Worth its
own investigation; it is not an endpointing question.

## Cost

319 sessions, paced at 1.0x because endpointing is silence-duration sensitive
and streaming is billed per session wall-clock. ~776 session-minutes; ~45
minutes for each of the first two sweeps at concurrency 8 and 6, ~18 for the
combo sweep. Raw messages are kept in
`events.jsonl` per sim, so rescoring is free: `endpoint_sweep.py --report <dir>`.
