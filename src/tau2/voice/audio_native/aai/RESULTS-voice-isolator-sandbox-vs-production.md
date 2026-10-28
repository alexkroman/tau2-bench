# Is the sandbox voice isolator better than production's?

**Date:** 2026-09-21 · **Method:** controlled-SNR replay, full-band synthetic stimulus
**Grid:** 2 models × 2 clusters × voice-focus {on, off} × 7 noise conditions
**n:** 10 items per cell per condition (70 paired observations pooled)
**Endpoints:** `streaming.assemblyai.com/v3/ws` ·
`streaming.sandbox000.assemblyai-labs.com/v3/ws`
**Setting under test:** `voice_focus=near-field`, `voice_focus_threshold=0.7`

## Answer

**No — and more precisely, it is not an isolator difference at all.**

The sandbox's `universal-3-6-pro` + voice-focus path *is* measurably different
and very slightly better: **−0.37 pp WER, 95% CI [−0.73, −0.01], n = 70**. But
the improvement is **the same size on clean audio with no background speech in
it** as it is under competing speech at equal loudness.

An isolator cannot improve audio that has nothing to isolate. Whatever the
`voice_focus` flag switches on in that path, it is not voice isolation.

| condition | bed cost vs clean | isolator contrast (3.6) |
| --- | --- | --- |
| **clean (no bed at all)** | — | **−0.58 pp** |
| people_talking @ 15 dB | −0.08 pp | +0.28 pp |
| people_talking @ 5 dB | +1.17 pp | −0.32 pp |
| **people_talking @ 0 dB** | **+3.79 pp** | **−0.39 pp** |
| tv_news @ 15 dB | +0.41 pp | −0.57 pp |
| tv_news @ 5 dB | +0.42 pp | −0.28 pp |
| **tv_news @ 0 dB** | **+1.62 pp** | **−0.74 pp** |

Negative contrast = sandbox better. Read the first row against the last two:
the effect is as large with **zero** background speech as with background
speech at the caller's own volume. **It does not scale with SNR**, which is the
defining signature of isolation and the test this experiment was built to run.

### Control behaves exactly as it should

`universal-3-5-pro`, where both clusters were already shown to return
byte-identical transcripts: **+0.09 pp, CI [−0.17, +0.36], n = 70** — and
exactly 0.00 pp in 5 of the 7 conditions. The method detects nothing where
there is nothing, which is what licenses believing the 3.6 number.

## Second finding: voice focus fails on both counts

Two separate facts, and the distinction matters — "the bed is soft enough that
isolation copes" is **not** what happened.

### (a) At benchmark SNR the bed costs nothing even with isolation OFF

WER with `voice_focus` disabled entirely, averaged over both models and both
clusters:

| condition | WER (focus OFF) | vs clean |
| --- | --- | --- |
| clean | 0.1418 | — |
| people_talking @ 15 dB | 0.1379 | −0.39 pp |
| tv_news @ 15 dB | 0.1434 | +0.16 pp |
| people_talking @ 5 dB | 0.1484 | +0.67 pp |
| tv_news @ 5 dB | 0.1426 | +0.08 pp |
| **people_talking @ 0 dB** | 0.1758 | **+3.41 pp** |
| **tv_news @ 0 dB** | 0.1862 | **+4.45 pp** |

The recognizer is unbothered by competing speech at 15 dB with no isolation
running. Nothing is being "handled" — there is no problem to handle. tau2-bench
plays its speech beds at exactly this level.

### (b) Where the bed DOES hurt, isolation barely helps

Focus gain (`WER(on) − WER(off)`, negative = helps), pooled over both models
and both clusters, n = 40 per condition:

| condition | bed damage | focus gain | CI | share recovered |
| --- | --- | --- | --- | --- |
| clean | — | −0.14 pp | [−0.44, +0.15] | — |
| people @ 15 dB | −0.39 pp | +0.07 pp | [−0.07, +0.21] | — |
| people @ 5 dB | +0.67 pp | −0.08 pp | [−0.39, +0.23] | ~12% |
| **people @ 0 dB** | **+3.41 pp** | **−0.32 pp** | [−0.81, +0.16] | **~9%** |
| tv_news @ 15 dB | +0.16 pp | +0.06 pp | [−0.31, +0.43] | — |
| tv_news @ 5 dB | +0.08 pp | −0.15 pp | [−0.61, +0.31] | — |
| **tv_news @ 0 dB** | **+4.45 pp** | **−0.14 pp** | [−1.23, +0.95] | **~3%** |

At the one SNR where competing speech genuinely costs accuracy, voice focus at
0.7 recovers **3–9% of the damage**, and every interval spans zero.

**This is a worse result for voice focus than "the bed is soft."** If the bed
were merely soft and isolation were coping, the 0 dB cells would show large
gains. They show none.

Caveats on (b): only `voice_focus_threshold=0.7` was tested, and the SDK
defaults to **0.9** precisely because 0.7 was judged too permissive — a more
aggressive setting may recover more at 0 dB, at a cost elsewhere. And these are
synthetic mixes of a looped bed over TTS speech, which is cleaner than a real
room.

## Why this experiment could answer the question when the last one could not

The previous replay (32 archived calls) returned −0.75 pp with CI [−3.23,
+1.74] and could not distinguish a perfect isolator from a useless one, because
the recoverable budget (1.6 pp) was smaller than the measurement precision
(±2.49 pp). Three changes fixed the sensitivity:

| | previous replay | this experiment |
| --- | --- | --- |
| stimulus | archived 8 kHz, upsampled | **AssemblyAI TTS at 16 kHz, full band** |
| push rate | 6× realtime | **1× realtime** |
| absolute WER | ~67% | **13–18%** (live runs: 21–24%) |
| noise | whatever the task happened to have | **controlled sweep, clean → 0 dB** |
| baseline | between-task speech vs non-speech (confounded with content) | **same item, clean vs bedded** |
| CI half-width | ±2.49 pp | **±0.36 pp** |

Precision improved ~7×, which is what turned an unresolvable question into a
resolvable one.

## Method

- **Stimulus:** 10 caller utterance groups (55–86 words) taken from real
  `audio_script_gold` scripts, so spelled names and digit-by-digit ZIPs are
  present. Synthesized with AssemblyAI TTS (`eve`, 16 kHz). 273 s total.
- **Noise:** `people_talking.wav` and
  `medium_size_room_tv_news_iphone_mic.wav`, both natively 16 kHz mono, looped
  and scaled to a target SNR against each item's speech RMS.
- **Delivery:** 100 ms frames at 1× realtime, 4 s drain, explicit `Terminate`.
- **Scoring:** Levenshtein WER over `scripts/stt_errors.py:_normalize` tokens
  (digit folding, spelled-run collapse), against the exact synthesized text.
- **Estimand (unchanged):** difference-in-differences against each cell's own
  focus-off baseline —
  `[sbx(on) − sbx(off)] − [prod(on) − prod(off)]` — so anything the clusters
  differ in outside voice focus cancels.
- **Statistics:** paired per item; CI = ±1.96 SE; exact two-sided sign test.

Raw output: `/tmp/settle.json`. Script: `/tmp/settle.py`.

## What is established, and what is not

**Established**

1. The sandbox runs a **different** code path for `universal-3-6-pro` when
   `voice_focus` is set — confirmed twice, by transcript divergence (23 of 32
   archived calls differ, vs 2–7 on every other path) and by the −0.37 pp WER
   contrast whose CI excludes zero.
2. That path is **not voice isolation**: its benefit is present at full
   strength on clean audio and does not grow as competing speech gets louder.
3. For `universal-3-5-pro`, **both clusters run the same thing** (+0.09 pp,
   CI spanning zero; byte-identical transcripts).
4. **At the benchmark's 15 dB SNR the speech bed costs ~0 pp of WER, with
   isolation off.** The recognizer does not need help at that level.
5. **At 0 dB, where the bed costs 3.4–4.5 pp, voice focus at 0.7 recovers only
   3–9% of it**, every CI spanning zero. It is ineffective where it matters,
   not merely unnecessary where it doesn't.

**Not established**

- What the divergent 3.6 path actually is. A decode-parameter difference, a
  different model revision gated behind the same flag, or a front-end change
  that happens to be flag-conditioned are all consistent with these data.
- Whether the effect holds at `voice_focus_threshold=0.9` (the SDK default) or
  with `far-field`. Only 0.7 / near-field was tested.
- Whether −0.37 pp matters operationally. It is ~2% relative on a 16% base, far
  below the run-to-run variance of the benchmark itself.

## Consequences

- **Do not adopt the sandbox endpoint expecting better noise robustness.** The
  measured advantage is real, tiny, and unrelated to noise.
- **The benchmark's background-speech condition is not currently a stress
  test.** If competing speech is meant to be part of the difficulty, the bed
  needs to run nearer 5 dB than 15 dB; at 15 dB it is decorative.
- **Voice focus is not earning its place at 0.7** on either cluster — in the
  archived-audio replay it roughly doubled the competing-speech penalty in all
  four cells, and here it moves WER by well under half a point. The SDK default
  of 0.9 remains untested and is the obvious next thing to check.
