# Two questions: is the 3.6 model better, and is the sandbox voice isolator better?

**Date:** 2026-09-21 · **Domain:** retail, tasks 0–113 (full set), 1 trial
**Runs:** `u35-prod-full` (universal-3-5-pro, production) · `u36-nearfield-full`
(universal-3-6-pro, sandbox000), both `near-field @ 0.7`
**Plus:** an offline replay of identical audio through a 2×2×2 grid
(model × cluster × voice-focus)
**tau2-bench:** `cef1ad9` + uncommitted (see "Reproducibility") · **agent SDK:** `f34290a5e`

> **This document was substantially rewritten on 2026-09-21 after four
> adversarial reviews.** The previous version's headline — "3.5-production
> transcribes words more accurately" — was **wrong**: it was an artifact of a
> per-utterance similarity threshold. Standard WER reverses it. A list of
> withdrawn claims is in "Corrections".

---

## Answers

### Q1 — Is the 3.6 model better? **No detectable difference.**

On **identical replayed audio**, holding the cluster and voice focus fixed,
`universal-3-6-pro` and `universal-3-5-pro` are indistinguishable:

| comparison (3.6 − 3.5), same audio | Δ WER | 95% CI |
| --- | --- | --- |
| production, focus on | −0.0014 | [−0.0055, +0.0027] |
| production, focus off | −0.0023 | [−0.0110, +0.0064] |
| sandbox, focus off | −0.0048 | [−0.0140, +0.0043] |
| sandbox, focus on | −0.0350 | [−0.0845, +0.0145] |

Three of four cells are flat to within 0.5 pp. The only non-trivial cell is
**sandbox + focus on** — which is a *voice-focus* result, not a model result
(see Q2). **Nothing here supports "3.6 is a better model."**

### Q2 — Is the sandbox voice isolator better? **Not on its own. It is a 3.6-specific voice-focus fix.**

The effect is **isolator-gated but model-specific** — a three-way interaction,
not a main effect of either variable.

**Bed penalty** (WER on the 5 speech-bed calls minus the 4 non-speech-bed
calls), all eight replay cells:

| model | cluster | focus | bed penalty |
| --- | --- | --- | --- |
| 3.5 | production | **off** | +8.54 pp |
| 3.5 | sandbox | **off** | +8.99 pp |
| 3.6 | production | **off** | +8.36 pp |
| 3.6 | sandbox | **off** | +8.36 pp |
| 3.5 | production | on | +13.29 pp |
| 3.5 | sandbox | on | +13.29 pp |
| 3.6 | production | on | +14.03 pp |
| 3.6 | sandbox | on | **+7.41 pp** |

Three readings, in order:

1. **With voice focus OFF, all four cells are ~8.4–9.0 pp.** Model and cluster
   both make no difference. This is what rules the *model* out — a model
   difference would appear here and does not.
2. **Voice focus at 0.7 makes competing speech WORSE** in three of four cells
   (+13 to +14 pp against the ~8.5 pp no-isolation baseline). On production it
   is actively harmful on this corpus, for both models.
3. **Only `3.6 + sandbox + focus on` beats the no-isolation baseline** (+7.41).
   And `3.5 + sandbox + focus on` is **+13.29 — byte-identical to
   production**, so the sandbox isolator does nothing for 3.5.

So the sandbox does not have a generally better isolator; it has something that
helps voice focus **specifically when paired with 3.6**. Whether that is a
retuned isolator, a different 3.6 build, or a different front-end pairing
cannot be separated from these runs.

The supporting evidence from the main grid follows.

**(a) Voice focus is harmful on production and neutral on sandbox** — replay,
identical audio, Δ WER of focus ON minus focus OFF (positive = focus hurts):

| | production | sandbox |
| --- | --- | --- |
| universal-3-5-pro | **+0.0296** | +0.0271 |
| universal-3-6-pro | **+0.0306** | **−0.0030** |

Interaction (sandbox focus effect − production focus effect) at 3.6:
**−0.0336, 95% CI [−0.0836, +0.0164]**, n=9 calls. Directionally clear,
**not significant**. At 3.5 the interaction is zero (−0.0025).

**(b) Production leaks background speech into the transcript; sandbox does
not** — word-insertion rate from the live runs, split by whether the task's
background bed carries human speech:

| | speech bed (57 tasks) | non-speech bed (57) | penalty |
| --- | --- | --- | --- |
| 3.5-prod @ 0.7 | **6.30%** | 4.21% | **+2.09 pp** |
| 3.6-sandbox @ 0.7 | 4.50% | 4.61% | **−0.11 pp** |

3.5-on-production inserts materially more words when a TV or another
conversation is playing under the caller; 3.6-on-sandbox shows no bed
sensitivity. The leaked text is visible in the transcripts — e.g.
`"Hello? Oh, oh, we don't have any officers at the border that would be
charged for traffic."` is a correct transcription of the TV-news bed, scored
as a caller mis-hearing.

**This live comparison cannot attribute the difference**, because it changes
model and cluster at once. It corroborates the *existence* of a bed-conditioned
gap; only the replay table above, which holds each variable fixed in turn,
shows that the gap is isolator-gated and 3.6-specific.

**Caveat that prevents a clean answer:** (a) and (b) both confound the isolator
with the whole cluster. A sandbox build could differ in decoder, endpointer, or
serving health as well as in voice focus. The decisive arm is **3.5-on-sandbox**
— if it also loses the bed penalty, the isolator (or the cluster) is
responsible and the model is not implicated at all.

### The finding neither question anticipated

**The large live-run difference is probably not about transcription at all.**
The arms differ by +0.114 reward, but on identical audio their WER differs by
<0.5 pp. What separates them in the live runs is the **agent failing to
respond**:

| | 3.5-prod | 3.6-sandbox |
| --- | --- | --- |
| caller turns with no reply before the next turn | 134 / 677 (20%) | 46 / 793 (6%) |
| — excluding each call's final turn (an artifact) | 98 / 558 (17.6%) | 28 / 647 (4.3%) |
| — of those, agent produced **neither speech nor a tool call** | **55** | **3** |
| caller turns after which the agent never speaks again | 84 (12.5%) | 17 (2.3%) |

A 20× gap in "agent went silent and did nothing" is a pipeline-stall signature,
not an ASR signature — and it points at cluster/serving health rather than at
either speech model.

---

## The offline replay (method)

`both.wav` is saved for every sim, so the same caller audio can be pushed
through any endpoint configuration. This removes conversation divergence, TTS
seed, barge-in, agent LLM, cluster load feedback and reward in one move.

- 9 calls with complete results across all 8 configs (5 speech-bed, 5
  non-speech-bed sampled; one dropped for a connection failure)
- caller channel extracted from the stereo mix, 8 kHz → 16 kHz, pushed at 5×
  realtime
- scored as **WER on concatenated text** (`scripts/stt_errors.py:_normalize`),
  so turn segmentation is irrelevant
- paired per call; CIs are ±1.96 SE on the paired differences

| config | mean WER |
| --- | --- |
| u3.5 · production · focus on | 0.6452 |
| u3.5 · production · focus off | 0.6156 |
| u3.5 · sandbox · focus on | 0.6452 |
| u3.5 · sandbox · focus off | 0.6181 |
| u3.6 · production · focus on | 0.6438 |
| u3.6 · production · focus off | 0.6133 |
| u3.6 · sandbox · focus on | **0.6103** |
| u3.6 · sandbox · focus off | 0.6133 |

**Absolute WER is inflated (~61–65%)** because the replay upsamples 8 kHz
archived audio and pushes at 5× realtime. That penalty is identical across all
eight cells, so the *contrasts* hold; the absolute level does not transfer.

Three cells are byte-identical across clusters (u3.5 focus-on, u3.6 focus-off:
Δ = 0.0000, CI [0,0]), which suggests the two endpoints share a backend for
those paths and diverge only on the 3.6 + voice-focus path.

**Limitations:** n=9 calls; no contrast clears significance; band-limited
audio; `voice_focus_threshold` fixed at 0.7.

---

## Live-run results

### Reward — real but fragile

| | 3.5 prod | 3.6 sandbox | delta |
| --- | --- | --- | --- |
| mean reward (binary; = pass rate) | 0.2281 | 0.3421 | **+0.1140** |
| 95% CI (Wald) | | | [+0.020, +0.208] |
| 95% CI (exact conditional) | | | **[+0.011, +0.195]** |
| DB match | 34/113 | 49/113 | paired McNemar p = 0.017 |
| NL assertions | 78/111 | 82/111 | +3.6 pp |
| action match (per-task rate, n=110) | 0.397 | 0.518 | Wilcoxon p = 0.003 |

McNemar exact on all 114 tasks: 17 both pass · 66 both fail · 22 to 3.6 · 9 to
3.5 → **p = 0.029**. Reward is strictly {0,1} here, so mean reward ≡ pass rate
and McNemar is the correct test.

**Two things weaken this.** It does not survive Holm correction across the ~15
tests reported here. And it is concentrated in the first quarter of the run:

| launch-order quartile | Q1 (n=25) | Q2 | Q3 | Q4 |
| --- | --- | --- | --- | --- |
| Δ reward | **+0.280** | +0.077 | +0.038 | +0.077 |

Dropping Q1: **Δ = +0.064, CI [−0.038, +0.167], p = 0.33.** Because both arms
dispatched tasks in the same id order, run position and task identity are
perfectly confounded — three quarters of the point estimate rides on one
quarter of the corpus.

### Transcription — 3.6-sandbox is better, not worse

Session-level WER (concatenate all gold and all hypothesis per task):

| | 3.5 prod | 3.6 sandbox |
| --- | --- | --- |
| **WER** (script normalization) | **24.08%** | **21.06%** |
| WER (plain lowercase) | 24.98% | 21.99% |
| CER | 21.76% | 18.78% |
| substitutions / deletions / insertions per 100 | 5.55 / 17.97 / 1.46 | 5.15 / 15.83 / 1.01 |

3.6-sandbox is better on all three edit types. The previous version of this
document reported the opposite, from a per-utterance binary flag
(`similarity < 0.95`); that metric measures *"fraction of utterances that are
flawless"*, rises with utterance length, and inverts below a 0.82 threshold.
See "Corrections".

### Where the words actually go

| | 3.5 prod | 3.6 sandbox |
| --- | --- | --- |
| gold words never delivered | **20.0%** | 17.7% |
| — excluding each call's trailing (unspoken) tail | 14.55% | 14.91% |
| VAD-detected segments returning no transcript | **56** | 23 |
| mid-word interruption markers (em dash) | **31** | 14 |
| digit-token error rate | 34.7% | 37.1% |

3.5-production delivers less speech overall, but the gap is **1.13×**, not the
2.4× the previous version claimed — and it vanishes once the early-termination
tail is removed. The one place 3.6-sandbox is worse is digit tokens
(37.1% vs 34.7%), which matters for ZIP and order-id lookups.

---

## Corrections — claims withdrawn from the previous version

| withdrawn claim | why it failed |
| --- | --- |
| "3.5-production transcribes words more accurately" | False on WER (24.1% vs 21.1%), CER, and WER within the same 1:1 pairs. The metric was a per-utterance threshold, not word accuracy. |
| "mis-hearing rate +4.9 pp worse, p = 0.010" | **The entire gap is the `MINOR` bucket** (filler/contraction rendering). Excluding MINOR: −0.5 pp, p = 0.918. |
| "speech never reaching the agent, 111 vs 46 (2.4×)" | That quantity is `len(gold) − len(heard)`, a count of transcript *events*. Real word loss is 1.13×. |
| "never transcribed (1:0) 58 vs 35" | `_align` has no deletion move; **100% of 1:0 pairs are trailing residue** in both arms. A mid-call drop is forced into MERGED. |
| "merged (2:1) = speech never reaching the agent" | Merges deliver **76% / 72%** of their words; 47 and 28 of them deliver ≥80%. Only the turn boundary is lost, and the LLM sees a string. |
| **"Structural loss predicts failure; content errors don't" — "the single most important result"** | The predictor is the trailing residue (r = −0.217); the merge component is r = −0.018. Under task fixed effects the association is null (conditional logit p = 0.52; discordant pairs 14 vs 12, p = 0.85). **Reverse causation via early termination.** |
| "content-error correlation is of the *wrong sign* (+0.094)" | Compositional artifact — the two rates share a denominator (r = −0.607). Partialling structural out gives −0.061, n.s. |
| "NON_LATIN_SCRIPT 22 vs 11 despite `languages:['en']`" | **Zero non-Latin characters exist.** All 45 occurrences are U+2014 EM DASH; the regex `[^\x00-\xff]` matches punctuation. Verified by enumerating every character. |
| "NEGATION / substitution 61 vs 111" | 93–94% is a curly-apostrophe bug: `_classify` runs an ASCII-apostrophe regex on raw text, and gold uses U+2019. Real count: 4 vs 7. |
| "Two different instruments agreeing on the same defect" | Not independent (same `task.log`, same transcript stream) and they measure **disjoint** events — a dropped turn requires a transcript to exist. Per-task r = 0.077. |
| "Latency ruled out directly" | Computed over *answered* turns only, censoring 20% of A and 6% of B. Unconditional: p80 **23.0 s vs 3.9 s**, p90 **∞ vs 7.2 s**. |
| "`user_stop` (caller gave up) 110 vs 101" | `user_stop` is the **normal** ending of a successful call. Explicit give-ups (`###OUT-OF-SCOPE###`) are **2 vs 2**. |
| "The tasks 3.6 wins are the longest" | Compared against both-pass, which is by construction the easy tasks. Flip-vs-flip: Mann-Whitney **p = 0.215**. |
| "Not a voice-isolation story in either direction" | Contradicted by the bed-conditioned insertion rate (Q2b). The earlier subgroup test was also underpowered — interaction p = 0.635. |
| "Run trust: 0 infrastructure errors" | **16 sim logs per arm carry a fatal `AssemblyAI TTS (3009): Too many concurrent sessions`**, clustered at cold start. Plus `user_hallucination` discards. See below. |

---

## Run trust — corrected

| | 3.5 prod | 3.6 sandbox |
| --- | --- | --- |
| tasks scored in final `results.json` | 114 / 114 | 114 / 114 |
| unscored in final results | 0 | 0 |
| **sim logs with fatal `TTS 3009` (quota)** | **16** | **16** |
| sims discarded as `user_hallucination` | 1 | 2 |
| keepalive deaths | 0 | 0 |
| `run-bench.sh` exit | 0 | 0 |

The harness exhausts **its own account's TTS concurrency budget** at cold start
(10 workers opening at once). Affected calls produce 0 transcripts, 0 turns,
reward 0.0 — and tau2 retried them, so the final `results.json` looks clean.
Removing them: reward delta +0.1165, McNemar p = 0.043 (survives), but the
structural-loss correlation falls to −0.108 (n.s.) and "never attempted auth"
goes 19→8 and 10→1.

`run-bench.sh` asserts no *unscored* sims; it does not fail on fatal wire
events, and its recommended `grep "synthesize_voice failed"` check is
ElevenLabs-specific so it returns 0 on the AssemblyAI caller path.

---

## Reproducibility — currently incomplete

**The stated repro does not reproduce these runs.** `git status` shows 13
modified files plus one untracked; the committed patch covers one of them.
At `cef1ad9`, `TAU2_TTS_PROVIDER` does not exist and
`src/tau2/voice/utils/assemblyai_utils.py` is absent, so `synthesize.py` will
not import — and if patched around, the caller's voice falls back to
ElevenLabs, which re-bases the comparison.

Also undeclared: `discrete_time_adapter.py` carries a playback-progress change
whose absence alters the very metric the verdict rests on, and two env switches
(`AAI_PLAYBACK_REPORT`, default ON; `TAU2_CLIENT_DUCK`, default off) are not in
the constants table. The agent project is not a git repository, `run-bench.sh`
hardcodes three absolute paths, and neither the SDK commit nor its working-tree
cleanliness is recorded at run time.

**To fix:** commit the branch and cite the SHA; add a prerequisites block
(repo URLs, commits, directory layout, `pnpm install && pnpm build`, and the
required secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `ASSEMBLYAI_API_KEY`,
`ASSEMBLYAI_STAGING_API_KEY`, seven `TAU2_VOICE_ID_*`); have `run-bench.sh` log
`git -C "$SDK_DIR" rev-parse HEAD` and `status --porcelain`.

---

## Limits

- **1 trial × 114 tasks.** The harness's own note: three near-identical builds
  over the same ten tasks scored 0.20 / 0.50 / 0.10.
- **Model, cluster, and build move together** — keys are environment-scoped.
- **Run position is confounded with task identity** (same dispatch order both
  arms); Q1 carries most of the reward effect.
- **The stimulus is not matched.** `speech_environment.voice_seed` differs on
  all 114 task pairs, so TTS realisation, injected frame drops, burst noise and
  vocal tics are independent draws per arm. Aggregate rates are balanced, so
  this is noise, not bias — but pinning the seed is free and would tighten
  every paired test.
- **Only 40 gold texts appear in both arms**; the conversations diverge.
- **One threshold (0.7), one agent LLM** (`qwen-3.8-27b`, not the default
  `gpt-5.6-terra`).
- **`auth_classification` is null on all 228 sims**; auth figures use a
  tool-call proxy.
- **The evaluator's golden actions error identically in both arms**, capping
  achievable reward.

## Instrument bugs found (fix before the next comparison)

1. `_classify`: normalize `’`→`'` before `NEGATION_RE` — removes ~161 of 172 rows.
2. `NON_LATIN_RE`: match non-Latin Unicode letter categories, not `[^\x00-\xff]`.
3. `_align`: add 1:0 / 0:1 moves so a mid-call drop can be located at all.
4. Report **WER and CER as primary**; keep the per-utterance flag as a
   secondary diagnostic, always beside mean utterance length.
5. `latency.py`: filter by `simulation_index` (it currently globs all logs,
   including crashed retries), exclude each call's final turn, and report the
   censored distribution rather than answered-turn percentiles.
6. Exclude utterances that never reach `is_final_chunk` (78 and 61 here — their
   audio was cut short, and they score as ASR errors) and out-of-turn injected
   phrases (62 vs 101, penalising B 1.6×).
7. Make `_normalize`'s address gate symmetric (computed on `gold or heard`).
8. Count `TTS error 3009` and `hallucination_discarded` in run-trust checks.

## Next experiments, in order

1. **3.5-on-sandbox.** Completes the 2×2 and tests the strongest alternative
   explanation: if it also stops stalling and loses the bed penalty, the effect
   is the cluster and neither speech model is implicated.
2. **3.6-on-production.** Separates model from cluster.
3. **Extend the replay** to all 114 calls at 1× realtime with `voice_focus`
   swept — this answers Q2 directly and costs no benchmark runs.
4. **Counterbalance dispatch order** (reverse arm B, or interleave) to break
   the position↔task confound; pin `voice_seed` across arms.
5. **Fix the TTS 3009 cold start** (backoff, or stagger worker startup) and
   make a fatal wire event set `infrastructure_error`.
6. **3 trials** once the above are in place, with reward pre-registered as the
   primary endpoint.
