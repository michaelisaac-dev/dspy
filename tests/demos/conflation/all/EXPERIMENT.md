# Conflation under an LLM-call penalty: a CAL (cost / accuracy / latency) sweep

Does penalizing LLM calls inside a GEPA objective push `dspy.Flex` to decompose an LLM classifier
into deterministic Python — and what does that cost in accuracy?

**Short answer.** Yes, dramatically, and it costs nothing in accuracy — but the two halves of the
sweep make different claims, and only one of them is about accuracy:

- **λ ≤ 0.05** — *significantly more accurate* than calling the LLM every time (+4.2 to +4.6 pp,
  McNemar p = 0.019 / 0.031), while being 1.4–2.2× cheaper and 1.7–2.6× faster.
- **λ ≥ 0.1** — accuracy **statistically indistinguishable** from the baseline (p = 0.56–1.00), at
  5–130× lower cost and 5–30× lower latency. The right claim here is *same accuracy, far cheaper*,
  not *more accurate*.

The fine-grained ranking *among* penalties is **not resolvable** at this sample size, and the stated
target (0 calls at ~100% accuracy) was **not reached** — ~92–95% is the ceiling here.

Total API spend: **$11.04**.

---

## 1. Task and data

Business-listing conflation: given place A (`input_name`, `input_address`), place B (`match_name`,
`match_address`) and the distance between their coordinates, decide whether they are the same
physical place.

- Source: `../conflation_coded.jsonl`, 1,029 labelled pairs, **769 positive / 260 negative
  (74.7% positive)**.
- The label is genuinely semantic in the tail: `CONCESSION #2 KEN MERCER SPORTS PARK` vs
  `KEN MERCER SPORTS PARK` at the same address is **false**; `KIN CAFE` vs `KIN` at the same address
  is **true**.

### Splits (`conflation_common.load_splits`, seed 0)

| split | positives | negatives | total | role |
|---|---|---|---|---|
| train | 30 | 30 | 60 | GEPA reflection minibatches |
| val | 15 | 15 | 30 | GEPA candidate selection |
| test | 120 | 120 | 240 | reported metrics |

**All three are class-balanced 50/50**, so chance is 0.500 and accuracy is readable without a
base-rate caveat. Disjointness verified: `train∩test = val∩test = train∩val = 0` on the full
(nameA, addrA, nameB, addrB) signature.

Two known data blemishes, both quantified and neither material:
- 15/240 test examples (6.2%) share a place-A with train/val. Accuracy on those is **lower**
  (0.867 vs 0.924 at λ=0.4), so this is not inflating results.
- 3 exact duplicate rows inside test → effective n = 237.

---

## 2. Method

### The objective

For each penalty λ, GEPA optimizes

```
score(example) = max(0, correct − λ · n_llm_calls)
```

`n_llm_calls` is the number of predictor calls in the dspy trace for that `forward()`. λ=0 is plain
accuracy (calls are free). As λ rises, each call must buy back more accuracy than it costs, so the
optimizer is pushed to settle cases in Python. The metric returns `ScoreWithFeedback`; the feedback
string is λ-aware so the reflection LM is told what a call actually costs.

Swept: **λ ∈ {0, 0.05, 0.1, 0.2, 0.4}**, one GEPA run each, `max_metric_calls=400`,
`reflection_minibatch_size=3`, `seed=0`, 8 threads.

### Models

| role | model | why |
|---|---|---|
| execution | `anthropic/claude-haiku-4-5` | weakest/cheapest Claude — makes the call penalty a real tradeoff |
| reflection | `anthropic/claude-opus-5` | writes the Python; strongest available |

The demo previously specified `anthropic/claude-haiku-4-7`, **which does not exist** — the API 404s.
Combined with a per-example `except Exception`, every LM call was being scored as "wrong answer,
0 calls" rather than crashing. Fixed, and the baseline now asserts `errors == 0` so it cannot recur
silently.

### Instrumentation

dspy's disk and memory caches are **disabled** for the whole sweep, so latency and cost are what a
cold production call costs.

- **Cost** — per-example, from `dspy.track_usage()` tokens priced through an explicit table, and in
  aggregate from litellm's own per-call `cost`. The two agree to the cent.
- **Latency** — `perf_counter` around each `forward()`, reported as mean/p50/p95 **per request**,
  plus wall-clock throughput (`wall_s / n`). These differ by 4–8× under the 8-thread pool and are
  reported as separate columns; do not read per-request latency as pipeline cost.
- **Calls** — `len(trace)`.

Every per-example record (gold, pred, n_calls, latency, cost, tokens, error) is persisted, so any
other metric can be recomputed without re-running anything.

---

## 3. Results

All figures on the 240-example balanced test split. `acc@pool` re-weights accuracy to the source
pool's 74.7% prevalence (see §5); `req ms` is per-request latency, `ms/ex` is wall-clock throughput.

| λ | acc | acc@pool | prec | rec | spec | F1 | calls/ex | $/1k | req ms | ms/ex | GEPA score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.904 | 0.894 | 0.922 | 0.883 | 0.925 | 0.902 | 1.000 | 0.98 | 1924 | 245 | — |
| 0 | 0.950 | 0.966 | 0.922 | 0.983 | 0.917 | 0.952 | 0.254 | 0.70 | 1155 | 153 | 0.950 |
| 0.05 | 0.946 | 0.952 | 0.935 | 0.958 | 0.933 | 0.947 | 0.167 | 0.45 | 726 | 99 | 0.938 |
| 0.1 | 0.908 | 0.950 | 0.850 | 0.992 | 0.825 | 0.915 | 0.071 | 0.18 | 347 | 53 | 0.903 |
| 0.2 | 0.917 | 0.929 | 0.897 | 0.942 | 0.892 | 0.919 | 0.075 | 0.09 | 135 | 19 | 0.907 |
| 0.4 | 0.921 | 0.944 | 0.885 | 0.967 | 0.875 | 0.924 | 0.004 | 0.01 | 65 | 15 | 0.921 |

![CAL frontier under an LLM-call penalty](cal_frontier.png)

*Regenerate with `python sweep_penalties.py --plot-only` (reads the JSON, makes no API calls).*

**Reading the figure.**

- **Top-left / top-right — cost → accuracy and latency → accuracy.** Points are joined in λ order,
  *not* sorted by x: λ is the independent variable, and sorting by cost would draw a smooth
  monotone "frontier" the data does not have (λ=0.1 and λ=0.2 invert). Error bars are Wilson 95%
  CIs. The orange square is the baseline. The x-axis on the right panel is **per-request** latency
  under 8-way concurrency — wall-clock throughput is 4–8× lower and is the `ms/ex` column in the
  table above.
- **Bottom-left — what the penalty buys.** Average LLM calls per example against λ, with the
  baseline's 1.0 calls/ex as the dashed reference. This is the cleanest signal in the whole sweep:
  0.25 → 0.00 as λ rises, i.e. the objective genuinely drives decomposition into Python.
- **Bottom-right — which penalties actually beat the baseline.** Accuracy with Wilson 95% CIs
  against the baseline's own CI band (shaded). The `p=` on each point is a **two-sided McNemar
  exact test of that program against the baseline**, paired on the same 240 examples: it counts
  only the examples where exactly one of the two is correct, and asks whether that split is more
  lopsided than chance. **Bold** (p<0.05) means a real accuracy gain — only λ=0 and λ=0.05 qualify.
  Plain (p≥0.05) means parity: the CI overlaps the baseline band, so the apparent difference is
  within noise. Parity here is still a large win, just on cost and latency (5–130× cheaper) rather
  than on accuracy. This panel is the corrected version of §3.1.

Raw data: `penalty_sweep.json`. Optimized programs: `sweep_programs/flex_penalty_*.json`.

### 3.1 The headline result

Cost and latency gains are unambiguous — call rate is a near-deterministic property of the program,
not a noisy estimate (1.000 vs 0.004 calls/ex is not a coin flip). **Accuracy gains are only
significant at λ ≤ 0.05.** McNemar exact test, each optimized program against the baseline on the
same 240 examples:

| λ | acc | Δ vs baseline | base-only correct | opt-only correct | p | verdict |
|---|---|---|---|---|---|---|
| 0 | 0.950 | +0.046 | 4 | 15 | **0.019** | **significant** |
| 0.05 | 0.946 | +0.042 | 4 | 14 | **0.031** | **significant** |
| 0.1 | 0.908 | +0.004 | 13 | 14 | 1.000 | not significant |
| 0.2 | 0.917 | +0.013 | 11 | 14 | 0.690 | not significant |
| 0.4 | 0.921 | +0.017 | 11 | 15 | 0.557 | not significant |

Repeating against the independent second baseline measurement (acc 0.892) gives the same split:
λ=0 and λ=0.05 significant (p = 0.004 / 0.007), λ≥0.1 not (p = 0.27–0.57).

So there are two distinct, differently-supported claims:

- **λ=0.05 is a genuine three-axis win**: +4.2 pp accuracy (p=0.031), 2.2× cheaper, 2.6× faster.
- **λ=0.4 is a cost/latency win at parity**: accuracy indistinguishable from the baseline
  (+1.7 pp, p=0.557) at **130× lower cost** and **30× lower per-request latency**, making one LLM
  call across 240 examples.

The second is arguably the more useful engineering result, but it must not be stated as an accuracy
improvement.

### 3.2 What the penalty actually buys

The monotone signal is in the routing, not in accuracy:

| λ | deterministic n | accuracy on those | LLM-routed n | accuracy on those |
|---|---|---|---|---|
| 0 | 179 | 97.8% | 61 | 86.9% |
| 0.05 | 200 | 95.0% | 40 | 92.5% |
| 0.1 | 223 | 91.5% | 17 | 82.4% |
| 0.2 | 222 | 93.7% | 18 | 66.7% |
| 0.4 | 239 | 92.5% | 1 | 0% |

Deterministic coverage climbs 179 → 239 while accuracy on the deterministic set falls
97.8% → 92.5%. As λ rises the rules must swallow cases they would rather defer, so their hit rate
on a *larger* set drops. Overall accuracy stays roughly flat because the two effects cancel. The
deferred cases are consistently harder than the rule-decided ones — which is what deferral is
supposed to mean.

### 3.3 Two regimes, one real transition

McNemar exact tests on paired predictions:

| comparison | discordant | p | verdict |
|---|---|---|---|
| λ=0 vs 0.05 | 4 vs 3 | 1.000 | not significant |
| **λ=0.05 vs 0.1** | **13 vs 4** | **0.049** | **significant** |
| λ=0.1 vs 0.2 | 6 vs 8 | 0.791 | not significant |
| λ=0.1 vs 0.4 | 4 vs 7 | 0.549 | not significant |
| λ=0.2 vs 0.4 | 4 vs 5 | 1.000 | not significant |

There are **two regimes** with one genuine step between them:
- **λ ≤ 0.05** — ~0.95 accuracy, 0.17–0.25 calls/ex, $0.45–0.70/1k
- **λ ≥ 0.1** — ~0.91–0.92 accuracy, 0–0.08 calls/ex, $0.01–0.18/1k

Within the λ≥0.1 plateau nothing is distinguishable. A stratified bootstrap (B=10,000) gives
pairwise P(A>B) of **0.33–0.80** under *both* weightings — every comparison is a coin flip.

### 3.4 More search budget does not help

λ=0.2 re-run at `max_metric_calls=1200` (3× the sweep):

| | 400 calls | 1200 calls |
|---|---|---|
| accuracy | 0.9167 | 0.9208 |
| F1 | 0.9187 | 0.9231 |
| avg calls/ex | 0.0750 | 0.0375 |
| $/1k | 0.086 | 0.057 |
| **optimization cost** | **$0.46** | **$2.80** |

McNemar: 7 vs 8 discordant, **p = 1.000**. Statistically identical accuracy for 6× the spend. The
call rate halved, which is the one plausibly real gain. **The binding constraint is the 30-example
val set, not search budget** — GEPA selects candidates at 1/30 = 0.033 granularity, so its choices
are noisy no matter how many rollouts it gets.

### 3.5 The stated target was not reached

The goal was 0 LLM calls at ~100% accuracy at λ=0.2 with a high `MAX_METRIC_CALLS`. Actual:
**0.0375 calls/ex at 92.1% accuracy**. Effectively-zero calls: yes. ~100% accuracy: no.

This was predictable and was flagged before spending. A decision tree over distance, the dataset's
precomputed similarities, and derived text features caps at **~95% cross-validated** — the residual
errors need world knowledge no rule set has. Haiku itself only manages 90.4% here, so ~92–95% from
free Python is arguably the better outcome, just not the requested one.

### 3.6 What Flex actually contributes: plain GEPA, no Flex

Everything above optimizes a `dspy.Flex(SamePlace)`, where GEPA rewrites the module's *source*. To
isolate what Flex adds, `run_plain_gepa.py` removes exactly that one thing: the program is a bare
`dspy.Predict(SamePlace)`, so GEPA can only rewrite the *instruction*. Everything else is held
identical — same splits and seed, same executor and reflection LM, `max_metric_calls=400`,
`reflection_minibatch_size=3`, same 240-example test set.

**λ is structurally inert here, and that is the finding.** A `dspy.Predict` makes exactly one
predictor call per `forward()`, so `n_calls ≡ 1` and the metric collapses to `max(0, correct − λ)` —
a monotone transform of accuracy for any λ<1, which shifts every candidate by a constant and cannot
reorder them. Sweeping λ would redraw one program at five y-offsets. So plain GEPA is a **point**,
not a frontier; it has no mechanism to trade calls for cost. The run asserts `avg_calls == 1.00`
exactly rather than assuming it.

| system | accuracy | 95% CI | calls/ex | $/1k | mean latency |
|---|---|---|---|---|---|
| baseline (un-optimized) | 90.4% (217/240) | [86.0, 93.5] | 1.00 | $0.98 | 1924 ms |
| **plain GEPA** (Predict, prompt-only) | **92.5%** (222/240) | [88.5, 95.2] | 1.00 | **$2.88** | **2841 ms** |
| **Flex + GEPA, λ=0** | **95.0%** (228/240) | [91.5, 97.1] | 0.25 | **$0.70** | **1155 ms** |

**Plain GEPA is Pareto-dominated by Flex+GEPA λ=0 on all three CAL axes at once** — lower accuracy,
4.1× the inference cost, 2.5× the latency. The accuracy difference alone is not significant
(8 vs 2 discordant, McNemar p=0.109) on n=240; the cost and latency differences are not estimates at
all, but direct consequences of call structure.

Two further observations:

- **Prompt-only optimization made inference 2.9× more expensive than doing nothing** ($2.88 vs
  $0.98 per 1k). GEPA's only lever is the instruction, so it lengthens it — and every example pays
  that token cost on every call, forever. It cannot trade the increase away. Flex+GEPA at λ=0 went
  the other direction *without being asked to*: it routed 75% of cases into Python and came out
  **cheaper than the un-optimized baseline** while also being more accurate.
- **Plain GEPA's accuracy gain over the baseline is not statistically significant** (5 vs 10
  discordant, p=0.302), whereas Flex+GEPA λ=0's is (4 vs 15, **p=0.019**).

Two caveats on the plain-GEPA arm specifically. Its 30-example valset **saturated at 1.00 (30/30)**,
so candidate selection ran out of signal — the same small-valset limit §3.4 hit. And with
`skip_perfect_score` at its default, 27 iterations were skipped as "all subsample scores perfect"
and only 4 candidates were ever proposed; at λ=0 with a 90%-accurate seed, a 3-example minibatch is
all-correct ~73% of the time. Both push the plain-GEPA number *down*, so 92.5% is best read as a
lower bound. Neither changes the cost or latency columns, which are structural.

Search cost for this arm: **$1.56**, 552 s, 405 LM calls, instructions changed.

---

## 4. Reproducibility

```bash
python sweep_penalties.py                                   # full sweep -> penalty_sweep.json + cal_frontier.png
python sweep_penalties.py --resume                          # skip penalties already present
python sweep_penalties.py --penalties 0.2 --max-metric-calls 1200 --out highbudget_0p2.json
python sweep_penalties.py --plot-only                       # re-render from JSON, no API calls
python run_plain_gepa.py                                    # §3.6 no-Flex arm -> merges into the same JSON + figure
```

| file | contents |
|---|---|
| `conflation_common.py` | signature, splits, metric factory, cost/latency instrumentation |
| `sweep_penalties.py` | sweep driver, console table, CAL figure |
| `test_flex_conflation.py` | single-run demo at λ=0.2 |
| `penalty_sweep.json` | all metrics + every per-example record |
| `highbudget_0p2.json` | the 1200-call λ=0.2 run |
| `cal_frontier.png` | four-panel figure |
| `sweep_programs/` | saved optimized programs, loadable via `dspy.Flex(...).load()` |

---

## 5. Validation performed

| check | result |
|---|---|
| split disjointness | 0 overlap across train/val/test |
| silent failures | **0 errors** in 1,440 evaluated examples |
| call counting | traced `n_calls` vs observed token usage: **0 discrepancies** in 1,440 records |
| cost arithmetic | re-derives **exactly** from tokens at published prices, all evals |
| cost meter | equals dspy's `GLOBAL_HISTORY` exactly (10 calls, $0.2212 both) in a controlled run |
| uncosted calls | 0 (cache disabled, every call priced) |
| noise floor | **±1.2 pp**, from re-running the identical baseline on the identical 240 examples |

---

## 6. Limitations

1. **One GEPA run per λ.** Measured run-to-run noise (±1.2 pp) is the same size as every difference
   within the λ≥0.1 plateau. Only the λ≤0.05 vs λ≥0.1 split is established. Resolving individual
   λ rankings needs 3–5 seeds per λ (~$25–35).
2. **Val set is 30 examples** — the dominant constraint (§3.4).
3. **n = 240** (237 effective). ±1.9 pp on any single accuracy.
4. **Prevalence.** The pipeline is balanced 50/50 end to end, and GEPA selected operating points
   against a 50/50 val set, so the balanced numbers are the coherent primary result. `acc@pool`
   re-weights recall/specificity (exact) to the source pool's 74.7% prevalence and answers a
   *different* question — "what if this were deployed unchanged on the natural mix" — for a program
   never tuned for that mix. It is a projection, not a correction. Re-weighted cost additionally
   assumes routing depends only on class, which is an approximation.
5. **Latency** is measured under 8-way concurrency. Per-request and throughput figures differ by
   4–8× and are reported separately.
6. **Single execution model.** The sweep runs Haiku 4.5 only; no strong-vs-weak execution-model
   axis was measured.
7. **One artifact was lost.** The high-budget run overwrote `sweep_programs/flex_penalty_0.2.json`
   with its own λ=0.2 program, because the save directory was a module-level constant rather than
   derived from `--out`. Fixed (`program_dir_for`). Nothing analytical was lost — both runs' metrics,
   per-example records and `module_src` are intact in their respective JSONs — but the sweep λ=0.2
   program's *predictor signature* is only recoverable by re-running that compile (~$0.46).

### Corrections made during the work

Recorded because they affected conclusions, not just presentation:

- Claimed the LLM fallback was "a net negative" and did "worse than chance." **Wrong** — generalized
  from λ=0.2 (66.7%, n=18) and λ=0.4 (0%, n=1) without checking sample size; at λ=0/0.05 the
  fallback scores 86.9%/92.5%.
- Claimed "λ=0.4 dominates λ=0.1 and λ=0.2." **Unsupported** — bootstrap P = 0.67 and 0.53.
- Then claimed prevalence re-weighting "reverses the ranking" to make λ=0.1 best. **Also
  unsupported** — P = 0.33 and 0.80. The prevalence effect on point estimates is real; it never
  licensed a ranking claim in either direction. Both readings were noise.
- Claimed every optimized program "dominates the baseline on all three CAL axes," i.e. is *more
  accurate*. **False for λ≥0.1** (p = 0.56–1.00); only λ≤0.05 clears significance. Found by finally
  running the optimized-vs-baseline test, which had never been run — every earlier significance test
  compared λ against λ.

**The common thread in all four:** each was a comparative claim asserted from point estimates
without a paired test. Point estimates in this experiment carry ~±1.9 pp sampling error on top of
~±1.2 pp run-to-run nondeterminism, which is larger than most differences discussed. Every
comparative claim in §3 is now anchored to a stated test; treat any comparison here that lacks one
as unsupported.

---

## 7. Cost

| item | cost |
|---|---|
| sweep: baseline eval | $0.2353 |
| sweep: GEPA compile λ=0 | $0.2795 |
| sweep: GEPA compile λ=0.05 | $3.4173 |
| sweep: GEPA compile λ=0.1 | $2.0062 |
| sweep: GEPA compile λ=0.2 | $0.4567 |
| sweep: GEPA compile λ=0.4 | $1.0192 |
| sweep: 5 × test eval | $0.3417 |
| high-budget: baseline eval | $0.2450 |
| high-budget: GEPA compile λ=0.2 (1200) | $2.7956 |
| high-budget: test eval | $0.0136 |
| meter verification run | $0.2212 |
| model-id probe + smoke test | $0.0086 |
| §3.6 plain GEPA (no Flex): compile + test eval | $1.5600 |
| **total** | **$12.60** |

Opus reflection is ~96% of the total; Haiku execution is rounding error. Compile cost is highly
uneven ($0.28–$3.42 at the same rollout budget) because GEPA keeps proposing while candidates keep
improving.

---

## 8. If continuing

1. **Seed replication before anything else** (~$25–35). Every open question in §6 is a sample-size
   question; no other change is worth making first.
2. **Enlarge the val set** (30 → 80+) and re-test whether search budget then matters. §3.4 says the
   val set, not the budget, is the ceiling.
3. **Strong-vs-weak execution model.** Sonnet 5 as executor would test whether a better judge shifts
   the optimal λ — the penalty's meaning depends on what a call is worth.
4. **Expose `name_similarity` / `address_similarity`.** They exist in the source data and are the
   strongest single features; deliberately withheld here to keep the task honest to the original
   demo, but they would raise the zero-call ceiling.
