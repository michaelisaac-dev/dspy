---
name: flex-eval
description: Run, extend, or re-analyze the Flex evaluation on LangProBe — add benchmarks/seeds/λ sweeps, regenerate the paper-comparison figures, and follow the methodology rules learned in the Sep 2026 pilot. Use when the user asks to evaluate Flex, run LangProBe benchmarks, extend the Flex paper eval, sweep λ penalties, or rebuild the results figures.
---

# Flex evaluation on LangProBe — playbook

Everything lives in `tests/demos/langprobe-flex/` (runner `run_bench.py`, plots
`aggregate_plots.py`, overview `RESULTS.md`, raw runs `results/results.jsonl`,
compiled artifacts `results/*_module_src.py`). The LangProBe clone with the
paper's released per-cell results is `tests/demos/langprobe/`
(`experiment_data/20250305/*.csv`). Read `RESULTS.md` first — it is the source
of truth for current numbers and findings.

## Running

```
.venv/bin/python tests/demos/langprobe-flex/run_bench.py <bench> <phase> [seed] [lambda] [warm]
#  bench: heart | iris | iris_fixed | scone   (extend registry() for new ones)
#  phase: baselines | gepa_cot | gepa_flex | gepa_flex_lam
#  gepa_flex_lam takes: seed, lambda, optional "warm" (starts from the saved λ=0 artifact)
.venv/bin/python tests/demos/langprobe-flex/aggregate_plots.py   # tables + all figures
```

- Long runs: launch with the harness background facility (never a bare shell `&`
  — orphaned children get reaped) and monitor the log with a stall alert
  (mtime > 10 min); post-sleep hangs leave live processes stuck on dead sockets —
  kill and relaunch, the LM cache replays completed rollouts nearly free.
- **Keys**: the repo `.env` wins; the exported shell `OPENAI_API_KEY` has been a
  stale, depleted key before. Deno must be installed (Flex sandbox).
- **Models**: task LM `gpt-4o-mini` (matches the paper — do not change without
  breaking comparability), reflection `gpt-5-mini`. Known asymmetry: the paper's
  optimizers self-optimize with the task model (their configs pass no
  prompt/teacher model); ours uses a stronger reflector. B1-vs-B2 stays
  controlled (same reflector); note the caveat when comparing to their matrix.
  The paper does NOT use GEPA (verified: camera-ready, repo, CSVs) — our runs
  are the first GEPA numbers on LangProBe.

## Experimental design rules (learned, non-negotiable)

1. **Arms are B0/B1/B2 only** — unoptimized baselines, GEPA-over-prompts (CoT
   student), GEPA-over-Flex. Per Isaac: no "one-shot codegen" arm — anything
   that writes `module_src` is *inside* the Flex parameter space, not a rival
   baseline. Evaluate Flex by the column comparison (code space vs prompt
   space), same optimizer, same budget (500 metric calls, valset 40, train 15).
2. **Replication gate before any comparison**: unoptimized baselines must
   reproduce the paper's numbers within a couple of points (they did: heart
   59.2/57.2, iris 60.0/58.7, scone 79.6/79.0, scone-CoT 89.0/88.8).
3. **Multi-seed is mandatory**: a 40-example valset can select overfit code
   (heart GEPA-Flex seeds: 82.9, 82.9, 68.4). Report accuracy as directional,
   cost separations ($0 vs per-call) as categorical.
4. **λ (per-LM-call penalty) only helps tasks with a cheap head.** Metric:
   `dspy.Prediction(score=em - lam*len(program_trace), feedback=...)` — declare
   `program_trace` in the metric signature; GEPA supplies it for Flex students.
   On delegation-bound tasks (Scone: every instance needs the model) call
   penalties only collapse accuracy (λ≥0.15 → pure code ≈ chance); the right
   currency there is dollars/tokens per call, not call count (open work).
   Warm-start (`... gepa_flex_lam <seed> <lam> warm`) restarts GEPA from the
   saved λ=0 artifact — the continual-recompilation demo.
5. **LM-cache traps**: any best-of-k sampling needs a per-sample nonce in the
   prompt or the cache silently collapses k to 1 (this bug shipped a wrong
   number once).

## Figure/analysis conventions

- `fig_paper_fig2_overlay.png` reconstructs the paper's Fig. 2 (mean score vs
  total inference cost, log-x, four config classes over gpt-4o-mini + gpt-4o)
  on the **shared-benchmark subset** — y-values are far higher than the
  published 15-dataset figure on both sides; explain that in the doc caption,
  NOT in a disclaimer title on the chart (user preference, stated twice).
  Paper side gets best-cell-per-config-per-benchmark (oracle per-task selection
  — favors the paper; say so in the footer).
- Paper cells with `cost==0` have missing cost data, not free runs: **impute**
  from the row's recorded token counts × OpenAI list prices (gpt-4o-mini
  .15/.60, gpt-4o 2.50/10.00 per M) — formula verified to reproduce their
  recorded costs exactly.
- Compiled programs cost $0 at test time — log axes need a dedicated "$0"
  column (map zeros to min-positive/4, tick it "$0", dotted separator).
- **Iris's official split is broken** (class-sorted file, never shuffled: train
  is all setosa, test has none — every paper Iris cell affected; GEPA correctly
  no-ops because the valset is 100%). Use `iris_fixed` for real optimization
  results; exclude Iris from aggregates on both sides. Degenerate-valset
  no-ops are a *benchmark-bug detector* — investigate, don't force.
- LangProBe upstream bug fixed in the clone: missing import in
  `HeartDisease_program.py` (one line, documented in RESULTS.md).

## Extending

New benchmark = one `registry()` entry (Bench class + Signature) — nothing
else changes. Next tier (self-contained, ~$2–3 + 30–60 min each): MMLU, GSM8K,
MATH, Judge, HumanEval. Retrieval tier (HotpotQA, hover, RAGQAArena) depends on
public ColBERTv2 endpoints being alive; SWE-annotation tasks are long-context
(~$4–7 each). Defer IReRa/AppWorld (own infra). The eval already caught one
real Flex bug (identity baseline dropped field desc/prefix — fixed with
dict-form signature reconstruction + shim `InputField`/`OutputField` + bridge
decode, regression tests in `tests/flex/test_flex_binding.py`): treat
benchmark integration failures as potential library bugs first.
