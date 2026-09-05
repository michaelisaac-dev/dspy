# Hiring-decision forensics with Flex + GEPA — Bertrand–Mullainathan résumé data

**Question** (Isaac, 2026-09-04): if Flex compiles a program that predicts an
employer's hiring decisions, does the artifact surface hidden discrimination —
e.g. explicit race/gender weighting readable in source?

**Dataset**: Bertrand & Mullainathan résumé audit (via OpenIntro; `resume.csv`,
4,870 rows). Real employer **callback decisions**; race/gender signaled by
applicant names and **randomly assigned** — so any race weight that predicts
callbacks is causal discrimination, not omitted-variable proxying. Ground
truth: 9.65% callbacks for white-coded names vs 6.45% for Black-coded (~1.5×),
gender effects ≈ 0. Base rate 8%; the effect is ~3pp — small by construction.

**Setup**: `run_hiring.py`. Task LM gpt-4o-mini, reflection gpt-5-mini,
Flex+GEPA at λ=0.4 (pure-code artifacts), 3,000 metric calls. Treatments:
`full` (17 qualification fields + firstname/race/gender) vs `masked` (quals
only). Test = 1,500 held-out; balanced accuracy; **group-rate audit** = the
compiled program's predicted callback rate by true race/gender on test.
Classical ceiling: pooled logistic regression (3,370 rows).

## Results

| arm | balanced acc | race term in artifact | predicted white vs black rate |
|---|---|---|---|
| logreg full (ceiling) | 57.4 | +0.30 log-odds (correct sign) | — |
| logreg masked | 60.7 | — | — |
| v2 per-example feedback, full s0 | 55.8 | none | 12.0% / 12.7% (no gap) |
| v2 full s1 | 50.0 | **hash seed only** (see F3) | 21.1% / 16.0% (hash noise) |
| v2 masked s0/s1 | 50.5 / 54.7 | — | no systematic gap |
| **v3 aggregate feedback, full s0** | 53.0 | **explicit: `{"black": 0.062, "white": 0.104}`** | **6.0% / 2.3% — study's direction, amplified** |
| v3 full s1 | 57.1 | **explicit refusal** ("Do not use protected attributes… to avoid introducing bias") | 13.6% / 16.9% (no gap) |

v3 = v2 plus neutral per-field crosstabs in the GEPA feedback (callback rate by
value for every field equally — the aggregate evidence any analyst starts from;
race got no special billing among 19 fields).

## Findings

**F1 — Individual-level fidelity is the wrong detector for a 3pp effect.**
Even the classical ceiling shows *negative* ΔFidelity (masked 60.7 > full 57.4):
name dummies add more variance than the true race signal adds prediction. The
right detectors are structural (terms in the artifact) and aggregate-behavioral
(group-rate audit) — both only possible because the artifact is code.

**F2 — Per-example reflective feedback is blind to population-level effects.**
v2's compiled programs are honest qualification rubrics with no race terms and
no rate gap: GEPA's minibatch-of-3 reflection cannot accumulate evidence for a
3pp effect that pooled regression sees at 3+σ. Discovery needs aggregate
evidence in the loop.

**F3 — The hash-gamer specimen (v2 full s1): why artifact auditing beats both
grep and behavioral auditing.** GEPA matched the 8% base rate by emitting "yes"
from a deterministic hash of `firstname|race|gender`. A grep audit flags it
(race appears in source); a behavioral audit flags it (its hash produces
white 21.1% vs black 16.0%!); the code proves the mechanism is arbitrary noise,
not systematic weighting. Term presence ≠ discriminatory use; behavioral
disparity ≠ systematic mechanism; only the artifact adjudicates.

**F4 — With aggregate evidence, discrimination becomes legible in source — but
the reflection model's values interfere.** Given identical crosstabs, seed 0
**encoded the race rates verbatim** into its scoring table and reproduces the
study's direction on held-out data (white 6.0% vs black 2.3% predicted
callbacks — amplified ~2.6× vs the study's 1.5×, encoded as too strong a
factor); seed 1 **explicitly refused** ("Do not use protected attributes"),
built a quals-only rubric — and scored *better* (57.1, ≈ the logreg ceiling).
Two consequences: (a) the user's hypothesis is demonstrated end-to-end — a
compiled program surfaced the dataset's hidden discrimination as two readable
lines of source; (b) alignment training makes the reflection model sometimes
*editorialize instead of describe* on a forensics task — descriptive-audit
framing in the instructions is the v4 knob, and the refusal behavior is itself
a finding about LM-driven decision modeling.

**F5 — Library bug found and fixed** (second one caught by eval work):
`DspyGEPAResult.from_gepa_result` rebuilt every history candidate and crashed
the whole compile when one rejected candidate carried unparseable code, losing
the winning program. Fixed defensively (unbuildable candidates → None + warning;
`best_candidate` still fails loudly); regression test in
`tests/flex/test_flex_gepa.py`.

## Utrecht calibration (synthetic, planted bias — company B)

Second dataset (`run_utrecht.py`; ICT Institute synthetic recruitment, via the
GitHub mirror of the Kaggle set). Company B's planted rule: male 45.5% vs
female 14.0% hired (3.25x). Strong-signal calibration of the instrument:

| arm | balanced acc | gender in artifact | predicted male vs female |
|---|---|---|---|
| logreg full / masked (ceiling) | 89.7 / 90.1 | +0.50 male | — |
| Flex+GEPA full s0 / s1 | 89.1 / 89.5 | **`{"male": 0.477, "female": 0.116}` encoded (both seeds)** | 56.7%/21.4% and 56.2%/18.8% |
| Flex+GEPA masked s0 | 88.5 | — (never saw gender) | **55.2% / 24.5%** |

**F7 — Large planted bias: recovered in code and in behavior, both seeds** —
ratios 2.65–3.0x vs planted 3.25x (direction AND approximate magnitude; large
effects encode faithfully where B&M's 3pp got distorted). Fidelity matches the
classical ceiling. No ethics refusals on synthetic gender data (vs 1/2 refusal
on B&M's real race data — alignment asymmetry across attributes/realism).

**F8 — Proxy discrimination survives masking, and the artifact names the
proxy.** The masked program (never saw gender) still predicts male 55.2% vs
female 24.5% (2.25x). Its own code says why: "Core strong positive: debate club
membership" — and debateclub=True is 75% male (sport is similarly gendered:
Rugby 78% male, Chess 90% female). Masking protected attributes removed almost
none of the discrimination; it just laundered it through correlated features —
legible only because the decision procedure is source code.

Figure: `results/fig_forensics_comparison.png` (B&M panel with the verbatim
AER-2004 quote + Utrecht panel).

## Caveats

- 2 seeds per arm; the v3 encode-vs-refuse split needs more seeds to estimate
  rates. Group-rate gaps carry sampling noise (~±1.5pp at n=1500, 8% base).
- The v3 program *over-weights* race relative to ground truth — recovery of
  direction, not magnitude. Magnitude-faithful encoding likely needs
  calibration-aware feedback.
- gpt-5-mini may know this famous study; the crosstabs we fed dominate any
  prior, but contamination can't be excluded for the refusal behavior either.
- v1 (balanced valset — miscalibrated artifacts) archived in
  `results/results_v1_balancedval.jsonl`; superseded by v2/v3 design.

## Reproduce

```
.venv/bin/python tests/demos/hiring-forensics/run_hiring.py <full|masked> <baseline|gepa|logreg> [seed] [agg]
```
