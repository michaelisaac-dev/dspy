# SWE-bench Pro: does an evolved harness make a small model a better software engineer?

Fourth in the series after conflation (classification), political emails (extraction) and
terminal-bench-2 (terminal-agent harness). This one runs the experiment TB2's writeup put first on
its "if continuing" list (§11.4): hold the execution model fixed and **small**, and ask whether
GEPA evolving the `dspy.Flex` harness raises the resolve rate — i.e. whether harness structure can
buy back capability the model alone lacks.

- Benchmark: [SWE-bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os) (Scale AI), public set —
  731 real GitHub issues across 11 repos, each with its own prebuilt Docker image, human-verified
  fail-to-pass / pass-to-pass test lists, and per-instance run scripts.
- Execution model: **Claude Haiku 4.5** (the weakest, cheapest Claude) for every episode in every
  arm. Reflection: Claude Opus 5, whose cost amortizes across the run.
- The comparison: `dspy.Flex(sig, tools)` out of the box (a single `dspy.RLM` — a real code-REPL
  agent, not a straw man) vs the same object after GEPA rewrote its source under a
  resolve-rate-only metric.

> **Status: pilot run complete (2026-08-03). Headline: baseline 0/12 → evolved harness 3/12 on
> held-out instances, same Haiku 4.5, zero contamination hits. Directional, not significant
> (n=12). Total spend $123.26.**
>
> | component | state |
> |---|---|
> | manifest (731 instances), pilot splits (python-only, 3 repos, disjoint, deterministic) | verified by `preflight.py` |
> | verifier contract vs the official eval | **verified against the oracle**: gold patch → `resolved=True` (4/4 f2p, 0/52 p2p broken); empty patch → `resolved=False` |
> | agent container + tools + patch extraction + Flex sandbox | verified by `preflight.py --smoke` |
> | the experiment itself | see `results.json` |

---

## 1. Task and data

Each SWE-bench Pro instance is a real repository at a pinned commit, a real issue
(`problem_statement`), an explicit statement of what a fix must do (`requirements`), the interface
it should expose (`interface`), and two hidden test sets: `fail_to_pass` (fail before the fix,
must pass after) and `pass_to_pass` (must keep passing). An instance is **resolved** only if every
test in both sets passes — the union check, exactly as the official eval computes it.

### The pilot subset (this run)

30 instances, **Python only**, stratified across the three Python repos so no single codebase
carries the result:

| split | n | ansible | openlibrary | qutebrowser | role |
|---|---|---|---|---|---|
| GEPA train | 10 | 4 | 3 | 3 | reflection minibatches (size 2) |
| GEPA val | 8 | 3 | 3 | 2 | candidate selection |
| test | 12 | 4 | 4 | 4 | reported metrics |

Python-only is deliberate: one language keeps the emulation/toolchain axis (Go compiles and npm
installs behave very differently under qemu on Apple Silicon) out of a 30-instance comparison, and
Python is the language a small model is least handicapped on — if the harness effect does not show
up here, more languages will not rescue it. The language axis belongs to a bigger run.

## 2. Method

### What runs where — the honesty structure

```
agent container  (instance image, repo at /app reset to base_commit)
    harness(problem, requirements, interface, session)   <- the ONLY optimizable part
    -> git add -A; git diff --cached base_commit          <- the patch is the deliverable
    -> container destroyed
verify container (fresh, same image; the agent never had access to it)
    apply patch -> check out updated test files -> run_script.sh -> parser.py -> output.json
    resolved = (fail_to_pass ∪ pass_to_pass) ⊆ PASSED
```

The verifier mirrors `scaleapi/SWE-bench_Pro-os`'s local-docker evaluation exactly: same
entryscript structure (ENV exports from the instance dockerfiles, `git apply -v`, the last line of
`before_repo_set_cmd`, the per-instance `run_script.sh` and `parser.py`), same resolution rule.
The agent cannot see the test lists, the updated test files, the run scripts, or the gold patch —
they exist only host-side and in the verify container, after the agent is gone.

### Objective

`make_resolve_metric`: score = 1 if resolved else 0. No call penalty — the question is
capability, not cost (TB2 carries the cost objective for this series). Feedback per episode: a
verdict line distinguishing failure modes (harness crashed / deadline / no commands / no patch /
patch failed tests), the command transcript with exit codes, and the verifier's per-test verdicts
(which issue tests still fail, which previously-passing tests broke) — evidence, not a scalar.

### Contamination

GEPA sees verifier output (failing test names, pytest tails) for the 18 train/val instances only —
an ordinary training signal, but real information about those instances. Before believing any
positive result, grep the winning `module_src` for instance-specific literals (file paths, test
names, repo-specific strings); the check is cheap and §3 of the TB2 writeup shows why it is not
optional.

### Instrumentation

dspy caches disabled; per-episode cost from `dspy.track_usage()` priced through an explicit table
and cross-checked against litellm's aggregate; `finish_reason == "length"` counted per phase
(silent reflection truncation destroyed an emails-demo run); every finished episode appended to
`episodes.jsonl` keyed by (arm, harness-source-hash, instance) so killed runs resume; containers
labelled and reaped at startup and exit.

## 3. Results (pilot run, 2026-08-03)

| arm | resolved | calls/task | $/task | mean latency | deadline hits | empty patches |
|---|---|---|---|---|---|---|
| baseline (Flex out of the box, Haiku 4.5) | **0/12** | 20.3 | $0.61 | 477 s | 0 | 5 |
| GEPA-evolved harness (same model) | **3/12** | 22.9 | $0.97 | 697 s | 3 | 4 |

Paired on the same 12 held-out instances: **gained 3, lost 0**. The three resolves span two of the
three repos (openlibrary 1, qutebrowser 2, ansible 0). Full per-episode records in `results.json`.

**Read the 3/12 honestly.** Three one-directional flips out of twelve give a two-sided exact
p ≈ 0.25 — *directional, not significant*. What the pilot establishes is (a) the pipeline holds
end to end, (b) the effect direction, and (c) that the winning harness generalizes: the
contamination grep found **zero** train/val-specific literals in its source (no file paths, no
test names, not even the repo names — the evolved harness is fully generic). The val→test drop
(5/8 selected-on vs 3/12 held-out) is ordinary selection optimism and is why the val number was
never the claim.

**What GEPA built** (311 lines, from a one-RLM seed): two `dspy.Predict` stages and two
`dspy.ReAct` agents, with plain-Python glue that parses test output for failure markers — i.e. it
grew localization, editing, and self-verification structure. The baseline's signature failure
mode (confident summary, zero-byte patch: 5/12 episodes) dropped to 4/12, and the calls became
adaptive (16–31 per episode) instead of pinned at the seed's fixed 21-call ceiling.

**Cost honesty.** The compile cost $104.29 — ~3× the estimate — because evolved candidates run
ReAct trajectories that re-send growing context every step (44.6M prompt tokens across the
compile) and the capability metric deliberately does not charge for calls. Total run: **$123.26**
(baseline $7.31, compile $104.29, optimized eval ~$11.66), exceeding the run's original $100
envelope with the operator's explicit approval before the final phase. 1,170 execution-side
truncations at `max_tokens=8000` were logged during the compile; raising the exec cap is a cheap
lever for a follow-up.

Checks any positive result must survive:

| check | how |
|---|---|
| memorized train/val instances? | grep winning `module_src` for instance-specific literals |
| gain concentrated in one repo? | `by_repo` in the JSON; 4 instances per repo cannot carry a result |
| reflection truncated? | `truncated_calls` in the compile meter |
| budget ran out mid-climb? | winning candidate's position in the GEPA log |
| paired flips | `gained` / `lost` lists printed by `run_experiment.py` |

## 4. What this demo is *not*

1. **Not leaderboard-comparable.** The official numbers use SWE-agent under its own scaffold and
   budgets. This demo's absolute resolve rates are for its *internal* comparison only — both arms
   run identical conditions on identical instances, which is what the experiment measures.
2. **Not the full benchmark.** 30 of 731 instances, one language. At n_test=12, only large effects
   are even visible (see §7).
3. **Stateless shell.** Commands run via `docker exec bash -lc` with the working directory carried
   across calls; exported variables and background jobs do not persist (same deviation as TB2).
4. **Emulation.** On Apple Silicon the amd64 images run under qemu/Rosetta; wall-clock latency is
   not representative of native hardware. Both arms pay it equally.

## 5. Files

| file | contents |
|---|---|
| `fetch_data.py` | HF dataset + upstream scripts → `swebp_data/manifest.jsonl`; `--pull-images pilot` |
| `swebp_common.py` | manifest, splits, docker layer, verifier, tools, `SWEAgent`, metric, cache |
| `preflight.py` | environment checks; `--oracle` / `--null` / `--oracle-split` / `--smoke` |
| `run_experiment.py` | baseline → compile → optimized; resume; results.json |
| `episodes.jsonl` | per-episode resume log (gitignored) |
| `programs/` | the evolved harness, loadable via `dspy.Flex(...).load()` |

## 6. Reproducibility

```bash
python fetch_data.py                          # dataset + upstream       (~100 MB)
python preflight.py                           # env + splits             ($0, seconds)
python fetch_data.py --pull-images pilot      # 30 images                (~10-30 GB total)
python preflight.py --oracle <iid> --null <iid>   # verifier vs gold     ($0, minutes)
python preflight.py --smoke <train-iid>       # one baseline episode     (~$0.2-0.5)
python run_experiment.py --phase all 2>&1 | tee experiment.log
python run_experiment.py --phase all --resume     # after any interruption
```

## 7. Budget and statistical honesty, before starting

Cost envelope (Haiku execution): 12 baseline + ~60 compile + 12 optimized ≈ 84 episodes. At an
observed ~$0.2–0.5/episode plus $3–8 Opus reflection, the pilot lands in **$25–55**, under the
run's `--max-cost-usd 90` guard. Wall clock at 3 concurrent episodes: several hours, dominated by
agent reasoning and emulated test runs.

At n_test=12, a two-sided comparison of paired resolve rates can only distinguish large effects
(roughly: 0/12 vs 4/12 is suggestive; 2/12 vs 4/12 is noise). The pilot's job is not significance
— it is (a) does the pipeline hold end to end, (b) does GEPA's evolved harness *directionally*
beat the baseline at equal model, and (c) what does it write. Scaling n_test comes after those
answers, with this same code.
