# Terminal-Bench 2.0 under an LLM-call penalty: evolving the agent harness with Flex+GEPA

Third in the series after `tests/demos/conflation/all/EXPERIMENT.md` (binary classification) and
`tests/demos/political-fundraising-emails/n50-all/EXPERIMENT.md` (string extraction). Same
machinery — `dspy.Flex` + `dspy.GEPA` under a penalized objective, the same CAL (cost / accuracy /
latency) instrumentation, the same figure — pointed at a different kind of object.

**What is being optimized is different, and that is the whole point.** In the first two demos the
Flex module *is* the task solver, and GEPA's win is moving work out of the LLM and into deterministic
Python. Here the Flex module is an **agent harness**: it receives a task instruction and three tools
that reach into a live Docker container, and it has to drive that container to a state where the
task's own hidden tests pass. GEPA rewrites the harness — the loop, the sub-predictors and their
instructions, the recon strategy, the stopping rule, the self-verification — while the *tasks* stay
untouched and unseen.

> **Status: built and validated, not yet swept.**
>
> | component | state |
> |---|---|
> | manifest, splits (disjoint / stratified / deterministic / covering all 89) | verified |
> | container runner + verifier contract | **verified against the oracle**: `fix-git` scores `reward=1` in 28 s |
> | metric, every feedback branch | verified against synthetic predictions |
> | console table + four-panel figure | verified against synthetic records |
> | Flex sandbox + tool bridge (`preflight.py --smoke`) | **unverified** — needs Deno installed |
> | the sweep itself | **not run** |
>
> The results tables in §3 are empty because the sweep has not been paid for; §8 estimates it at
> **$500–1,000 and ~20 wall-clock hours**. Read §8 before starting one. Two Windows-only bugs that
> would each have silently ruined a full sweep were caught by the oracle at $0 — see §10.

---

## 1. Task and data

[Terminal-Bench 2.0](https://github.com/harbor-framework/terminal-bench-2) is 89 containerized
terminal tasks — "assemble a genome from reads", "recover a truncated SQLite database", "fix this
Git repository", "find the mate-in-one from a PNG of a chess board". Each ships:

| file | role |
|---|---|
| `task.toml` | prebuilt Docker image, CPU/memory caps, agent + verifier timeouts, difficulty |
| `instruction.md` | the prompt handed to the agent — the only thing it sees |
| `tests/test.sh` | the verifier; writes `1` or `0` to `/logs/verifier/reward.txt` |
| `solution/solve.sh` | the oracle, used here only to smoke-test the runner |

Because **every** task pins a prebuilt `docker_image`, the checkout can skip Git-LFS blobs entirely
(they are only build context for images we never build): ~1 GB becomes ~80 MB. That is what
`fetch_data.py` does, and it is why `tb2_data/` is `.gitignore`d — it is reproducible byte-for-byte
from one command.

- **89 tasks**, 4 easy / 55 medium / 30 hard, across 16 categories (software-engineering 26,
  system-administration 9, scientific-computing 8, security 8, …).
- Instructions run ~715 characters median, 4,345 max.
- **The label is produced by the environment, not by a matcher.** There is no fuzzy matching and no
  judgment call anywhere in the metric — unlike both earlier demos, where the matcher was itself a
  limitation worth arguing about.

### Splits

Stratified **within each difficulty band**, because the benchmark is lopsided (4/55/30) and an
unstratified shuffle can hand a split zero easy tasks — which would make resolve rates across splits
incomparable for reasons that have nothing to do with the harness.

| split | n | easy | medium | hard | role |
|---|---|---|---|---|---|
| GEPA train | 24 | 1 | 15 | 8 | reflection minibatches (size 2) |
| GEPA val | 20 | 1 | 12 | 7 | candidate selection |
| **test** | **45** | **2** | **28** | **15** | reported metrics |

Disjoint by construction, deterministic under a fixed seed, and their union is all 89 tasks — all
four properties are asserted by `preflight.py`.

---

## 2. Method

### What GEPA actually rewrites

`TerminalAgent` (in `tb2_common.py`) runs one episode:

```
start container  →  harness(instruction, session)  →  copy tests in  →  bash /tests/test.sh  →  tear down
                    ^^^^^^^ the only optimizable part
```

`self.harness` is a `dspy.Flex` over `SolveTerminalTask`, with three host tools bridged into its
sandbox: `terminal_exec`, `write_file`, `read_file`. The wrapper is what keeps the experiment honest:
the reward is produced by the task's own verifier **after** the harness has returned and can no
longer act, and `tests/` is copied into the container only at that moment. No harness, however it is
rewritten, can read or edit the thing that grades it.

**Baseline** = whatever `dspy.Flex(sig, tools=...)` gives you before optimization, which is a single
`dspy.RLM` over the signature — a code-REPL agent with tool access. That is a real agent, not a straw
man, and it is deliberately *not* hand-tuned: the comparison is "out-of-the-box Flex" against "Flex
after GEPA", which is the comparison a user of this library actually faces.

### Objective

For each penalty λ, GEPA optimizes

```
score = max(0, resolved − λ · n_llm_calls / STEP_BUDGET)          STEP_BUDGET = 30
```

**The normalizer is not cosmetic.** A terminal episode is tens of LLM calls, not one, so the
per-call λ used in the other two demos (0.05–0.4) would drive every score to 0 and the optimizer
would learn only "make no calls". Dividing by a reference step budget makes λ read as *"the fraction
of the score forfeited by a harness that spends 30 LLM calls"*, which is both interpretable and
comparable across the three demos. It is a unit of account, not a cap; the cap is
`max_predictor_calls=80`, enforced by `dspy.Flex` itself.

Swept **λ ∈ {0, 0.1, 0.25, 0.5}** — four points rather than five, because each one is a full GEPA
compile plus a 45-episode evaluation, and a fifth point buys less than the seed replication §9 says
this needs more.

### Feedback is a transcript, not a scalar

A harness can only be improved from evidence about its behavior. `make_metric` therefore returns, per
episode:

1. a **verdict line** distinguishing the failure modes that call for different fixes — crashed,
   ran out of wall clock, ran *no commands at all*, verifier returned 0, solved but expensive;
2. the **command transcript** — last 18 commands with exit codes, durations, output sizes and
   200-character tails;
3. the **verifier's own output**, which the agent never saw during the episode;
4. the **objective**, stating that shell commands are free and only model calls are charged, without
   saying how to structure the loop.

An infrastructure failure (Docker could not give us a container) is reported as such and explicitly
labelled *not a harness failure*, so the optimizer is not taught from noise.

### Contamination

The verifier tail quoted in the feedback contains pytest output from a task's hidden tests. GEPA sees
this **only for the 44 train/val tasks, never for the 45 test tasks** — an ordinary training reward
signal. It is still real information about those 44, so §3.3 requires inspecting the winning harness
source for task-specific literals before any result is believed. The same check caught nothing in the
emails demo and is cheap; skipping it here would be indefensible given the harness sees far richer
per-task text than a committee name.

### Instrumentation

dspy caches disabled, so cost and latency are cold-call values. Per-episode cost from
`dspy.track_usage()` priced through an explicit table, cross-checked against litellm's own per-call
cost. Truncation detection counts `finish_reason == "length"` — added after silent reflection
truncation destroyed an entire run of the emails demo ($3.99) and looked exactly like "the optimizer
found nothing".

---

## 3. Results

**Not yet run.** The tables below are the shape the sweep fills in; `sweep_penalties.py` prints the
first one verbatim and `--plot-only` renders the figure from `penalty_sweep.json`.

| λ | resolved | rate | easy | med | hard | calls/task | cmds/task | $/task | sec | dl | score | p vs base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | | | | | | | | | | | — | — |
| 0 | | | | | | | | | | | | |
| 0.1 | | | | | | | | | | | | |
| 0.25 | | | | | | | | | | | | |
| 0.5 | | | | | | | | | | | | |

`dl` = episodes cut off by the wall-clock cap. **A nonzero `dl` column caps the resolve rate**, and a
sweep where it differs across λ is not a clean comparison — the cheap harnesses would be winning
partly by finishing.

### 3.1 The question λ=0 answers

Whether GEPA can improve a terminal agent *at all* when cost is not a consideration. Both earlier
demos found their λ=0 result surprising in opposite directions — conflation's λ=0 was already
decomposing to 0.254 calls/example unprompted, while the emails λ=0 kept exactly 1.000 calls/email
and won purely on prompt and signature structure. Neither prior predicts this one.

### 3.2 The question λ>0 answers

Whether the *shape* the first two demos found — two regimes with nothing resolvable inside the
penalized one — survives when "spend fewer LLM calls" means "do more in the shell" rather than "write
a regex". The mechanism is genuinely different: shell commands are free under this metric, so a
penalized harness is not forced to become deterministic, only to stop *thinking* between actions.

### 3.3 Checks any positive result must survive

Copied from the emails demo, which needed all four:

| check | how |
|---|---|
| Did the harness memorize the training tasks? | grep the winning `module_src` for task names, file paths and literals from the 44 train/val tasks |
| Is the gain concentrated in one difficulty band? | `by_difficulty` in the JSON; 2 easy tasks cannot carry a result |
| Was the reflection truncated? | `truncated_calls` and `max_completion_tokens_seen` per λ |
| Did the budget run out mid-climb? | `python budget_check.py sweep.log` (§7) |

---

## 4. What this demo is *not*

**This is not the official Harbor harness, and its resolve rates are not comparable to the public
Terminal-Bench leaderboard.** Three deliberate deviations, each of which lowers the achievable
number:

1. **No persistent shell.** Commands run through `docker exec bash -lc`, which is stateless. The one
   piece of state a terminal agent constantly relies on — the working directory — is carried across
   calls through a state file, so `cd src` then `ls` behaves as expected. Exported variables, shell
   functions and background job control do **not** persist. Terminal-Bench proper drives a tmux
   session. This is the largest deviation.
2. **A tighter clock.** Tasks declare 900–1800 s for the agent; the demo caps episodes at 600 s by
   default (`--episode-timeout`) because the official budget is unaffordable across a sweep. The cap
   is reported as `deadline_hits` rather than hidden.
3. **No `storage_mb` enforcement and no GPU.** `--cpus` and `--memory` are honored; the storage cap
   needs a storage driver that Docker Desktop does not provide by default, and no task in the
   manifest requests a GPU.

None of these affect the *comparison* between baseline and optimized harness — both run under
identical conditions on identical tasks — which is what the experiment measures. They do mean the
absolute rate understates what the same models achieve under Harbor.

---

## 5. Files

| file | contents |
|---|---|
| `fetch_data.py` | clones TB2 without LFS blobs, flattens `task.toml` into `tb2_data/tasks.jsonl`, optionally pre-pulls images |
| `tb2_common.py` | signature, splits, Docker session layer, the three host tools, `TerminalAgent`, metric factory, CAL instrumentation, episode cache |
| `preflight.py` | environment checks; **oracle mode**, which validates the whole runner for $0 |
| `pilot.py` | measures candidate execution models on an 8-task train probe |
| `sweep_penalties.py` | sweep driver, console table, CAL figure |
| `budget_check.py` | budget-adequacy diagnosis from a run log |
| `penalty_sweep.json` | all metrics, every per-episode record |
| `episodes.jsonl` | per-episode resume log (gitignored) |
| `cal_frontier.png` | four-panel figure |
| `sweep_programs/` | saved harnesses, loadable via `dspy.Flex(...).load()` |
| `gepa_log_*/` | GEPA candidate checkpoints |

---

## 6. Reproducibility

```bash
python fetch_data.py                      # clone + manifest         (~80 MB, no LFS)
python preflight.py                       # docker, deno, splits     ($0, seconds)
python fetch_data.py --pull-images --split test   # optional; avoids pulls inside episodes

# Validate the runner before spending anything. This replaces the agent with each task's own
# solve.sh and runs the real verifier: a PASS proves image pull, container start, working directory,
# command execution, test copy, test.sh and reward parsing are all correct.
python preflight.py --oracle fix-git      # one task                 ($0)
python preflight.py --oracle-split val    # all 20 val tasks         ($0, slow)
python preflight.py --smoke fix-git       # Deno sandbox + tool bridge ($0)

python pilot.py                           # choose the execution model by measurement
python sweep_penalties.py --penalties 0 --test-limit 6 --max-metric-calls 20   # cheap dry run
python sweep_penalties.py 2>&1 | tee sweep.log                                 # the real thing
python sweep_penalties.py --resume        # skip penalties already in the JSON
python sweep_penalties.py --plot-only     # re-render, no API calls
python budget_check.py sweep.log          # was the budget enough?
```

**A failing oracle is a bug in this runner, not an agent failure.** Finding it after a $600 sweep
instead of before is the difference `preflight.py` exists to make.

### Recoverability

The other two demos lost runs to crashes and learned from it; here an episode costs minutes and
dollars instead of milliseconds and cents, so the same protections are stronger:

- the JSON is rewritten after **every** λ, not at the end;
- `--resume` skips any λ already present;
- GEPA checkpoints every candidate to `gepa_log_<λ>/` — `compile()` only returns at the very end, so
  without this a killed run discards everything it found;
- **every finished episode** is appended to `episodes.jsonl`, keyed by
  `(λ, sha256(harness source), task)`. An evaluation killed at task 30 of 45 resumes at task 31. The
  source hash is what makes this safe: change the harness by one character and every key changes, so
  a cached record can never be attributed to a program that did not produce it. Episodes are
  stochastic, so this is *resumption*, not memoization — `--no-episode-cache` forces a clean
  re-measurement;
- containers are labelled `dspy_tb2_demo=1` and reaped at startup and via `atexit`, because a killed
  run cannot run its own `finally` block.

### Prerequisites

- **Docker daemon running.** ~50 GB of free disk for the images if you pull all 89.
- **Deno**, for the `dspy.Flex` sandbox: `irm https://deno.land/install.ps1 | iex` (PowerShell) or
  `curl -fsSL https://deno.land/install.sh | sh`.
- `ANTHROPIC_API_KEY`.

---

## 7. Budget adequacy

`max_metric_calls=120` is small — 120 *container episodes*, roughly 6 reflection rounds over a
20-task valset. The emails demo showed exactly what a starved run looks like: at λ=0.2 with 200 calls
it reported 0.890 accuracy / 0.270 calls / $0.51, and at 600 the same configuration produced 0.930 /
0.090 / $0.24 — simultaneously more accurate, three times more deterministic and half the cost. The
starved run was not wrong, it was *early*, and nothing in its output said so.

`budget_check.py` reads the sweep log and reports where in each run the winning candidate appeared:
found early → plateaued; found in the final quarter → still climbing, raise the budget. It also flags
the case that matters most here — the base harness never being beaten — which at this budget is the
likeliest outcome and which is indistinguishable from "every proposal failed to bind" unless you
check `truncated_calls` too.

---

## 8. Cost, before you start

Per episode: an RLM agent on a TB2 task runs tens of LLM calls with growing context. At Sonnet-5
pricing that is roughly **$0.7–1.5 per episode** and **5–10 minutes of wall clock**.

| item | episodes | estimate |
|---|---|---|
| baseline evaluation | 45 | $35–65 |
| per λ: GEPA compile | 120 | $85–180 |
| per λ: reflection (Opus-5) | — | $3–8 |
| per λ: test evaluation | 45 | $35–65 |
| **per λ total** | 165 | **$120–250** |
| **four λ + baseline** | **705** | **$500–1,000** |

Wall clock at 4 concurrent containers: **~5 h per λ, ~20 h for the sweep.**

Two ways to spend less:

- `--test-limit 6 --max-metric-calls 20 --penalties 0 0.25` exercises the entire pipeline for roughly
  $30. A truncated test set is recorded as `test_limit` in the JSON so it can never be mistaken for a
  result.
- `--episode-timeout 300` halves the ceiling. It will also lower the resolve rate; watch
  `deadline_hits`.

This is the first demo in the series where the cost is dominated by *execution*, not reflection —
the inverse of the emails sweep, where Opus-5 reflection dominated and Haiku execution was
negligible.

---

## 9. Limitations

1. **n_test = 45.** At a ~30% resolve rate one standard error is ~6.8 pp, so only differences above
   ~15 pp are resolvable. This is the binding constraint on every conclusion, and unlike the emails
   demo there is no second source of tasks to pool in — the benchmark is 89 tasks and half of them
   are already in the test split.
2. **One GEPA run per λ, one episode per task.** Both the optimizer and the agent are stochastic, and
   neither variance is measured. A single-episode-per-task resolve rate on a hard benchmark is a
   noisy quantity; pass@1 with no replication is the weakest part of this design.
3. **The harness deviates from Harbor** (§4). Absolute rates are not leaderboard-comparable.
4. **Execution-model choice is a pilot, not a proof.** `pilot.py` probes 8 tasks; a model that clears
   the floor there could still be the wrong tier for the full sweep.
5. **Verifier flakiness is untested.** Every `test.sh` apt-installs and downloads `uv` from the
   network. A network hiccup during verification scores a solved task as unresolved, and nothing
   currently distinguishes that from a genuine failure. Re-running the oracle over a split twice
   would measure it; that has not been done.
6. **Contamination is bounded but not zero** (§2). GEPA reads verifier output for 44 tasks.

---

## 10. Bugs found before the first run, and what each would have cost

The emails demo recorded its bugs after they had cost $24.51. The oracle path exists so this one
records them at $0. Both of these were found by the *first* oracle run, and both are invisible on
Linux and macOS.

**1. CRLF verifiers — would have scored 0 on all 89 tasks, indistinguishable from a useless agent.**
Git's `core.autocrlf=true`, the default on many Windows installs, rewrites every `.sh` in the
checkout to CRLF. `docker cp` copies bytes, so the container receives a verifier bash cannot parse:

```
/tests/test.sh: line 2: $'\r': command not found
/tests/test.sh: line 30: syntax error: unexpected end of file
```

`reward.txt` is never written, so every task reports `resolved=False`. The sweep would have run for
20 hours, cost several hundred dollars, and produced a uniform 0.000 resolve rate at every λ — which
reads as "the agent can't do terminal work" and not as "the grader was broken". Fixed by pinning
`core.autocrlf=false core.eol=lf` at clone, an auto-repair path
(`fetch_data.py --fix-line-endings`), and a **fatal** preflight check, because this failure mode is
too expensive to warn about.

**2. `subprocess(text=True, input=...)` silently CRLF-ifies every file the agent writes.** Python's
text-mode stdin wrapper translates `\n` to `os.linesep` on write, so on Windows

```python
subprocess.run([...], text=True, input="a\nb\n")   # the child receives b'a\r\nb\r\n'
```

Every file `write_file` put into a Linux container — which for a terminal agent is mostly shell
scripts and source files — would have arrived with CRLF endings and failed for reasons the agent
could not see or fix. This one would *not* have zeroed the benchmark; it would have quietly lowered
the resolve rate by an unknown amount and been attributed to the harness. Fixed by encoding stdin to
bytes in `_docker` and decoding the output by hand.

Two further hazards were fixed by inspection rather than by the oracle:

**3. The episode clock started before the container did.** `deadline` was set in `__init__`, so a
first-time image pull (minutes) was charged against the agent's 600 s. Whether a harness had a full
budget or a tenth of one would have depended on whether the image happened to be in the local cache
— a hidden variable that could make one λ look better than another for no real reason. The clock now
starts at the end of `start()`.

**4. `docker exec -w /app` on an image with no `WORKDIR`.** The bootstrap that creates the working
directory was itself running *in* that directory. Now bootstraps from `/`.

---

## 11. If continuing

1. **Seed replication** — the only way to separate optimizer noise from a real effect at n=45.
2. **A persistent tmux-backed shell**, closing the largest deviation from Harbor in §4.
3. **`k` episodes per task** to turn a noisy pass@1 into an estimate with an error bar of its own.
4. **A weaker execution model under the evolved harness** — the interesting version of this
   experiment is whether a better harness lets a cheaper model reach the same resolve rate, which is
   the CAL question this benchmark is actually good at answering.
