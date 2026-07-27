"""Sweep the LLM-call penalty on Terminal-Bench 2.0 and trace the CAL frontier.

Same design as `tests/demos/political-fundraising-emails/n50-all/sweep_penalties.py`, but what GEPA
rewrites here is the AGENT HARNESS, not a task solver. For each penalty lambda, GEPA optimizes
`max(0, resolved - lambda * n_llm_calls / STEP_BUDGET)` over 24 training tasks with 20 for candidate
selection, and the harness it produces is then run over 45 held-out tasks with resolve rate reported
alongside cost, latency and call rate.

Everything lands in `penalty_sweep.json` -- including per-episode records -- so any other metric can
be recomputed later without re-running the sweep.

    python preflight.py --oracle-split val    # ALWAYS do this first; it costs nothing
    python sweep_penalties.py                 # full sweep
    python sweep_penalties.py --resume        # skip penalties already in the JSON
    python sweep_penalties.py --penalties 0 --max-metric-calls 120
    python sweep_penalties.py --plot-only     # re-render the figure from the JSON

Recoverability, because an episode here costs minutes and dollars rather than milliseconds and cents:

* the JSON is rewritten after every lambda;
* `--resume` skips any lambda already in it;
* GEPA checkpoints every candidate into `gepa_log_<lambda>/` (`compile()` only returns at the very
  end, so without this a killed run discards everything it found);
* every finished episode is appended to `episodes.jsonl` keyed by (lambda, harness source hash,
  task), so an evaluation killed at task 30 of 45 resumes at task 31 rather than starting over.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import dspy

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from tb2_common import (  # noqa: E402
    EPISODE_TIMEOUT_S,
    EXEC_MODEL,
    MAX_PREDICTOR_CALLS,
    REFLECTION_MODEL,
    STEP_BUDGET,
    EpisodeCache,
    TerminalAgent,
    disable_cache,
    docker_available,
    fmt,
    load_splits,
    make_lms,
    make_metric,
    make_resolve_metric,
    meter,
    new_harness,
    reap_orphans,
    run_program,
    summarize,
)

DEMO_DIR = Path(__file__).parent
SWEEP_PATH = DEMO_DIR / "penalty_sweep.json"
PLOT_PATH = DEMO_DIR / "cal_frontier.png"
PROGRAM_DIR = DEMO_DIR / "sweep_programs"
EPISODES_PATH = DEMO_DIR / "episodes.jsonl"

# Four points, not five. Each lambda is a full GEPA compile plus a 45-episode evaluation, so a fifth
# point buys less than the seed replication that section 8 of EXPERIMENT.md says this needs more.
DEFAULT_PENALTIES = [0.0, 0.1, 0.25, 0.5]
# One metric call = one container episode, so this budget is measured in wall-clock hours, not
# seconds. 120 gives roughly 6 reflection rounds over a 20-task valset; `budget_check.py` reports
# whether that was enough for each lambda rather than leaving it to assumption.
DEFAULT_MAX_METRIC_CALLS = 120
# 2, not 4: the reflective record for one episode carries a command transcript and a verifier tail,
# so a 4-episode minibatch would push the code proposer's prompt past what is useful to reason over.
REFLECTION_MINIBATCH = 2
# Containers, not HTTP requests. Each holds a CPU and up to 2 GB, so this is bounded by the host,
# not by the API. Raise it only if `docker stats` says you have headroom.
EVAL_THREADS = 4

# From the dataviz reference palette, light mode; validated all-pairs (worst dE 24.7 protan).
C_OPT, C_BASE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c3"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval -- the honest error bar on a resolve rate from n tasks."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_p(ra: list[dict], rb: list[dict]) -> float:
    """Two-sided exact McNemar p for two harnesses scored on the same tasks (paired)."""
    by_task = {r["task"]: r for r in rb}
    pairs = [(a, by_task[a["task"]]) for a in ra if a["task"] in by_task]
    n01 = sum(1 for x, y in pairs if x["resolved"] and not y["resolved"])
    n10 = sum(1 for x, y in pairs if not x["resolved"] and y["resolved"])
    n = n01 + n10
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) for i in range(0, min(n01, n10) + 1)) / 2**n * 2)


def plot_path_for(out: Path) -> Path:
    return PLOT_PATH if out.resolve() == SWEEP_PATH.resolve() else out.with_suffix(".png")


def _relpath(p: Path) -> str:
    """Path relative to the demo dir when possible, else absolute. Never raises."""
    import os
    try:
        return str(Path(os.path.relpath(p, DEMO_DIR)))
    except ValueError:
        return str(p)


def program_dir_for(out: Path) -> Path:
    """Isolate saved programs per --out so a side run cannot overwrite the sweep's artifacts."""
    return PROGRAM_DIR if out.resolve() == SWEEP_PATH.resolve() else out.with_suffix("") / "programs"


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_penalty(penalty, train, val, test, max_metric_calls, exec_lm, reflection_lm, threads,
                objective: str = "penalty",
                program_dir: Path = PROGRAM_DIR, log_dir: Path | None = None,
                cache: EpisodeCache | None = None, episode_timeout_s: float = EPISODE_TIMEOUT_S):
    """Optimize the harness at one penalty and evaluate it. Returns a JSON-ready dict."""
    program = TerminalAgent(harness=new_harness(), episode_timeout_s=episode_timeout_s)
    baseline_src = program.harness.module_src

    started = time.perf_counter()
    with meter(exec_lm, reflection_lm) as opt_cost:
        optimized = dspy.GEPA(
            metric=(make_resolve_metric() if objective == "resolve" else make_metric(penalty)),
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            reflection_minibatch_size=REFLECTION_MINIBATCH,
            num_threads=threads,
            seed=0,
            log_dir=str(log_dir) if log_dir else None,
            # At lambda=0 the score IS the resolve rate, which is 0 or 1 per episode. A 2-task
            # minibatch that happens to be all-solved would be skipped by the default, throwing away
            # exactly the rounds that could teach the harness what worked.
            skip_perfect_score=False,
        ).compile(program, trainset=train, valset=val)
    opt_wall_s = time.perf_counter() - started

    records, meta = run_program(optimized, test, threads=threads, penalty=penalty, cache=cache)
    row = summarize(records, penalty)
    row.update(meta)

    program_dir.mkdir(parents=True, exist_ok=True)
    program_path = program_dir / f"harness_penalty_{penalty:g}.json"
    optimized.save(str(program_path))

    return {
        "penalty": penalty,
        "max_metric_calls": max_metric_calls,
        "optimization": {"wall_s": opt_wall_s, **opt_cost,
                         "changed_code": optimized.harness.module_src != baseline_src},
        "test": row,
        "records": [r._asdict() for r in records],
        "module_src": optimized.harness.module_src,
        "program_path": _relpath(program_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", choices=("penalty", "resolve"), default="penalty",
                    help="'penalty' sweeps the LLM-call cost objective (the original design). "
                         "'resolve' optimises ONLY the Terminal-Bench verifier pass rate -- no cost "
                         "term, and feedback that asks what capability the harness is missing "
                         "rather than how to be cheaper. Forces a single run at lambda=0.")
    ap.add_argument("--penalties", type=float, nargs="+", default=DEFAULT_PENALTIES)
    ap.add_argument("--max-metric-calls", type=int, default=DEFAULT_MAX_METRIC_CALLS)
    ap.add_argument("--threads", type=int, default=EVAL_THREADS)
    ap.add_argument("--episode-timeout", type=float, default=EPISODE_TIMEOUT_S,
                    help="wall-clock ceiling per episode, in seconds")
    ap.add_argument("--train-limit", type=int, default=0,
                    help="use only the first N GEPA train tasks (0 = all 24). Like --test-limit, "
                         "this makes the run a scoped exercise rather than a measurement.")
    ap.add_argument("--val-limit", type=int, default=0,
                    help="use only the first N GEPA val tasks (0 = all 20). The val split is what "
                         "GEPA selects candidates on, and every accepted candidate costs one "
                         "episode per val task -- so this is the main lever on compile cost.")
    ap.add_argument("--test-limit", type=int, default=0,
                    help="evaluate on only the first N test tasks (0 = all 45). For a cheap dry run "
                         "of the whole pipeline; a truncated test set is NOT a reportable result and "
                         "is recorded as `test_limit` in the JSON so it cannot be mistaken for one.")
    ap.add_argument("--out", type=Path, default=SWEEP_PATH)
    ap.add_argument("--resume", action="store_true", help="skip penalties already present in --out")
    ap.add_argument("--plot-only", action="store_true", help="re-render the figure from --out")
    ap.add_argument("--no-episode-cache", action="store_true",
                    help="re-run every episode instead of reusing finished ones from episodes.jsonl")
    args = ap.parse_args()
    # Resolve immediately: plot_path_for / program_dir_for derive paths from this, and a relative
    # --out made `program_path.relative_to(DEMO_DIR)` raise in the emails demo AFTER a full GEPA
    # compile had run -- losing the run's metrics to a bookkeeping line.
    args.out = args.out.resolve()
    if args.objective == "resolve":
        # There is no penalty to sweep: the objective is the verifier pass rate alone. lambda=0
        # keeps the record shape (score == resolve_rate) so the table, figure and JSON all work.
        args.penalties = [0.0]

    if args.plot_only:
        plot(json.loads(args.out.read_text(encoding="utf-8")), plot_path_for(args.out))
        return

    reachable, detail = docker_available()
    if not reachable:
        raise SystemExit(f"docker is not usable ({detail}); run `python preflight.py` for the fix")
    reaped = reap_orphans()
    if reaped:
        print(f"reaped {reaped} orphaned container(s) from an earlier run")

    disable_cache()  # so latency and cost are what a cold production call would cost
    exec_lm, reflection_lm = make_lms()
    dspy.configure(lm=exec_lm)
    train, val, test = load_splits()
    if args.train_limit:
        train = train[: args.train_limit]
    if args.val_limit:
        val = val[: args.val_limit]
    if args.test_limit:
        test = test[: args.test_limit]
    if args.train_limit or args.val_limit or args.test_limit:
        print(f"!! scoped run (train={len(train)} val={len(val)} test={len(test)}): "
              f"a pipeline exercise, not a measurement")
    print(f"splits: gepa_train={len(train)} gepa_val={len(val)} test={len(test)}")
    cache = None if args.no_episode_cache else EpisodeCache(EPISODES_PATH)
    if cache:
        print(f"episode cache: {len(cache)} finished episode(s) in {EPISODES_PATH.name}")

    data = json.loads(args.out.read_text(encoding="utf-8")) if (args.resume and args.out.exists()) else {}
    data.setdefault("meta", {}).update({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exec_model": EXEC_MODEL, "reflection_model": REFLECTION_MODEL,
        "max_metric_calls": args.max_metric_calls, "reflection_minibatch": REFLECTION_MINIBATCH,
        "skip_perfect_score": False,
        "objective": args.objective,
        "step_budget": STEP_BUDGET, "max_predictor_calls": MAX_PREDICTOR_CALLS,
        "episode_timeout_s": args.episode_timeout,
        "eval_threads": args.threads,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "test_limit": args.test_limit or None,
        "train_limit": args.train_limit or None,
        "val_limit": args.val_limit or None,
        "benchmark": "Terminal-Bench 2.0 (89 tasks), harbor-framework/terminal-bench-2",
        "split_rule": "stratified by difficulty: 50% test, 22% gepa_val, remainder gepa_train",
        "cache": "disabled (dspy disk + memory)",
        "penalty_note": (
            "score = max(0, resolved - lambda * n_llm_calls / step_budget). The normalizer makes "
            "lambda read as 'fraction of the score forfeited by a harness that spends step_budget "
            "LLM calls', which keeps it comparable to the per-call lambda in the other two demos."
        ),
        "latency_note": (
            "latency_* is per-EPISODE agent wall time -- harness reasoning plus every container "
            "command -- measured inside a 4-thread pool on a shared host, so it is a throughput "
            "figure, not a clean per-request latency. verify_mean_s is excluded from it."
        ),
        "power_note": (
            "n_test=45; at a ~30% resolve rate one standard error is ~6.8pp, so only differences "
            "above ~15pp are resolvable. Every comparison is significance-tested against the baseline."
        ),
        "deviation_note": (
            "This is NOT the official Harbor harness. Commands run via `docker exec` with the "
            "working directory carried across calls but no other shell state; the episode wall clock "
            "is capped at episode_timeout_s rather than the task's own 900-1800s. Resolve rates are "
            "therefore not comparable to the public Terminal-Bench leaderboard."
        ),
        "dspy_version": dspy.__version__, "python": platform.python_version(),
    })

    # The baseline is penalty-independent (the same un-optimized harness whatever lambda is), so it
    # runs once and is re-scored at each lambda from its own per-episode records.
    if "baseline" not in data:
        print("\n=== baseline (un-optimized dspy.Flex harness = a single dspy.RLM over the tools) ===")
        base_program = TerminalAgent(harness=new_harness(), episode_timeout_s=args.episode_timeout)
        base_records, base_meta = run_program(base_program, test, threads=args.threads,
                                              penalty=0.0, cache=cache)
        data["baseline"] = {
            "records": [r._asdict() for r in base_records],
            "module_src": base_program.harness.module_src,
            "by_penalty": {f"{p:g}": {**summarize(base_records, p), **base_meta} for p in args.penalties},
        }
        args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")
        first = data["baseline"]["by_penalty"][f"{args.penalties[0]:g}"]
        print("  " + fmt(first))
        if first["errors"]:
            print("  !! baseline had errors: " + str(first["first_error"]))

    runs = data.setdefault("runs", {})
    for penalty in args.penalties:
        key = f"{penalty:g}"
        if args.resume and key in runs:
            print(f"\n=== penalty {key}: cached, skipping ===")
            continue
        print(f"\n=== penalty {key}  (max_metric_calls={args.max_metric_calls}) ===")
        runs[key] = run_penalty(penalty, train, val, test, args.max_metric_calls,
                                exec_lm, reflection_lm, args.threads, args.objective, program_dir_for(args.out),
                                program_dir_for(args.out).parent / f"gepa_log_{key}",
                                cache=cache, episode_timeout_s=args.episode_timeout)
        args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")  # write after every lambda
        r = runs[key]
        print("  " + fmt(r["test"]))
        o = r["optimization"]
        print(f"  optimization: {o['wall_s'] / 60:.0f}min, {o['calls']} LM calls, "
              f"${o['cost_usd_litellm']:.2f}, code_changed={o['changed_code']}")
        print(f"  max completion tokens seen: {o.get('max_completion_tokens_seen', '?')}")
        if o.get("truncated_calls"):
            print(f"  !! {o['truncated_calls']} TRUNCATED call(s) -- raise REFLECTION_MAX_TOKENS; "
                  f"truncated proposals fail to bind and score 0, which looks identical to "
                  f"'the optimizer found nothing'")

    print(f"\nwrote {args.out}")
    print_table(data)
    plot(data, plot_path_for(args.out))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(data: dict) -> None:
    runs = data.get("runs", {})
    keys = sorted(runs, key=float)
    base_by = data.get("baseline", {}).get("by_penalty", {})
    brecs = data.get("baseline", {}).get("records")
    hdr = (f"{'lam':>6} {'resolved':>9} {'rate':>6} {'easy':>5} {'med':>5} {'hard':>5} "
           f"{'calls':>6} {'cmds':>6} {'$/task':>7} {'sec':>6} {'dl':>3} {'score':>6} {'p vs base':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))

    def line(label, r, score, pv):
        d = r.get("by_difficulty", {})
        def rate(k):
            return f"{d[k]['resolve_rate']:.2f}" if k in d else "  -  "
        print(f"{label:>6} {r['n_resolved']:>4}/{r['n']:<4} {r['resolve_rate']:6.3f} "
              f"{rate('easy'):>5} {rate('medium'):>5} {rate('hard'):>5} "
              f"{r['avg_calls']:6.1f} {r['avg_commands']:6.1f} {r['cost_usd_per_task']:7.2f} "
              f"{r['latency_mean_s']:6.0f} {r['deadline_hits']:3d} {score:>6} {pv:>10}")

    if base_by:
        b = base_by[keys[0]] if keys and keys[0] in base_by else next(iter(base_by.values()))
        line("base", b, "-", "-")
    for k in keys:
        pv = mcnemar_p(brecs, runs[k]["records"]) if brecs else float("nan")
        line(k, runs[k]["test"], f"{runs[k]['test']['score']:.3f}", f"{pv:.3f}")
    opt = sum(runs[k]["optimization"]["cost_usd_litellm"] for k in keys)
    ev = sum(runs[k]["test"]["cost_usd_total"] for k in keys)
    base_ev = sum(r.get("cost_usd", 0.0) for r in (brecs or []))
    print(f"\nspend: ${opt:.2f} optimizing + ${ev + base_ev:.2f} evaluating "
          f"= ${opt + ev + base_ev:.2f} across {len(keys)} penalties")
    print("dl = episodes cut off by the wall-clock cap; a nonzero column caps the resolve rate.")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def plot(data: dict, path: Path | None = None) -> None:
    # dspy registers a lazy numpy proxy in sys.modules; matplotlib's `from numpy.exceptions import
    # ...` trips that proxy into a recursive import. Materialize the real numpy first.
    import numpy as np
    _ = np.ndarray
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = data["runs"]
    keys = sorted(runs, key=float)
    lam = [float(k) for k in keys]
    t = [runs[k]["test"] for k in keys]
    acc = [x["resolve_rate"] for x in t]
    cost = [x["cost_usd_per_task"] for x in t]
    lat = [x["latency_mean_s"] for x in t]
    calls = [x["avg_calls"] for x in t]
    ci = [wilson(x["n_resolved"], x["n"]) for x in t]
    # Wilson intervals are not centred on the point estimate, so at a resolve rate of 0 or 1 an
    # endpoint lands on the wrong side of it and matplotlib refuses the negative error bar.
    acc_err = [[max(0.0, a - lo) for a, (lo, _) in zip(acc, ci, strict=True)],
               [max(0.0, hi - a) for a, (_, hi) in zip(acc, ci, strict=True)]]

    base_by = data.get("baseline", {}).get("by_penalty", {})
    b = base_by.get(keys[0]) if keys else None
    brecs = data.get("baseline", {}).get("records")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    fig.patch.set_facecolor(SURFACE)
    (ax_cost, ax_lat), (ax_calls, ax_sig) = axes

    def frontier(ax, xs, xlabel, title, base_x):
        # Joined in lambda order, not sorted by x: lambda is the independent variable, and sorting by
        # cost would draw a smooth monotone "frontier" the data may not have.
        ax.errorbar(xs, acc, yerr=acc_err, color=C_OPT, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=1.2, capsize=4,
                    ecolor=C_OPT, zorder=3, label="GEPA-evolved harness (95% CI)")
        # Penalized points cluster in the cheap corner at near-identical resolve rates, so a fixed
        # offset overprints their labels into unreadable mush. Stagger by index.
        offsets = [(9, -3), (9, 9), (9, -14), (-9, 9), (-9, -14)]
        for i in range(len(xs)):
            dx, dy = offsets[i % len(offsets)]
            ax.annotate(f"λ={lam[i]:g}", (xs[i], acc[i]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8.5, color=INK_2, zorder=4,
                        ha=("left" if dx > 0 else "right"))
        if b is not None:
            ax.plot([base_x], [b["resolve_rate"]], color=C_BASE, marker="s", markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none",
                    zorder=3, label="baseline harness (un-optimized dspy.Flex)")
            ax.annotate("baseline", (base_x, b["resolve_rate"]), textcoords="offset points",
                        xytext=(0, -17), fontsize=8.5, color=INK_2, ha="center", zorder=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("resolve rate")
        ax.set_title(title, fontsize=11, loc="left", pad=10)
        # "best", not a fixed corner: where the points land on these two panels depends entirely on
        # what the sweep measured, and a fixed corner overprinted the baseline marker.
        ax.legend(loc="best", fontsize=8.5, frameon=False, labelcolor=INK_2)
        _style(ax)

    frontier(ax_cost, cost, "inference cost  (USD per task attempted)",
             "Cost → resolve-rate frontier", b["cost_usd_per_task"] if b else 0)
    frontier(ax_lat, lat, "mean agent wall clock per task  (s; 4-way concurrency)",
             "Latency → resolve-rate frontier", b["latency_mean_s"] if b else 0)

    ax_calls.plot(lam, calls, color=C_OPT, linewidth=2, marker="o", markersize=8,
                  markeredgecolor=SURFACE, markeredgewidth=2, zorder=3, label="GEPA-evolved harness")
    if b is not None:
        ax_calls.axhline(b["avg_calls"], color=C_BASE, linewidth=2, linestyle=(0, (5, 3)),
                         zorder=2, label="baseline harness")
    step_budget = data.get("meta", {}).get("step_budget", 30)
    ax_calls.axhline(step_budget, color=INK_2, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
    # Anchored right: the left end of this line is where the lambda=0 point and its label live.
    ax_calls.annotate("step budget (the penalty's unit of account)", (lam[-1], step_budget),
                      textcoords="offset points", xytext=(-2, 4), ha="right", fontsize=7.5, color=INK_2)
    for x, y in zip(lam, calls, strict=True):
        ax_calls.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 10),
                          ha="center", fontsize=8.5, color=INK_2, zorder=4)
    ax_calls.set_xlabel("LLM-call penalty  λ")
    ax_calls.set_ylabel("avg LLM calls per task")
    ax_calls.set_title("What the penalty buys: shell work instead of model steps",
                       fontsize=11, loc="left", pad=10)
    ax_calls.set_ylim(-0.5, max([*calls, b["avg_calls"] if b else 0, step_budget]) * 1.2 + 0.5)
    ax_calls.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=INK_2)
    _style(ax_calls)

    # Which penalties actually beat the baseline? Points whose CI clears the baseline band do.
    ax_sig.errorbar(lam, acc, yerr=acc_err, color=C_OPT, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=1.2, capsize=4,
                    ecolor=C_OPT, zorder=3, label="GEPA-evolved harness (95% CI)")
    if b is not None:
        blo, bhi = wilson(b["n_resolved"], b["n"])
        ax_sig.axhspan(blo, bhi, color=C_BASE, alpha=0.13, zorder=0)
        ax_sig.axhline(b["resolve_rate"], color=C_BASE, linewidth=2, linestyle=(0, (5, 3)), zorder=2,
                       label="baseline harness (95% CI band)")
        for x, k, (_, hi_i) in zip(lam, keys, ci, strict=True):
            pv = mcnemar_p(brecs, runs[k]["records"])
            ax_sig.annotate(f"p={pv:.3f}" if pv >= 0.001 else "p<0.001", (x, hi_i),
                            textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8,
                            color=(C_OPT if pv < 0.05 else INK_2), zorder=4,
                            fontweight=("bold" if pv < 0.05 else "normal"))
    ax_sig.set_xlabel("LLM-call penalty  λ")
    ax_sig.set_ylabel("resolve rate")
    lo = min([*[c[0] for c in ci], b["resolve_rate"] if b else 1.0])
    hi = max([*[c[1] for c in ci], b["resolve_rate"] if b else 0.0])
    pad = max(0.012, (hi - lo) * 0.16)
    ax_sig.set_ylim(max(-0.02, lo - pad * 2.6), min(1.02, hi + pad * 1.7))  # truncated axis
    ax_sig.set_title(
        "Which penalties actually beat the baseline harness?\n"
        "p = McNemar exact test vs baseline, paired on the same tasks.\n"
        "bold p<0.05 = real gain · plain = parity (CI overlaps shaded band)",
        fontsize=9.5, loc="left", pad=8)
    ax_sig.legend(loc="lower left", fontsize=8.5, frameon=False, labelcolor=INK_2)
    _style(ax_sig)

    meta = data.get("meta", {})
    fig.suptitle(
        f"Terminal-Bench 2.0: CAL frontier as GEPA evolves the agent harness  "
        f"(exec {meta.get('exec_model', '?').split('/')[-1]}, "
        f"reflection {meta.get('reflection_model', '?').split('/')[-1]}, "
        f"n_test={meta.get('n_test', '?')})",
        fontsize=13, color=INK, x=0.01, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = path or PLOT_PATH
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved plot -> {out_png}")


if __name__ == "__main__":
    main()
