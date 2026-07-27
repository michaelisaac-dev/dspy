"""Sweep the LLM-call penalty on committee attribution and trace the CAL frontier.

Same design as `tests/demos/conflation/all/sweep_penalties.py`, adapted for string extraction:
for each penalty λ, GEPA optimizes `max(0, correct − λ·n_llm_calls)`, then the resulting program is
evaluated on a held-out test set with accuracy / macro-P/R/F1 alongside cost, latency and call rate.

Everything lands in `penalty_sweep.json` — including per-example records — so any other metric can
be recomputed later without re-running the sweep.

    python sweep_penalties.py                        # full sweep
    python sweep_penalties.py --resume               # skip penalties already in the JSON
    python sweep_penalties.py --penalties 0 --max-metric-calls 600
    python sweep_penalties.py --plot-only            # re-render the figure from the JSON
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

sys.path.insert(0, str(Path(__file__).parent))
from emails_common import (
    EXEC_MODEL,
    REFLECTION_MODEL,
    IdentifyCommittee,
    disable_cache,
    fmt,
    load_splits,
    make_lms,
    make_metric,
    meter,
    run_program,
    summarize,
)

DEMO_DIR = Path(__file__).parent
SWEEP_PATH = DEMO_DIR / "penalty_sweep.json"
PLOT_PATH = DEMO_DIR / "cal_frontier.png"
PROGRAM_DIR = DEMO_DIR / "sweep_programs"

DEFAULT_PENALTIES = [0.0, 0.05, 0.1, 0.2, 0.4]
# 600 was wasteful on this dataset: GEPA hit a perfect val score (40/40) at iteration 2 and
# then spent ~430 further rollouts proposing candidates it could not rank. With 148 usable
# training rows and a ~90% baseline, a 40-row val slice saturates after ~4 corrections.
DEFAULT_MAX_METRIC_CALLS = 200
REFLECTION_MINIBATCH = 4
EVAL_THREADS = 8

# From the dataviz reference palette, light mode; validated all-pairs (worst ΔE 24.7 protan).
C_OPT, C_BASE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c3"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval — the honest error bar on an accuracy from n examples."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_p(ra: list[dict], rb: list[dict]) -> float:
    """Two-sided exact McNemar p for two programs scored on the same examples (paired)."""
    n01 = sum(1 for x, y in zip(ra, rb) if x["correct"] and not y["correct"])
    n10 = sum(1 for x, y in zip(ra, rb) if not x["correct"] and y["correct"])
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
                program_dir: Path = PROGRAM_DIR, log_dir: Path | None = None):
    """Optimize at one penalty and evaluate the result. Returns a JSON-ready dict."""
    program = dspy.Flex(IdentifyCommittee)
    baseline_src = program.module_src

    started = time.perf_counter()
    with meter(exec_lm, reflection_lm) as opt_cost:
        optimized = dspy.GEPA(
            metric=make_metric(penalty),
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            reflection_minibatch_size=REFLECTION_MINIBATCH,
            num_threads=threads,
            seed=0,
            # Checkpoint every candidate: compile() only returns (and only then does the program get
            # saved) at the very end, so killing a long run previously threw away everything it had
            # found. With log_dir, a stopped run leaves its candidates on disk.
            log_dir=str(log_dir) if log_dir else None,
            # At λ=0 the score IS accuracy, the baseline is ~92%, and a 4-email minibatch is
            # all-correct ~72% of the time — so the default (True) skipped 32 of 43 iterations and
            # GEPA never optimized anything. Only affects λ=0: at λ>0 an LLM call costs λ, so no
            # minibatch containing a call can score perfectly.
            skip_perfect_score=False,
        ).compile(program, trainset=train, valset=val)
    opt_wall_s = time.perf_counter() - started

    records, meta = run_program(optimized, test, threads=threads)
    row = summarize(records, penalty)
    row.update(meta)

    program_dir.mkdir(parents=True, exist_ok=True)
    program_path = program_dir / f"flex_penalty_{penalty:g}.json"
    optimized.save(str(program_path))

    return {
        "penalty": penalty,
        "max_metric_calls": max_metric_calls,
        "optimization": {"wall_s": opt_wall_s, **opt_cost,
                         "changed_code": optimized.module_src != baseline_src},
        "test": row,
        "records": [r._asdict() for r in records],
        "module_src": optimized.module_src,
        "program_path": _relpath(program_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--penalties", type=float, nargs="+", default=DEFAULT_PENALTIES)
    ap.add_argument("--max-metric-calls", type=int, default=DEFAULT_MAX_METRIC_CALLS)
    ap.add_argument("--threads", type=int, default=EVAL_THREADS)
    ap.add_argument("--out", type=Path, default=SWEEP_PATH)
    ap.add_argument("--resume", action="store_true", help="skip penalties already present in --out")
    ap.add_argument("--plot-only", action="store_true", help="re-render the figure from --out")
    args = ap.parse_args()
    # Resolve immediately: plot_path_for / program_dir_for derive paths from this, and a
    # relative --out made program_path.relative_to(DEMO_DIR) raise after a full GEPA compile
    # had already run — losing the run's metrics to a bookkeeping line.
    args.out = args.out.resolve()

    if args.plot_only:
        plot(json.loads(args.out.read_text(encoding="utf-8")), plot_path_for(args.out))
        return

    disable_cache()  # so latency and cost are what a cold production call would cost
    exec_lm, reflection_lm = make_lms()
    dspy.configure(lm=exec_lm)
    train, val, test = load_splits()
    print(f"splits: gepa_train={len(train)} gepa_val={len(val)} test={len(test)}")

    data = json.loads(args.out.read_text(encoding="utf-8")) if (args.resume and args.out.exists()) else {}
    data.setdefault("meta", {}).update({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exec_model": EXEC_MODEL, "reflection_model": REFLECTION_MODEL,
        "max_metric_calls": args.max_metric_calls, "reflection_minibatch": REFLECTION_MINIBATCH,
        "skip_perfect_score": False, "reflection_max_tokens": 24000,
        "eval_threads": args.threads,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "test_composition": "val.jsonl (50, held out by the original demo) + 50 rows held out of train.jsonl",
        "cache": "disabled (dspy disk + memory)",
        "latency_note": (
            "latency_* are PER-REQUEST wall times measured inside an 8-thread pool. Throughput per "
            "example is wall_s/n, which is several times lower. Do not read latency_mean_s as "
            "pipeline cost."
        ),
        "power_note": (
            "n_test=100; at ~95% accuracy one standard error is ~2.2pp, so differences under ~5pp "
            "are not resolvable. Every comparison is significance-tested against the baseline."
        ),
        "dspy_version": dspy.__version__, "python": platform.python_version(),
    })

    # The baseline is penalty-independent (one Predict call per email whatever λ is), so it runs
    # once and is re-scored at each λ from its own per-example records.
    if "baseline" not in data:
        print("\n=== baseline (un-optimized dspy.Flex) ===")
        base_records, base_meta = run_program(dspy.Flex(IdentifyCommittee), test, threads=args.threads)
        data["baseline"] = {
            "records": [r._asdict() for r in base_records],
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
                                exec_lm, reflection_lm, args.threads, program_dir_for(args.out),
                                program_dir_for(args.out).parent / f"gepa_log_{key}")
        args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")  # write after every λ
        r = runs[key]
        print("  " + fmt(r["test"]))
        o = r["optimization"]
        print(f"  optimization: {o['wall_s']:.0f}s, {o['calls']} LM calls, "
              f"${o['cost_usd_litellm']:.2f}, code_changed={o['changed_code']}")
        print(f"  max completion tokens seen: {o.get('max_completion_tokens_seen', '?')}")
        if o.get("truncated_calls"):
            print(f"  !! {o['truncated_calls']} TRUNCATED call(s) — raise REFLECTION_MAX_TOKENS; "
                  f"truncated proposals are unparseable and score 0, which looks identical to "
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
    hdr = (f"{'λ':>6} {'acc':>6} {'acc@val':>8} {'mP':>6} {'mR':>6} {'mF1':>6} {'calls/ex':>9} "
           f"{'$/1k':>7} {'req ms':>7} {'ms/ex':>7} {'score':>6} {'p vs base':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))

    def line(label, r, score, pv):
        av = r["by_source"].get("val_jsonl", {}).get("accuracy", float("nan"))
        print(f"{label:>6} {r['accuracy']:6.3f} {av:8.3f} {r['macro_precision']:6.3f} "
              f"{r['macro_recall']:6.3f} {r['macro_f1']:6.3f} {r['avg_calls']:9.3f} "
              f"{r['cost_usd_per_1k_examples']:7.2f} {r['latency_mean_s']*1000:7.0f} "
              f"{r['wall_s']/r['n']*1000:7.0f} {score:>6} {pv:>10}")

    if base_by:
        b = base_by[keys[0]] if keys and keys[0] in base_by else next(iter(base_by.values()))
        line("base", b, "—", "—")
    for k in keys:
        pv = mcnemar_p(brecs, runs[k]["records"]) if brecs else float("nan")
        line(k, runs[k]["test"], f"{runs[k]['test']['score']:.3f}", f"{pv:.3f}")
    total = sum(runs[k]["optimization"]["cost_usd_litellm"] for k in keys)
    print(f"\ntotal GEPA optimization spend: ${total:.2f} across {len(keys)} runs")


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
    acc = [x["accuracy"] for x in t]
    cost = [x["cost_usd_per_1k_examples"] for x in t]
    lat = [x["latency_mean_s"] * 1000 for x in t]
    calls = [x["avg_calls"] for x in t]
    ci = [wilson(round(x["accuracy"] * x["n"]), x["n"]) for x in t]
    # Wilson intervals are not centred on the point estimate: at accuracy 1.0 the upper bound lands
    # marginally BELOW 1.0, which yields a negative error bar and matplotlib refuses to draw it.
    # Clamp the interval to contain the point estimate.
    acc_err = [[max(0.0, a - lo) for a, (lo, _) in zip(acc, ci, strict=True)],
               [max(0.0, hi - a) for a, (_, hi) in zip(acc, ci, strict=True)]]

    base_by = data.get("baseline", {}).get("by_penalty", {})
    b = base_by.get(keys[0]) if keys else None
    brecs = data.get("baseline", {}).get("records")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    fig.patch.set_facecolor(SURFACE)
    (ax_cost, ax_lat), (ax_calls, ax_sig) = axes

    def frontier(ax, xs, xlabel, title, base_x):
        # Joined in λ order, not sorted by x: λ is the independent variable, and sorting by cost
        # would draw a smooth monotone "frontier" the data may not have.
        ax.errorbar(xs, acc, yerr=acc_err, color=C_OPT, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=1.2, capsize=4,
                    ecolor=C_OPT, zorder=3, label="GEPA-optimized (95% CI)")
        # Penalized points cluster in the cheap corner at near-identical accuracy, so a fixed
        # offset overprints their labels into unreadable mush. Stagger by index.
        offsets = [(9, -3), (9, 9), (9, -14), (-9, 9), (-9, -14)]
        for i in range(len(xs)):
            dx, dy = offsets[i % len(offsets)]
            ax.annotate(f"λ={lam[i]:g}", (xs[i], acc[i]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8.5, color=INK_2, zorder=4,
                        ha=("left" if dx > 0 else "right"))
        if b is not None:
            ax.plot([base_x], [b["accuracy"]], color=C_BASE, marker="s", markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none",
                    zorder=3, label="baseline (un-optimized)")
            ax.annotate("baseline", (base_x, b["accuracy"]), textcoords="offset points",
                        xytext=(0, -17), fontsize=8.5, color=INK_2, ha="center", zorder=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("accuracy")
        ax.set_title(title, fontsize=11, loc="left", pad=10)
        ax.legend(loc="lower right", fontsize=8.5, frameon=False, labelcolor=INK_2)
        _style(ax)

    frontier(ax_cost, cost, "inference cost  (USD per 1,000 emails)",
             "Cost → accuracy frontier", b["cost_usd_per_1k_examples"] if b else 0)
    frontier(ax_lat, lat, "mean per-request latency  (ms; 8-way concurrency)",
             "Latency → accuracy frontier", b["latency_mean_s"] * 1000 if b else 0)

    ax_calls.plot(lam, calls, color=C_OPT, linewidth=2, marker="o", markersize=8,
                  markeredgecolor=SURFACE, markeredgewidth=2, zorder=3, label="GEPA-optimized")
    if b is not None:
        ax_calls.axhline(b["avg_calls"], color=C_BASE, linewidth=2, linestyle=(0, (5, 3)),
                         zorder=2, label="baseline (un-optimized)")
    for x, y in zip(lam, calls, strict=True):
        ax_calls.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10),
                          ha="center", fontsize=8.5, color=INK_2, zorder=4)
    ax_calls.set_xlabel("LLM-call penalty  λ")
    ax_calls.set_ylabel("avg LLM calls per email")
    ax_calls.set_title("What the penalty buys: decomposition into Python", fontsize=11, loc="left", pad=10)
    ax_calls.set_ylim(-0.08, max([*calls, b["avg_calls"] if b else 0]) * 1.25 + 0.05)
    ax_calls.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=INK_2)
    _style(ax_calls)

    # Which penalties actually beat the baseline? Points whose CI clears the baseline band do.
    ax_sig.errorbar(lam, acc, yerr=acc_err, color=C_OPT, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=1.2, capsize=4,
                    ecolor=C_OPT, zorder=3, label="GEPA-optimized (95% CI)")
    if b is not None:
        blo, bhi = wilson(round(b["accuracy"] * b["n"]), b["n"])
        ax_sig.axhspan(blo, bhi, color=C_BASE, alpha=0.13, zorder=0)
        ax_sig.axhline(b["accuracy"], color=C_BASE, linewidth=2, linestyle=(0, (5, 3)), zorder=2,
                       label="baseline, always call the LLM (95% CI band)")
        for x, k, (_, hi_i) in zip(lam, keys, ci, strict=True):
            pv = mcnemar_p(brecs, runs[k]["records"])
            ax_sig.annotate(f"p={pv:.3f}" if pv >= 0.001 else "p<0.001", (x, hi_i),
                            textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8,
                            color=(C_OPT if pv < 0.05 else INK_2), zorder=4,
                            fontweight=("bold" if pv < 0.05 else "normal"))
    ax_sig.set_xlabel("LLM-call penalty  λ")
    ax_sig.set_ylabel("accuracy")
    lo = min([*[c[0] for c in ci], b["accuracy"] if b else 1.0])
    hi = max([*[c[1] for c in ci], b["accuracy"] if b else 0.0])
    pad = max(0.012, (hi - lo) * 0.16)
    ax_sig.set_ylim(lo - pad * 2.6, min(1.02, hi + pad * 1.7))  # truncated axis
    ax_sig.set_title(
        "Which penalties actually beat the baseline on accuracy?\n"
        "p = McNemar exact test vs baseline, paired on the same emails.\n"
        "bold p<0.05 = real gain · plain = parity (CI overlaps shaded band)",
        fontsize=9.5, loc="left", pad=8)
    ax_sig.legend(loc="lower left", fontsize=8.5, frameon=False, labelcolor=INK_2)
    _style(ax_sig)

    meta = data.get("meta", {})
    fig.suptitle(
        f"Political fundraising emails: CAL frontier under an LLM-call penalty  "
        f"(exec {meta.get('exec_model','?').split('/')[-1]}, "
        f"reflection {meta.get('reflection_model','?').split('/')[-1]}, "
        f"n_test={meta.get('n_test','?')})",
        fontsize=13, color=INK, x=0.01, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = path or PLOT_PATH
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved plot -> {out_png}")


if __name__ == "__main__":
    main()
