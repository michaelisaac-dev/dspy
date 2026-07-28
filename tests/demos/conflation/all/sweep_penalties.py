"""Sweep the LLM-call penalty and trace how the CAL frontier (cost, accuracy, latency) moves.

For each penalty λ we run GEPA against `max(0, correct − λ·n_llm_calls)`, then evaluate the
resulting program on a held-out balanced test split, recording accuracy / precision / recall / F1
alongside the three CAL axes. λ=0 is plain accuracy (LLM calls are free); as λ rises each call has
to buy back more accuracy than it costs, and GEPA is pushed to decide cases in Python instead.

Everything lands in `penalty_sweep.json` — including per-example records — so any other metric can
be recomputed later without re-running the sweep.

    python sweep_penalties.py                        # full sweep
    python sweep_penalties.py --resume               # skip penalties already in the JSON
    python sweep_penalties.py --penalties 0.2 --max-metric-calls 900
    python sweep_penalties.py --plot-only            # re-render the figure from the JSON
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).parent))
from conflation_common import (
    EXEC_MODEL,
    REFLECTION_MODEL,
    SamePlace,
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
DEFAULT_MAX_METRIC_CALLS = 400
EVAL_THREADS = 8

# From references/palette.md, light mode. Validated for all-pairs CVD separation with
# `validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all` — all checks pass
# (worst pair ΔE 9.2 deutan, normal-vision 24.0). Aqua carries a contrast WARN (2.74:1 on this
# surface), so the relief rule applies and its point is always directly labelled, never
# identified by colour alone.
C_OPT = "#2a78d6"      # categorical slot 1 — the GEPA-optimized Flex programs
C_BASE = "#eb6834"     # categorical slot 2 — the un-optimized baseline
C_PLAIN = "#1baf7a"    # categorical slot 3 — plain GEPA (dspy.Predict, instruction-only)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
AXIS = "#c9c8c3"


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_penalty(penalty, train, val, test, max_metric_calls, exec_lm, reflection_lm, threads,
                program_dir: Path = PROGRAM_DIR):
    """Optimize at one penalty and evaluate the result. Returns a JSON-ready dict."""
    program = dspy.Flex(SamePlace)
    baseline_src = program.module_src

    started = time.perf_counter()
    with meter(exec_lm, reflection_lm) as opt_cost:
        optimized = dspy.GEPA(
            metric=make_metric(penalty),
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            reflection_minibatch_size=3,
            num_threads=threads,
            seed=0,
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
        "optimization": {
            "wall_s": opt_wall_s,
            **opt_cost,
            "changed_code": optimized.module_src != baseline_src,
        },
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
    print(f"splits: train={len(train)} val={len(val)} test={len(test)} (class-balanced)")

    data = json.loads(args.out.read_text(encoding="utf-8")) if (args.resume and args.out.exists()) else {}
    data.setdefault("meta", {}).update({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exec_model": EXEC_MODEL,
        "reflection_model": REFLECTION_MODEL,
        "max_metric_calls": args.max_metric_calls,
        "eval_threads": args.threads,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "test_split": "class-balanced subset",
        "cache": "disabled (dspy disk + memory)",
        "latency_note": (
            f"per-example wall time measured inside a {args.threads}-thread pool; "
            "concurrency inflates absolute values slightly but consistently across rows"
        ),
        "dspy_version": dspy.__version__,
        "python": platform.python_version(),
    })

    # The baseline is penalty-independent (one Predict call per example whatever λ is), so it runs
    # once and gets re-scored at each λ from its own per-example records.
    if "baseline" not in data:
        print("\n=== baseline (un-optimized dspy.Flex) ===")
        base_records, base_meta = run_program(dspy.Flex(SamePlace), test, threads=args.threads)
        data["baseline"] = {
            "records": [r._asdict() for r in base_records],
            "by_penalty": {f"{p:g}": {**summarize(base_records, p), **base_meta} for p in args.penalties},
        }
        args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print("  " + fmt(data["baseline"]["by_penalty"][f"{args.penalties[0]:g}"]))
        if data["baseline"]["by_penalty"][f"{args.penalties[0]:g}"]["errors"]:
            print("  !! baseline had errors: "
                  + str(data["baseline"]["by_penalty"][f"{args.penalties[0]:g}"]["first_error"]))

    runs = data.setdefault("runs", {})
    for penalty in args.penalties:
        key = f"{penalty:g}"
        if args.resume and key in runs:
            print(f"\n=== penalty {key}: cached, skipping ===")
            continue
        print(f"\n=== penalty {key}  (max_metric_calls={args.max_metric_calls}) ===")
        runs[key] = run_penalty(penalty, train, val, test, args.max_metric_calls,
                                exec_lm, reflection_lm, args.threads, program_dir_for(args.out))
        # Write after every penalty so a mid-sweep failure doesn't throw away the spend.
        args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")
        r = runs[key]
        print("  " + fmt(r["test"]))
        print(f"  optimization: {r['optimization']['wall_s']:.0f}s, "
              f"{r['optimization']['calls']} LM calls, "
              f"${r['optimization']['cost_usd_litellm']:.2f}")

    print(f"\nwrote {args.out}")
    print_table(data)
    plot(data, plot_path_for(args.out))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(data: dict) -> None:
    runs = data.get("runs", {})
    keys = sorted(runs, key=float)
    hdr = (f"{'λ':>6} {'acc':>6} {'acc@pool':>9} {'prec':>6} {'rec':>6} {'spec':>6} {'F1':>6} "
           f"{'calls/ex':>9} {'$/1k':>7} {'$/1k@pool':>10} {'req ms':>7} {'ms/ex':>7} {'score':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))

    def line(label, r, score):
        print(f"{label:>6} {r['accuracy']:6.3f} {r['accuracy_at_pool_prevalence']:9.3f} "
              f"{r['precision']:6.3f} {r['recall']:6.3f} {r['specificity']:6.3f} {r['f1']:6.3f} "
              f"{r['avg_calls']:9.3f} {r['cost_usd_per_1k_examples']:7.2f} "
              f"{r['cost_usd_per_1k_at_pool_prevalence']:10.2f} "
              f"{r['latency_mean_s']*1000:7.0f} {r['wall_s']/r['n']*1000:7.0f} {score:>6}")

    base = data.get("baseline", {}).get("by_penalty", {})
    if base:
        b = base[keys[0]] if keys and keys[0] in base else next(iter(base.values()))
        line("base", b, "—")
    for k in keys:
        line(k, runs[k]["test"], f"{runs[k]['test']['score']:.3f}")
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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a proportion — the honest error bar on an accuracy from n examples.

    Without this the figure reads as a trend across λ, but λ=0.1/0.2/0.4 differ by 4–8 examples out
    of 240 (McNemar exact p = 0.79 / 0.55 / 1.00). Only λ=0.05 → λ=0.1 is a real step (p = 0.049).
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfwidth = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


def mcnemar_p(ra: list[dict], rb: list[dict]) -> float:
    """Two-sided exact McNemar p for two programs scored on the same examples (paired)."""
    import math
    n01=sum(1 for x,y in zip(ra,rb) if (x["pred"]==x["gold"]) and not (y["pred"]==y["gold"]))
    n10=sum(1 for x,y in zip(ra,rb) if not (x["pred"]==x["gold"]) and (y["pred"]==y["gold"]))
    n=n01+n10
    if n==0:
        return 1.0
    return min(1.0, sum(math.comb(n,i) for i in range(0,min(n01,n10)+1))/2**n*2)


def plot_path_for(out: Path) -> Path:
    """Keep each --out's figure beside it, so a side run can't clobber the main sweep's plot."""
    return PLOT_PATH if out.resolve() == SWEEP_PATH.resolve() else out.with_suffix(".png")


def _relpath(p: Path) -> str:
    """Path relative to the demo dir when possible, else absolute. Never raises."""
    import os
    try:
        return str(Path(os.path.relpath(p, DEMO_DIR)))
    except ValueError:
        return str(p)


def program_dir_for(out: Path) -> Path:
    """Same isolation for saved programs: a side run must not overwrite the sweep's artifacts.

    (It did once — a --out=highbudget run at λ=0.2 clobbered sweep_programs/flex_penalty_0.2.json,
    because this path used to be a module-level constant.)
    """
    return PROGRAM_DIR if out.resolve() == SWEEP_PATH.resolve() else out.with_suffix("") / "programs"


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
    f1 = [x["f1"] for x in t]
    cost = [x["cost_usd_per_1k_examples"] for x in t]
    # Mean, not p50: once most examples are decided in Python the median collapses to ~4ms and
    # hides the LLM-call tail entirely. Mean is what actually governs throughput. p50/p95 are in
    # the JSON and the console table.
    lat = [x["latency_mean_s"] * 1000 for x in t]
    calls = [x["avg_calls"] for x in t]
    # Wilson 95% CIs, as asymmetric +/- offsets for errorbar().
    ci = [wilson(round(x["accuracy"] * x["n"]), x["n"]) for x in t]
    # Wilson intervals are not centred on the point estimate: at accuracy 1.0 the upper bound lands
    # marginally BELOW 1.0, which yields a negative error bar and matplotlib refuses to draw it.
    # Clamp the interval to contain the point estimate.
    acc_err = [[max(0.0, a - lo) for a, (lo, _) in zip(acc, ci, strict=True)],
               [max(0.0, hi - a) for a, (_, hi) in zip(acc, ci, strict=True)]]

    base_by = data.get("baseline", {}).get("by_penalty", {})
    b = base_by.get(keys[0]) if keys else None

    # Plain GEPA (dspy.Predict, instruction-only) is optional — present once `run_plain_gepa.py` has
    # run. It is a single POINT, not a curve: a Predict makes exactly one call per example whatever
    # λ is, so the penalty cannot reorder its candidates and there is no trajectory to draw.
    plain = data.get("plain_gepa")
    p_test = plain["test"] if plain else None

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    fig.patch.set_facecolor(SURFACE)
    (ax_cost, ax_lat), (ax_calls, ax_qual) = axes

    def frontier(ax, xs, xlabel, title, base_x, plain_x=None):
        # Connect in λ order, not sorted-by-x: λ is the independent variable, and sorting by cost
        # would draw a smooth monotone "frontier" the data does not actually have (λ=0.1 and λ=0.2
        # invert). The line is the trajectory as the penalty rises.
        ax.errorbar(xs, acc, yerr=acc_err, color=C_OPT, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=1.2, capsize=4,
                    ecolor=C_OPT, zorder=3, label="GEPA-optimized (95% CI)")
        for i in range(len(xs)):
            ax.annotate(f"λ={lam[i]:g}", (xs[i], acc[i]), textcoords="offset points",
                        xytext=(9, -3), fontsize=8.5, color=INK_2, zorder=4)
        if b is not None:
            ax.plot([base_x], [b["accuracy"]], color=C_BASE, marker="s", markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none",
                    zorder=3, label="baseline (un-optimized)")
            # Below the marker: the legend occupies the mid-right band, so a label above or to the
            # left of the baseline point collides with it.
            ax.annotate("baseline", (base_x, b["accuracy"]), textcoords="offset points",
                        xytext=(0, -17), fontsize=8.5, color=INK_2, ha="center", zorder=4)
        if p_test is not None and plain_x is not None:
            ax.plot([plain_x], [p_test["accuracy"]], color=C_PLAIN, marker="D", markersize=8.5,
                    markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", zorder=3,
                    label="plain GEPA (no Flex, prompt-only)")
            # Direct label, not colour alone: aqua carries a contrast WARN on this surface.
            ax.annotate("plain GEPA", (plain_x, p_test["accuracy"]), textcoords="offset points",
                        xytext=(0, 12), fontsize=8.5, color=INK_2, ha="center", zorder=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("accuracy")
        ax.set_title(title, fontsize=11, loc="left", pad=10)
        # Lower-left collides with the λ labels on the cheap end, lower-right with the baseline
        # marker; the mid-right band under the rising line is the one clear region in both panels.
        ax.legend(loc="center right", fontsize=8.5, frameon=False, labelcolor=INK_2)
        _style(ax)

    frontier(ax_cost, cost, "inference cost  (USD per 1,000 examples)",
             "Cost → accuracy frontier", b["cost_usd_per_1k_examples"] if b else 0,
             p_test["cost_usd_per_1k_examples"] if p_test else None)
    frontier(ax_lat, lat, "mean per-request latency  (ms; 8-way concurrency)",
             "Latency → accuracy frontier", b["latency_mean_s"] * 1000 if b else 0,
             p_test["latency_mean_s"] * 1000 if p_test else None)

    ax_calls.plot(lam, calls, color=C_OPT, linewidth=2, marker="o", markersize=8,
                  markeredgecolor=SURFACE, markeredgewidth=2, zorder=3, label="GEPA-optimized")
    if b is not None:
        ax_calls.axhline(b["avg_calls"], color=C_BASE, linewidth=2, linestyle=(0, (5, 3)),
                         zorder=2, label="baseline (un-optimized)")
    for x, y in zip(lam, calls, strict=True):
        ax_calls.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10),
                          ha="center", fontsize=8.5, color=INK_2, zorder=4)
    if p_test is not None:
        # No curve across λ: for a plain Predict the call count is structurally 1, so λ has nothing
        # to act on. Drawn as a flat line spanning the axis, which is literally what it is — and it
        # lands on top of the baseline's line, which is the point worth seeing.
        ax_calls.axhline(p_test["avg_calls"], color=C_PLAIN, linewidth=2, linestyle=(0, (1, 2)),
                         zorder=3, label="plain GEPA (no Flex) — fixed at 1.00, λ cannot move it")
    ax_calls.set_xlabel("LLM-call penalty  λ")
    ax_calls.set_ylabel("avg LLM calls per example")
    ax_calls.set_title("What the penalty buys: decomposition into Python", fontsize=11, loc="left", pad=10)
    ax_calls.set_ylim(-0.08, max([*calls, b["avg_calls"] if b else 0]) * 1.25 + 0.05)
    ax_calls.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=INK_2)
    _style(ax_calls)

    # Panel 4 answers the one question the frontier panels cannot: which penalties actually beat
    # the baseline on accuracy? Points whose CI clears the baseline band do; the rest are at parity.
    # (This panel previously plotted accuracy against a re-weighting of itself, which illustrated a
    # ranking difference that later turned out to be noise. Not a question worth a panel.)
    ax_qual.errorbar(lam, acc, yerr=acc_err, color=C_OPT, linewidth=2, marker="o", markersize=8,
                     markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=1.2, capsize=4,
                     ecolor=C_OPT, zorder=3, label="GEPA-optimized (95% CI)")
    if b is not None:
        blo, bhi = wilson(round(b["accuracy"] * b["n"]), b["n"])
        ax_qual.axhspan(blo, bhi, color=C_BASE, alpha=0.13, zorder=0)
        ax_qual.axhline(b["accuracy"], color=C_BASE, linewidth=2, linestyle=(0, (5, 3)), zorder=2,
                        label="baseline, always call the LLM (95% CI band)")
        brecs = data["baseline"]["records"]
        # Show the actual p on every point — "n.s." alone discards the information and forces the
        # reader out to the prose to decode the panel. Anchor to the top of the interval, not the
        # point, or the label lands on the whisker.
        for x, k, (_, hi_i) in zip(lam, keys, ci, strict=True):
            pv = mcnemar_p(brecs, runs[k]["records"])
            ax_qual.annotate(f"p={pv:.3f}" if pv >= 0.001 else "p<0.001", (x, hi_i),
                             textcoords="offset points", xytext=(0, 6), ha="center",
                             fontsize=8, color=(C_OPT if pv < 0.05 else INK_2), zorder=4,
                             fontweight=("bold" if pv < 0.05 else "normal"))
    if plain is not None:
        pv_plain = mcnemar_p(data["baseline"]["records"], plain["records"])
        ax_qual.axhline(plain["test"]["accuracy"], color=C_PLAIN, linewidth=2, linestyle=(0, (1, 2)),
                        zorder=3,
                        label=f"plain GEPA (no Flex), p={pv_plain:.3f} vs baseline")
    ax_qual.set_xlabel("LLM-call penalty  λ")
    ax_qual.set_ylabel("accuracy")
    plain_acc = [p_test["accuracy"]] if p_test else []
    lo = min([*[c[0] for c in ci], *plain_acc, b["accuracy"] if b else 1.0]) if b else min(c[0] for c in ci)
    hi = max([*[c[1] for c in ci], *plain_acc, b["accuracy"] if b else 0.0]) if b else max(c[1] for c in ci)
    pad = max(0.012, (hi - lo) * 0.16)
    # Extra headroom below clears a strip for the legend under the lowest interval.
    ax_qual.set_ylim(lo - pad * 2.6, hi + pad * 1.7)  # truncated axis: y starts well above 0
    ax_qual.set_title(
        "Which penalties actually beat the baseline on accuracy?\n"
        "p = McNemar exact test vs baseline, paired on the same 240 examples.\n"
        "bold p<0.05 = real gain · plain = parity (CI overlaps shaded band)",
        fontsize=9.5, loc="left", pad=8)
    ax_qual.legend(loc="lower left", fontsize=8.5, frameon=False, labelcolor=INK_2)
    _style(ax_qual)

    meta = data.get("meta", {})
    fig.suptitle(
        f"Conflation: CAL frontier under an LLM-call penalty  "
        f"(exec {meta.get('exec_model', '?').split('/')[-1]}, "
        f"reflection {meta.get('reflection_model', '?').split('/')[-1]}, "
        f"n_test={meta.get('n_test', '?')})",
        fontsize=13, color=INK, x=0.01, ha="left", y=0.992,
    )
    if p_test is not None:
        fig.text(0.01, 0.951,
                 "overlaid: plain GEPA on a bare dspy.Predict — instruction-only, no Flex. One "
                 "point, not a curve: a Predict always makes exactly 1 call, so λ has nothing to "
                 "act on.",
                 fontsize=9.5, color=INK_2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.932 if p_test is not None else 0.96))
    out_png = path or PLOT_PATH
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved plot -> {out_png}")


if __name__ == "__main__":
    main()
