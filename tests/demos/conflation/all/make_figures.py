"""Render each CAL panel of the conflation sweep as its own standalone vector figure.

    python make_figures.py            # reads penalty_sweep.json -> figures/*.svg (+ .png previews)

The four-panel `cal_frontier.png` stays the demo's canonical artifact; these are the same panels
split for publication (each must stand alone, so each carries its own context footer). Vector
output keeps text as text (`svg.fonttype: none`) so the blog's own font stack applies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# dspy registers a lazy numpy proxy; materialize numpy before matplotlib imports it.
import numpy as np

_ = np.ndarray
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt  # noqa: E402
from sweep_penalties import mcnemar_p, wilson  # noqa: E402

DEMO_DIR = Path(__file__).parent
SWEEP_PATH = DEMO_DIR / "penalty_sweep.json"
FIG_DIR = DEMO_DIR / "figures"

# Categorical slots 1-3 of the dataviz reference palette (light mode); this trio passes the
# validator all-pairs. Aqua carries a contrast WARN on this surface, so it is always direct-labeled.
C_OPT = "#2a78d6"
C_BASE = "#eb6834"
C_PLAIN = "#1baf7a"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


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


def _save(fig, name: str, footer: str) -> None:
    fig.text(0.01, 0.012, footer, fontsize=8, color=MUTED, ha="left")
    FIG_DIR.mkdir(exist_ok=True)
    for ext, kw in (("svg", {}), ("png", {"dpi": 150})):
        fig.savefig(FIG_DIR / f"{name}.{ext}", facecolor=SURFACE, **kw)
    plt.close(fig)
    print(f"saved figures/{name}.svg (+.png)")


def main() -> None:
    data = json.loads(SWEEP_PATH.read_text(encoding="utf-8"))
    runs = data["runs"]
    keys = sorted(runs, key=float)
    lam = [float(k) for k in keys]
    t = [runs[k]["test"] for k in keys]
    acc = [x["accuracy"] for x in t]
    cost = [x["cost_usd_per_1k_examples"] for x in t]
    lat = [x["latency_mean_s"] * 1000 for x in t]
    calls = [x["avg_calls"] for x in t]
    ci = [wilson(round(x["accuracy"] * x["n"]), x["n"]) for x in t]
    acc_err = [[max(0.0, a - lo) for a, (lo, _) in zip(acc, ci, strict=True)],
               [max(0.0, hi - a) for a, (_, hi) in zip(acc, ci, strict=True)]]

    b = data["baseline"]["by_penalty"][keys[0]]
    brecs = data["baseline"]["records"]
    plain = data.get("plain_gepa")
    p_test = plain["test"] if plain else None

    meta = data.get("meta", {})
    base_footer = (f"execution {meta.get('exec_model', '?').split('/')[-1]} · "
                   f"reflection {meta.get('reflection_model', '?').split('/')[-1]} · "
                   f"n = {meta.get('n_test', '?')} held-out examples")
    footer = base_footer + " · error bars: 95% Wilson CIs"

    def frontier(name: str, xs, xlabel, title, base_x, plain_x) -> None:
        fig, ax = plt.subplots(figsize=(6.8, 5.0))
        fig.patch.set_facecolor(SURFACE)
        # Points connect in λ order, not sorted by x: λ is the independent variable, and sorting
        # by x would draw a monotone frontier the data does not have (λ=0.1 and 0.2 invert).
        ax.errorbar(xs, acc, yerr=acc_err, color=C_OPT, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=1.2, capsize=4,
                    ecolor=C_OPT, zorder=3, label="Flex + GEPA (95% CI)")
        for i in range(len(xs)):
            ax.annotate(f"λ={lam[i]:g}", (xs[i], acc[i]), textcoords="offset points",
                        xytext=(9, -3), fontsize=8.5, color=INK_2, zorder=4)
        ax.plot([base_x], [b["accuracy"]], color=C_BASE, marker="s", markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none",
                zorder=3, label="baseline (un-optimized)")
        ax.annotate("baseline", (base_x, b["accuracy"]), textcoords="offset points",
                    xytext=(0, -17), fontsize=8.5, color=INK_2, ha="center", zorder=4)
        if p_test is not None and plain_x is not None:
            ax.plot([plain_x], [p_test["accuracy"]], color=C_PLAIN, marker="D", markersize=8.5,
                    markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", zorder=3,
                    label="plain GEPA (prompt-only)")
            ax.annotate("plain GEPA", (plain_x, p_test["accuracy"]), textcoords="offset points",
                        xytext=(0, 12), fontsize=8.5, color=INK_2, ha="center", zorder=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("accuracy")
        ax.set_title(title, fontsize=11.5, loc="left", pad=10)
        ax.legend(loc="center right", fontsize=8.5, frameon=False, labelcolor=INK_2)
        _style(ax)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        _save(fig, name, footer)

    frontier("cal_cost_accuracy", cost, "inference cost  (USD per 1,000 examples)",
             "Cost vs accuracy, as the LLM-call penalty rises",
             b["cost_usd_per_1k_examples"], p_test["cost_usd_per_1k_examples"] if p_test else None)
    frontier("cal_latency_accuracy", lat, "mean per-request latency  (ms; 8-way concurrency)",
             "Latency vs accuracy, as the LLM-call penalty rises",
             b["latency_mean_s"] * 1000, p_test["latency_mean_s"] * 1000 if p_test else None)

    # -- calls vs penalty ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    fig.patch.set_facecolor(SURFACE)
    ax.plot(lam, calls, color=C_OPT, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3, label="Flex + GEPA")
    ax.axhline(b["avg_calls"], color=C_BASE, linewidth=2, linestyle=(0, (5, 3)),
               zorder=2, label="baseline (always 1 call)")
    if p_test is not None:
        ax.axhline(p_test["avg_calls"], color=C_PLAIN, linewidth=2, linestyle=(0, (1, 2)),
                   zorder=3, label="plain GEPA (fixed at 1.00; λ cannot move it)")
    for x, y in zip(lam, calls, strict=True):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color=INK_2, zorder=4)
    ax.set_xlabel("LLM-call penalty  λ")
    ax.set_ylabel("avg LLM calls per example")
    ax.set_title("What the penalty buys: work moves into Python", fontsize=11.5, loc="left", pad=10)
    ax.set_ylim(-0.08, max([*calls, b["avg_calls"]]) * 1.25 + 0.05)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=INK_2)
    _style(ax)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _save(fig, "cal_calls_vs_penalty", base_footer)

    # -- accuracy vs baseline, with paired significance ----------------------
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    fig.patch.set_facecolor(SURFACE)
    ax.errorbar(lam, acc, yerr=acc_err, color=C_OPT, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=1.2, capsize=4,
                ecolor=C_OPT, zorder=3, label="Flex + GEPA (95% CI)")
    blo, bhi = wilson(round(b["accuracy"] * b["n"]), b["n"])
    ax.axhspan(blo, bhi, color=C_BASE, alpha=0.13, zorder=0)
    ax.axhline(b["accuracy"], color=C_BASE, linewidth=2, linestyle=(0, (5, 3)), zorder=2,
               label="baseline (95% CI band)")
    for x, k, (_, hi_i) in zip(lam, keys, ci, strict=True):
        pv = mcnemar_p(brecs, runs[k]["records"])
        ax.annotate(f"p={pv:.3f}" if pv >= 0.001 else "p<0.001", (x, hi_i),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=8, color=(C_OPT if pv < 0.05 else INK_2), zorder=4,
                    fontweight=("bold" if pv < 0.05 else "normal"))
    if plain is not None:
        pv_plain = mcnemar_p(brecs, plain["records"])
        ax.axhline(plain["test"]["accuracy"], color=C_PLAIN, linewidth=2, linestyle=(0, (1, 2)),
                   zorder=3, label=f"plain GEPA, p={pv_plain:.3f} vs baseline")
    ax.set_xlabel("LLM-call penalty  λ")
    ax.set_ylabel("accuracy")
    plain_acc = [p_test["accuracy"]] if p_test else []
    lo = min([*[c[0] for c in ci], *plain_acc, b["accuracy"]])
    hi = max([*[c[1] for c in ci], *plain_acc, b["accuracy"]])
    pad = max(0.012, (hi - lo) * 0.16)
    ax.set_ylim(lo - pad * 2.6, hi + pad * 1.7)  # truncated axis, well above 0
    ax.set_title("Which penalties beat the baseline on accuracy?\n"
                 "p = exact McNemar vs baseline, paired on the same examples; bold = p<0.05",
                 fontsize=10, loc="left", pad=8)
    ax.legend(loc="lower left", fontsize=8.5, frameon=False, labelcolor=INK_2)
    _style(ax)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _save(fig, "cal_accuracy_significance", footer)


if __name__ == "__main__":
    main()
