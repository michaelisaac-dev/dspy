from __future__ import annotations

import sys
from pathlib import Path

import pytest

import dspy

sys.path.insert(0, str(Path(__file__).parent))
from conflation_common import (
    EXEC_MODEL,
    REFLECTION_MODEL,
    SamePlace,
    fmt,
    load_splits,
    make_lms,
    make_metric,
    run_program,
    summarize,
)

# dspy registers a lazy numpy proxy in sys.modules; matplotlib's `from numpy.exceptions import ...`
# trips that proxy into a recursive import. Materialize the real numpy first. (banking77/pajama get
# this for free by importing pandas/datasets before matplotlib; this demo depends on neither.)
np = pytest.importorskip("numpy")
_ = np.ndarray  # force the proxy to load the real module before matplotlib imports it
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

DEMO_DIR = Path(__file__).parent
SAVE_PATH = DEMO_DIR / "conflation_flex.json"
PLOT_PATH = DEMO_DIR / "conflation_improvement.png"

# The penalty is the knob `sweep_penalties.py` sweeps; this single-run demo pins it at the value
# that trades one LLM call against 0.2 of the 1.0 available for a correct answer.
PENALTY = 0.20
MAX_METRIC_CALLS = 400
EVAL_THREADS = 8

EXEC_LM, STRONG_LM = make_lms()
dspy.configure(lm=EXEC_LM)


def _showcase(program: dspy.Module, label: str) -> None:
    """Print the flexed module's clean dspy.Module source and its flat predictors."""
    print(f"--- {label} ---")
    print("predictors on the module:", [n for n, _ in program.named_predictors()])
    print(program.module_src)


def test_loader():
    program = dspy.Flex(SamePlace)
    program.load(str(SAVE_PATH))
    return program


def test_flex_conflation() -> None:
    dspy.configure(lm=EXEC_LM)
    train, val, test = load_splits()
    print(f"splits: train={len(train)} val={len(val)} test={len(test)} (class-balanced)")
    print(f"models: exec={EXEC_MODEL} reflection={REFLECTION_MODEL}  penalty={PENALTY}")

    program = dspy.Flex(SamePlace)

    # Fresh baseline: a clean dspy.Module subclass that delegates to one dspy.Predict.
    src = program.module_src or ""
    assert src.lstrip().startswith("class ")
    assert "dspy.Predict(" in src
    _showcase(program, "baseline (un-optimized flex)")

    base_records, base_meta = run_program(program, test, threads=EVAL_THREADS)
    base = summarize(base_records, PENALTY) | base_meta
    print(f"[baseline] {fmt(base)}")
    # A model id that 404s used to surface as "0% accuracy, 0 calls" because every call raised and
    # was swallowed per-example. Fail loudly instead.
    assert base["errors"] == 0, f"baseline had {base['errors']} failed examples: {base['first_error']}"

    optimized = dspy.GEPA(
        metric=make_metric(PENALTY),
        reflection_lm=STRONG_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        reflection_minibatch_size=3,
        num_threads=EVAL_THREADS,
        seed=0,
    ).compile(program, trainset=train, valset=val)

    # The metric penalizes LLM calls, so GEPA should push most logic into plain Python —
    # watch the module_src shift from one Predict to focused predictors + deterministic code.
    _showcase(optimized, "optimized by GEPA")
    print(f"GEPA changed the code: {optimized.module_src != program.module_src}")

    opt_records, opt_meta = run_program(optimized, test, threads=EVAL_THREADS)
    opt = summarize(opt_records, PENALTY) | opt_meta
    print(f"[optimized] {fmt(opt)}  (eval wall {opt['wall_s']:.0f}s vs baseline {base['wall_s']:.0f}s)")
    print(
        f"score improvement: {opt['score'] - base['score']:+.3f}  |  "
        f"F1 improvement: {opt['f1'] - base['f1']:+.3f}  |  "
        f"calls/ex: {base['avg_calls']:.2f} -> {opt['avg_calls']:.2f}"
    )

    # Persist the optimized program with the standard Module.save/load (code round-trips).
    optimized.save(str(SAVE_PATH))
    reloaded = dspy.Flex(SamePlace)
    reloaded.load(str(SAVE_PATH))
    assert reloaded.module_src == optimized.module_src
    print(f"saved + reloaded optimized program -> {SAVE_PATH}")

    _plot(base, opt, len(test))


def _plot(base: dict, opt: dict, n_test: int) -> None:
    """Before/after in three panels: the optimized objective, the classification metrics, the cost.

    Panel 1 is the number GEPA actually optimizes (accuracy − λ·LLM calls). Panel 2 is what
    validates the classifier. Panel 3 is the decomposition win — a one-call Predict versus
    deterministic Python that settles clear cases without touching the LM.
    """
    # Palette slots 1 and 2 from the dataviz reference palette (light mode); validated all-pairs.
    labels_xy = ["baseline\n(flex / Predict)", "optimized\n(GEPA code)"]
    colors = ["#eb6834", "#2a78d6"]
    surface, ink2, grid = "#fcfcfb", "#52514e", "#e6e5e1"
    fig, (ax_score, ax_cls, ax_calls) = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor(surface)

    def style(ax):
        ax.set_facecolor(surface)
        ax.grid(True, axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=ink2, labelsize=9)

    score_bars = ax_score.bar(labels_xy, [base["score"], opt["score"]], color=colors, width=0.6)
    ax_score.set_ylabel("mean metric score")
    ax_score.set_ylim(0, 1.1)  # headroom so the on-bar labels clear the title at ~1.0
    ax_score.set_title(f"Score (accuracy − {PENALTY:g}·LLM calls)", fontsize=11, loc="left")
    for bar, s in zip(score_bars, [base["score"], opt["score"]], strict=True):
        ax_score.text(bar.get_x() + bar.get_width() / 2, s + 0.02, f"{s:.2f}", ha="center",
                      va="bottom", color=ink2)
    style(ax_score)

    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    base_vals = [base[k] for k in ("accuracy", "precision", "recall", "f1")]
    opt_vals = [opt[k] for k in ("accuracy", "precision", "recall", "f1")]
    xpos = np.arange(len(metric_names))
    width = 0.38
    b_bars = ax_cls.bar(xpos - width / 2, base_vals, width, label="baseline", color=colors[0])
    o_bars = ax_cls.bar(xpos + width / 2, opt_vals, width, label="optimized", color=colors[1])
    ax_cls.set_xticks(xpos)
    ax_cls.set_xticklabels(metric_names)
    ax_cls.set_ylabel("score (0–1)")
    ax_cls.set_ylim(0, 1.15)  # headroom for the on-bar labels near 1.0
    ax_cls.set_title("Classification metrics (positive = same place)", fontsize=11, loc="left")
    ax_cls.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=ink2)
    for bars, vals in ((b_bars, base_vals), (o_bars, opt_vals)):
        for bar, v in zip(bars, vals, strict=True):
            ax_cls.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7, color=ink2)
    style(ax_cls)

    call_vals = [base["avg_calls"], opt["avg_calls"]]
    call_bars = ax_calls.bar(labels_xy, call_vals, color=colors, width=0.6)
    ax_calls.set_ylabel("avg LLM calls / example")
    ax_calls.set_ylim(0, max(*call_vals, 1) * 1.2)
    ax_calls.set_title("LLM calls (lower = more deterministic)", fontsize=11, loc="left")
    for bar, n in zip(call_bars, call_vals, strict=True):
        ax_calls.text(bar.get_x() + bar.get_width() / 2, n, f"{n:.2f}", ha="center",
                      va="bottom", color=ink2)
    style(ax_calls)

    fig.suptitle(f"Conflation: same-place matching (n={n_test})", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PLOT_PATH, dpi=150, facecolor=surface)
    plt.close(fig)
    print(f"saved plot -> {PLOT_PATH}")
    assert PLOT_PATH.exists()


if __name__ == "__main__":
    # test_loader()
    test_flex_conflation()
