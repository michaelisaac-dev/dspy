from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import pytest
from dotenv import load_dotenv

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()

# dspy registers a lazy numpy proxy in sys.modules; matplotlib's `from numpy.exceptions import ...`
# trips that proxy into a recursive import. Materialize the real numpy first. (banking77/pajama get
# this for free by importing pandas/datasets before matplotlib; this demo depends on neither.)
np = pytest.importorskip("numpy")
_ = np.ndarray  # force the proxy to load the real module before matplotlib imports it
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

DEMO_DIR = Path(__file__).parent
DATA_PATH = DEMO_DIR.parent / "conflation_coded.jsonl"
SAVE_PATH = DEMO_DIR / "conflation_flex.json"
PLOT_PATH = DEMO_DIR / "conflation_improvement.png"

EXEC_LM = dspy.LM("anthropic/claude-haiku-4-5", max_tokens=1000)
STRONG_LM = dspy.LM("anthropic/claude-opus-4-7", max_tokens=8000)
dspy.configure(lm=EXEC_LM)

# Small class-balanced train/val sets drive GEPA (its budget is MAX_METRIC_CALLS, so val must stay
# small); the test split is the ENTIRE remaining dataset (every pos/neg not consumed by train/val)
# for the most complete accuracy/cost estimate. The full pools are imbalanced (769 pos / 260 neg),
# so the test split is majority-positive — read accuracy against that base rate, not a 50% chance line.
N_TRAIN_POS, N_TRAIN_NEG = 8, 8
N_VAL_POS, N_VAL_NEG = 5, 5
MAX_METRIC_CALLS = 45
EVAL_THREADS = 8


class SamePlace(dspy.Signature):
    """Decide whether two business listings refer to the same physical place.

    You compare place A (input_name / input_address) with place B (match_name /
    match_address), plus the geographic distance between them.

    Prefer a deterministic Python algorithm.
    Reserve an LLM call only for ambiguous ones that simple logic can't decide confidently.
    """

    input_name: str = dspy.InputField(desc="Name of place A.")
    input_address: str = dspy.InputField(desc="Street address of place A.")
    match_name: str = dspy.InputField(desc="Name of place B.")
    match_address: str = dspy.InputField(desc="Street address of place B.")
    distance: float = dspy.InputField(desc="Distance between the two coordinates.")
    is_same: bool = dspy.OutputField(desc="True if A and B are the same physical place, else False.")


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return bool(value)


def _to_example(row: dict) -> dspy.Example:
    return dspy.Example(
        input_name=row["input_name"],
        input_address=row["input_address"],
        match_name=row["match_name"],
        match_address=row["match_address"],
        distance=float(row["distance"]),
        is_same=(row["judgment"] == "true"),
    ).with_inputs("input_name", "input_address", "match_name", "match_address", "distance")


def _load_splits() -> tuple[list, list, list]:
    rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    pos = [r for r in rows if r["judgment"] == "true"]
    neg = [r for r in rows if r["judgment"] == "false"]
    rng = random.Random(0)
    rng.shuffle(pos)
    rng.shuffle(neg)

    def take(seq, start, count):
        return [_to_example(r) for r in seq[start : start + count]]

    train = take(pos, 0, N_TRAIN_POS) + take(neg, 0, N_TRAIN_NEG)
    val = take(pos, N_TRAIN_POS, N_VAL_POS) + take(neg, N_TRAIN_NEG, N_VAL_NEG)
    # Test = the entire remaining dataset: every pos/neg not consumed by the train/val slices.
    test = take(pos, N_TRAIN_POS + N_VAL_POS, len(pos)) + take(neg, N_TRAIN_NEG + N_VAL_NEG, len(neg))
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None) -> ScoreWithFeedback:
    """Reward correct + deterministic, penalize LLM calls."""
    example, prediction = gold, pred
    llm_call_penalty = 0.15
    # GEPA delivers the execution trace via `program_trace` at scoring time (declaring the
    # parameter opts in); `trace` still carries it on feedback calls and from _evaluate below.
    exec_trace = program_trace if program_trace is not None else trace
    n_calls = len(exec_trace) if exec_trace else 0  # predictor calls during this forward()
    try:
        pred = _as_bool(prediction.is_same)
    except Exception:
        return ScoreWithFeedback(
            score=0.0,
            feedback="`is_same` was missing or unreadable. Return dspy.Prediction(is_same=<bool>).",
        )
    gold = bool(example.is_same)
    correct = pred == gold
    score = max(0.0, (1.0 if correct else 0.0) - llm_call_penalty * n_calls)

    if not correct:
        fb = (
            f"WRONG: predicted is_same={pred}, expected {gold}. Use the input fields (name, address, distance)"
            f"to ideally decide deterministically in Python whether the two location are the same."
        )
        if n_calls == 0:
            fb += " If this case is truly ambiguous for rules, route it to the LLM judge instead."
    elif n_calls > 0:
        fb = (
            f"Correct, but used {n_calls} LLM call(s) (cost {llm_call_penalty * n_calls:.2f}). If the "
            "normalized name/address similarity and distance already make this clear, decide it in "
            "Python and skip the LLM. Reserve LLM calls for genuinely ambiguous cases only."
        )
    else:
        fb = "Correct with no LLM call. This is great! Keep settling clear cases deterministically."
    return ScoreWithFeedback(score=score, feedback=fb)


class EvalResult(NamedTuple):
    """Headline metrics for one program on one dataset (positive class = same place)."""

    score: float       # mean metric score = accuracy − 0.15·(LLM calls); GEPA's objective
    accuracy: float    # fraction correct — weak on the ~75%-positive test split (see below)
    precision: float   # TP / (TP + FP): of predicted-same pairs, how many truly are
    recall: float      # TP / (TP + FN): of truly-same pairs, how many we caught
    f1: float          # harmonic mean of precision & recall — the balanced headline
    avg_calls: float   # avg LLM calls / example — the decomposition (cost) win


def _evaluate(program: dspy.Module, dataset: list) -> EvalResult:
    """Return the metric score plus full classification metrics for `program` over `dataset`.

    Score is the number the optimizer actually optimizes — accuracy minus the 0.15-per-LLM-call
    penalty — not raw accuracy. On the imbalanced full test split (~75% positive) accuracy is a
    weak signal: a predict-always-same classifier already scores ~0.75, so precision / recall / F1
    (positive class = is_same True) are what actually validate the classifier. avg_calls exposes
    GEPA's win — the un-optimized Predict makes one traced LLM call per example; deterministic
    Python that settles clear cases drives that toward 0.
    """
    def run_one(ex):
        gold = bool(ex.is_same)
        try:
            with dspy.context(trace=[]):
                pred_obj = program(**ex.inputs())
                trace = list(dspy.settings.trace or [])
            score = float(metric(ex, pred_obj, trace=trace).score)
            return score, _as_bool(pred_obj.is_same), gold, len(trace)
        except Exception:
            # An unreadable prediction counts as wrong in both accuracy and the confusion matrix
            # (pred = not gold keeps them consistent); errors should be ~0 in practice.
            return 0.0, (not gold), gold, 0

    with ThreadPoolExecutor(max_workers=EVAL_THREADS) as pool:
        results = list(pool.map(run_one, dataset))
    n = len(dataset) or 1
    total_score = sum(s for s, _, _, _ in results)
    calls = sum(c for _, _, _, c in results)
    correct = sum(1 for _, p, g, _ in results if p == g)
    tp = sum(1 for _, p, g, _ in results if p and g)
    fp = sum(1 for _, p, g, _ in results if p and not g)
    fn = sum(1 for _, p, g, _ in results if not p and g)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return EvalResult(
        score=total_score / n,
        accuracy=correct / n,
        precision=precision,
        recall=recall,
        f1=f1,
        avg_calls=calls / n,
    )


def _showcase(program: dspy.Module, label: str) -> None:
    """Print the flexed module's clean dspy.Module source and its flat predictors."""
    print("predictors on the module:", [n for n, _ in program.named_predictors()])
    print(program.module_src)


def test_loader():
    program = dspy.Flex(SamePlace)
    program.load(str(SAVE_PATH))
    return program


def test_flex_conflation() -> None:
    dspy.configure(lm=EXEC_LM)
    train, val, test = _load_splits()
    print(f"splits: train={len(train)} val={len(val)} test={len(test)}")

    program = dspy.Flex(SamePlace)

    # Fresh baseline: a clean dspy.Module subclass that delegates to one dspy.Predict.
    assert program.module_src.lstrip().startswith("class ")
    assert "dspy.Predict(" in program.module_src
    _showcase(program, "baseline (un-optimized flex)")

    base = _evaluate(program, test)
    print(
        f"[baseline] score={base.score:.2f} acc={base.accuracy:.2f} "
        f"P={base.precision:.2f} R={base.recall:.2f} F1={base.f1:.2f} "
        f"calls/ex={base.avg_calls:.2f}"
    )

    optimized = dspy.GEPA(
        metric=metric,
        reflection_lm=STRONG_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        reflection_minibatch_size=3,
        num_threads=EVAL_THREADS,
    ).compile(program, trainset=train, valset=val)

    # The metric penalizes LLM calls, so GEPA should push most logic into plain Python —
    # watch the module_src shift from one Predict to focused predictors + deterministic code.
    _showcase(optimized, "optimized by GEPA")
    print(f"GEPA changed the code: {optimized.module_src != program.module_src}")

    opt = _evaluate(optimized, test)
    print(
        f"[optimized] score={opt.score:.2f} acc={opt.accuracy:.2f} "
        f"P={opt.precision:.2f} R={opt.recall:.2f} F1={opt.f1:.2f} "
        f"calls/ex={opt.avg_calls:.2f}"
    )
    print(
        f"score improvement: {opt.score - base.score:+.2f}  |  "
        f"F1 improvement: {opt.f1 - base.f1:+.2f}"
    )

    # Persist the optimized program with the standard Module.save/load (code round-trips).
    optimized.save(str(SAVE_PATH))
    reloaded = dspy.Flex(SamePlace)
    reloaded.load(str(SAVE_PATH))
    assert reloaded.module_src == optimized.module_src
    print(f"saved + reloaded optimized program -> {SAVE_PATH}")

    # Plot the before/after (a la banking77). Panel 1: the score GEPA optimizes
    # (accuracy − 0.15/LLM-call). Panel 2: the classification metrics that actually validate the
    # classifier on the imbalanced test split — accuracy/precision/recall/F1 (positive = same place),
    # since accuracy alone is fooled by the ~75% majority-positive base rate. Panel 3: the
    # decomposition win (one-call Predict -> deterministic Python that settles clear cases at 0 calls).
    labels_xy = ["baseline\n(flex / Predict)", "optimized\n(GEPA code)"]
    colors = ["#9aa0a6", "#1a73e8"]
    fig, (ax_score, ax_cls, ax_calls) = plt.subplots(1, 3, figsize=(13, 4.5))

    score_bars = ax_score.bar(labels_xy, [base.score, opt.score], color=colors)
    ax_score.set_ylabel("mean metric score")
    ax_score.set_ylim(0, 1.1)  # headroom so the on-bar labels clear the title at ~1.0
    ax_score.set_title("Score (accuracy − call penalty)")
    for bar, s in zip(score_bars, [base.score, opt.score], strict=True):
        ax_score.text(bar.get_x() + bar.get_width() / 2, s + 0.02, f"{s:.2f}", ha="center", va="bottom")

    # Grouped bars: one group per classification metric, baseline vs optimized side by side.
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    base_vals = [base.accuracy, base.precision, base.recall, base.f1]
    opt_vals = [opt.accuracy, opt.precision, opt.recall, opt.f1]
    xpos = np.arange(len(metric_names))
    width = 0.38
    b_bars = ax_cls.bar(xpos - width / 2, base_vals, width, label="baseline", color=colors[0])
    o_bars = ax_cls.bar(xpos + width / 2, opt_vals, width, label="optimized", color=colors[1])
    ax_cls.set_xticks(xpos)
    ax_cls.set_xticklabels(metric_names)
    ax_cls.set_ylabel("score (0–1)")
    ax_cls.set_ylim(0, 1.15)  # headroom for the on-bar labels near 1.0
    ax_cls.set_title("Classification metrics (positive = same place)")
    ax_cls.legend(loc="lower right", fontsize=8)
    for bars, vals in ((b_bars, base_vals), (o_bars, opt_vals)):
        for bar, v in zip(bars, vals, strict=True):
            ax_cls.text(
                bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    call_bars = ax_calls.bar(labels_xy, [base.avg_calls, opt.avg_calls], color=colors)
    ax_calls.set_ylabel("avg LLM calls / example")
    ax_calls.set_ylim(0, max(base.avg_calls, opt.avg_calls, 1) * 1.2)
    ax_calls.set_title("LLM calls (lower = more deterministic)")
    for bar, n in zip(call_bars, [base.avg_calls, opt.avg_calls], strict=True):
        ax_calls.text(bar.get_x() + bar.get_width() / 2, n, f"{n:.1f}", ha="center", va="bottom")

    fig.suptitle(f"Conflation: same-place matching (n={len(test)})")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved plot -> {PLOT_PATH}")
    assert PLOT_PATH.exists()


if __name__ == "__main__":
    test_loader()
    # test_flex_conflation()
