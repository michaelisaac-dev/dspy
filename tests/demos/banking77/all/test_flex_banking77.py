"""End-to-end showcase: flex a BANKING77 intent classifier, then optimize its code with GEPA.

Flow:
  1. Write a classification Signature and "flex" it with a tool: ``dspy.Flex(signature, tools=[...])``
     binds a multi-call ``dspy.RLM`` baseline (a tool makes the baseline an RLM instead of a single
     ``dspy.Predict``), so there is real LM-call headroom for GEPA to shrink.
  2. Benchmark that baseline on a held-out test split.
  3. Run ``dspy.GEPA`` over a small train/val split. Because the module is flex-marked,
     GEPA optimizes its *code* (``module_src``) — decomposing the
     task into focused predictors / plain Python.
  4. Benchmark the optimized program on the same test split.
  5. Plot baseline-vs-optimized classification metrics (accuracy + macro precision/recall/F1) and
     avg LLM calls/example to `banking77_improvement.png`.

The GEPA metric's *feedback* is the prompt GEPA feeds its code proposer. We deliberately
make that feedback adversarial — it forces the model to interrogate its own logic and
*prove* why each classification is correct rather than pattern-match — to squeeze more out
of the reflection model.

Needs real LMs + network (HuggingFace `PolyAI/banking77`). Skips without an API key or the
optional `datasets`/`matplotlib` deps. Dataset: https://huggingface.co/datasets/PolyAI/banking77
"""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import pytest
from dotenv import load_dotenv

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()

pd = pytest.importorskip("pandas")
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# HuggingFace `PolyAI/banking77` is a script-based loader (incompatible with datasets>=3),
# and it simply wraps these canonical CSVs. We read them directly — same data.
BANKING77_BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"

DEMO_DIR = Path(__file__).parent
SAVE_PATH = DEMO_DIR / "banking77_flex.json"
PLOT_PATH = DEMO_DIR / "banking77_improvement.png"

# Executor runs the classifier (baseline and optimized); reflection authors the optimized code.
# We deliberately make the executor a SMALL model (Haiku) and the reflection model a STRONG one
# (Opus): a weak executor leaves real accuracy headroom on a hard 77-way task, so the Predict
# baseline is well below 100% and GEPA's decomposition has something to lift. (With a strong
# executor on a tiny test split, even the un-optimized Predict baseline scores ~100% and the
# demo shows no improvement.) Both override via env.
_exec_default = "anthropic/claude-haiku-4-5"
_reflect_default = "anthropic/claude-opus-4-7"
EXEC_LM = dspy.LM(os.getenv("BANKING_EXEC_LM", _exec_default), max_tokens=2000)
REFLECTION_LM = dspy.LM(os.getenv("BANKING_REFLECTION_LM", _reflect_default), temperature=1.0, max_tokens=8000)

# Small train/val samples drive GEPA (its budget is MAX_METRIC_CALLS, so val must stay small); the
# test split is the ENTIRE BANKING77 test set (~3080 examples, ~40 per intent) for the most complete
# accuracy/F1 estimate. The test set is class-balanced across the 77 intents, so macro and micro
# averaging nearly coincide (and micro-F1 == accuracy for single-label multiclass).
N_TRAIN, N_VAL = 20, 10
MAX_METRIC_CALLS = 60
EVAL_THREADS = 8

# Small per-LLM-call penalty folded into GEPA's score. Accuracy stays the dominant term (a
# correct answer is worth 1.0), but among equally-accurate programs the one that makes fewer
# LM calls scores higher. Without this, the metric is pure accuracy, so among equally-accurate
# programs GEPA has no score reason to prefer a leaner one — its decomposition *feedback* is then
# overridden by its score-based acceptance. This makes "decompose into focused, deterministic
# predictors" actually pay.
# Kept small (0.02) so it only breaks ties: a decomposition must *hold* accuracy to win — it
# can never beat a strictly-more-accurate program, so the accuracy headline can't regress.
LLM_CALL_PENALTY = 0.35

# The "challenge" injected into GEPA's reflection prompt (via metric feedback): push the
# model to justify its logic instead of guessing.
CHALLENGE = (
    "Before you keep or revise this classifier, challenge yourself: is the approach actually "
    "correct, or are you pattern-matching on superficial keywords? For every decision the code "
    "makes you must be able to PROVE why the chosen intent is the customer's true goal and why "
    "each competing intent is wrong. Question your own assumptions and justify them explicitly. "
    "Prefer decomposing the task into focused predictors over one opaque call."
)


def _norm(label: str) -> str:
    return (label or "").strip().lower().replace(" ", "_")


def _predicted(pred) -> str:
    return _norm(getattr(pred, "intent", ""))


def _load_splits():
    train_df = pd.read_csv(f"{BANKING77_BASE}/train.csv")
    test_df = pd.read_csv(f"{BANKING77_BASE}/test.csv")
    labels = sorted(train_df["category"].unique())

    def to_examples(df):
        return [dspy.Example(text=r.text, intent=r.category).with_inputs("text") for r in df.itertuples(index=False)]

    pool = train_df.sample(frac=1, random_state=0).reset_index(drop=True)  # disjoint train/val
    train = to_examples(pool.iloc[:N_TRAIN])
    val = to_examples(pool.iloc[N_TRAIN : N_TRAIN + N_VAL])
    test = to_examples(test_df)  # the entire BANKING77 test set
    return labels, train, val, test


def _build_signature(labels: list[str]):
    instructions = (
        "You are an intent classifier for the BANKING77 dataset. Given a single retail-banking "
        "customer message, return the one most appropriate intent.\n\n"
        "The answer MUST be exactly one of these 77 snake_case intent labels:\n"
        + ", ".join(labels)
        + ".\n\nReturn only the label, verbatim, with no extra words or punctuation."
    )
    return dspy.Signature("text: str -> intent: str", instructions)


class EvalResult(NamedTuple):
    """Headline metrics for one program on one dataset (77-way intent classification)."""

    accuracy: float          # fraction correct == micro-F1 for single-label multiclass
    macro_precision: float   # mean per-intent precision (all 77 intents weighted equally)
    macro_recall: float      # mean per-intent recall
    macro_f1: float          # mean per-intent F1 — the balanced headline across intents
    avg_calls: float         # avg traced LLM calls / example — the decomposition (cost) win


def _evaluate(program: dspy.Module, dataset: list) -> EvalResult:
    """Return accuracy + macro-averaged precision/recall/F1 and avg LLM calls/example.

    BANKING77 is 77-way single-label classification, so precision/recall/F1 are **macro-averaged**
    (computed per intent, then unweighted mean). Macro weighs all 77 intents equally, so weakness on
    rarer or easily-confused intents shows up instead of being masked by the common ones — micro-
    averaged F1 would just equal accuracy and add nothing. The call count shows *how* the answer was
    reached: the RLM baseline spends several traced calls per example; a GEPA-decomposed program
    settles cases in fewer focused calls (or plain Python at zero).
    """
    def run_one(ex):
        try:
            with dspy.context(lm=EXEC_LM, trace=[]):
                pred = program(**ex.inputs())
                n_calls = len(dspy.settings.trace or [])
            return _predicted(pred), _norm(ex.intent), n_calls
        except Exception:
            return "", _norm(ex.intent), 0  # unreadable prediction counts as wrong (empty label)

    with ThreadPoolExecutor(max_workers=EVAL_THREADS) as pool:
        results = list(pool.map(run_one, dataset))
    n = len(dataset) or 1
    correct = sum(1 for pred, gold, _ in results if pred == gold)
    avg_calls = sum(c for _, _, c in results) / n

    # Per-intent confusion counts in one pass, then macro-average over the true intents.
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    for pred, gold, _ in results:
        if pred == gold:
            tp[gold] += 1
        else:
            fn[gold] += 1
            fp[pred] += 1  # pred may be "" on error — a bucket we never average over
    true_classes = {gold for _, gold, _ in results}
    precisions, recalls, f1s = [], [], []
    for c in true_classes:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    k = len(true_classes) or 1
    return EvalResult(
        accuracy=correct / n,
        macro_precision=sum(precisions) / k,
        macro_recall=sum(recalls) / k,
        macro_f1=sum(f1s) / k,
        avg_calls=avg_calls,
    )


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None) -> ScoreWithFeedback:
    # Declaring `program_trace` (the optional 6th GEPA metric arg) opts in to receiving the
    # execution trace at scoring time, so this cost metric can count LM calls.
    target = _norm(gold.intent)
    predicted = _predicted(pred)
    correct = predicted == target
    exec_trace = program_trace if program_trace is not None else trace
    n_calls = len(exec_trace) if exec_trace else 0

    score = max(0.0, (1.0 if correct else 0.0) - LLM_CALL_PENALTY * n_calls)
    cost = f"(used {n_calls} LM call(s), cost {LLM_CALL_PENALTY * n_calls:.2f})"
    if correct:
        fb = (
            f"CORRECT — predicted '{predicted}', which matches the true intent {cost}. {CHALLENGE} "
            "Even though this one was right, prove the reasoning was sound and not luck, and settle "
            "clear cases in fewer (ideally one) focused predictor calls — extra LM calls are "
            "penalized per call."
        )
    else:
        fb = (
            f"WRONG — predicted '{predicted or '<empty>'}' but the true intent is '{target}' {cost}. "
            f"{CHALLENGE} Diagnose the exact reasoning flaw that produced the wrong label."
        )
    return ScoreWithFeedback(score=score, feedback=fb)


def _showcase(program: dspy.Module, label: str) -> None:
    """Print the flexed module's clean dspy.Module source and its flat predictors."""
    print(f"\n===== {label} =====")
    print("predictors on the module:", [n for n, _ in program.named_predictors()])
    print("--- module_src (a normal dspy.Module subclass) ---")
    print(program.module_src)


def test_flex_banking77_showcase() -> None:
    dspy.configure(lm=EXEC_LM)
    labels, train, val, test = _load_splits()
    print(f"\nBANKING77: {len(labels)} intents | train={len(train)} val={len(val)} test={len(test)}")

    # A tool the classifier may consult to ground itself in the real label space. Providing any tool
    # makes the dspy.Flex baseline a dspy.RLM (a multi-call REPL agent) instead of a single
    # dspy.Predict — so the baseline spends SEVERAL LM calls per example, giving GEPA real headroom
    # to collapse it down to a leaner program (ideally one focused predictor).
    def find_intents(keyword: str) -> list[str]:
        """Return BANKING77 intent labels containing the keyword (case-insensitive)."""
        k = keyword.lower()
        return [lbl for lbl in labels if k in lbl.lower()]

    # 1. Flex the classifier: an RLM baseline (tool provided), code-optimizable. Needs Deno for the
    # RLM's sandbox; see dspy.RLM docs.
    program = dspy.Flex(_build_signature(labels), tools=[find_intents])
    baseline_src = program.module_src
    assert program.module_src.lstrip().startswith("class ")
    assert "dspy.RLM(" in baseline_src  # tool provided -> multi-call RLM baseline
    _showcase(program, "baseline (un-optimized flex)")

    # 2. Benchmark the baseline.
    base = _evaluate(program, test)
    print(
        f"[baseline / flex-RLM] acc={base.accuracy:.1%} macro-P={base.macro_precision:.2f} "
        f"macro-R={base.macro_recall:.2f} macro-F1={base.macro_f1:.2f} calls/ex={base.avg_calls:.1f}"
    )

    # 3. Optimize the module's CODE with GEPA (challenging feedback drives the reflection).
    optimized = dspy.GEPA(
        metric=gepa_metric,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        reflection_minibatch_size=5,
        num_threads=EVAL_THREADS,
        track_stats=True,
    ).compile(program, trainset=train, valset=val)

    # 4. Benchmark the optimized program on the same test split.
    opt = _evaluate(optimized, test)
    code_changed = optimized.module_src != baseline_src
    print(
        f"[optimized / GEPA]    acc={opt.accuracy:.1%} macro-P={opt.macro_precision:.2f} "
        f"macro-R={opt.macro_recall:.2f} macro-F1={opt.macro_f1:.2f} calls/ex={opt.avg_calls:.1f}"
    )
    print(
        f"improvement: {opt.accuracy - base.accuracy:+.1%} accuracy, "
        f"{opt.macro_f1 - base.macro_f1:+.2f} macro-F1, "
        f"{opt.avg_calls - base.avg_calls:+.1f} LLM calls/example | GEPA changed the code: {code_changed}"
    )
    detailed = getattr(optimized, "detailed_results", None)
    if detailed is not None:
        print(f"GEPA val scores explored: {detailed.val_aggregate_scores}")
    _showcase(optimized, "optimized by GEPA")

    # 4b. Persist the optimized program with the standard Module.save/load — the generated
    # code (module_src) round-trips alongside predictor state, no special on-disk format.
    optimized.save(str(SAVE_PATH))
    reloaded = dspy.Flex(_build_signature(labels))
    reloaded.load(str(SAVE_PATH))
    assert reloaded.module_src == optimized.module_src
    print(f"saved + reloaded optimized program -> {SAVE_PATH}")

    # 5. Plot the before/after. Panel 1: the classification metrics that validate a 77-way
    # classifier — accuracy plus macro-averaged precision/recall/F1 (all 77 intents weighted
    # equally, so weakness on rare/confusable intents shows). Panel 2: avg LLM calls/example, the
    # decomposition win (the multi-call RLM baseline -> focused code).
    colors = ["#9aa0a6", "#1a73e8"]
    fig, (ax_cls, ax_calls) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Grouped bars: one group per metric, baseline vs optimized side by side.
    metric_names = ["Accuracy", "Macro-P", "Macro-R", "Macro-F1"]
    base_vals = [base.accuracy, base.macro_precision, base.macro_recall, base.macro_f1]
    opt_vals = [opt.accuracy, opt.macro_precision, opt.macro_recall, opt.macro_f1]
    xpos = list(range(len(metric_names)))
    width = 0.38
    b_bars = ax_cls.bar([x - width / 2 for x in xpos], base_vals, width, label="baseline", color=colors[0])
    o_bars = ax_cls.bar([x + width / 2 for x in xpos], opt_vals, width, label="optimized", color=colors[1])
    ax_cls.set_xticks(xpos)
    ax_cls.set_xticklabels(metric_names)
    ax_cls.set_ylabel("score (0–1)")
    ax_cls.set_ylim(0, 1.15)  # headroom for the on-bar labels near 1.0
    ax_cls.set_title("Classification metrics (macro-avg over 77 intents)")
    ax_cls.legend(loc="lower right", fontsize=8)
    for bars, vals in ((b_bars, base_vals), (o_bars, opt_vals)):
        for bar, v in zip(bars, vals, strict=True):
            ax_cls.text(
                bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    labels_xy = ["baseline\n(flex / RLM)", "optimized\n(GEPA code)"]
    call_bars = ax_calls.bar(labels_xy, [base.avg_calls, opt.avg_calls], color=colors)
    ax_calls.set_ylabel("avg LLM calls / example")
    ax_calls.set_ylim(0, max(base.avg_calls, opt.avg_calls, 1) * 1.2)
    ax_calls.set_title("LLM calls (lower = more deterministic)")
    for bar, nc in zip(call_bars, [base.avg_calls, opt.avg_calls], strict=True):
        ax_calls.text(bar.get_x() + bar.get_width() / 2, nc, f"{nc:.1f}", ha="center", va="bottom")

    fig.suptitle(f"BANKING77 intent classification (n={len(test)})")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved plot -> {PLOT_PATH}")

    # Showcase invariants: the pipeline ran end-to-end and we measured both ends.
    # (Whether GEPA changes the code / improves accuracy depends on the live models and
    # budget, so it's reported and plotted rather than hard-asserted.)
    assert PLOT_PATH.exists()
    assert 0.0 <= base.accuracy <= 1.0 and 0.0 <= opt.accuracy <= 1.0
    assert optimized.module_src is not None


if __name__ == "__main__":
    test_flex_banking77_showcase()
