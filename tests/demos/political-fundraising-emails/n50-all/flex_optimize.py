from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import NamedTuple

import pytest
from dotenv import load_dotenv

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()

# dspy registers a lazy numpy proxy in sys.modules; matplotlib's `from numpy.exceptions import ...`
# trips that proxy into a recursive import. Materialize the real numpy first (same guard the
# conflation demo uses — this demo also depends on neither pandas nor datasets).
np = pytest.importorskip("numpy")
_ = np.ndarray  # force the proxy to load the real module before matplotlib imports it
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

DEMO_DIR = Path(__file__).parent
DATA_DIR = DEMO_DIR.parent / "eval_data"
TRAIN_PATH = DATA_DIR / "train.jsonl"
TEST_PATH = DATA_DIR / "val.jsonl"  # held out — never seen by GEPA
SAVE_PATH = DEMO_DIR / "committee_flex.json"
PLOT_PATH = DEMO_DIR / "committee_improvement.png"

EXEC_LM = dspy.LM("anthropic/claude-opus-4-7")
STRONG_LM = dspy.LM("anthropic/claude-opus-4-8")
dspy.configure(lm=EXEC_LM)

# GEPA sees only train.jsonl (split into a train pool it minibatches over + a val pool it scores
# candidates on). train + val together cover ALL of train.jsonl: we hold out N_VAL rows for
# candidate selection and train on every remaining row, so no example is dropped and the two never
# overlap. val.jsonl is a fully held-out test set — and importantly 19 of its committees never
# appear in train, so a program can't win by memorizing labels; it has to actually read the email
# text. A bigger val pool makes candidate selection less noisy (a lucky 1-of-10 run won't win).
N_VAL = 20  # held out of train.jsonl for candidate selection; the rest becomes the train pool
N_TEST = 50  # all of val.jsonl
MAX_METRIC_CALLS = 250  # room for several code-proposal rounds to converge toward the recoverable ceiling
REFLECTION_MINIBATCH = 4
EVAL_THREADS = 8

# Score = accuracy − LLM_CALL_PENALTY * (#LLM calls). The un-optimized baseline spends exactly one
# traced call per email, so its score is capped at (accuracy − 0.20). The penalty plus the metric
# feedback push toward the target: a correct answer produced by DETERMINISTIC code with zero LLM
# calls. The feedback names that goal outright but never says HOW to reach it — what to read in the
# raw text, when to fall back to an LLM — leaving the algorithm for GEPA to discover from the emails
# and gold labels.
LLM_CALL_PENALTY = 0.20


class IdentifyCommittee(dspy.Signature):
    """Identify the political committee that sponsored a fundraising email.

    The committee's name is present in the email text itself, so most emails can be resolved
    deterministically in code with no model call; reserve a model call only as a fallback for
    emails the code cannot resolve.
    """

    email_body: str = dspy.InputField(desc="Full parsed text of the political fundraising email.")
    committee: str = dspy.OutputField(desc="Legal name of the sponsoring political committee.")


def _norm(s: str) -> str:
    """Lowercase and strip everything but [a-z0-9] — kills the PDF-parser whitespace glitches
    ('FRIENDS OFSHERROD BROWN'), punctuation ('INC.' vs 'INC'), and casing noise in the labels."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _match(pred: str, gold: str) -> bool:
    """True if the predicted committee matches gold up to normalization / a legal suffix.

    Tolerant of parser whitespace, punctuation, case, and 'INC.'-style suffixes, but strict
    enough to reject a whole-disclaimer dump: a substring only counts when it covers >=70% of
    the longer string, otherwise we require a >=0.9 character-ratio.
    """
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return p == g
    if p == g:
        return True
    short, long = (p, g) if len(p) <= len(g) else (g, p)
    if short in long and len(short) / len(long) >= 0.7:
        return True
    return SequenceMatcher(None, p, g).ratio() >= 0.9


def _pred_committee(prediction) -> str:
    value = getattr(prediction, "committee", None)
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _to_example(row: dict) -> dspy.Example:
    return dspy.Example(
        email_body=row["email_body"],
        committee=row["committee"],
    ).with_inputs("email_body")


def _load_splits() -> tuple[list, list, list]:
    def read(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    train_rows = read(TRAIN_PATH)
    test_rows = read(TEST_PATH)
    rng = random.Random(0)
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    # train + val partition ALL of train.jsonl: hold out the first N_VAL rows for candidate
    # selection, train GEPA on every remaining row. No overlap, nothing dropped.
    val = [_to_example(r) for r in train_rows[:N_VAL]]
    train = [_to_example(r) for r in train_rows[N_VAL:]]
    test = [_to_example(r) for r in test_rows[:N_TEST]]
    return train, val, test


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None) -> ScoreWithFeedback:
    """Score = correctness − a per-LLM-call penalty. The feedback names the objective — a correct
    answer produced by DETERMINISTIC code with NO LLM call — and treats a correct-but-LLM answer as
    leaving score on the table. It tells GEPA only that the committee is recoverable from the email
    text; it never says WHERE in the text or HOW to extract it, so GEPA still has to discover the
    algorithm itself from the full emails in its reflective dataset."""
    example, prediction = gold, pred
    # GEPA delivers the execution trace via `program_trace` at scoring time (declaring the
    # parameter opts in); `trace` still carries it on feedback calls and from _evaluate below.
    exec_trace = program_trace if program_trace is not None else trace
    n_calls = len(exec_trace) if exec_trace else 0  # predictor calls during this forward()

    pred_name_str = _pred_committee(prediction)
    if not pred_name_str.strip():
        return ScoreWithFeedback(
            score=0.0,
            feedback="No committee returned (score 0.00). Return dspy.Prediction(committee=<str>).",
        )

    gold_name = example.committee
    correct = _match(pred_name_str, gold_name)
    score = max(0.0, (1.0 if correct else 0.0) - LLM_CALL_PENALTY * n_calls)
    cost = f"{n_calls} LLM call(s), cost {LLM_CALL_PENALTY * n_calls:.2f}, score {score:.2f}"

    # The objective, stated without giving away the algorithm: a correct answer from deterministic
    # code with no LLM call. Naming only the data property (the name is in the text) keeps the "how"
    # — where to look, how to extract, when to fall back — for GEPA to discover.
    goal = (
        "Target: return the correct committee with a DETERMINISTIC, no-LLM algorithm. The committee "
        "name is present in the email text, so it is recoverable in pure Python — reserve an LLM call "
        "only for emails where code genuinely cannot recover it. (Where in the text it is, and how to "
        "extract it, is for you to work out.)"
    )
    if not correct and n_calls == 0:
        # The worst outcome: a confident deterministic guess that's wrong (0.00). Deferring THIS
        # email to the LLM would have scored ~(1 - penalty). Surface that asymmetry so GEPA learns to
        # route low-confidence cases to the fallback instead of guessing — without saying how to
        # detect low confidence or how to parse.
        fb = (
            f"WRONG with NO LLM call ({cost}) — the worst outcome. Predicted committee={pred_name_str!r}; "
            f"correct committee={gold_name!r}. A confident-but-wrong code guess scores 0.00, whereas "
            f"deferring THIS email to the LLM would have scored about {1 - LLM_CALL_PENALTY:.2f}. When the "
            f"code cannot recover the name with high confidence, fall back to the LLM instead of guessing. "
            + goal
        )
    elif not correct:
        fb = (
            f"Incorrect ({cost}). Predicted committee={pred_name_str!r}; correct committee={gold_name!r}. "
            + goal
        )
    elif n_calls == 0:
        fb = f"Ideal — correct with no LLM call ({cost}). This is the target; keep resolving emails like this in code."
    else:
        fb = (
            f"Correct but used an LLM call ({cost}); that costs {LLM_CALL_PENALTY * n_calls:.2f}. Full "
            f"score needs this SAME answer with zero LLM calls. " + goal
        )
    return ScoreWithFeedback(score=score, feedback=fb)


class EvalResult(NamedTuple):
    """Headline metrics for one program on one dataset (committee attribution)."""

    score: float            # mean metric score = accuracy − 0.20·(LLM calls); GEPA's objective
    accuracy: float         # fuzzy-match correctness rate (via _match); the headline
    macro_precision: float  # mean per-committee precision (all committees weighted equally)
    macro_recall: float     # mean per-committee recall (== per-committee accuracy)
    macro_f1: float         # mean per-committee F1
    avg_calls: float        # avg LLM calls / example — the decomposition (cost) win


def _evaluate(program: dspy.Module, dataset: list) -> EvalResult:
    """Return the metric score plus accuracy and macro-averaged precision/recall/F1 and avg calls.

    Score is the number GEPA optimizes — accuracy minus the 0.20-per-call penalty — not raw
    accuracy; the Predict baseline reads every email with one traced call, so its score is capped
    below its accuracy, and GEPA's win is settling clear cases in deterministic Python at 0 calls.

    This is many-way attribution (one committee per email), so precision/recall/F1 are
    **macro-averaged** over committees — computed per committee, then unweighted mean, weighing every
    committee equally rather than letting the few repeated ones dominate. Committees are matched
    *fuzzily* (`_match` tolerates parser/punctuation/suffix noise), so each prediction is canonicalized
    to the committee it matches: its own gold when correct, else the first other known committee it
    matches, else its own (wrong) bucket. NB: the test set is singleton-heavy (most committees appear
    once), so macro-recall tracks accuracy closely; the metrics diverge on the repeated committees and
    on wrong-committee confusions (which cost precision).
    """

    def run_one(ex):
        try:
            with dspy.context(trace=[]):
                pred = program(**ex.inputs())
                trace = list(dspy.settings.trace or [])
            score = float(metric(ex, pred, trace=trace).score)
            return score, _pred_committee(pred), ex.committee, len(trace)
        except Exception:
            return 0.0, "", ex.committee, 0

    with ThreadPoolExecutor(max_workers=EVAL_THREADS) as pool:
        results = list(pool.map(run_one, dataset))
    n = len(dataset) or 1
    total_score = sum(s for s, _, _, _ in results)
    calls = sum(c for _, _, _, c in results)
    correct = sum(1 for _, pred, gold, _ in results if _match(pred, gold))

    # Macro precision/recall/F1 over committee classes. Because matching is fuzzy, canonicalize each
    # prediction to the committee it matches before building the confusion matrix.
    gold_variants = sorted({gold for _, _, gold, _ in results if gold}, key=_norm)
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    for _, pred, gold, _ in results:
        gold_c = _norm(gold)
        if _match(pred, gold):
            pred_c = gold_c
        else:
            pred_c = next((_norm(g) for g in gold_variants if _match(pred, g)), _norm(pred))
        if pred_c == gold_c:
            tp[gold_c] += 1
        else:
            fn[gold_c] += 1
            fp[pred_c] += 1  # pred_c may be "" (empty/error) — a bucket we never average over
    true_classes = {_norm(gold) for _, _, gold, _ in results}
    precisions, recalls, f1s = [], [], []
    for c in true_classes:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    k = len(true_classes) or 1
    return EvalResult(
        score=total_score / n,
        accuracy=correct / n,
        macro_precision=sum(precisions) / k,
        macro_recall=sum(recalls) / k,
        macro_f1=sum(f1s) / k,
        avg_calls=calls / n,
    )


def _showcase(program: dspy.Module, label: str) -> None:
    """Print the flexed module's clean dspy.Module source and its flat predictors."""
    print(f"\n----- {label} -----")
    print("predictors on the module:", [n for n, _ in program.named_predictors()])
    print(program.module_src)


def test_loader():
    program = dspy.Flex(IdentifyCommittee)
    program.load(str(SAVE_PATH))
    return program


def test_flex_political() -> None:
    dspy.configure(lm=EXEC_LM)
    train, val, test = _load_splits()
    print(f"splits: train={len(train)} val={len(val)} test={len(test)} (test committees held out)")

    program = dspy.Flex(IdentifyCommittee)

    # Fresh baseline: a clean dspy.Module subclass delegating to one dspy.Predict.
    base_src = program.module_src or ""
    assert base_src.lstrip().startswith("class ")
    assert "dspy.Predict(" in base_src
    _showcase(program, "baseline (un-optimized flex)")

    base = _evaluate(program, test)
    print(
        f"[baseline] score={base.score:.2f} acc={base.accuracy:.2f} "
        f"macro-P={base.macro_precision:.2f} macro-R={base.macro_recall:.2f} "
        f"macro-F1={base.macro_f1:.2f} calls/ex={base.avg_calls:.2f}"
    )

    optimized = dspy.GEPA(
        metric=metric,
        reflection_lm=STRONG_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        reflection_minibatch_size=REFLECTION_MINIBATCH,
        num_threads=EVAL_THREADS,
    ).compile(program, trainset=train, valset=val)

    # The penalty pushes logic into plain Python — watch module_src shift from one Predict to a
    # disclaimer parser + a reserved LLM fallback.
    _showcase(optimized, "optimized by GEPA")
    print(f"GEPA changed the code: {optimized.module_src != program.module_src}")

    opt = _evaluate(optimized, test)
    print(
        f"[optimized] score={opt.score:.2f} acc={opt.accuracy:.2f} "
        f"macro-P={opt.macro_precision:.2f} macro-R={opt.macro_recall:.2f} "
        f"macro-F1={opt.macro_f1:.2f} calls/ex={opt.avg_calls:.2f}"
    )
    print(
        f"score improvement: {opt.score - base.score:+.2f}  |  "
        f"macro-F1 improvement: {opt.macro_f1 - base.macro_f1:+.2f}"
    )

    # Persist with the standard Module.save/load (code round-trips).
    optimized.save(str(SAVE_PATH))
    reloaded = dspy.Flex(IdentifyCommittee)
    reloaded.load(str(SAVE_PATH))
    assert reloaded.module_src == optimized.module_src
    print(f"saved + reloaded optimized program -> {SAVE_PATH}")

    # Before/after plot (a la conflation). Panel 1: the score GEPA optimizes (accuracy −
    # 0.20/LLM-call). Panel 2: the classification metrics that validate the attribution —
    # accuracy plus macro-averaged precision/recall/F1 over committees (each committee weighted
    # equally). Panel 3: LLM calls/example, the decomposition win (one-call Predict ->
    # deterministic disclaimer parsing).
    labels_xy = ["baseline\n(flex / Predict)", "optimized\n(GEPA code)"]
    colors = ["#9aa0a6", "#1a73e8"]
    fig, (ax_score, ax_cls, ax_calls) = plt.subplots(1, 3, figsize=(13, 4.5))

    score_bars = ax_score.bar(labels_xy, [base.score, opt.score], color=colors)
    ax_score.set_ylabel("mean metric score")
    ax_score.set_ylim(0, 1.1)
    ax_score.set_title("Score (accuracy − call penalty)")
    for bar, s in zip(score_bars, [base.score, opt.score], strict=True):
        ax_score.text(bar.get_x() + bar.get_width() / 2, s + 0.02, f"{s:.2f}", ha="center", va="bottom")

    # Grouped bars: one group per metric, baseline vs optimized side by side.
    metric_names = ["Accuracy", "Macro-P", "Macro-R", "Macro-F1"]
    base_vals = [base.accuracy, base.macro_precision, base.macro_recall, base.macro_f1]
    opt_vals = [opt.accuracy, opt.macro_precision, opt.macro_recall, opt.macro_f1]
    xpos = np.arange(len(metric_names))
    width = 0.38
    b_bars = ax_cls.bar(xpos - width / 2, base_vals, width, label="baseline", color=colors[0])
    o_bars = ax_cls.bar(xpos + width / 2, opt_vals, width, label="optimized", color=colors[1])
    ax_cls.set_xticks(xpos)
    ax_cls.set_xticklabels(metric_names)
    ax_cls.set_ylabel("score (0–1)")
    ax_cls.set_ylim(0, 1.15)  # headroom for the on-bar labels near 1.0
    ax_cls.set_title("Classification metrics (macro-avg over committees)")
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

    fig.suptitle(f"Political email committee attribution (n={len(test)})")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved plot -> {PLOT_PATH}")
    assert PLOT_PATH.exists()


if __name__ == "__main__":
    test_flex_political()
