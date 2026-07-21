from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

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
DATA_DIR = DEMO_DIR / "eval_data"
TRAIN_PATH = DATA_DIR / "train.jsonl"
TEST_PATH = DATA_DIR / "val.jsonl"  # held out — never seen by GEPA
SAVE_PATH = DEMO_DIR / "committee_flex.json"
PLOT_PATH = DEMO_DIR / "committee_improvement.png"

EXEC_LM = dspy.LM("anthropic/claude-opus-4-7", max_tokens=2000)
STRONG_LM = dspy.LM("anthropic/claude-opus-4-8", max_tokens=8000)
dspy.configure(lm=EXEC_LM)

# GEPA sees only train.jsonl (split into a train pool it minibatches over + a val pool it scores
# candidates on). train + val together cover ALL of train.jsonl: we hold out N_VAL rows for
# candidate selection and train on every remaining row, so no example is dropped and the two never
# overlap. val.jsonl is a fully held-out test set — and importantly 19 of its committees never
# appear in train, so a program can't win by memorizing labels; it has to actually read the email
# text. A bigger val pool makes candidate selection less noisy (a lucky 1-of-10 run won't win).
N_VAL = 20  # held out of train.jsonl for candidate selection; the rest becomes the train pool
N_TEST = 50  # all of val.jsonl
MAX_METRIC_CALLS = 100  # room for several code-proposal rounds to converge toward the recoverable ceiling
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


def _evaluate(program: dspy.Module, dataset: list) -> tuple[float, float, float]:
    """Return (mean metric score, accuracy, avg LLM calls/example).

    The headline number is the metric score GEPA optimizes — accuracy minus the 0.15-per-call
    penalty — not raw accuracy. The un-optimized Predict baseline reads every email with one
    traced LLM call, so its score is capped below its accuracy. GEPA's win is settling the clear
    cases (a disclaimer is present) in deterministic Python at 0 calls.
    """

    def run_one(ex):
        try:
            with dspy.context(trace=[]):
                pred = program(**ex.inputs())
                trace = list(dspy.settings.trace or [])
            score = float(metric(ex, pred, trace=trace).score)
            ok = _match(_pred_committee(pred), ex.committee)
            return score, int(ok), len(trace)
        except Exception:
            return 0.0, 0, 0

    with ThreadPoolExecutor(max_workers=EVAL_THREADS) as pool:
        results = list(pool.map(run_one, dataset))
    n = len(dataset)
    total_score = sum(s for s, _, _ in results)
    correct = sum(ok for _, ok, _ in results)
    calls = sum(c for _, _, c in results)
    return total_score / n, correct / n, calls / n


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

    base_score, base_acc, base_calls = _evaluate(program, test)
    print(
        f"[baseline] score={base_score:.2f} "
        f"(accuracy={base_acc:.2f}, avg LLM calls/example={base_calls:.2f})"
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

    opt_score, opt_acc, opt_calls = _evaluate(optimized, test)
    print(
        f"[optimized] score={opt_score:.2f} "
        f"(accuracy={opt_acc:.2f}, avg LLM calls/example={opt_calls:.2f})"
    )
    print(f"score improvement: {opt_score - base_score:+.2f}")

    # Persist with the standard Module.save/load (code round-trips).
    optimized.save(str(SAVE_PATH))
    reloaded = dspy.Flex(IdentifyCommittee)
    reloaded.load(str(SAVE_PATH))
    assert reloaded.module_src == optimized.module_src
    print(f"saved + reloaded optimized program -> {SAVE_PATH}")

    # Before/after plot (a la conflation), one panel per metric. Score is what GEPA optimizes
    # (accuracy − 0.15/LLM-call); accuracy shows it's held; LLM calls/example shows the
    # decomposition win (one-call Predict -> deterministic disclaimer parsing).
    labels_xy = ["baseline\n(flex / Predict)", "optimized\n(GEPA code)"]
    colors = ["#9aa0a6", "#1a73e8"]
    fig, (ax_score, ax_acc, ax_calls) = plt.subplots(1, 3, figsize=(11, 4))

    score_bars = ax_score.bar(labels_xy, [base_score, opt_score], color=colors)
    ax_score.set_ylabel("mean metric score")
    ax_score.set_ylim(0, 1.1)
    ax_score.set_title("Score (accuracy − call penalty)")
    for bar, s in zip(score_bars, [base_score, opt_score], strict=True):
        ax_score.text(bar.get_x() + bar.get_width() / 2, s + 0.02, f"{s:.2f}", ha="center", va="bottom")

    acc_bars = ax_acc.bar(labels_xy, [base_acc, opt_acc], color=colors)
    ax_acc.set_ylabel("test accuracy")
    ax_acc.set_ylim(0, 1.1)
    ax_acc.set_title("Accuracy")
    for bar, a in zip(acc_bars, [base_acc, opt_acc], strict=True):
        ax_acc.text(bar.get_x() + bar.get_width() / 2, a + 0.02, f"{a:.1%}", ha="center", va="bottom")

    call_bars = ax_calls.bar(labels_xy, [base_calls, opt_calls], color=colors)
    ax_calls.set_ylabel("avg LLM calls / example")
    ax_calls.set_ylim(0, max(base_calls, opt_calls, 1) * 1.2)
    ax_calls.set_title("LLM calls")
    for bar, n in zip(call_bars, [base_calls, opt_calls], strict=True):
        ax_calls.text(bar.get_x() + bar.get_width() / 2, n, f"{n:.1f}", ha="center", va="bottom")

    fig.suptitle(f"Political email committee attribution (n={len(test)})")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved plot -> {PLOT_PATH}")
    assert PLOT_PATH.exists()


if __name__ == "__main__":
    test_flex_political()
