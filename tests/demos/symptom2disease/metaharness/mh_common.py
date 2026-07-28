"""Head-to-head with Meta-Harness on its own Symptom2Disease split.

Meta-Harness (Lee, Nair, Zhang, Lee, Khattab, Finn -- arXiv 2603.28052, "Meta-Harness: End-to-End
Optimization of Model Harnesses") searches over *harnesses*: the code around a fixed base model that
decides what to store, retrieve and show. Its text-classification experiment reports **86.8% on
Symptom2Disease**, against zero-shot 63.2 / few-shot(32) 72.2 / few-shot(all) 78.3 / ACE 77.8 /
MCE 83.0.

Two things were read out of its reference implementation
(github.com/stanford-iris-lab/meta-harness, `reference_examples/text_classification`) rather than
guessed, because both decide what a fair comparison looks like:

**1. The data.** `data/symptom_diagnosis/{train,val,test}.jsonl` -- 200 / 50 / 212 over **22**
classes (the Kaggle 24 minus Acne and Dimorphic Hemorrhoids), fixed by `config.yaml`
(`Symptom2Disease: {num_train: 200, num_val: 50, num_test: 212}`). Those three files are vendored
verbatim into `eval_data/` and read-only. Using the paper's own split is the whole point: the
sibling demo's 24-class / 360-example split produces a number that cannot be put next to 86.8.

**2. The scoring function.** `data/evaluators.py::eval_symptom2disease` is a normalized exact-string
match returning a bool. That is all of it -- **plain 0/1 accuracy, no penalty term.** Context length
enters only as a *secondary* Pareto axis ("Ctx" in Table 2), never scalarized into the score. So the
LLM-call penalty this demo family has been sweeping (`max(0, correct - lambda * n_calls)`) is not
the objective Meta-Harness optimized, and any run meant to beat 86.8 should drop it.

Dropping it, though, only recovers the paper's objective -- it does not beat it. Under DSPy the
metric is not just a number: GEPA consumes `ScoreWithFeedback`, and the *feedback* string is what
the reflection LM reads when it rewrites the program. That gives a lever Meta-Harness's scalar
`bool` does not have, and this module implements two metrics that differ **only** in that string:

  * `metric_accuracy`  -- the control. Score = 1/0 exact match; feedback names the wrong label.
    This is the paper's scoring function, transcribed.
  * `metric_contrastive` -- identical score, richer feedback: the confusion is contrasted against
    labelled *training* exemplars of both classes, plus a running confusion table and a coverage
    report over classes never yet answered correctly.

The score is byte-identical between the two arms, so the experiment isolates the feedback channel
and nothing else. The mechanisms the feedback is meant to induce are the ones the paper's own
discovered harness ("Label-Primed Query": label primer, coverage block, query-anchored contrastive
pairs) converged on -- the difference is that here they have to be *discovered* by GEPA from metric
feedback rather than assembled by the proposer.

Caveats that belong next to any number this produces, stated once here:
  * **Base model differs.** Meta-Harness ran `openrouter/openai/gpt-oss-120b` at temperature 0.
    No OpenRouter credentials exist in this repo, so the executor is `claude-haiku-4-5`, as in the
    sibling demos. This is the load-bearing confound; `run_compare.py` therefore always measures the
    un-optimized zero-shot baseline on the *same* 212 examples, so the model's contribution and the
    optimizer's contribution can be read separately.
  * **Output is enum-constrained.** `disease` is a `Literal` over the 22 labels, so an off-label
    string is impossible; the paper's harness emits free text into a `[DIAGNOSIS]...[/DIAGNOSIS]`
    tag and can lose points to formatting. Measured on the sibling demo, that was worth a lot
    (0.475 -> 0.725), so it is a real advantage and is reported, not buried.
  * **One example of the paper's own val set (1/212) also appears in its test set.** It is their
    split; both methods inherit it. Reported by `check_data()`, not silently repaired.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

sys.path.insert(0, str(Path(__file__).parent.parent))
from s2d_common import (
    EXEC_MODEL,
    REFLECTION_MODEL,
    disable_cache,
    make_lms,
    meter,
    pred_disease,
    run_program,
    summarize,
)

MH_DIR = Path(__file__).parent
DATA_DIR = MH_DIR / "eval_data"

# Provenance for anything written into the results JSON.
PAPER = {
    "title": "Meta-Harness: End-to-End Optimization of Model Harnesses",
    "arxiv": "2603.28052",
    "code": "github.com/stanford-iris-lab/meta-harness",
    "split_source": "reference_examples/text_classification/data/symptom_diagnosis/*.jsonl",
    "scoring_source": "reference_examples/text_classification/data/evaluators.py::eval_symptom2disease",
    "their_model": "openrouter/openai/gpt-oss-120b @ temperature 0.0",
    # Table 2, Symptom2Disease column.
    "s2d_table2": {"zero_shot": 63.2, "fewshot_8": 67.9, "fewshot_32": 72.2,
                   "fewshot_all": 78.3, "MCE": 83.0, "ACE": 77.8, "meta_harness": 86.8},
}
TARGET = PAPER["s2d_table2"]["meta_harness"] / 100.0

__all__ = [
    "DATA_DIR", "EXEC_MODEL", "LABELS", "PAPER", "REFLECTION_MODEL", "TARGET",
    "DiagnoseFromSymptoms", "METRICS", "canonical", "check_data", "disable_cache",
    "load_paper_splits", "make_lms", "meter", "paper_eval", "run_program", "summarize",
]


# ---------------------------------------------------------------------------
# Data -- the paper's split, verbatim
# ---------------------------------------------------------------------------


def _read(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}; see the module docstring for provenance")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


_RAW = {s: _read(s) for s in ("train", "val", "test")} if DATA_DIR.exists() else {}
LABELS: list[str] = sorted({r["answer"] for r in _RAW.get("test", [])})


def _to_example(row: dict) -> dspy.Example:
    return dspy.Example(symptoms=row["question"], disease=row["answer"]).with_inputs("symptoms")


def load_paper_splits() -> tuple[list, list, list]:
    """(train, val, test) exactly as Meta-Harness defines them: 200 / 50 / 212, order preserved.

    No reshuffling and no restratifying. The comparison is only worth making if the 212 test
    examples are the same 212 examples, and re-deriving splits from the raw Kaggle CSV would not
    reproduce them -- the paper's texts are largely a distinct corpus (only 5 of its 212 test items
    appear in this demo's own manifest).
    """
    return ([_to_example(r) for r in _RAW["train"]],
            [_to_example(r) for r in _RAW["val"]],
            [_to_example(r) for r in _RAW["test"]])


def check_data() -> dict:
    """Split hygiene, reported rather than repaired -- these are the paper's files."""
    def keys(split): return {re.sub(r"[^a-z0-9]+", " ", r["question"].lower()).strip() for r in _RAW[split]}
    tr, va, te = keys("train"), keys("val"), keys("test")
    return {
        "n_train": len(_RAW["train"]), "n_val": len(_RAW["val"]), "n_test": len(_RAW["test"]),
        "n_classes": len(LABELS),
        "dupes_within_test": len(_RAW["test"]) - len(te),
        "train_test_overlap": len(tr & te),
        "val_test_overlap": len(va & te),  # 1 in the paper's own files
        "train_val_overlap": len(tr & va),
    }


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------


class DiagnoseFromSymptoms(dspy.Signature):
    """Given a patient's description of their symptoms, name the disease."""

    # One line, deliberately. Strategy in the seed prompt is task-solving information the optimizer
    # should have had to discover, and it inflates the baseline -- a lesson the political-emails
    # demo in this family paid for. The label set is not strategy: it is the task definition, and
    # Meta-Harness's own zero-shot prompt lists all 22 diagnoses too ("Possible diagnoses include:
    # drug reaction, allergy, chicken pox, ...", loaders.py::_load_symptom2disease).
    symptoms: str = dspy.InputField(desc="A patient's free-text description of their symptoms.")
    disease: Literal[tuple(LABELS)] = dspy.OutputField()  # type: ignore[valid-type]


# ---------------------------------------------------------------------------
# Label matching -- and the paper's evaluator, kept alongside as a cross-check
# ---------------------------------------------------------------------------


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


_NORM_TO_LABEL = {norm(x): x for x in LABELS}


def canonical(pred: str) -> str:
    """Map a prediction onto one of the 22 labels, or "" if it is not one of them."""
    p = norm(pred)
    if not p:
        return ""
    if p in _NORM_TO_LABEL:
        return _NORM_TO_LABEL[p]
    hits = [lab for key, lab in _NORM_TO_LABEL.items() if key in p]
    return hits[0] if len(hits) == 1 else ""


def paper_eval(prediction: str, target: str) -> bool:
    """`evaluators.py::eval_symptom2disease`, transcribed.

    Runs beside `canonical` so the headline number can be reported under the paper's *own* matcher
    as well as ours. They can only disagree when the model emits something outside the enum, which
    the `Literal` makes impossible -- `run_compare.py` asserts the two agree on every example, which
    is what turns "we used their evaluator" from a claim into a check.
    """
    text = prediction or ""
    match = re.search(r"\[DIAGNOSIS\](.*?)\[/DIAGNOSIS\]", text, re.I | re.S)
    if match:
        text = match.group(1).strip()
    else:
        match = re.search(r"(?:diagnosis|final diagnosis|conclusion)[:：]\s*([^\n]+)", text, re.I)
        if match:
            text = match.group(1).strip()

    def normalize(value: str) -> str:
        value = re.sub(r"\s+", " ", value.lower().strip())
        return re.sub(r"[.!?]+$", "", value)

    return normalize(text) == normalize(target)


# ---------------------------------------------------------------------------
# Training exemplars, for the contrastive feedback
# ---------------------------------------------------------------------------

# Train split only. The 50 val examples GEPA scores against are excluded so feedback can never
# quote the thing being graded, and the 212 test examples are never loaded here at all.
_BY_LABEL: dict[str, list[str]] = defaultdict(list)
for _row in _RAW.get("train", []):
    _BY_LABEL[_row["answer"]].append(_row["question"])
for _lab in _BY_LABEL:
    _BY_LABEL[_lab].sort()  # deterministic: no RNG in the metric


def exemplars(label: str, k: int = 2) -> list[str]:
    return _BY_LABEL.get(label, [])[:k]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
#
# Both return score = 1.0 / 0.0 on normalized exact match -- the paper's objective, no penalty.
# They differ only in `feedback`, which is the channel Meta-Harness's scalar bool does not have.


def _score(gold, pred) -> tuple[str, str, str, bool]:
    raw = pred_disease(pred)
    predicted = canonical(raw)
    return raw, predicted, gold.disease, predicted == gold.disease


def metric_accuracy(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
    """CONTROL: the paper's scoring function. 1/0 accuracy; feedback names the miss and stops."""
    raw, predicted, gold_label, correct = _score(gold, pred)
    if not raw.strip():
        return ScoreWithFeedback(score=0.0, feedback="No disease returned. Return dspy.Prediction(disease=<str>).")
    if not predicted:
        return ScoreWithFeedback(
            score=0.0,
            feedback=f"OFF-LABEL: returned {raw!r}, which is not one of the 22 allowed labels "
                     f"(correct was {gold_label!r}). Answer with exactly one label, verbatim.")
    if not correct:
        return ScoreWithFeedback(
            score=0.0,
            feedback=f"WRONG: predicted {predicted!r}, correct is {gold_label!r}.\n"
                     f"Symptom text: {gold.symptoms}")
    return ScoreWithFeedback(score=1.0, feedback="Correct. Keep whatever produced this.")


class _ConfusionState:
    """Running (gold -> predicted) tally shared across a GEPA run.

    A single metric call sees one example; what actually fixes a 22-way classifier is knowing which
    confusions are *systematic*. GEPA evaluates minibatches and valsets through a thread pool, so
    this is locked. It is monotonic -- never reset -- which is deliberate: the reflection LM should
    see the confusion structure accumulated over the whole search, not just the current minibatch.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.pairs: Counter = Counter()      # (gold, predicted) -> n, wrong answers only
        self.seen: Counter = Counter()       # gold -> n
        self.hit: Counter = Counter()        # gold -> n correct

    def record(self, gold: str, predicted: str, correct: bool) -> None:
        with self._lock:
            self.seen[gold] += 1
            if correct:
                self.hit[gold] += 1
            elif predicted:
                self.pairs[(gold, predicted)] += 1

    def top_confusions(self, k: int = 5, min_n: int = 2) -> list[tuple[str, str, int]]:
        with self._lock:
            items = [(g, p, n) for (g, p), n in self.pairs.most_common() if n >= min_n]
        return items[:k]

    def never_right(self, min_attempts: int = 2) -> list[tuple[str, int]]:
        with self._lock:
            return sorted((lab, n) for lab, n in self.seen.items()
                          if n >= min_attempts and self.hit[lab] == 0)


def make_contrastive_metric(n_exemplars: int = 2, n_confusions: int = 5):
    """NEW: same 1/0 score, feedback carrying the evidence needed to fix the *class boundary*.

    Three additions over the control, each aimed at a mechanism the paper's discovered harness
    ("Label-Primed Query") ended up containing -- here they have to be discovered rather than
    assembled:

      1. *Contrastive exemplars.* A miss names two classes; the feedback shows labelled TRAIN
         examples of both, side by side. "Predicted typhoid, correct is malaria" is not actionable
         on its own -- the reflection LM cannot know what separates them without seeing them. This
         is the metric-side analogue of the paper's query-anchored contrastive pairs.
      2. *Running confusion table.* Which mistakes repeat, across the whole search so far. A
         5-example minibatch over 22 classes cannot show this, so without it the reflector is
         chasing singletons.
      3. *Coverage report.* Classes attempted but never once correct -- the failure mode the
         paper's label primer and per-class coverage block exist to prevent.

    Cost: feedback strings get longer, so the reflective dataset does too. Exemplars here average
    ~150 characters and only wrong answers carry them, which is why REFLECTION_MAX_TOKENS=32000 in
    `s2d_common` matters -- an 8000-token cap once truncated a proposal mid-class and the whole run
    looked like "the optimizer found nothing".
    """
    state = _ConfusionState()

    def _contrast(gold_label: str, predicted: str) -> str:
        lines = ["\nWhat each of these two actually looks like (labelled training cases):"]
        for label in (gold_label, predicted):
            for text in exemplars(label, n_exemplars):
                lines.append(f"  [{label}] {text}")
        lines.append(f"Encode what separates {gold_label!r} from {predicted!r}. A rule that fixes "
                     f"this pair without breaking the other 20 classes is worth more than a rule "
                     f"that memorises this one case.")
        return "\n".join(lines)

    def _global(prefix: str = "") -> str:
        parts = []
        confusions = state.top_confusions(n_confusions)
        if confusions:
            parts.append("Repeated confusions so far: "
                         + "; ".join(f"{g!r} called {p!r} x{n}" for g, p, n in confusions))
        missing = state.never_right()
        if missing:
            parts.append("Never once correct: "
                         + ", ".join(f"{lab!r} (0/{n})" for lab, n in missing))
        return (prefix + "\n".join(parts)) if parts else ""

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
        raw, predicted, gold_label, correct = _score(gold, pred)
        state.record(gold_label, predicted, correct)

        if not raw.strip():
            return ScoreWithFeedback(
                score=0.0, feedback="No disease returned. Return dspy.Prediction(disease=<str>).")
        if not predicted:
            return ScoreWithFeedback(
                score=0.0,
                feedback=f"OFF-LABEL: returned {raw!r}, which is not one of the 22 allowed labels "
                         f"(correct was {gold_label!r}). Answer with exactly one label, verbatim. "
                         f"If code picks the label, have it choose from the fixed list rather than "
                         f"generating a name.")
        if not correct:
            return ScoreWithFeedback(
                score=0.0,
                feedback=(f"WRONG: predicted {predicted!r}, correct is {gold_label!r}.\n"
                          f"Symptom text: {gold.symptoms}"
                          + _contrast(gold_label, predicted)
                          + _global("\n")))
        return ScoreWithFeedback(
            score=1.0,
            feedback=(f"Correct ({gold_label!r}). Keep whatever produced this."
                      + _global("\n")))

    metric.state = state  # type: ignore[attr-defined]  # driver dumps the final confusion table
    return metric


METRICS = {
    "accuracy": lambda: metric_accuracy,          # control = the paper's scoring function
    "contrastive": make_contrastive_metric,       # the change under test
}
