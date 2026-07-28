"""Shared pieces for the Symptom2Disease Flex+GEPA demo.

Third classification-style demo in the series, after `tests/demos/conflation/all` (binary) and
`tests/demos/political-fundraising-emails/n50-all` (string extraction). Same machinery: a penalized
objective, the same CAL (cost / accuracy / latency) instrumentation, the same figure.

What is different here is the *shape of the task*, and it is the shape that makes the penalty
interesting. This is **24-way closed-set classification over short free-text symptom descriptions**
(~42 tokens each). Both earlier demos had a deterministic path available — a disclaimer line to
regex, a street number to compare — so "spend fewer LLM calls" had an obvious mechanism. Here the
deterministic path is keyword evidence over a 24-class vocabulary, which may or may not be enough.

Every fix the previous two demos paid for is carried forward:
  * dedup before splitting (a duplicate straddling train/test is leakage) -- done in fetch_data.py;
  * a val slice checked for representativeness, not just taken as a contiguous shuffle;
  * reflection max_tokens well clear of the cap, plus a `finish_reason == "length"` detector,
    because silent truncation once destroyed a whole run and looked like "the optimizer found
    nothing";
  * per-example records persisted so any metric can be recomputed without re-running;
  * per-request latency reported separately from wall-clock throughput.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, NamedTuple

from dotenv import load_dotenv

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()

DEMO_DIR = Path(__file__).parent
MANIFEST_PATH = DEMO_DIR / "eval_data" / "symptom2disease.jsonl"

# Chosen by pilot in the previous two demos and re-probed here by `pilot.py`. Haiku matched Opus
# example-for-example on the emails task at 6.5x lower cost; these inputs are ~42 tokens, so the
# execution side is close to free either way and the reflection LM dominates the bill.
EXEC_MODEL = "anthropic/claude-haiku-4-5"
REFLECTION_MODEL = "anthropic/claude-opus-5"

# USD per 1M tokens (input, output). litellm's own per-call cost is recorded alongside as
# `cost_usd_litellm`; this table exists so cost can be attributed to individual examples, which an
# aggregate history cannot do under a thread pool. The two agree to the cent.
PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
    "anthropic/claude-sonnet-5": (3.00, 15.00),
    "anthropic/claude-opus-5": (5.00, 25.00),
    "anthropic/claude-opus-4-8": (5.00, 25.00),
    "anthropic/claude-opus-4-7": (5.00, 25.00),
}

EXEC_MAX_TOKENS = 1000        # a class label plus dspy's field markers; measured ~15 tokens
REFLECTION_MAX_TOKENS = 32000  # see the module docstring: 8000 silently truncated a whole run


def make_lms(exec_model: str = EXEC_MODEL,
             exec_max_tokens: int = EXEC_MAX_TOKENS,
             reflection_max_tokens: int = REFLECTION_MAX_TOKENS) -> tuple[dspy.LM, dspy.LM]:
    """Return (execution LM, reflection LM) and make history unbounded for cost accounting."""
    # `meter()` slices lm.history by index; the default 10k cap pops from the FRONT, which would
    # shift those indices mid-run and silently drop calls from the totals.
    dspy.configure(max_history_size=10**9)
    return (dspy.LM(exec_model, max_tokens=exec_max_tokens),
            dspy.LM(REFLECTION_MODEL, max_tokens=reflection_max_tokens))


def disable_cache() -> None:
    """Turn off dspy's caches so latency and cost are what a cold production call would be."""
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_rows() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"no manifest at {MANIFEST_PATH}; run `python fetch_data.py` first")
    return [json.loads(line) for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


ROWS = load_rows() if MANIFEST_PATH.exists() else []
LABELS: list[str] = sorted({r["label"] for r in ROWS})

# Per class. The benchmark is balanced by construction (50 each before dedup, 38-50 after), so
# stratifying keeps every split balanced and makes chance exactly 1/24 = 4.2% everywhere.
N_TEST_PER_CLASS = 15   # 24 x 15 = 360 -> +/-1.6pp standard error at ~90% accuracy
# 2/class = 48. Every accepted GEPA candidate costs one metric call per val example, so val size
# trades directly against how many candidates a budget buys: 48 leaves room for ~11 candidates at
# 600 calls, where 72 would allow ~8. At the measured 0.725 baseline that is still ~13 model
# errors of headroom -- far clear of the saturation that stalled the emails demo at 1 error.
N_VAL_PER_CLASS = 2


def _to_example(row: dict) -> dspy.Example:
    return dspy.Example(symptoms=row["text"], disease=row["label"]).with_inputs("symptoms")


def load_splits(seed: int = 0) -> tuple[list, list, list]:
    """Stratified per class, disjoint by construction, deterministic under `seed`.

    Stratified because a 24-class problem with ~48 examples per class can easily hand a split zero
    instances of a class under an unstratified shuffle, which makes macro-averaged metrics
    incomparable across splits for reasons that have nothing to do with the program.
    """
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in ROWS:
        by_label[row["label"]].append(row)

    train, val, test = [], [], []
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda r: r["text"])
        random.Random(f"{seed}:{label}").shuffle(group)
        test += [_to_example(r) for r in group[:N_TEST_PER_CLASS]]
        val += [_to_example(r) for r in group[N_TEST_PER_CLASS:N_TEST_PER_CLASS + N_VAL_PER_CLASS]]
        train += [_to_example(r) for r in group[N_TEST_PER_CLASS + N_VAL_PER_CLASS:]]

    rng = random.Random(seed)
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------


class DiagnoseFromSymptoms(dspy.Signature):
    """Given a patient's description of their symptoms, name the disease."""

    # The docstring is one line, for the reason the emails demo learned the hard way: strategy in
    # the seed prompt is task-solving information the optimizer should have had to discover, and it
    # inflates the baseline.
    #
    # The 24 labels ARE in the output field description, and that is not a hint -- it is the task
    # definition. This is closed-set classification; without the label space the task is
    # underdetermined and the model would be graded on guessing an exact surface form
    # ("athlete's foot" vs "Fungal infection"), which measures vocabulary luck rather than
    # diagnosis. Every serious classification setup states its label set.
    symptoms: str = dspy.InputField(desc="A patient's free-text description of their symptoms.")
    # A Literal, not a str. Measured on 40 test cases: with a plain `str` field the model returned
    # something outside the label set on 19 of them ("Measles (Rubeola)", "Based on the symptoms
    # described (chronic...") and scored 0.475; with the enum, off-label predictions went to ZERO
    # and accuracy to 0.725. The str version was mostly measuring output-format compliance, which
    # is not the task. Every remaining error is now a genuine misdiagnosis.
    disease: Literal[tuple(LABELS)] = dspy.OutputField()  # type: ignore[valid-type]


# ---------------------------------------------------------------------------
# Label matching
# ---------------------------------------------------------------------------


def norm(s: str) -> str:
    """Lowercase, strip everything but [a-z0-9]. Kills casing, spacing and punctuation noise."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


_NORM_TO_LABEL = {norm(x): x for x in LABELS}


def canonical(pred: str) -> str:
    """Map a prediction onto one of the 24 labels, or "" if it is not one of them.

    Deliberately strict compared to the emails demo's fuzzy matcher, and it can afford to be: the
    label space is closed and stated in the prompt, so there is no legitimate reason to emit a
    surface form outside it. Two liberties are taken, both about formatting rather than meaning:

      * normalization (case, spaces, punctuation) -- "Chicken Pox" == "Chicken pox";
      * a label embedded in a longer string resolves via containment, but ONLY when exactly one
        label matches -- so "the patient likely has Psoriasis" resolves, while a hedge naming two
        diseases does not silently get credited with one of them.

    No edit distance, no synonyms. An unmatched prediction scores 0 and is counted separately as
    `off_label`. The strictness is the point: a fuzzy matcher was a standing caveat in both earlier
    demos, and a closed label set lets it be removed from the argument entirely.
    """
    p = norm(pred)
    if not p:
        return ""
    if p in _NORM_TO_LABEL:
        return _NORM_TO_LABEL[p]
    hits = [lab for key, lab in _NORM_TO_LABEL.items() if key in p]
    return hits[0] if len(hits) == 1 else ""


def pred_disease(prediction) -> str:
    value = getattr(prediction, "disease", None)
    return value if isinstance(value, str) else ("" if value is None else str(value))


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def make_metric(llm_call_penalty: float):
    """Reward a correct label, charge per LLM call: score = max(0, correct - lambda * n_calls).

    Same per-call unit as the conflation and emails demos (the baseline spends exactly one call per
    example), so the sweep is directly comparable to both.
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None) -> ScoreWithFeedback:
        exec_trace = program_trace if program_trace is not None else trace
        n_calls = len(exec_trace) if exec_trace else 0

        raw = pred_disease(pred)
        predicted = canonical(raw)
        gold_label = gold.disease
        correct = predicted == gold_label
        score = max(0.0, (1.0 if correct else 0.0) - llm_call_penalty * n_calls)
        cost = f"{n_calls} LLM call(s), cost {llm_call_penalty * n_calls:.2f}, score {score:.2f}"

        if llm_call_penalty == 0.0:
            goal = ("LLM calls are free under the current objective, so accuracy is all that "
                    "matters. Use whatever mix of code and model calls maximizes correctness.")
        else:
            goal = (
                f"Target: name the right disease with a DETERMINISTIC, no-LLM algorithm wherever the "
                f"symptom text makes that possible (each call costs {llm_call_penalty:.2f} of the "
                f"score). The label set is fixed and known and the descriptions are short, so much "
                f"of this may be separable in plain Python -- reserve a model call for descriptions "
                f"that are genuinely ambiguous between two diseases. Which evidence to key on, and "
                f"how to break ties, is for you to work out."
            )

        if not raw.strip():
            return ScoreWithFeedback(
                score=0.0,
                feedback=f"No disease returned ({cost}). Return dspy.Prediction(disease=<str>). " + goal)
        if not predicted:
            # Off-vocabulary gets its own branch: it is a formatting failure, not a diagnosis
            # failure, and the fix for it is different.
            return ScoreWithFeedback(
                score=0.0,
                feedback=(f"OFF-LABEL: returned {raw!r} ({cost}), which is not one of the 24 allowed "
                          f"labels (correct was {gold_label!r}). The answer must be exactly one "
                          f"label, verbatim. If code picks the label, have it choose from the fixed "
                          f"list rather than generating a name. " + goal))
        if not correct:
            fb = (f"WRONG: predicted {predicted!r}, correct is {gold_label!r} ({cost}).\n"
                  f"Symptom text: {gold.symptoms}\n"
                  f"What in this description separates {gold_label!r} from {predicted!r}? Encode "
                  f"that distinction. " + goal)
        elif n_calls > 0 and llm_call_penalty > 0:
            fb = (f"Correct, but used {n_calls} LLM call(s) ({cost}). Full score needs this same "
                  f"answer with no model call. " + goal)
        else:
            fb = f"Correct ({cost}). Keep whatever produced this. " + goal
        return ScoreWithFeedback(score=score, feedback=fb)

    return metric


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


@contextmanager
def meter(*lms: dspy.LM):
    """Accumulate calls / tokens / litellm cost across `lms` for the block.

    Snapshots each LM's history index on entry and reads the tail on exit; history is append-only
    (see `make_lms`, which lifts the trim cap) and list.append is atomic under the GIL, so this is
    safe across the evaluation thread pool. Verified equal to dspy's GLOBAL_HISTORY.
    """
    totals: dict[str, Any] = {
        "calls": 0, "cost_usd_litellm": 0.0,
        "prompt_tokens": 0, "completion_tokens": 0, "uncosted_calls": 0,
        # A truncated reflection emits half a Python class, scores 0 everywhere and gets rejected --
        # indistinguishable from "the optimizer found nothing" unless counted. That cost a whole run
        # and $3.99 on the emails demo before anyone noticed.
        "truncated_calls": 0, "max_completion_tokens_seen": 0,
    }
    starts = [len(lm.history) for lm in lms]
    yield totals
    for lm, start in zip(lms, starts, strict=True):
        for entry in lm.history[start:]:
            totals["calls"] += 1
            cost = entry.get("cost")
            if cost is None:
                totals["uncosted_calls"] += 1
            else:
                totals["cost_usd_litellm"] += cost
            usage = entry.get("usage") or {}
            totals["prompt_tokens"] += usage.get("prompt_tokens") or 0
            completion = usage.get("completion_tokens") or 0
            totals["completion_tokens"] += completion
            totals["max_completion_tokens_seen"] = max(totals["max_completion_tokens_seen"], completion)
            if _hit_length_cap(entry):
                totals["truncated_calls"] += 1


def _hit_length_cap(entry: dict) -> bool:
    """True if this call stopped because it ran out of max_tokens rather than finishing."""
    response = entry.get("response")
    if response is None:
        return False
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    for choice in choices or []:
        reason = getattr(choice, "finish_reason", None)
        if reason is None and isinstance(choice, dict):
            reason = choice.get("finish_reason")
        if reason in ("length", "max_tokens"):
            return True
    return False


def price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rate_in, rate_out = PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * rate_in + completion_tokens * rate_out) / 1e6


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class Record(NamedTuple):
    """One test example's outcome -- enough to recompute any penalty's score offline."""

    gold: str
    pred: str
    raw: str
    correct: bool
    n_calls: int
    latency_s: float
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None


def run_program(program: dspy.Module, dataset: list, threads: int = 8,
                canonical_fn=None) -> tuple[list[Record], dict]:
    """Run `program` over `dataset`, returning per-example records plus wall-clock metadata.

    `canonical_fn` overrides the label matcher, so a run over a different label set (the
    metaharness/ comparison uses the paper's 22 classes) reuses this loop instead of copying it.
    """
    to_label = canonical_fn or canonical

    def run_one(ex) -> Record:
        started = time.perf_counter()
        try:
            with dspy.context(trace=[]), dspy.track_usage() as usage:
                out = program(**ex.inputs())
                trace = list(dspy.settings.trace or [])
            elapsed = time.perf_counter() - started
            p_tok = c_tok = 0
            cost = 0.0
            for model, totals in usage.get_total_tokens().items():
                p = totals.get("prompt_tokens") or 0
                c = totals.get("completion_tokens") or 0
                p_tok += p
                c_tok += c
                cost += price(model, p, c)
            raw = pred_disease(out)
            pred = to_label(raw)
            return Record(ex.disease, pred, raw, pred == ex.disease, len(trace),
                          elapsed, cost, p_tok, c_tok, None)
        except Exception as exc:
            # Counts as wrong. `error` is surfaced so a run that silently fails every call -- as a
            # bad model id once did -- is visible in the JSON instead of looking like 0% accuracy.
            return Record(ex.disease, "", "", False, 0, time.perf_counter() - started,
                          0.0, 0, 0, f"{type(exc).__name__}: {exc}"[:200])

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        records = list(pool.map(run_one, dataset))
    return records, {"wall_s": time.perf_counter() - started, "threads": threads}


def _macro_prf(records: list[Record]) -> tuple[float, float, float]:
    """Macro-averaged precision / recall / F1, averaged over the classes present in the gold labels.

    An off-label prediction ("" after `canonical`) costs recall on the class that should have been
    predicted but is never itself averaged over as a class, so a model that invents vocabulary is
    penalised without inventing spurious classes in the denominator.
    """
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    for r in records:
        if r.correct:
            tp[r.gold] += 1
        else:
            fn[r.gold] += 1
            if r.pred:
                fp[r.pred] += 1
    classes = sorted({r.gold for r in records})
    ps, rs, fs = [], [], []
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        ps.append(p)
        rs.append(rec)
        fs.append(2 * p * rec / (p + rec) if (p + rec) else 0.0)
    k = len(classes) or 1
    return sum(ps) / k, sum(rs) / k, sum(fs) / k


def summarize(records: list[Record], penalty: float) -> dict:
    """Accuracy + macro P/R/F1 + the three CAL axes, for one penalty, from per-example records."""
    n = len(records) or 1
    lat = sorted(r.latency_s for r in records)
    correct = sum(1 for r in records if r.correct)
    macro_p, macro_r, macro_f1 = _macro_prf(records)
    score = sum(max(0.0, (1.0 if r.correct else 0.0) - penalty * r.n_calls) for r in records) / n

    def pct(p: float) -> float:
        return lat[min(len(lat) - 1, int(round(p * (len(lat) - 1))))] if lat else 0.0

    per_class = {}
    for c in sorted({r.gold for r in records}):
        sub = [r for r in records if r.gold == c]
        per_class[c] = {"n": len(sub), "accuracy": sum(1 for r in sub if r.correct) / len(sub)}

    return {
        "n": len(records),
        "penalty": penalty,
        "score": score,
        "accuracy": correct / n,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "n_classes": len({r.gold for r in records}),
        "chance": 1.0 / max(1, len({r.gold for r in records})),
        "off_label": sum(1 for r in records if r.raw.strip() and not r.pred),
        "avg_calls": sum(r.n_calls for r in records) / n,
        "calls_total": sum(r.n_calls for r in records),
        "frac_examples_using_llm": sum(1 for r in records if r.n_calls > 0) / n,
        "cost_usd_total": sum(r.cost_usd for r in records),
        "cost_usd_per_example": sum(r.cost_usd for r in records) / n,
        "cost_usd_per_1k_examples": 1000 * sum(r.cost_usd for r in records) / n,
        "latency_mean_s": sum(lat) / n,
        "latency_p50_s": pct(0.50),
        "latency_p95_s": pct(0.95),
        "prompt_tokens": sum(r.prompt_tokens for r in records),
        "completion_tokens": sum(r.completion_tokens for r in records),
        "errors": sum(1 for r in records if r.error),
        "first_error": next((r.error for r in records if r.error), None),
        "per_class_accuracy": per_class,
    }


def fmt(row: dict) -> str:
    return (
        f"score={row['score']:.3f} acc={row['accuracy']:.3f} macroF1={row['macro_f1']:.3f} "
        f"off_label={row['off_label']} calls/ex={row['avg_calls']:.3f} "
        f"${row['cost_usd_per_1k_examples']:.2f}/1k lat_mean={row['latency_mean_s']*1000:.0f}ms"
    )
