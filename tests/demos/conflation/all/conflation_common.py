"""Shared pieces for the conflation Flex+GEPA demos.

Both the single-run demo (``test_flex_conflation.py``) and the penalty sweep
(``sweep_penalties.py``) import from here so they measure the same things the
same way: the task signature, the class-balanced splits, the penalized metric,
and an evaluator that reports the three CAL axes (cost, accuracy, latency)
alongside the usual classification metrics.
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from dotenv import load_dotenv

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()

DEMO_DIR = Path(__file__).parent
DATA_PATH = DEMO_DIR.parent / "conflation_coded.jsonl"

# `anthropic/claude-haiku-4-7` (the previous EXEC_LM) is not a real model id — the API 404s on it,
# and `_evaluate`'s broad `except Exception` turned that into "wrong answer, 0 calls" rather than a
# crash. Haiku 4.5 is the weakest Claude available and the cheapest per call, which is what makes
# the call penalty interesting: GEPA has to decide whether a weak-but-cheap judge is worth 0.2.
EXEC_MODEL = "anthropic/claude-haiku-4-5"
REFLECTION_MODEL = "anthropic/claude-opus-5"

# USD per 1M tokens (input, output), used for the per-example cost attribution below. litellm's own
# per-call cost is recorded alongside these as `cost_usd_litellm` and is the authoritative figure;
# this table exists so cost can be attributed to individual examples, which litellm's aggregate
# history cannot do under a thread pool.
PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
    "anthropic/claude-sonnet-5": (3.00, 15.00),
    "anthropic/claude-opus-5": (5.00, 25.00),
    "anthropic/claude-opus-4-8": (5.00, 25.00),
}


def make_lms() -> tuple[dspy.LM, dspy.LM]:
    """Return (execution LM, reflection LM) and make history unbounded for cost accounting."""
    # `meter()` below slices lm.history by index; the default 10k cap pops from the *front*, which
    # would shift those indices mid-run and silently drop calls from the totals.
    dspy.configure(max_history_size=10**9)
    exec_lm = dspy.LM(EXEC_MODEL, max_tokens=2000)
    reflection_lm = dspy.LM(REFLECTION_MODEL, max_tokens=8000)
    return exec_lm, reflection_lm


def disable_cache() -> None:
    """Turn off dspy's caches so latency and cost are what a cold production call would be."""
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)


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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Train/val drive GEPA (its budget is max_metric_calls, so val must stay small). The test split is a
# CLASS-BALANCED subset rather than the full 1029-row remainder: the full pool is 769 pos / 260 neg,
# where a predict-always-same classifier already scores 0.75 accuracy. Balancing puts the chance
# line at 0.50 so accuracy is readable on its own, and caps the per-evaluation LM spend.
N_TRAIN_POS, N_TRAIN_NEG = 30, 30
N_VAL_POS, N_VAL_NEG = 15, 15
N_TEST_POS, N_TEST_NEG = 120, 120


def _to_example(row: dict) -> dspy.Example:
    return dspy.Example(
        input_name=row["input_name"],
        input_address=row["input_address"],
        match_name=row["match_name"],
        match_address=row["match_address"],
        distance=float(row["distance"]),
        is_same=(row["judgment"] == "true"),
    ).with_inputs("input_name", "input_address", "match_name", "match_address", "distance")


def load_splits(seed: int = 0) -> tuple[list, list, list]:
    """Class-balanced train / val / test splits, disjoint by construction."""
    rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    pos = [r for r in rows if r["judgment"] == "true"]
    neg = [r for r in rows if r["judgment"] == "false"]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    def take(seq, start, count):
        return [_to_example(r) for r in seq[start : start + count]]

    train = take(pos, 0, N_TRAIN_POS) + take(neg, 0, N_TRAIN_NEG)
    val = take(pos, N_TRAIN_POS, N_VAL_POS) + take(neg, N_TRAIN_NEG, N_VAL_NEG)
    test = take(pos, N_TRAIN_POS + N_VAL_POS, N_TEST_POS) + take(neg, N_TRAIN_NEG + N_VAL_NEG, N_TEST_NEG)
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def pool_prevalence() -> float:
    """Fraction of the FULL dataset that is positive (~0.747).

    The test split is deliberately rebalanced to 50/50 so accuracy is readable against a 0.50
    chance line, but that is not the distribution the classifier would actually see. Accuracy and
    precision are prevalence-dependent, so `summarize` reports both: the directly measured value on
    the balanced split, and the value re-weighted to this prevalence. Recall and specificity are
    prevalence-invariant and need no correction.
    """
    rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sum(r["judgment"] == "true" for r in rows) / len(rows)


def _as_bool(value) -> bool:
    return value if isinstance(value, bool) else bool(value)


def make_metric(llm_call_penalty: float):
    """Build the GEPA metric for one penalty: reward correct + deterministic, charge per LLM call.

    The penalty is the sweep's independent variable. At 0.0 the metric is plain accuracy and GEPA
    has no reason to write Python; as it rises, each LLM call has to buy back more accuracy than it
    costs, and the optimizer is pushed toward deciding cases in code.
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None) -> ScoreWithFeedback:
        example, prediction = gold, pred
        # GEPA delivers the execution trace via `program_trace` at scoring time (declaring the
        # parameter opts in); `trace` still carries it on feedback calls and from evaluate() below.
        exec_trace = program_trace if program_trace is not None else trace
        n_calls = len(exec_trace) if exec_trace else 0  # predictor calls during this forward()
        try:
            predicted = _as_bool(prediction.is_same)
        except Exception:
            return ScoreWithFeedback(
                score=0.0,
                feedback="`is_same` was missing or unreadable. Return dspy.Prediction(is_same=<bool>).",
            )
        gold_label = bool(example.is_same)
        correct = predicted == gold_label
        score = max(0.0, (1.0 if correct else 0.0) - llm_call_penalty * n_calls)

        if not correct:
            fb = (
                f"WRONG: predicted is_same={predicted}, expected {gold_label}. Use the input fields "
                "(name, address, distance) to ideally decide deterministically in Python whether the "
                "two locations are the same."
            )
            if n_calls == 0 and llm_call_penalty < 1.0:
                fb += " If this case is truly ambiguous for rules, route it to the LLM judge instead."
        elif n_calls > 0:
            if llm_call_penalty == 0.0:
                fb = (
                    f"Correct, using {n_calls} LLM call(s). LLM calls are free under the current "
                    "objective, so accuracy is all that matters here."
                )
            else:
                fb = (
                    f"Correct, but used {n_calls} LLM call(s), costing {llm_call_penalty * n_calls:.2f} "
                    f"of the 1.00 available for this example (penalty {llm_call_penalty:.2f}/call). If "
                    "the normalized name/address similarity and distance already make this clear, "
                    "decide it in Python and skip the LLM. Reserve LLM calls for genuinely ambiguous "
                    "cases only."
                )
        else:
            fb = "Correct with no LLM call. This is great! Keep settling clear cases deterministically."
        return ScoreWithFeedback(score=score, feedback=fb)

    return metric


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


@contextmanager
def meter(*lms: dspy.LM):
    """Accumulate calls / tokens / litellm cost across `lms` for the duration of the block.

    Works by snapshotting each LM's history index on entry and reading the tail on exit. `history`
    is append-only (see `make_lms`, which lifts the trim cap), and list.append is atomic under the
    GIL, so this is safe across the evaluation thread pool.
    """
    totals: dict[str, Any] = {
        "calls": 0,
        "cost_usd_litellm": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "uncosted_calls": 0,  # cache hits and any call litellm could not price
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
            totals["completion_tokens"] += usage.get("completion_tokens") or 0


def price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """USD for one call, from the PRICES table. Unknown models price at 0 (and are reported as such)."""
    rate_in, rate_out = PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * rate_in + completion_tokens * rate_out) / 1e6


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class Record(NamedTuple):
    """One test example's outcome — enough to recompute any penalty's score offline."""

    gold: bool
    pred: bool
    n_calls: int
    latency_s: float
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None


def run_program(program: dspy.Module, dataset: list, threads: int = 8) -> tuple[list[Record], dict]:
    """Run `program` over `dataset`, returning per-example records plus aggregate LM totals.

    Per-example cost comes from `dspy.track_usage()` (thread-local) priced through PRICES; the
    aggregate `meter` totals carry litellm's own cost for cross-checking.
    """

    def run_one(ex) -> Record:
        gold = bool(ex.is_same)
        started = time.perf_counter()
        try:
            with dspy.context(trace=[]), dspy.track_usage() as usage:
                pred_obj = program(**ex.inputs())
                trace = list(dspy.settings.trace or [])
            elapsed = time.perf_counter() - started
            prompt_tokens = completion_tokens = 0
            cost = 0.0
            for model, totals in usage.get_total_tokens().items():
                p = totals.get("prompt_tokens") or 0
                c = totals.get("completion_tokens") or 0
                prompt_tokens += p
                completion_tokens += c
                cost += price(model, p, c)
            return Record(gold, _as_bool(pred_obj.is_same), len(trace), elapsed, cost,
                          prompt_tokens, completion_tokens, None)
        except Exception as exc:
            # An unreadable prediction counts as wrong (pred = not gold keeps accuracy and the
            # confusion matrix consistent). `error` is surfaced so a run that silently fails every
            # call — as the bad model id did — is visible in the JSON instead of looking like 0%.
            return Record(gold, (not gold), 0, time.perf_counter() - started, 0.0, 0, 0,
                          f"{type(exc).__name__}: {exc}"[:200])

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        records = list(pool.map(run_one, dataset))
    wall_s = time.perf_counter() - started
    return records, {"wall_s": wall_s, "threads": threads}


def summarize(records: list[Record], penalty: float) -> dict:
    """Classification + CAL metrics for one penalty, computed from per-example records.

    Records are penalty-independent, so a single run can be scored at every penalty in the sweep —
    which is how the baseline row is produced without re-evaluating it five times.
    """
    n = len(records) or 1
    lat = sorted(r.latency_s for r in records)
    correct = sum(1 for r in records if r.pred == r.gold)
    tp = sum(1 for r in records if r.pred and r.gold)
    fp = sum(1 for r in records if r.pred and not r.gold)
    fn = sum(1 for r in records if not r.pred and r.gold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    score = sum(max(0.0, (1.0 if r.pred == r.gold else 0.0) - penalty * r.n_calls) for r in records) / n

    def pct(p: float) -> float:
        return lat[min(len(lat) - 1, int(round(p * (len(lat) - 1))))] if lat else 0.0

    # Re-weight to the real class mix. Accuracy/precision measured on the balanced split do not
    # transfer to a 74.7%-positive pool; recall and specificity do. Cost and call rate shift too,
    # because which cases get routed to the LLM depends on the class mix.
    tn = len(records) - tp - fp - fn
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    pi = pool_prevalence()
    acc_pool = recall * pi + specificity * (1 - pi)
    denom_pool = recall * pi + (1 - specificity) * (1 - pi)
    prec_pool = (recall * pi) / denom_pool if denom_pool else 0.0
    f1_pool = 2 * prec_pool * recall / (prec_pool + recall) if (prec_pool + recall) else 0.0
    pos = [r for r in records if r.gold] or records
    neg = [r for r in records if not r.gold] or records
    mean = lambda seq, f: sum(f(r) for r in seq) / len(seq)  # noqa: E731
    calls_pool = pi * mean(pos, lambda r: r.n_calls) + (1 - pi) * mean(neg, lambda r: r.n_calls)
    cost_pool = pi * mean(pos, lambda r: r.cost_usd) + (1 - pi) * mean(neg, lambda r: r.cost_usd)

    return {
        "n": len(records),
        "penalty": penalty,
        "score": score,
        "accuracy": correct / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        # --- re-weighted to the real ~74.7%-positive pool (see pool_prevalence) ---
        "pool_prevalence": pi,
        "accuracy_at_pool_prevalence": acc_pool,
        "precision_at_pool_prevalence": prec_pool,
        "f1_at_pool_prevalence": f1_pool,
        "avg_calls_at_pool_prevalence": calls_pool,
        "cost_usd_per_1k_at_pool_prevalence": 1000 * cost_pool,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
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
    }


def evaluate(program: dspy.Module, dataset: list, penalty: float, threads: int = 8) -> dict:
    """Convenience wrapper: run and summarize at a single penalty."""
    records, meta = run_program(program, dataset, threads=threads)
    out = summarize(records, penalty)
    out.update(meta)
    return out


def fmt(row: dict) -> str:
    """One-line summary for console output."""
    return (
        f"score={row['score']:.3f} acc={row['accuracy']:.3f} P={row['precision']:.3f} "
        f"R={row['recall']:.3f} F1={row['f1']:.3f} calls/ex={row['avg_calls']:.3f} "
        f"${row['cost_usd_per_1k_examples']:.2f}/1k lat_p50={row['latency_p50_s']*1000:.1f}ms"
    )
