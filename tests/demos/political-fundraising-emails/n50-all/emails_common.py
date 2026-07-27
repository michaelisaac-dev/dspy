"""Shared pieces for the political-fundraising-email Flex+GEPA demos.

Mirrors `tests/demos/conflation/all/conflation_common.py`: same task signature, splits, penalized
metric factory and CAL (cost / accuracy / latency) instrumentation, adapted for a *string
extraction* task rather than binary classification.

Two structural differences from the conflation demo:

* **No class balance / prevalence question.** The label is a committee name, not a boolean, so
  there is nothing to re-weight. Precision / recall / F1 are macro-averaged over committees.
* **Statistical power is the binding constraint.** `val.jsonl` alone is 50 rows — ±4.2 pp standard
  error at ~90% accuracy, so nothing under ~10 pp would be resolvable. `load_splits` therefore also
  holds out rows from `train.jsonl` to double the test set, and `summarize` reports the two halves
  separately so the clean "unseen committee" property of `val.jsonl` is not lost.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, NamedTuple

from dotenv import load_dotenv

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()

DEMO_DIR = Path(__file__).parent
DATA_DIR = DEMO_DIR.parent / "eval_data"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"

# The demo shipped with `claude-opus-4-7` as executor (which does exist — verified). A 30-email
# pilot of all three tiers on this task said to drop to Haiku:
#
#     opus-4-7    acc 0.967   $9.27/1k   2303 ms
#     sonnet-5    acc 0.933   $5.53/1k   2103 ms
#     haiku-4-5   acc 0.967   $1.43/1k   1338 ms
#
# Haiku matched Opus example-for-example (both missed only "Georgia Blue PAC") at 6.5x lower cost
# and 1.7x lower latency; Sonnet was worst, appending a parenthetical gloss the fuzzy matcher
# rejects. Finding the sponsor in a disclaimer line does not need a frontier model. n=30, so a
# small true gap could hide — but identical predictions on 29/30 rules out a large one.
EXEC_MODEL = "anthropic/claude-haiku-4-5"
REFLECTION_MODEL = "anthropic/claude-opus-5"

# USD per 1M tokens (input, output). litellm's own per-call cost is recorded alongside as
# `cost_usd_litellm` and is authoritative; this table exists so cost can be attributed to
# individual examples, which an aggregate history cannot do under a thread pool.
PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
    "anthropic/claude-sonnet-5": (3.00, 15.00),
    "anthropic/claude-opus-5": (5.00, 25.00),
    "anthropic/claude-opus-4-8": (5.00, 25.00),
    "anthropic/claude-opus-4-7": (5.00, 25.00),
}


# Execution completions are a committee name plus dspy's field markers — ~18 tokens measured, so
# 2000 is ample. Reflection is the one that bites: it has to emit an entire rewritten module.
EXEC_MAX_TOKENS = 2000
# The first run of this sweep used 8000 here and SILENTLY TRUNCATED every proposal. Measured
# reflection output averaged 7,643 completion tokens against the 8,000 cap (96%), so proposals were
# cut off mid-class; the resulting unparseable code scored 0 on every example, GEPA rejected all 11
# proposals, and λ=0 returned the base program byte-identical after $3.71. The conflation demo used
# the same 8000 and was never affected — its reflection outputs ran 1,004–2,769 tokens (13–35% of
# cap) because its reflective dataset holds short records, not four full ~650-token emails.
# Raised 24000 -> 32000: the λ=0 run peaked at 21,714 completion tokens (90% of the 24k cap),
# and penalized runs emit LONGER programs (deterministic parsing logic on top of the prompt).
# Truncation here is silent and catastrophic — it produced unparseable code that scored 0.
REFLECTION_MAX_TOKENS = 32000


def make_lms(exec_max_tokens: int = EXEC_MAX_TOKENS,
             reflection_max_tokens: int = REFLECTION_MAX_TOKENS) -> tuple[dspy.LM, dspy.LM]:
    """Return (execution LM, reflection LM) and make history unbounded for cost accounting."""
    # `meter()` slices lm.history by index; the default 10k cap pops from the *front*, which would
    # shift those indices mid-run and silently drop calls from the totals.
    dspy.configure(max_history_size=10**9)
    return (dspy.LM(EXEC_MODEL, max_tokens=exec_max_tokens),
            dspy.LM(REFLECTION_MODEL, max_tokens=reflection_max_tokens))


def disable_cache() -> None:
    """Turn off dspy's caches so latency and cost are what a cold production call would be."""
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)


class IdentifyCommittee(dspy.Signature):
    """Identify the political committee that sponsored a fundraising email."""

    # Deliberately MINIMAL — one line, no hints.
    #
    # The original seed carried a second paragraph: "The committee's name is present in the email
    # text itself, so most emails can be resolved deterministically in code with no model call;
    # reserve a model call only as a fallback." That is two problems at once. It leaks a substantive
    # task hint (it turns an inference problem into a copy-from-text problem), and it is guidance
    # aimed at GEPA that was landing in the *execution* prompt. Together they put the un-optimized
    # baseline at 91% test / 97.5% on GEPA's val slice — leaving the optimizer ~1 example of
    # headroom, so it could not demonstrate any prompt improvement at λ=0.
    #
    # The decomposition objective still reaches GEPA, through the metric feedback ("Target: return
    # the correct committee with a DETERMINISTIC, no-LLM algorithm..."), which is where it belongs.
    email_body: str = dspy.InputField()
    committee: str = dspy.OutputField()


# ---------------------------------------------------------------------------
# Fuzzy label matching
# ---------------------------------------------------------------------------


def norm(s: str) -> str:
    """Lowercase and strip everything but [a-z0-9] — kills PDF-parser whitespace glitches
    ('FRIENDS OFSHERROD BROWN'), punctuation ('INC.' vs 'INC'), and casing noise in the labels."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def match(pred: str, gold: str) -> bool:
    """True if the predicted committee matches gold up to normalization / a legal suffix.

    Tolerant of parser whitespace, punctuation, case and 'INC.'-style suffixes, but strict enough to
    reject a whole-disclaimer dump: a substring only counts when it covers >=70% of the longer
    string, otherwise a >=0.9 character-ratio is required.
    """
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return p == g
    if p == g:
        return True
    short, long = (p, g) if len(p) <= len(g) else (g, p)
    if short in long and len(short) / len(long) >= 0.7:
        return True
    return SequenceMatcher(None, p, g).ratio() >= 0.9


def pred_committee(prediction) -> str:
    value = getattr(prediction, "committee", None)
    return value if isinstance(value, str) else ("" if value is None else str(value))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# 20 was too small: at λ=0.2 GEPA scored 0.89 on its val set and 0.79 on the 100 held-out
# emails — an 11-point generalization gap, because selection granularity was 1/20 = 0.05.
N_GEPA_VAL = 40    # held out of the train pool for GEPA candidate selection
N_TRAIN_HOLDOUT = 50  # held out of train.jsonl and added to the test set, purely for power


def _to_example(row: dict, source: str) -> dspy.Example:
    return dspy.Example(
        email_body=row["email_body"],
        committee=row["committee"],
        source=source,
    ).with_inputs("email_body")


def load_splits(seed: int = 0) -> tuple[list, list, list]:
    """Return (gepa_train, gepa_val, test).

    `test` is `val.jsonl` (50, the demo's original held-out set, rich in committees that never
    appear in training) PLUS 50 rows held out of `train.jsonl`. GEPA never sees either. The union
    exists only to halve the sampling error on accuracy; `summarize` still reports the `val.jsonl`
    half on its own so the unseen-committee property remains legible.
    """
    def read(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    train_rows, val_rows = read(TRAIN_PATH), read(VAL_PATH)

    # train.jsonl contains 2 email bodies twice (labels agree in both cases). Without dedup a
    # duplicate can straddle the holdout boundary, putting the same email in both the GEPA train
    # pool and the test set. Nothing crosses between train.jsonl and val.jsonl, so this is the only
    # leakage path — but it is a real one, so close it.
    seen: set[str] = set()
    deduped = []
    for r in train_rows:
        if r["email_body"] not in seen:
            seen.add(r["email_body"])
            deduped.append(r)
    train_rows = deduped

    rng = random.Random(seed)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    holdout = [_to_example(r, "train_holdout") for r in train_rows[:N_TRAIN_HOLDOUT]]
    remaining = train_rows[N_TRAIN_HOLDOUT:]

    # Re-randomize before splitting val/train. Taking a CONTIGUOUS slice of the already-shuffled
    # list drew a pathologically easy val set: 1 error in 40 (97.5%) against a pool rate of 9/98
    # (90.8%) and a test rate of 89.0% — a ~1-in-20 draw. That left GEPA's entire candidate-
    # selection signal with one fixable example, so a 4-6pp improvement was invisible to it and it
    # returned the base program. An independent shuffle puts the val slice at 4 errors / 0.900,
    # matching the pool and the test set. The test split is untouched by this and no model output
    # is used to build it.
    random.Random(seed + 1_000).shuffle(remaining)

    gepa_val = [_to_example(r, "gepa_val") for r in remaining[:N_GEPA_VAL]]
    gepa_train = [_to_example(r, "gepa_train") for r in remaining[N_GEPA_VAL:]]
    test = [_to_example(r, "val_jsonl") for r in val_rows] + holdout
    rng.shuffle(test)
    return gepa_train, gepa_val, test


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def make_metric(llm_call_penalty: float):
    """Build the GEPA metric for one penalty: reward correct + deterministic, charge per LLM call.

    The feedback names the objective — a correct answer from deterministic code with no LLM call —
    and states only the data property that makes it possible (the name is in the text). It never
    says *where* in the text or *how* to extract it; that is for GEPA to discover.
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None) -> ScoreWithFeedback:
        example, prediction = gold, pred
        exec_trace = program_trace if program_trace is not None else trace
        n_calls = len(exec_trace) if exec_trace else 0

        name = pred_committee(prediction)
        if not name.strip():
            return ScoreWithFeedback(
                score=0.0,
                feedback="No committee returned (score 0.00). Return dspy.Prediction(committee=<str>).",
            )

        gold_name = example.committee
        correct = match(name, gold_name)
        score = max(0.0, (1.0 if correct else 0.0) - llm_call_penalty * n_calls)
        cost = f"{n_calls} LLM call(s), cost {llm_call_penalty * n_calls:.2f}, score {score:.2f}"

        goal = (
            "Target: return the correct committee with a DETERMINISTIC, no-LLM algorithm. The "
            "committee name is present in the email text, so it is recoverable in pure Python — "
            "reserve an LLM call only for emails where code genuinely cannot recover it. (Where in "
            "the text it is, and how to extract it, is for you to work out.)"
        )
        if llm_call_penalty == 0.0:
            goal = (
                "LLM calls are free under the current objective, so accuracy is all that matters. "
                "Use whatever mix of code and model calls maximizes correctness."
            )

        if not correct and n_calls == 0:
            # The worst outcome: a confident deterministic guess that is wrong. Surface the
            # asymmetry so GEPA learns to route low-confidence cases to the fallback.
            fb = (
                f"WRONG with NO LLM call ({cost}) — the worst outcome. Predicted committee={name!r}; "
                f"correct committee={gold_name!r}. A confident-but-wrong code guess scores 0.00, "
                f"whereas deferring THIS email to the LLM would have scored about "
                f"{max(0.0, 1 - llm_call_penalty):.2f}. When the code cannot recover the name with "
                f"high confidence, fall back to the LLM instead of guessing. " + goal
            )
        elif not correct:
            fb = f"Incorrect ({cost}). Predicted committee={name!r}; correct committee={gold_name!r}. " + goal
        elif n_calls == 0:
            fb = (f"Ideal — correct with no LLM call ({cost}). This is the target; keep resolving "
                  f"emails like this in code.")
        else:
            fb = (
                f"Correct but used an LLM call ({cost}); that costs {llm_call_penalty * n_calls:.2f}. "
                f"Full score needs this SAME answer with zero LLM calls. " + goal
            )
        return ScoreWithFeedback(score=score, feedback=fb)

    return metric


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


@contextmanager
def meter(*lms: dspy.LM):
    """Accumulate calls / tokens / litellm cost across `lms` for the duration of the block.

    Snapshots each LM's history index on entry and reads the tail on exit. `history` is append-only
    (see `make_lms`, which lifts the trim cap) and list.append is atomic under the GIL, so this is
    safe across the evaluation thread pool. Verified equal to dspy's GLOBAL_HISTORY.
    """
    totals: dict[str, Any] = {
        "calls": 0, "cost_usd_litellm": 0.0,
        "prompt_tokens": 0, "completion_tokens": 0, "uncosted_calls": 0,
        # A truncated reflection emits half a Python class, which scores 0 everywhere and gets
        # rejected — indistinguishable from "the optimizer found nothing" unless you count it.
        # That cost $3.71 and a whole run before it was noticed. Never again.
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
    """One test example's outcome — enough to recompute any penalty's score offline."""

    gold: str
    pred: str
    correct: bool
    source: str
    n_calls: int
    latency_s: float
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None


def run_program(program: dspy.Module, dataset: list, threads: int = 8) -> tuple[list[Record], dict]:
    """Run `program` over `dataset`, returning per-example records plus wall-clock metadata."""

    def run_one(ex) -> Record:
        started = time.perf_counter()
        source = getattr(ex, "source", "?")
        try:
            with dspy.context(trace=[]), dspy.track_usage() as usage:
                pred_obj = program(**ex.inputs())
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
            name = pred_committee(pred_obj)
            return Record(ex.committee, name, match(name, ex.committee), source,
                          len(trace), elapsed, cost, p_tok, c_tok, None)
        except Exception as exc:
            # Counts as wrong. `error` is surfaced so a run that silently fails every call — as a
            # bad model id would — is visible in the JSON instead of looking like 0% accuracy.
            return Record(ex.committee, "", False, source, 0, time.perf_counter() - started,
                          0.0, 0, 0, f"{type(exc).__name__}: {exc}"[:200])

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        records = list(pool.map(run_one, dataset))
    return records, {"wall_s": time.perf_counter() - started, "threads": threads}


def _macro_prf(records: list[Record]) -> tuple[float, float, float]:
    """Macro-averaged precision / recall / F1 over committees.

    Many-way attribution with fuzzy matching, so each prediction is canonicalized to the committee
    it matches: its own gold when correct, else the first other known committee it matches, else
    its own (wrong) bucket. The test set is singleton-heavy, so macro-recall tracks accuracy
    closely; the metrics diverge on repeated committees and on wrong-committee confusions.
    """
    gold_variants = sorted({r.gold for r in records if r.gold}, key=norm)
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    for r in records:
        gold_c = norm(r.gold)
        pred_c = gold_c if r.correct else next(
            (norm(g) for g in gold_variants if match(r.pred, g)), norm(r.pred))
        if pred_c == gold_c:
            tp[gold_c] += 1
        else:
            fn[gold_c] += 1
            fp[pred_c] += 1
    classes = {norm(r.gold) for r in records}
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

    by_source = {}
    for src in sorted({r.source for r in records}):
        sub = [r for r in records if r.source == src]
        by_source[src] = {
            "n": len(sub),
            "accuracy": sum(1 for r in sub if r.correct) / len(sub),
            "avg_calls": sum(r.n_calls for r in sub) / len(sub),
        }

    return {
        "n": len(records),
        "penalty": penalty,
        "score": score,
        "accuracy": correct / n,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "n_committees": len({norm(r.gold) for r in records}),
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
        "by_source": by_source,
    }


def fmt(row: dict) -> str:
    return (
        f"score={row['score']:.3f} acc={row['accuracy']:.3f} macroF1={row['macro_f1']:.3f} "
        f"calls/ex={row['avg_calls']:.3f} ${row['cost_usd_per_1k_examples']:.2f}/1k "
        f"lat_mean={row['latency_mean_s']*1000:.0f}ms"
    )
