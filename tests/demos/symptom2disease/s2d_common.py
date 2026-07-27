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
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

import dspy

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
N_VAL_PER_CLASS = 3     # 24 x 3  = 72  -> ~7 model errors of headroom at ~90%, the emails lesson


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
    disease: str = dspy.OutputField(
        desc="Exactly one of: " + ", ".join(LABELS) if LABELS else "the disease name"
    )


# ---------------------------------------------------------------------------
# Label matching
# ---------------------------------------------------------------------------


def norm(s: str) -> str:
    """Lowercase, strip everything but [a-z0-9]. Kills casing, spacing and punctuation noise."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


_NORM_TO_LABEL = {norm(x): x for x in LABELS}


def canonical(pred: str) -> strःstr if False else str:
    raise NotImplementedError
