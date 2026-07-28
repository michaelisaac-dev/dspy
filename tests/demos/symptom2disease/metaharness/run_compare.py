"""Optimize a dspy.Flex diagnoser on Meta-Harness's own Symptom2Disease split and score it there.

    python run_compare.py                          # baseline + both metric arms
    python run_compare.py --arms contrastive       # one arm
    python run_compare.py --max-metric-calls 300   # cheaper
    python run_compare.py --report-only            # re-print the table from results.json

Everything the comparison rests on -- the split, the evaluator, the 86.8 target and the confounds --
is documented in `mh_common`. The one number to keep in view while reading the output: the paper's
Meta-Harness result is 86.8% on these same 212 test examples, and its zero-shot baseline on them was
63.2%.

Two arms, identical in every respect except the metric's `feedback` string (the `score` is the same
1/0 exact match in both). That is the ablation: same data, same budget, same seed, same executor,
same reflection LM, same minibatch. Per-example records for every arm land in `results.json`, so any
other statistic can be recomputed without paying for the run again.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from mh_common import (
    EXEC_MODEL,
    LABELS,
    METRICS,
    PAPER,
    REFLECTION_MODEL,
    TARGET,
    DiagnoseFromSymptoms,
    canonical,
    check_data,
    disable_cache,
    load_paper_splits,
    make_lms,
    meter,
    paper_eval,
    run_program,
    summarize,
)
from s2d_common import REFLECTION_MAX_TOKENS

MH_DIR = Path(__file__).parent
RESULTS_PATH = MH_DIR / "results.json"
PROGRAM_DIR = MH_DIR / "programs"

DEFAULT_MAX_METRIC_CALLS = 600
# 22 classes and a 5-example minibatch means most minibatches touch each class zero times, so the
# reflection LM sees singletons rather than structure. 10 is still cheap (each is one ~40-token
# classification) and is held identical across arms, so it cannot explain a difference between them.
REFLECTION_MINIBATCH = 10
EVAL_THREADS = 8
SEED = 0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval -- the honest error bar on an accuracy from n examples."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_p(ra: list[dict], rb: list[dict]) -> float:
    """Two-sided exact McNemar p for two programs scored on the same examples (paired)."""
    n01 = sum(1 for x, y in zip(ra, rb, strict=True) if x["correct"] and not y["correct"])
    n10 = sum(1 for x, y in zip(ra, rb, strict=True) if not x["correct"] and y["correct"])
    n = n01 + n10
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) for i in range(0, min(n01, n10) + 1)) / 2**n * 2)


def binom_p_greater(k: int, n: int, p0: float) -> float:
    """One-sided exact binomial P(X >= k | n, p0): 'could 86.8% have produced a score this high?'

    Unpaired, because Meta-Harness's 212 per-example outcomes are not published -- only the
    aggregate 86.8. That makes this test strictly weaker than the McNemar used between our own arms,
    and it also treats 86.8 as a fixed constant rather than an estimate with its own +/-2.3pp
    interval. Both facts are printed with the number.
    """
    if k <= 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) * p0**i * (1 - p0) ** (n - i) for i in range(k, n + 1)))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def evaluate(program: dspy.Module, test: list, threads: int) -> tuple[list[dict], dict]:
    """Test-set pass, scored under both our matcher and the paper's, which must agree."""
    records, meta = run_program(program, test, threads=threads, canonical_fn=canonical)
    row = summarize(records, penalty=0.0)
    row.update(meta)

    # The `Literal` output field makes an off-label string impossible, which is precisely why the
    # two matchers can be expected to agree -- so assert it rather than assume it. A mismatch would
    # mean the headline number depends on which evaluator you pick, and that has to be visible.
    disagreements = sum(1 for r in records if paper_eval(r.raw, r.gold) != r.correct)
    row["paper_evaluator_disagreements"] = disagreements
    row["accuracy_paper_evaluator"] = sum(1 for r in records if paper_eval(r.raw, r.gold)) / len(records)
    return [r._asdict() for r in records], row


def run_arm(name: str, train, val, test, max_metric_calls, exec_lm, reflection_lm, threads) -> dict:
    program = dspy.Flex(DiagnoseFromSymptoms)
    baseline_src = program.module_src
    metric = METRICS[name]()

    started = time.perf_counter()
    with meter(exec_lm, reflection_lm) as opt_cost:
        optimized = dspy.GEPA(
            metric=metric,
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            reflection_minibatch_size=REFLECTION_MINIBATCH,
            num_threads=threads,
            seed=SEED,
            log_dir=str(MH_DIR / f"gepa_log_{name}"),
            # The score is plain accuracy here, so a minibatch the current program already gets
            # right is skipped by default -- and on a task where the seed program is decent that
            # silently burns most of the iteration budget. The sibling demo lost 32 of 43
            # iterations this way before it was caught.
            skip_perfect_score=False,
        ).compile(program, trainset=train, valset=val)
    opt_wall_s = time.perf_counter() - started

    records, row = evaluate(optimized, test, threads)

    PROGRAM_DIR.mkdir(parents=True, exist_ok=True)
    program_path = PROGRAM_DIR / f"flex_{name}.json"
    optimized.save(str(program_path))

    out = {
        "arm": name,
        "max_metric_calls": max_metric_calls,
        "optimization": {"wall_s": opt_wall_s, **opt_cost,
                         "changed_code": optimized.module_src != baseline_src},
        "test": row,
        "records": records,
        "module_src": optimized.module_src,
        "program_path": program_path.name,
    }
    state = getattr(metric, "state", None)
    if state is not None:
        out["confusion_during_search"] = {
            "top_pairs": [{"gold": g, "predicted": p, "n": n} for g, p, n in state.top_confusions(15, 1)],
            "never_right": [{"label": lab, "attempts": n} for lab, n in state.never_right()],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(METRICS), choices=list(METRICS))
    ap.add_argument("--max-metric-calls", type=int, default=DEFAULT_MAX_METRIC_CALLS)
    ap.add_argument("--threads", type=int, default=EVAL_THREADS)
    ap.add_argument("--out", type=Path, default=RESULTS_PATH)
    ap.add_argument("--resume", action="store_true", help="skip arms already present in --out")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    args.out = args.out.resolve()

    if args.report_only:
        report(json.loads(args.out.read_text(encoding="utf-8")))
        return

    hygiene = check_data()
    print("split hygiene:", json.dumps(hygiene))
    if hygiene["train_test_overlap"]:
        raise SystemExit("train/test overlap in the vendored split -- refusing to report a number")

    disable_cache()
    exec_lm, reflection_lm = make_lms()
    dspy.configure(lm=exec_lm)
    train, val, test = load_paper_splits()
    print(f"splits (paper's own): train={len(train)} val={len(val)} test={len(test)} "
          f"classes={len(LABELS)}")

    data = json.loads(args.out.read_text(encoding="utf-8")) if (args.resume and args.out.exists()) else {}
    data.setdefault("meta", {}).update({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paper": PAPER,
        "split_hygiene": hygiene,
        "exec_model": EXEC_MODEL, "reflection_model": REFLECTION_MODEL,
        "max_metric_calls": args.max_metric_calls,
        "reflection_minibatch": REFLECTION_MINIBATCH,
        "reflection_max_tokens": REFLECTION_MAX_TOKENS,
        "skip_perfect_score": False, "seed": SEED, "eval_threads": args.threads,
        "cache": "disabled (dspy disk + memory)",
        "confound": ("executor is claude-haiku-4-5, not the paper's gpt-oss-120b (no OpenRouter "
                     "credentials here). The zero-shot row isolates how much of any gap is the "
                     "model rather than the optimizer."),
        "dspy_version": dspy.__version__, "python": platform.python_version(),
    })

    if "baseline" not in data:
        print("\n=== baseline: un-optimized dspy.Flex (one Predict, zero-shot) ===")
        records, row = evaluate(dspy.Flex(DiagnoseFromSymptoms), test, args.threads)
        data["baseline"] = {"test": row, "records": records}
        args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(f"  acc={row['accuracy']:.4f}  (paper's zero-shot on this split: "
              f"{PAPER['s2d_table2']['zero_shot']:.1f}%)")

    arms = data.setdefault("arms", {})
    for name in args.arms:
        if args.resume and name in arms:
            print(f"\n=== arm {name}: cached, skipping ===")
            continue
        print(f"\n=== arm {name}  (max_metric_calls={args.max_metric_calls}) ===")
        arms[name] = run_arm(name, train, val, test, args.max_metric_calls,
                             exec_lm, reflection_lm, args.threads)
        args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")
        r = arms[name]
        o = r["optimization"]
        print(f"  acc={r['test']['accuracy']:.4f}  macroF1={r['test']['macro_f1']:.4f}  "
              f"calls/ex={r['test']['avg_calls']:.2f}")
        print(f"  optimization: {o['wall_s']:.0f}s, {o['calls']} LM calls, "
              f"${o['cost_usd_litellm']:.2f}, code_changed={o['changed_code']}")
        if o.get("truncated_calls"):
            print(f"  !! {o['truncated_calls']} TRUNCATED reflection call(s) -- raise "
                  f"REFLECTION_MAX_TOKENS; a truncated proposal is unparseable, scores 0 and looks "
                  f"exactly like 'the optimizer found nothing'")

    print(f"\nwrote {args.out}")
    report(data)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(data: dict) -> None:
    base = data.get("baseline")
    arms = data.get("arms", {})
    n = base["test"]["n"] if base else next(iter(arms.values()))["test"]["n"]

    print("\n" + "=" * 96)
    print(f"Symptom2Disease -- Meta-Harness's split (n_test={n}, {len(LABELS)} classes, "
          f"chance {100/len(LABELS):.1f}%)")
    print("=" * 96)

    hdr = f"{'system':<34} {'acc %':>7} {'95% CI':>16} {'macroF1':>8} {'calls/ex':>9} {'$/1k':>8}"
    print(hdr)
    print("-" * len(hdr))

    for label, value in PAPER["s2d_table2"].items():
        tag = "Meta-Harness (paper)" if label == "meta_harness" else f"  {label} (paper)"
        star = "  <- target" if label == "meta_harness" else ""
        print(f"{tag:<34} {value:7.1f} {'—':>16} {'—':>8} {'—':>9} {'—':>8}{star}")
    print(f"{'  ':<34} {'':>7} {'(gpt-oss-120b)':>16}")
    print("-" * len(hdr))

    def line(name: str, row: dict) -> None:
        k = round(row["accuracy"] * row["n"])
        lo, hi = wilson(k, row["n"])
        print(f"{name:<34} {row['accuracy']*100:7.1f} {f'[{lo*100:.1f}, {hi*100:.1f}]':>16} "
              f"{row['macro_f1']:8.3f} {row['avg_calls']:9.2f} {row['cost_usd_per_1k_examples']:8.2f}")

    if base:
        line("ours: zero-shot Flex (haiku-4.5)", base["test"])
    for name, arm in arms.items():
        line(f"ours: GEPA / metric={name}", arm["test"])

    print("\nsignificance")
    print("-" * 60)
    for name, arm in arms.items():
        row = arm["test"]
        k, nn = round(row["accuracy"] * row["n"]), row["n"]
        p_paper = binom_p_greater(k, nn, TARGET)
        verdict = ("beats 86.8 (p<0.05)" if p_paper < 0.05 and row["accuracy"] > TARGET
                   else "above 86.8, not significant" if row["accuracy"] > TARGET
                   else "at or below 86.8")
        print(f"  {name:<14} vs Meta-Harness 86.8: {k}/{nn} = {row['accuracy']*100:.1f}%, "
              f"one-sided exact binomial p={p_paper:.3f}  -> {verdict}")
        if base:
            p_base = mcnemar_p(base["records"], arm["records"])
            print(f"  {'':<14} vs our zero-shot baseline: McNemar p={p_base:.4f} (paired)")
    if len(arms) == 2:
        a, b = list(arms)
        p_ab = mcnemar_p(arms[a]["records"], arms[b]["records"])
        d = (arms[b]["test"]["accuracy"] - arms[a]["test"]["accuracy"]) * 100
        print(f"  {b} vs {a} (the scoring-function ablation): {d:+.1f}pp, McNemar p={p_ab:.4f}")

    print("\ncaveats")
    print("-" * 60)
    print("  * executor is claude-haiku-4-5; the paper's rows are gpt-oss-120b. The zero-shot row")
    print("    above is the like-for-like anchor for how much of any gap is the base model.")
    print("  * the binomial test against 86.8 is unpaired and treats 86.8 as exact; the paper's own")
    print(f"    86.8 on n=212 carries a 95% CI of roughly "
          f"[{wilson(round(0.868*212), 212)[0]*100:.1f}, {wilson(round(0.868*212), 212)[1]*100:.1f}].")
    dis = sum(a["test"].get("paper_evaluator_disagreements", 0) for a in arms.values()) \
        + (base["test"].get("paper_evaluator_disagreements", 0) if base else 0)
    print(f"  * scored under the paper's own evaluators.py matcher as a cross-check: "
          f"{dis} disagreement(s) with ours across all rows.")
    spend = sum(a["optimization"]["cost_usd_litellm"] for a in arms.values())
    print(f"\ntotal GEPA optimization spend: ${spend:.2f} across {len(arms)} arm(s)")


if __name__ == "__main__":
    main()
