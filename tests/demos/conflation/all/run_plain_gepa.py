"""What does GEPA buy WITHOUT Flex? One run of instruction-only optimization, on the same task.

    python run_plain_gepa.py                 # optimize + evaluate + merge into penalty_sweep.json
    python run_plain_gepa.py --max-metric-calls 400

The sweep in `sweep_penalties.py` optimizes a `dspy.Flex(SamePlace)` -- GEPA rewrites the module's
*source*, so it can decompose the task into Python plus a routed LLM call, and the LLM-call penalty
lambda buys real movement along the cost and latency axes. This script removes exactly one thing:
Flex. The program is a bare `dspy.Predict(SamePlace)`, so GEPA can only rewrite the *instruction*.

**Lambda is structurally inert here, which is the point.** A `dspy.Predict` makes exactly one
predictor call per `forward()`, always -- so `n_calls == 1` for every example and the metric reduces
to `max(0, correct - lambda)`. For any lambda < 1 that is a monotone transform of accuracy: it
shifts every candidate's score by the same constant and cannot reorder them. Sweeping lambda over a
plain Predict would redraw the identical program five times at five different y-offsets. So this
runs once, at lambda = 0 (where the metric *is* accuracy), and the result is a single point on the
CAL plane rather than a frontier -- prompt-only optimization has no mechanism to trade calls for
cost. `--penalty` exists to let that claim be re-checked rather than taken on faith.

Everything else is held identical to the sweep so the point is comparable: same splits and seed,
same executor and reflection LM, same `max_metric_calls`, same `reflection_minibatch_size=3`, same
240-example test set, same evaluator. Results merge into `penalty_sweep.json` under `plain_gepa`,
which `sweep_penalties.py --plot-only` picks up and overlays onto `cal_frontier.png`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).parent))
from conflation_common import (
    EXEC_MODEL,
    REFLECTION_MODEL,
    SamePlace,
    disable_cache,
    fmt,
    load_splits,
    make_lms,
    make_metric,
    meter,
    run_program,
    summarize,
)
from sweep_penalties import (
    DEFAULT_MAX_METRIC_CALLS,
    EVAL_THREADS,
    SWEEP_PATH,
    plot,
    plot_path_for,
)

PROGRAM_PATH = Path(__file__).parent / "plain_gepa_predict.json"
REFLECTION_MINIBATCH = 3  # same as the sweep
SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-metric-calls", type=int, default=DEFAULT_MAX_METRIC_CALLS)
    ap.add_argument("--threads", type=int, default=EVAL_THREADS)
    ap.add_argument("--penalty", type=float, default=0.0,
                    help="see the module docstring: inert for a plain Predict, since n_calls is "
                         "always 1. Exposed so that can be verified, not assumed.")
    ap.add_argument("--out", type=Path, default=SWEEP_PATH)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    args.out = args.out.resolve()

    disable_cache()
    exec_lm, reflection_lm = make_lms()
    dspy.configure(lm=exec_lm)
    train, val, test = load_splits()
    print(f"splits: gepa_train={len(train)} gepa_val={len(val)} test={len(test)}")

    program = dspy.Predict(SamePlace)
    baseline_instructions = program.signature.instructions

    print(f"\n=== plain GEPA (dspy.Predict, no Flex), λ={args.penalty:g}, "
          f"max_metric_calls={args.max_metric_calls} ===")
    started = time.perf_counter()
    with meter(exec_lm, reflection_lm) as opt_cost:
        optimized = dspy.GEPA(
            metric=make_metric(args.penalty),
            reflection_lm=reflection_lm,
            max_metric_calls=args.max_metric_calls,
            reflection_minibatch_size=REFLECTION_MINIBATCH,
            num_threads=args.threads,
            seed=SEED,
        ).compile(program, trainset=train, valset=val)
    opt_wall_s = time.perf_counter() - started

    records, meta = run_program(optimized, test, threads=args.threads)
    row = summarize(records, args.penalty)
    row.update(meta)
    optimized.save(str(PROGRAM_PATH))

    # A plain Predict cannot route around its own call, so this should be exactly 1.00. Asserting it
    # is what makes "prompt-only optimization cannot move along the cost axis" a measurement rather
    # than an assumption.
    if abs(row["avg_calls"] - 1.0) > 1e-9:
        print(f"  !! avg_calls={row['avg_calls']:.4f}, expected exactly 1.00 for a plain Predict — "
              f"the lambda-is-inert argument in the docstring does not hold; investigate before "
              f"reading the overlay.")

    entry = {
        "kind": "plain GEPA (dspy.Predict, instruction-only)",
        "penalty": args.penalty,
        "max_metric_calls": args.max_metric_calls,
        "optimization": {"wall_s": opt_wall_s, **opt_cost,
                         "changed_instructions": optimized.signature.instructions != baseline_instructions},
        "test": row,
        "records": [r._asdict() for r in records],
        "instructions_before": baseline_instructions,
        "instructions_after": optimized.signature.instructions,
        "program_path": PROGRAM_PATH.name,
        "lambda_note": ("n_calls is structurally 1 for a plain Predict, so max(0, correct - lambda) "
                        "is a monotone transform of accuracy and lambda cannot reorder candidates. "
                        "One point, not a frontier."),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exec_model": EXEC_MODEL, "reflection_model": REFLECTION_MODEL,
    }

    data = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    data["plain_gepa"] = entry
    args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")

    print("  " + fmt(row))
    o = entry["optimization"]
    print(f"  optimization: {o['wall_s']:.0f}s, {o['calls']} LM calls, ${o['cost_usd_litellm']:.2f}, "
          f"instructions_changed={o['changed_instructions']}")
    print(f"\nwrote {args.out}")

    if not args.no_plot:
        plot(data, plot_path_for(args.out))


if __name__ == "__main__":
    main()
