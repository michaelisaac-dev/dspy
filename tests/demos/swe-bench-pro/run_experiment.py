"""Baseline vs GEPA-evolved harness on the SWE-bench Pro pilot subset, small execution model.

    python run_experiment.py --phase baseline          # un-optimized Flex on the test split
    python run_experiment.py --phase compile           # GEPA evolves the harness on train/val
    python run_experiment.py --phase optimized         # evolved harness on the test split
    python run_experiment.py --phase all               # the three in order
    python run_experiment.py --phase all --resume      # skip finished episodes / reuse programs

Every finished episode is appended to episodes.jsonl keyed by (arm, harness-source hash,
instance), so a killed run resumes where it stopped. results.json is rewritten after every phase.
`--max-cost-usd` is checked between phases against the run's own meter; it cannot stop GEPA
mid-compile, so the compile budget is bounded by --max-metric-calls instead (each metric call is
one container episode).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from swebp_common import (
    EPISODE_TIMEOUT_S,
    EXEC_MODEL,
    REFLECTION_MODEL,
    EpisodeCache,
    SWEAgent,
    disable_cache,
    fmt,
    load_splits,
    make_lms,
    make_resolve_metric,
    meter,
    new_harness,
    reap_orphans,
    run_program,
    summarize,
)

import dspy

DEMO_DIR = Path(__file__).parent
RESULTS = DEMO_DIR / "results.json"
PROGRAM_DIR = DEMO_DIR / "programs"
PROGRAM_PATH = PROGRAM_DIR / "harness_gepa_resolve.json"
EPISODES = DEMO_DIR / "episodes.jsonl"


def load_results() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text(encoding="utf-8"))
    return {"meta": {}, "arms": {}}


def save_results(results: dict) -> None:
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def spend_so_far(results: dict) -> float:
    return sum(arm.get("meter", {}).get("cost_usd_litellm", 0.0) for arm in results["arms"].values())


def eval_arm(name: str, agent: SWEAgent, test: list, args, results: dict, lms) -> None:
    print(f"\n=== {name}: evaluating on {len(test)} test instance(s) ===", flush=True)
    cache = None if args.no_episode_cache else EpisodeCache(EPISODES)
    with meter(*lms) as m:
        records, run_meta = run_program(agent, test, threads=args.threads, arm=name, cache=cache)
    row = summarize(records)
    print(f"[{name}] {fmt(row)}")
    results["arms"][name] = {
        "summary": row,
        "records": [r._asdict() for r in records],
        "meter": m,
        "wall_s": run_meta["wall_s"],
        "module_src": agent.harness.module_src,
        "test_limit": args.test_limit,
    }
    save_results(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["baseline", "compile", "optimized", "all"], default="all")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--test-limit", type=int, default=0, help="truncate the test split (recorded in the JSON)")
    ap.add_argument("--max-metric-calls", type=int, default=60,
                    help="GEPA rollout budget; every metric call is one container episode")
    ap.add_argument("--reflection-minibatch", type=int, default=2)
    ap.add_argument("--episode-timeout", type=float, default=EPISODE_TIMEOUT_S)
    ap.add_argument("--max-cost-usd", type=float, default=90.0,
                    help="stop before starting a phase that would exceed this (checked between phases)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-episode-cache", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    reaped = reap_orphans()
    if reaped:
        print(f"reaped {reaped} orphaned container(s)")

    exec_lm, reflection_lm = make_lms()
    dspy.configure(lm=exec_lm)
    disable_cache()

    train, val, test = load_splits(seed=args.seed)
    if args.test_limit:
        test = test[: args.test_limit]
    print(f"splits: train={len(train)} val={len(val)} test={len(test)}  "
          f"exec={EXEC_MODEL} reflection={REFLECTION_MODEL}")

    results = load_results()
    results["meta"] = {
        "exec_model": EXEC_MODEL, "reflection_model": REFLECTION_MODEL,
        "max_metric_calls": args.max_metric_calls, "seed": args.seed,
        "episode_timeout_s": args.episode_timeout, "test_limit": args.test_limit,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
    }

    phases = [args.phase] if args.phase != "all" else ["baseline", "compile", "optimized"]

    for phase in phases:
        spent = spend_so_far(results)
        if spent > args.max_cost_usd:
            print(f"STOP: spend so far ${spent:.2f} exceeds --max-cost-usd {args.max_cost_usd}; "
                  f"not starting phase {phase!r}")
            break

        if phase == "baseline":
            if args.resume and "baseline" in results["arms"]:
                print("baseline already present; --resume skips it")
                continue
            agent = SWEAgent(episode_timeout_s=args.episode_timeout)
            eval_arm("baseline", agent, test, args, results, (exec_lm, reflection_lm))

        elif phase == "compile":
            if args.resume and PROGRAM_PATH.exists():
                print(f"compiled program already at {PROGRAM_PATH}; --resume skips the compile")
                continue
            print(f"\n=== compile: GEPA, resolve metric, {args.max_metric_calls} metric calls "
                  f"(each one is a container episode) ===", flush=True)
            agent = SWEAgent(episode_timeout_s=args.episode_timeout)
            started = time.perf_counter()
            with meter(exec_lm, reflection_lm) as m:
                optimized = dspy.GEPA(
                    metric=make_resolve_metric(),
                    reflection_lm=reflection_lm,
                    max_metric_calls=args.max_metric_calls,
                    reflection_minibatch_size=args.reflection_minibatch,
                    num_threads=args.threads,
                    seed=args.seed,
                    log_dir=str(DEMO_DIR / "gepa_log_resolve"),
                    track_stats=True,
                ).compile(agent, trainset=train, valset=val)
            PROGRAM_DIR.mkdir(exist_ok=True)
            optimized.harness.save(str(PROGRAM_PATH))
            changed = optimized.harness.module_src != agent.harness.module_src
            results["arms"]["compile"] = {
                "meter": m, "wall_s": time.perf_counter() - started,
                "changed_code": changed, "module_src": optimized.harness.module_src,
            }
            save_results(results)
            print(f"[compile] changed_code={changed} wall={time.perf_counter() - started:.0f}s "
                  f"cost=${m['cost_usd_litellm']:.2f} truncated_calls={m['truncated_calls']}")

        elif phase == "optimized":
            if not PROGRAM_PATH.exists():
                print(f"no compiled program at {PROGRAM_PATH}; run --phase compile first")
                break
            harness = new_harness()
            harness.load(str(PROGRAM_PATH))
            agent = SWEAgent(harness=harness, episode_timeout_s=args.episode_timeout)
            eval_arm("optimized", agent, test, args, results, (exec_lm, reflection_lm))

    # Console comparison, if both arms exist.
    arms = results["arms"]
    if "baseline" in arms and "optimized" in arms:
        b, o = arms["baseline"]["summary"], arms["optimized"]["summary"]
        print("\n=== comparison (same test split, same execution model) ===")
        print(f"{'arm':<12} {'resolved':<12} {'calls/task':<11} {'$/task':<8} {'lat mean':<9}")
        for label, row in (("baseline", b), ("optimized", o)):
            print(f"{label:<12} {row['n_resolved']}/{row['n']:<10} {row['avg_calls']:<11.1f} "
                  f"{row['cost_usd_per_task']:<8.2f} {row['latency_mean_s']:<9.0f}")
        both = {r["instance"]: r["resolved"] for r in arms["baseline"]["records"]}
        flips_up = [r["instance"] for r in arms["optimized"]["records"]
                    if r["resolved"] and not both.get(r["instance"], False)]
        flips_down = [r["instance"] for r in arms["optimized"]["records"]
                      if not r["resolved"] and both.get(r["instance"], False)]
        print(f"gained: {len(flips_up)}  lost: {len(flips_down)}  (n={len(both)} paired episodes)")

    total = spend_so_far(results)
    print(f"\ntotal metered spend this results.json: ${total:.2f}")


if __name__ == "__main__":
    main()
