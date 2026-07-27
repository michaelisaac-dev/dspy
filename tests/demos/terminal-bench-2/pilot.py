"""Pick the execution model by measuring it, not by assuming it.

The emails demo's model choice was a 30-example pilot that overturned the demo's original default
(Opus) in favour of one 6.5x cheaper. The same question is sharper here, in the opposite direction:
Terminal-Bench 2.0 is hard, and an execution model that resolves ~0 tasks gives GEPA no gradient at
all -- every minibatch is uniformly wrong, so the reflective dataset has no contrast to learn from.
The pilot therefore has to answer two things, not one:

    1. cost / resolve-rate / latency per model, as in the other demos;
    2. whether the model clears the FLOOR -- at least a couple of resolves on the probe. A model
       that scores 0/8 cannot be optimized against at any budget.

It runs the un-optimized harness (a single `dspy.RLM` over the container tools) on a fixed probe
drawn from the TRAIN split only, so nothing here touches the reported test set.

    python pilot.py                                  # default model list, 8 tasks
    python pilot.py --models anthropic/claude-sonnet-5 anthropic/claude-haiku-4-5
    python pilot.py --n 4 --threads 2                # cheaper probe
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import dspy

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from tb2_common import (  # noqa: E402
    EPISODE_TIMEOUT_S,
    EXEC_MAX_TOKENS,
    TerminalAgent,
    disable_cache,
    docker_available,
    load_splits,
    new_harness,
    reap_orphans,
    run_program,
    summarize,
)

DEMO_DIR = Path(__file__).parent
PILOT_PATH = DEMO_DIR / "pilot.json"

DEFAULT_MODELS = [
    "anthropic/claude-sonnet-5",
    "anthropic/claude-haiku-4-5",
]
DEFAULT_N = 8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="probe size, drawn from the TRAIN split")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--episode-timeout", type=float, default=EPISODE_TIMEOUT_S)
    ap.add_argument("--out", type=Path, default=PILOT_PATH)
    args = ap.parse_args()

    reachable, detail = docker_available()
    if not reachable:
        raise SystemExit(f"docker is not usable ({detail}); run `python preflight.py`")
    reap_orphans()
    disable_cache()

    train, _, _ = load_splits()
    # A fixed prefix of the (already seeded-shuffled) train split: the same probe for every model, so
    # the comparison is paired, and never a test task.
    probe = train[: args.n]
    print(f"probe: {args.n} train tasks -- " + ", ".join(f"{e.task_name}({e.difficulty})" for e in probe))

    results = {}
    for model in args.models:
        print(f"\n=== {model} ===")
        lm = dspy.LM(model, max_tokens=EXEC_MAX_TOKENS)
        dspy.configure(lm=lm, max_history_size=10**9)
        program = TerminalAgent(harness=new_harness(), episode_timeout_s=args.episode_timeout)
        started = time.perf_counter()
        # The episode cache is deliberately NOT used: a probe is a measurement, and reusing an
        # episode run under a different model would silently mix the two.
        records, meta = run_program(program, probe, threads=args.threads, penalty=0.0, cache=None)
        row = summarize(records, 0.0)
        row.update(meta, wall_min=(time.perf_counter() - started) / 60)
        results[model] = {"summary": row, "records": [r._asdict() for r in records]}
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")

    hdr = f"{'model':<32} {'resolved':>9} {'calls':>7} {'cmds':>7} {'$/task':>8} {'sec':>7} {'floor':>7}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for model, res in results.items():
        r = res["summary"]
        floor = "ok" if r["n_resolved"] >= 2 else "TOO LOW"
        print(f"{model.split('/')[-1]:<32} {r['n_resolved']:>4}/{r['n']:<4} {r['avg_calls']:7.1f} "
              f"{r['avg_commands']:7.1f} {r['cost_usd_per_task']:8.2f} {r['latency_mean_s']:7.0f} "
              f"{floor:>7}")
    print(f"\nwrote {args.out}")
    print("A model in the TOO LOW column resolves too little for GEPA to get a gradient from this "
          "probe; either move up a tier or raise --episode-timeout before concluding it cannot work.")


if __name__ == "__main__":
    main()
