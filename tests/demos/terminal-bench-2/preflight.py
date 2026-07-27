"""Check the environment, and prove the container runner works -- without spending a cent on LLMs.

The sweep is expensive and slow, so every failure mode that can be found for free should be found
here first. Three levels, cheapest first:

    python preflight.py                    # environment: docker, deno, manifest, splits
    python preflight.py --oracle fix-git   # run ONE task's own solution and verify it -> must PASS
    python preflight.py --oracle-split val # run the oracle over a whole split (slow, still $0)

`--oracle` is the important one. It replaces the agent with the task's shipped `solution/solve.sh`,
then runs the real verifier. A PASS proves the whole runner is correct: image pull, container start,
working directory, command execution, test copy, `test.sh`, reward parsing. A FAIL there is a bug in
this demo, not in the agent -- and finding it after a $40 sweep instead of before is the difference
this script exists to make.

`--smoke` additionally binds a trivial hand-written harness into `dspy.Flex` and runs one episode
with NO model configured, which checks the sandbox bridge (Deno) and the three host tools end to end.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from tb2_common import (  # noqa: E402
    TASKS,
    ContainerSession,
    DockerError,
    crlf_tasks,
    docker_available,
    load_splits,
    reap_orphans,
)

OK, BAD, WARN = "  ok  ", " FAIL ", " warn "


def check_environment() -> dict[str, bool]:
    """Report every prerequisite at once rather than dying on the first one.

    Returns which capabilities are available, because they gate different things: the oracle needs
    only Docker and the manifest, while the sandbox smoke test and the sweep also need Deno. Failing
    the oracle over a missing Deno would block the one check that costs nothing and catches the most.
    """
    fatal = False

    reachable, detail = docker_available()
    print(f"[{OK if reachable else BAD}] docker daemon: {detail}")
    if not reachable:
        print("         Docker Desktop is installed but its daemon is not running. Start it, or run")
        print("         `wsl -d docker-desktop` / launch Docker Desktop, then re-run this script.")
        fatal = True

    deno = shutil.which("deno")
    print(f"[{OK if deno else WARN}] deno: {deno or 'not on PATH'}")
    if not deno:
        print("         dspy.Flex executes optimizer-authored code in a Deno/Pyodide sandbox, so the")
        print("         SWEEP cannot run without it (--oracle can). Install with:")
        print("           irm https://deno.land/install.ps1 | iex        (PowerShell)")
        print("           curl -fsSL https://deno.land/install.sh | sh   (bash)")

    print(f"[{OK if TASKS else BAD}] manifest: {len(TASKS)} tasks")
    if not TASKS:
        print("         run `python fetch_data.py` first")
        return {"oracle": False, "sandbox": False}

    train, val, test = load_splits()
    print(f"[{OK}] splits: train={len(train)} val={len(val)} test={len(test)}")
    for label, split in (("train", train), ("val", val), ("test", test)):
        counts: dict[str, int] = {}
        for ex in split:
            counts[ex.difficulty] = counts.get(ex.difficulty, 0) + 1
        print(f"         {label:<6} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    overlap = ({e.task_name for e in train} | {e.task_name for e in val}) & {e.task_name for e in test}
    print(f"[{OK if not overlap else BAD}] train/val vs test overlap: {len(overlap)}")
    fatal = fatal or bool(overlap)

    missing = [n for n, s in TASKS.items() if not s.tests_dir.is_dir()]
    print(f"[{OK if not missing else BAD}] every task has tests/: {len(TASKS) - len(missing)}/{len(TASKS)}")
    fatal = fatal or bool(missing)

    # Fatal, and worth its own check: a CRLF verifier fails on every task while looking exactly like
    # an agent that solved nothing. This is the single most expensive way to waste a sweep.
    crlf = crlf_tasks()
    print(f"[{OK if not crlf else BAD}] verifier scripts are LF: {len(TASKS) - len(crlf)}/{len(TASKS)}")
    if crlf:
        print(f"         {len(crlf)} task(s) have CRLF test.sh (git core.autocrlf), e.g. {crlf[:3]}.")
        print("         bash in the container cannot parse them. Fix with:")
        print("           python fetch_data.py --fix-line-endings")
        fatal = True

    key = "ANTHROPIC_API_KEY"
    print(f"[{OK if os.getenv(key) else WARN}] {key}: {'set' if os.getenv(key) else 'missing (needed for the sweep, not for --oracle)'}")

    reaped = reap_orphans()
    if reaped:
        print(f"[{WARN}] reaped {reaped} orphaned container(s) from an earlier run")
    return {"oracle": not fatal, "sandbox": not fatal and bool(deno)}


def run_oracle(task_name: str, keep: bool = False) -> bool:
    """Solve one task with its own shipped solution, then verify. Uses zero LLM calls."""
    spec = TASKS.get(task_name)
    if spec is None:
        print(f"unknown task {task_name!r}")
        return False
    if not spec.solution_path.exists():
        print(f"{task_name}: no solution/solve.sh, cannot run the oracle")
        return False

    started = time.perf_counter()
    session = ContainerSession(spec, episode_timeout_s=spec.agent_timeout_s)
    try:
        session.start()
        # The oracle is a shell script, not an agent: push it in and run it exactly as-is.
        session.write("/oracle/solve.sh", spec.solution_path.read_text(encoding="utf-8"))
        out = session.exec("bash /oracle/solve.sh", timeout_s=int(spec.agent_timeout_s))
        resolved, tail, verify_s = session.verify()
    except DockerError as exc:
        print(f"[{BAD}] {task_name}: {exc}")
        return False
    finally:
        if not keep:
            session.stop()

    wall = time.perf_counter() - started
    print(f"[{OK if resolved else BAD}] {task_name:<34} reward={int(resolved)} "
          f"({wall:.0f}s total, {verify_s:.0f}s verifying)")
    if not resolved:
        print("         --- oracle output (tail) ---")
        print("         " + "\n         ".join(out.strip().splitlines()[-12:]))
        print("         --- verifier output (tail) ---")
        print("         " + "\n         ".join(tail.strip().splitlines()[-15:]))
    return resolved


def run_smoke(task_name: str) -> bool:
    """Bind a hand-written harness into dspy.Flex and run one episode with no LM configured.

    This exercises the part the oracle does not: the Deno sandbox, the tool bridge, and
    `TerminalAgent`'s start/verify/teardown. The harness is deliberately dumb -- it runs two fixed
    commands -- so the episode is expected to FAIL verification. What is being checked is that it
    fails cleanly, having actually run commands in the container.
    """
    from tb2_common import TerminalAgent, new_harness

    harness = new_harness()
    harness._bind_code(
        "class SolveTerminalTaskModule(dspy.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "\n"
        "    def forward(self, **inputs):\n"
        "        s = inputs['session']\n"
        "        listing = terminal_exec(session=s, command='ls -la', timeout_s=30)\n"
        "        write_file(session=s, path='/tmp/preflight.txt', content='hello from the sandbox')\n"
        "        back = read_file(session=s, path='/tmp/preflight.txt')\n"
        "        return dspy.Prediction(summary=listing[:200] + ' | roundtrip=' + back)\n"
    )
    agent = TerminalAgent(harness=harness, episode_timeout_s=180)
    pred = agent(task_name=task_name, instruction=TASKS[task_name].instruction)
    ok = pred.n_commands >= 3 and "roundtrip=hello from the sandbox" in (pred.summary or "")
    print(f"[{OK if ok else BAD}] sandbox + tool bridge: {pred.n_commands} command(s) ran, "
          f"resolved={pred.resolved} (a fail here is expected; a 0 command count is not)")
    if not ok:
        print(f"         summary: {pred.summary[:300]!r}")
        print(f"         error:   {pred.harness_error}")
    harness.close()
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle", nargs="*", metavar="TASK", help="run these tasks' shipped solutions")
    ap.add_argument("--oracle-split", choices=["train", "val", "test"], help="run the oracle over a split")
    ap.add_argument("--smoke", metavar="TASK", nargs="?", const="fix-git",
                    help="run one episode with a hand-written harness (checks Deno + tool bridge)")
    ap.add_argument("--keep", action="store_true", help="leave containers running for inspection")
    args = ap.parse_args()

    can = check_environment()
    if not (args.oracle is not None or args.oracle_split or args.smoke):
        raise SystemExit(0 if can["sandbox"] else 1)
    if not can["oracle"]:
        print("\nenvironment is not healthy; fix the FAILs above before running anything else")
        raise SystemExit(1)
    if args.smoke and not can["sandbox"]:
        print("\n--smoke needs Deno (see above); skipping it. --oracle still runs.")
        args.smoke = None

    names: list[str] = list(args.oracle or [])
    if args.oracle_split:
        train, val, test = load_splits()
        names += [ex.task_name for ex in {"train": train, "val": val, "test": test}[args.oracle_split]]

    if names:
        print(f"\n=== oracle over {len(names)} task(s) -- $0 in LLM spend, but slow ===")
        results = [run_oracle(n, keep=args.keep) for n in names]
        passed = sum(results)
        print(f"\noracle: {passed}/{len(results)} passed")
        if passed < len(results):
            print("A failing oracle is a bug in this runner (or a task that needs a GPU / a flaky "
                  "network install in its verifier) -- NOT an agent failure. Investigate before sweeping.")

    if args.smoke:
        print(f"\n=== sandbox smoke test on {args.smoke} ===")
        run_smoke(args.smoke)

    reap_orphans()


if __name__ == "__main__":
    main()
