"""Validate the whole pipeline before spending anything on model calls.

    python preflight.py                     # environment + splits checks              ($0, seconds)
    python preflight.py --oracle IID        # gold patch through the real verifier     ($0, minutes)
    python preflight.py --null IID          # empty patch must NOT resolve             ($0, minutes)
    python preflight.py --oracle-split val  # gold patch on every pilot-val instance   ($0, slow)
    python preflight.py --smoke IID         # one baseline episode end to end          (~$0.2-0.5)

A failing oracle is a bug in this runner, not an agent failure. TB2's oracle caught two bugs that
would each have silently zeroed a full sweep; finding them here costs nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from swebp_common import (
    MANIFEST,
    RUN_SCRIPTS,
    UPSTREAM,
    docker_available,
    load_instances,
    load_splits,
    run_verifier,
)


def check_environment() -> bool:
    ok = True

    reachable, detail = docker_available()
    print(f"[{'ok' if reachable else 'FAIL'}] docker: {detail}")
    ok &= reachable

    present = UPSTREAM.exists() and RUN_SCRIPTS.exists()
    print(f"[{'ok' if present else 'FAIL'}] upstream scripts: {UPSTREAM}")
    ok &= present

    manifest = MANIFEST.exists()
    print(f"[{'ok' if manifest else 'FAIL'}] manifest: {MANIFEST}" + ("" if manifest else "  (run fetch_data.py)"))
    ok &= manifest

    key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    print(f"[{'ok' if key else 'FAIL'}] ANTHROPIC_API_KEY " + ("set" if key else "missing (needed for --smoke and the experiment)"))

    import shutil
    deno = shutil.which("deno") is not None
    print(f"[{'ok' if deno else 'FAIL'}] deno on PATH (dspy.Flex sandbox)")

    if not manifest:
        return ok

    train, val, test = load_splits()
    ids = lambda xs: {ex.instance_id for ex in xs}  # noqa: E731
    disjoint = not (ids(train) & ids(val)) and not (ids(train) & ids(test)) and not (ids(val) & ids(test))
    print(f"[{'ok' if disjoint else 'FAIL'}] splits disjoint: train={len(train)} val={len(val)} test={len(test)}")
    ok &= disjoint

    t2 = load_splits()
    deterministic = [ex.instance_id for ex in t2[0]] == [ex.instance_id for ex in train]
    print(f"[{'ok' if deterministic else 'FAIL'}] splits deterministic under seed 0")
    ok &= deterministic

    specs = load_instances()
    langs = {specs[i].language for i in ids(train) | ids(val) | ids(test)}
    print(f"[{'ok' if langs == {'python'} else 'FAIL'}] pilot is python-only: {langs}")
    ok &= langs == {"python"}

    from collections import Counter
    repos = Counter(specs[i].repo for i in ids(train) | ids(val) | ids(test))
    print(f"[info] pilot repos: {dict(repos)}")
    return ok


def oracle(iid: str) -> bool:
    """The instance's own gold patch must resolve; anything else is a runner bug."""
    spec = load_instances()[iid]
    print(f"oracle: {iid}\n  image {spec.image}")
    v = run_verifier(spec, spec.gold_patch)
    print(f"  resolved={v.resolved} f2p={v.f2p_passed}/{v.f2p_total} "
          f"p2p_broken={v.p2p_failed}/{v.p2p_total} wall={v.wall_s:.0f}s"
          + (f" infra_error={v.infra_error}" if v.infra_error else ""))
    if not v.resolved:
        print("  --- verifier tail ---")
        print("  " + v.tail.replace("\n", "\n  ")[:2500])
    print(f"  [{'PASS' if v.resolved else 'FAIL'}] gold patch should resolve")
    return v.resolved


def null_patch(iid: str) -> bool:
    """An empty patch must NOT resolve -- proves the fail_to_pass tests actually fail at base."""
    spec = load_instances()[iid]
    print(f"null-patch: {iid}")
    v = run_verifier(spec, "")
    print(f"  resolved={v.resolved} f2p={v.f2p_passed}/{v.f2p_total} wall={v.wall_s:.0f}s")
    print(f"  [{'PASS' if not v.resolved else 'FAIL'}] empty patch should NOT resolve")
    return not v.resolved


def smoke(iid: str) -> bool:
    """One baseline episode end to end: container, tools, Flex sandbox, patch, verifier."""
    from swebp_common import SWEAgent, disable_cache, make_lms

    import dspy

    exec_lm, _ = make_lms()
    dspy.configure(lm=exec_lm)
    disable_cache()
    agent = SWEAgent()
    pred = agent(instance_id=iid)
    print(f"  resolved={pred.resolved} commands={pred.n_commands} patch_chars={pred.patch_chars} "
          f"agent={pred.agent_wall_s:.0f}s verify={pred.verify_wall_s:.0f}s "
          f"error={pred.harness_error} infra={pred.infra_error}")
    print(f"  summary: {str(pred.summary)[:300]}")
    healthy = not pred.infra_error and pred.n_commands > 0
    print(f"  [{'PASS' if healthy else 'FAIL'}] episode ran and the harness touched the container")
    return healthy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", metavar="IID")
    ap.add_argument("--null", metavar="IID")
    ap.add_argument("--oracle-split", choices=["train", "val", "test"])
    ap.add_argument("--smoke", metavar="IID")
    args = ap.parse_args()

    ok = check_environment()
    if args.oracle:
        ok &= oracle(args.oracle)
    if args.null:
        ok &= null_patch(args.null)
    if args.oracle_split:
        train, val, test = load_splits()
        split = {"train": train, "val": val, "test": test}[args.oracle_split]
        results = [oracle(ex.instance_id) for ex in split]
        print(f"oracle over {args.oracle_split}: {sum(results)}/{len(results)} passed")
        ok &= all(results)
    if args.smoke:
        ok &= smoke(args.smoke)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
