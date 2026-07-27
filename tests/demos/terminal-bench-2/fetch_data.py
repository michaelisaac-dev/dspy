"""Fetch Terminal-Bench 2.0 and flatten it into a manifest the demo can load.

Terminal-Bench 2.0 is 89 containerized terminal tasks (https://github.com/harbor-framework/terminal-bench-2).
Each task is a directory holding:

    task.toml         metadata: prebuilt docker image, cpu/memory caps, agent + verifier timeouts
    instruction.md    the prompt handed to the agent
    environment/      Dockerfile + build context (NOT needed here -- every task pins a prebuilt image)
    tests/test.sh     the verifier; writes 1 or 0 to /logs/verifier/reward.txt
    solution/         the oracle solve.sh, used by preflight.py to smoke-test the runner

Because every task pins a prebuilt `docker_image`, the repo clone can skip Git-LFS blobs entirely
(they are only build context for images we never build). That takes the checkout from ~1 GB to
~80 MB and is why this script sets GIT_LFS_SKIP_SMUDGE=1.

    python fetch_data.py                      # clone (if needed) + write tasks.jsonl
    python fetch_data.py --update             # git pull first
    python fetch_data.py --manifest-only      # rebuild tasks.jsonl from an existing checkout
    python fetch_data.py --pull-images        # docker pull every image (~large; do this once)
    python fetch_data.py --pull-images --split test   # ...only the split you are about to run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import tomllib

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEMO_DIR = Path(__file__).parent
DATA_DIR = DEMO_DIR / "tb2_data"
REPO_DIR = DATA_DIR / "terminal-bench-2"
MANIFEST_PATH = DATA_DIR / "tasks.jsonl"

REPO_URL = "https://github.com/harbor-framework/terminal-bench-2.git"


# Tasks are shell scripts destined for a Linux container. Git's `core.autocrlf=true` -- the default
# on many Windows installs -- rewrites every one of them to CRLF in the working tree. `docker cp`
# copies bytes, so the container gets a script bash cannot parse (`$'\r': command not found`, then
# `syntax error: unexpected end of file`): the verifier dies, every task scores 0, and it looks
# exactly like an agent that solved nothing. Pin the checkout to LF regardless of global config.
LF_CONFIG = ["-c", "core.autocrlf=false", "-c", "core.eol=lf"]


def clone(update: bool = False) -> None:
    """Shallow-clone the task repo without LFS blobs, or pull if it is already there."""
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    if REPO_DIR.exists():
        if update:
            print(f"updating {REPO_DIR}")
            subprocess.run(["git", "-C", str(REPO_DIR), *LF_CONFIG, "pull", "--ff-only"],
                           check=True, env=env)
        else:
            print(f"using existing checkout {REPO_DIR}")
        fix_line_endings()
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"cloning {REPO_URL} -> {REPO_DIR} (LFS blobs skipped, line endings pinned to LF)")
    subprocess.run(["git", *LF_CONFIG, "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
                   check=True, env=env)
    fix_line_endings()


def fix_line_endings() -> bool:
    """Re-check out the working tree with LF endings if anything came out CRLF. Returns True if repaired."""
    probe = sorted(REPO_DIR.glob("*/tests/test.sh"))
    crlf = [p for p in probe if b"\r\n" in p.read_bytes()]
    if not crlf:
        return False
    print(f"  !! {len(crlf)}/{len(probe)} verifier scripts have CRLF endings (git core.autocrlf); "
          f"re-checking out with LF")
    git = ["git", "-C", str(REPO_DIR)]
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    # Pin it in the repo's own config so a later `git pull` here cannot reintroduce the problem,
    # then drop and restore the index so every file is written out again through the new filter.
    subprocess.run([*git, "config", "core.autocrlf", "false"], check=True, env=env)
    subprocess.run([*git, "config", "core.eol", "lf"], check=True, env=env)
    subprocess.run([*git, "rm", "--cached", "-r", "-q", "."], check=True, env=env)
    subprocess.run([*git, "reset", "--hard", "-q"], check=True, env=env)
    still = [p for p in sorted(REPO_DIR.glob("*/tests/test.sh")) if b"\r\n" in p.read_bytes()]
    print(f"  {'repaired' if not still else f'STILL BROKEN for {len(still)} task(s)'}")
    return True


def _read_task(task_dir: Path) -> dict | None:
    """Flatten one task directory into a manifest row, or None if it is not a task."""
    toml_path = task_dir / "task.toml"
    instr_path = task_dir / "instruction.md"
    if not (toml_path.exists() and instr_path.exists()):
        return None
    cfg = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    env = cfg.get("environment", {})
    meta = cfg.get("metadata", {})
    return {
        "name": task_dir.name,
        "qualified_name": cfg.get("task", {}).get("name", task_dir.name),
        "instruction": instr_path.read_text(encoding="utf-8").strip(),
        "description": cfg.get("task", {}).get("description", ""),
        "docker_image": env.get("docker_image"),
        "cpus": env.get("cpus", 1),
        "memory_mb": env.get("memory_mb", 2048),
        "gpus": env.get("gpus", 0),
        "allow_internet": bool(env.get("allow_internet", True)),
        "agent_timeout_s": float(cfg.get("agent", {}).get("timeout_sec", 900.0)),
        "verifier_timeout_s": float(cfg.get("verifier", {}).get("timeout_sec", 900.0)),
        "difficulty": meta.get("difficulty", "unknown"),
        "category": meta.get("category", "unknown"),
        "expert_time_min": meta.get("expert_time_estimate_min"),
        "task_dir": str(task_dir.relative_to(DATA_DIR)).replace("\\", "/"),
        "has_solution": (task_dir / "solution" / "solve.sh").exists(),
        "n_test_files": len(list((task_dir / "tests").glob("*"))) if (task_dir / "tests").is_dir() else 0,
    }


def build_manifest() -> list[dict]:
    if not REPO_DIR.exists():
        raise SystemExit(f"no checkout at {REPO_DIR}; run without --manifest-only first")
    rows = sorted(
        (r for r in (_read_task(d) for d in REPO_DIR.iterdir() if d.is_dir()) if r),
        key=lambda r: r["name"],
    )
    # A task with no prebuilt image would need `docker build` against LFS content this script
    # deliberately skipped, so it cannot be run here. Drop it loudly rather than failing later.
    usable = [r for r in rows if r["docker_image"]]
    for r in rows:
        if not r["docker_image"]:
            print(f"  !! {r['name']}: no docker_image in task.toml -- excluded")
    gpu = [r for r in usable if r["gpus"]]
    for r in gpu:
        print(f"  !! {r['name']}: needs {r['gpus']} GPU(s) -- kept, but will fail on a CPU host")
    MANIFEST_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in usable) + "\n", encoding="utf-8"
    )
    return usable


def summarize(rows: list[dict]) -> None:
    print(f"\n{len(rows)} tasks -> {MANIFEST_PATH}")
    for field in ("difficulty", "category"):
        counts = Counter(r[field] for r in rows).most_common()
        print(f"  {field:<11} " + ", ".join(f"{k}={v}" for k, v in counts))
    no_sol = [r["name"] for r in rows if not r["has_solution"]]
    print(f"  images      {len({r['docker_image'] for r in rows})} unique")
    print(f"  no internet {sum(1 for r in rows if not r['allow_internet'])}")
    print(f"  no solution {len(no_sol)}" + (f" ({', '.join(no_sol[:5])})" if no_sol else ""))
    chars = sorted(len(r["instruction"]) for r in rows)
    print(f"  instruction chars: median {chars[len(chars) // 2]}, max {chars[-1]}")


def pull_images(rows: list[dict]) -> None:
    """Pre-pull images so a first `docker run` does not blow an episode's wall-clock budget."""
    images = sorted({r["docker_image"] for r in rows})
    print(f"\npulling {len(images)} image(s) -- this is slow and large the first time")
    failed = []
    for i, image in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {image}", flush=True)
        # 30 min: some of these images carry model checkpoints and datasets.
        proc = subprocess.run(["docker", "pull", image], capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            failed.append(image)
            print(f"      FAILED: {proc.stderr.strip().splitlines()[-1:] or proc.stdout[-200:]}")
    print(f"pulled {len(images) - len(failed)}/{len(images)}" + (f", failed: {failed}" if failed else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="git pull an existing checkout")
    ap.add_argument("--manifest-only", action="store_true", help="skip git, just rebuild tasks.jsonl")
    ap.add_argument("--fix-line-endings", action="store_true",
                    help="only re-check out the tree with LF endings (see the LF_CONFIG comment)")
    ap.add_argument("--pull-images", action="store_true", help="docker pull the task images")
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="all",
                    help="with --pull-images, restrict to one split of the demo's fixed partition")
    args = ap.parse_args()

    if args.fix_line_endings:
        if not fix_line_endings():
            print("line endings are already LF; nothing to do")
        return
    if not args.manifest_only:
        clone(update=args.update)
    rows = build_manifest()
    summarize(rows)

    if args.pull_images:
        if args.split != "all":
            # Import late: tb2_common reads the manifest this script just wrote.
            sys.path.insert(0, str(DEMO_DIR))
            from tb2_common import load_splits

            train, val, test = load_splits()
            chosen = {"train": train, "val": val, "test": test}[args.split]
            names = {ex.task_name for ex in chosen}
            rows = [r for r in rows if r["name"] in names]
            print(f"\nrestricted to split '{args.split}': {len(rows)} tasks")
        pull_images(rows)


if __name__ == "__main__":
    main()
