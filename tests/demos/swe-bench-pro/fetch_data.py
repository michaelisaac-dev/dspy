"""Fetch SWE-bench Pro (public set) and the official run scripts; build the local manifest.

    python fetch_data.py                      # dataset + upstream clone -> swebp_data/manifest.jsonl
    python fetch_data.py --pull-images pilot  # pre-pull the 30 pilot images (~0.5-2 GB each)

The manifest keeps every field the demo needs, including the gold patch -- which is used only by
`preflight.py --oracle` to validate the verifier at $0 and is never shown to any model. Everything
under swebp_data/ and swebp_upstream/ is reproducible from this one command and is gitignored.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).parent
DATA_DIR = DEMO_DIR / "swebp_data"
MANIFEST = DATA_DIR / "manifest.jsonl"
UPSTREAM = DEMO_DIR / "swebp_upstream"
UPSTREAM_URL = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
HF_DATASET = "ScaleAI/SWE-bench_Pro"


def ensure_upstream() -> None:
    if (UPSTREAM / "run_scripts").exists():
        print(f"upstream present: {UPSTREAM}")
        return
    print(f"cloning {UPSTREAM_URL} (shallow)...")
    subprocess.run(["git", "clone", "--depth", "1", UPSTREAM_URL, str(UPSTREAM)], check=True)


def build_manifest() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="test")
    DATA_DIR.mkdir(exist_ok=True)
    rows, skipped = [], []
    for r in ds:
        iid = r["instance_id"]
        # An instance without local run scripts cannot be verified; record and skip rather than
        # letting it fail mid-run. (The upstream repo ships scripts for more instances than the
        # public HF set, so this is expected to skip nothing.)
        if not (UPSTREAM / "run_scripts" / iid / "run_script.sh").exists():
            skipped.append(iid)
            continue
        rows.append({
            "instance_id": iid,
            "repo": r["repo"],
            "language": {"py": "python"}.get(r["repo_language"], r["repo_language"]),
            "base_commit": r["base_commit"],
            "dockerhub_tag": r["dockerhub_tag"],
            "problem_statement": r["problem_statement"],
            "requirements": r["requirements"],
            "interface": r["interface"],
            "before_last_cmd": r["before_repo_set_cmd"].strip().split("\n")[-1],
            "selected_test_files": eval(r["selected_test_files_to_run"]),
            "fail_to_pass": eval(r["fail_to_pass"]),
            "pass_to_pass": eval(r["pass_to_pass"]),
            "gold_patch": r["patch"],
        })
    with MANIFEST.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"manifest: {len(rows)} instances -> {MANIFEST}")
    if skipped:
        print(f"WARNING: {len(skipped)} instances skipped (no run scripts), e.g. {skipped[:3]}")
    return rows


def pull_images(split: str) -> None:
    sys.path.insert(0, str(DEMO_DIR))
    from swebp_common import PLATFORM, _docker, load_instances, load_splits

    if split == "pilot":
        train, val, test = load_splits()
        ids = [ex.instance_id for ex in train + val + test]
    else:
        ids = list(load_instances())
    specs = load_instances()
    print(f"pulling {len(ids)} images...")
    for i, iid in enumerate(ids, 1):
        image = specs[iid].image
        if _docker(["image", "inspect", image], timeout=120).returncode == 0:
            print(f"  [{i}/{len(ids)}] cached  {image.split(':')[1][:60]}")
            continue
        args = ["pull"] + ([f"--platform={PLATFORM}"] if PLATFORM else []) + [image]
        proc = _docker(args, timeout=3600)
        state = "ok" if proc.returncode == 0 else f"FAILED: {proc.stderr.strip()[-120:]}"
        print(f"  [{i}/{len(ids)}] {state:<6}  {image.split(':')[1][:60]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull-images", metavar="SPLIT", help="'pilot' (30 images) or 'all' (731)")
    args = ap.parse_args()
    ensure_upstream()
    build_manifest()
    if args.pull_images:
        pull_images(args.pull_images)


if __name__ == "__main__":
    main()
