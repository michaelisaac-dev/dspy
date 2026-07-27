"""Fetch Symptom2Disease and write a deduplicated manifest.

    python fetch_data.py            # download + dedup + manifest
    python fetch_data.py --stats    # just report on what is already there

The raw CSV contains 1200 rows over 24 diseases, 50 each. 43 of its texts appear more than once
(90 rows involved). Every duplicate carries a consistent label, so dropping the later copies loses
no information -- but leaving them in would let the same symptom description land in both the
training pool and the test set, which is exactly the leakage the emails demo had to fix after the
fact. Dedup happens here, once, and the manifest is what everything downstream reads.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

DEMO_DIR = Path(__file__).parent
DATA_DIR = DEMO_DIR / "eval_data"
CSV_PATH = DATA_DIR / "Symptom2Disease.csv"
MANIFEST_PATH = DATA_DIR / "symptom2disease.jsonl"

REPO_ID = "NeuronZero/Symptom2Disease"
CSV_NAME = "Symptom2Disease.csv"


def download() -> Path:
    from huggingface_hub import hf_hub_download
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        print(f"already present: {CSV_PATH.name}")
        return CSV_PATH
    src = hf_hub_download(REPO_ID, CSV_NAME, repo_type="dataset")
    CSV_PATH.write_bytes(Path(src).read_bytes())
    print(f"downloaded {REPO_ID}/{CSV_NAME} -> {CSV_PATH}")
    return CSV_PATH


def build_manifest() -> list[dict]:
    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    raw = len(df)

    # Deduplicate on the symptom text. Verified: no duplicate text carries two different labels,
    # so keeping the first copy is lossless. `keep="first"` on a stable read makes this
    # deterministic run to run.
    conflicts = [t for t, g in df.groupby("text") if g["label"].nunique() > 1]
    if conflicts:
        raise SystemExit(f"{len(conflicts)} duplicate text(s) carry conflicting labels; "
                         f"resolve before building a manifest: {conflicts[:3]}")
    df = df.drop_duplicates("text", keep="first")

    rows = [{"text": str(r.text).strip(), "label": str(r.label).strip()}
            for r in df.itertuples()]
    rows.sort(key=lambda r: (r["label"], r["text"]))  # deterministic order, independent of CSV order
    MANIFEST_PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                             encoding="utf-8")
    print(f"{raw} raw rows -> {len(rows)} unique ({raw - len(rows)} duplicate texts dropped) "
          f"-> {MANIFEST_PATH}")
    return rows


def stats(rows: list[dict]) -> None:
    counts = collections.Counter(r["label"] for r in rows)
    lengths = [len(r["text"]) for r in rows]
    print(f"  classes     {len(counts)}  (per class: min={min(counts.values())} "
          f"max={max(counts.values())})")
    thin = {k: v for k, v in sorted(counts.items()) if v < 45}
    if thin:
        print(f"  thin classes {thin}")
    print(f"  text chars  median={int(statistics.median(lengths))} max={max(lengths)} "
          f"(~{int(statistics.median(lengths) / 4)} tokens median)")
    print(f"  duplicates  {len(rows) - len({r['text'] for r in rows})}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="store_true", help="report on the existing manifest and exit")
    args = ap.parse_args()

    if args.stats:
        if not MANIFEST_PATH.exists():
            raise SystemExit(f"no manifest at {MANIFEST_PATH}; run `python fetch_data.py` first")
        rows = [json.loads(line) for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        stats(rows)
        return

    download()
    stats(build_manifest())


if __name__ == "__main__":
    main()
