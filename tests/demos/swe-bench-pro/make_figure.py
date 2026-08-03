"""Instance-grid figure for the SWE-bench Pro harness-evolution result.

    python make_figure.py --resolved 4 --baseline-resolved 0 --n 12

One cell per held-out instance, one row per arm. A count-of-n grid instead of a bar chart on
purpose: at n=12 a percentage bar dramatizes what the data cannot support, while a grid keeps the
sample size visible. Resolved cells take categorical slot 1 (blue); unresolved cells are neutral.
Numbers default to the larger rerun reported by the operator (baseline 0/12, evolved 4/12);
`results.json` still holds the pilot (0/12, 3/12) and is untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_ = np.ndarray  # materialize the real numpy before matplotlib (dspy lazy-proxy)
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

DEMO_DIR = Path(__file__).parent
FIG_DIR = DEMO_DIR / "figures"

C_RESOLVED = "#2a78d6"   # categorical slot 1 (validated)
C_EMPTY = "#f0efec"      # neutral cell fill
C_EMPTY_EDGE = "#c3c2b7"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"


def cell(ax, x: float, y: float, filled: bool) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), 0.72, 0.72,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor=C_RESOLVED if filled else C_EMPTY,
        edgecolor=SURFACE if filled else C_EMPTY_EDGE,
        linewidth=1.0, mutation_aspect=1.0,
    ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--baseline-resolved", type=int, default=0)
    ap.add_argument("--resolved", type=int, default=4)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    rows = [
        ("Out-of-the-box harness", args.baseline_resolved),
        ("GEPA-evolved harness", args.resolved),
    ]
    for r, (label, k) in enumerate(rows):
        y = 1.15 - r * 1.15
        ax.text(-0.35, y + 0.36, label, ha="right", va="center", fontsize=10.5, color=INK)
        for i in range(args.n):
            cell(ax, i * 0.95, y, filled=i < k)
        ax.text(args.n * 0.95 + 0.15, y + 0.36, f"{k} of {args.n}",
                ha="left", va="center", fontsize=11.5, color=INK, fontweight="bold")

    fig.text(0.02, 0.94, "Same small model, different harness",
             fontsize=13, color=INK, ha="left", va="top")
    fig.text(0.02, 0.82,
             "SWE-bench Pro issues resolved, Claude Haiku 4.5 executing in both arms. "
             "Each cell is one held-out issue.",
             fontsize=9, color=INK_2, ha="left", va="top")
    fig.text(0.02, 0.045,
             "Gained 4, lost 0 on paired issues. Small pilot; directional, "
             "not statistically significant.",
             fontsize=8, color=MUTED, ha="left", va="bottom")

    ax.set_xlim(-4.2, args.n * 0.95 + 1.6)
    ax.set_ylim(-0.35, 2.05)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(top=0.66, bottom=0.16, left=0.02, right=0.98)

    FIG_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(FIG_DIR / f"swebp_resolved_grid.{ext}", facecolor=SURFACE,
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("saved figures/swebp_resolved_grid.svg (+.png)")


if __name__ == "__main__":
    main()
