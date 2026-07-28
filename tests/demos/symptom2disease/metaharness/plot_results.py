"""Render the Meta-Harness comparison figure from results.json + run.log.

    python plot_results.py

Three panels, in the order the argument runs:

  A  Where everything landed against the paper's Table 2. The question the run asked.
  B  Per-class delta between the two metric arms. What the scoring-function change actually did.
  C  Valset trajectory vs final test accuracy. Why B did not add up to a win.

Palette: dataviz reference palette, light mode. Two categorical hues only -- blue `#2a78d6`
(slot 1) and red `#e34948` (slot 8), which clear every gate under `--pairs all`
(CVD dE 21.6 protan, normal-vision 32.3, both >= floor) -- plus the muted/gridline inks. Orange
was the obvious third accent and was dropped: `#eb6834` against `#e34948` fails both the CVD
separation (dE 5.6 deutan) and the normal-vision floor (dE 7.1), and the two would have shared
panel A. Red is used consistently for one meaning throughout -- "the thing the change cost you"
in B and C, "the bar we did not clear" in A.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MH_DIR = Path(__file__).parent
RESULTS = MH_DIR / "results.json"
LOG = MH_DIR / "run.log"
OUT = MH_DIR / "metaharness_comparison.png"

BLUE, RED = "#2a78d6", "#e34948"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CONTEXT = "#d8d7d1"  # de-emphasised fill for the paper's context rows

ARM_LABEL = {"accuracy": "metric=accuracy (paper's)", "contrastive": "metric=contrastive (new)"}

# Panel A and panel B share a left margin, so every y-label has to fit the same gutter. The one
# genuinely long class name is shortened; the rest are already short enough.
SHORT_CLASS = {"gastroesophageal reflux disease": "GERD (reflux)"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def val_trajectories(log_path: Path) -> dict[str, dict]:
    """Per-arm valset score of every candidate GEPA accepted, straight out of the run log.

    Parsed rather than stored because GEPA reports it to stdout, not into results.json -- and it
    is the panel that explains the other two, so it is worth the regex.
    """
    txt = log_path.read_text(errors="ignore")
    parts = re.split(r"=== arm (\w+)", txt)
    out = {}
    for i in range(1, len(parts), 2):
        name, body = parts[i], parts[i + 1]
        base = re.findall(r"valset score: ([0-9.]+) over", body)
        out[name] = {
            "base": float(base[0]) if base else None,
            "candidates": [float(x) for x in re.findall(r"Valset score for new program: ([0-9.]+)", body)],
        }
    return out


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def main() -> None:
    # dspy registers a lazy numpy proxy in sys.modules; matplotlib's `from numpy.exceptions import
    # ...` trips that proxy into a recursive import. Materialize the real numpy first.
    import numpy as np
    _ = np.ndarray
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    paper = data["meta"]["paper"]["s2d_table2"]
    base, arms = data["baseline"], data["arms"]
    n_test = base["test"]["n"]

    fig = plt.figure(figsize=(14.5, 10.5))
    fig.patch.set_facecolor(SURFACE)
    # Explicit margins rather than tight_layout: panel A spans two columns, which tight_layout
    # cannot solve, and the y-labels in A and B need a shared, generous left gutter.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.30,
                          left=0.168, right=0.985, top=0.885, bottom=0.065)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---- Panel A: everything against the paper's Table 2 -------------------------------------
    # Emphasis form, not categorical: the paper's rows are context (gray), ours are the subject
    # (blue), and the one number the run was aiming at is the single red bar.
    rows: list[tuple[str, float, str, tuple[float, float] | None]] = [
        ("zero-shot", paper["zero_shot"], "ctx", None),
        ("few-shot (8)", paper["fewshot_8"], "ctx", None),
        ("few-shot (32)", paper["fewshot_32"], "ctx", None),
        ("ACE", paper["ACE"], "ctx", None),
        ("few-shot (all)", paper["fewshot_all"], "ctx", None),
        ("MCE", paper["MCE"], "ctx", None),
        ("Meta-Harness", paper["meta_harness"], "target", None)]
    for key, lab in [(None, "ours: zero-shot Flex"), ("accuracy", f"ours: {ARM_LABEL['accuracy']}"),
                     ("contrastive", f"ours: {ARM_LABEL['contrastive']}")]:
        row = base["test"] if key is None else arms[key]["test"]
        k = round(row["accuracy"] * row["n"])
        lo, hi = wilson(k, row["n"])
        rows.append((lab, row["accuracy"] * 100, "ours", (lo * 100, hi * 100)))

    rows.sort(key=lambda r: r[1])
    ys = range(len(rows))
    colors = {"ctx": CONTEXT, "target": RED, "ours": BLUE}
    for y, (lab, v, kind, ci) in zip(ys, rows):
        ax_a.barh(y, v, height=0.62, color=colors[kind], zorder=3,
                  edgecolor=SURFACE, linewidth=2)  # 2px surface gap between adjacent bars
        if ci:
            ax_a.plot([ci[0], ci[1]], [y, y], color=INK_2, linewidth=1.4, zorder=4,
                      solid_capstyle="butt")
            for x in ci:
                ax_a.plot([x, x], [y - 0.13, y + 0.13], color=INK_2, linewidth=1.4, zorder=4)
        # Anchor the value label past the CI whisker, not past the bar end, or the two collide.
        label_x = ci[1] if ci else v
        ax_a.annotate(f"{v:.1f}", (label_x, y), xytext=(8, 0),
                      textcoords="offset points", va="center", fontsize=9.5,
                      color=INK if kind != "ctx" else INK_2,
                      fontweight="bold" if kind != "ctx" else "normal", zorder=5)

    ax_a.axvline(paper["meta_harness"], color=RED, linewidth=1.6, linestyle=(0, (5, 3)), zorder=2)
    ax_a.annotate("Meta-Harness 86.8", (paper["meta_harness"], len(rows) - 0.35),
                  xytext=(6, 0), textcoords="offset points", fontsize=9, color=RED, va="center")
    ax_a.set_yticks(list(ys))
    ax_a.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK_2)
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("accuracy on the paper's 212-example test set  (%)  ·  chance = 4.5%")
    ax_a.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax_a.set_title(
        f"A · Nobody's scoring function got there. Same 212 examples, same 22 classes.\n"
        f"Bars for our three rows carry Wilson 95% intervals (n={n_test}); the paper reports "
        f"point estimates only.",
        fontsize=10.5, loc="left", pad=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CONTEXT),
               plt.Rectangle((0, 0), 1, 1, color=RED),
               plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    ax_a.legend(handles, ["Meta-Harness paper, Table 2", "the target", "this run (dspy.Flex + GEPA)"],
                loc="lower right", fontsize=9, frameon=False, labelcolor=INK_2)
    _style(ax_a)

    # ---- Panel B: what the scoring-function change actually did ------------------------------
    # Diverging: polarity around zero, documented blue<->red pair, gray zero line.
    pa = arms["accuracy"]["test"]["per_class_accuracy"]
    pb = arms["contrastive"]["test"]["per_class_accuracy"]
    deltas = sorted(((c, (pb[c]["accuracy"] - pa[c]["accuracy"]) * 100) for c in pa),
                    key=lambda t: t[1])
    deltas = [d for d in deltas if abs(d[1]) > 1e-9]
    ys = range(len(deltas))
    for y, (lab, d) in zip(ys, deltas):
        ax_b.barh(y, d, height=0.66, color=(BLUE if d > 0 else RED), zorder=3,
                  edgecolor=SURFACE, linewidth=2)
        ax_b.annotate(f"{d:+.0f}", (d, y), xytext=(6 if d > 0 else -6, 0),
                      textcoords="offset points", va="center",
                      ha="left" if d > 0 else "right", fontsize=9, color=INK_2, zorder=4)
    ax_b.axvline(0, color=AXIS, linewidth=1.2, zorder=2)
    ax_b.set_yticks(list(ys))
    ax_b.set_yticklabels([SHORT_CLASS.get(d[0], d[0]) for d in deltas], fontsize=9, color=INK_2)
    ax_b.set_xlim(-115, 115)
    ax_b.set_xlabel("per-class accuracy, contrastive − accuracy  (percentage points)")
    ax_b.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax_b.set_title(
        "B · The mechanism worked, then paid for itself.\n"
        "Contrastive exemplars fixed bronchial asthma outright (+90pp) and\n"
        "peptic ulcer (+50pp), then gave it back across twelve other classes —\n"
        "five of which the control had answered perfectly.",
        fontsize=10.5, loc="left", pad=10)
    ax_b.legend([plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=RED)],
                ["contrastive better", "contrastive worse"],
                loc="lower right", fontsize=9, frameon=False, labelcolor=INK_2)
    _style(ax_b)

    # ---- Panel C: why B did not add up ------------------------------------------------------
    traj = val_trajectories(LOG)
    arm_color = {"accuracy": BLUE, "contrastive": RED}
    for name in ("accuracy", "contrastive"):
        t = traj[name]
        xs = list(range(1, len(t["candidates"]) + 1))
        ax_c.plot(xs, [v * 100 for v in t["candidates"]], color=arm_color[name], linewidth=2,
                  marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
                  label=f"{ARM_LABEL[name]} — valset (n=50)")
        test_acc = arms[name]["test"]["accuracy"] * 100
        ax_c.axhline(test_acc, color=arm_color[name], linewidth=1.6, linestyle=(0, (5, 3)),
                     zorder=2)
        ax_c.annotate(f"test {test_acc:.1f}", (1.0, test_acc), xytext=(2, 6 if name == "accuracy" else -14),
                      textcoords="offset points", fontsize=9, color=arm_color[name], ha="left")
    # The peak is candidate 7 of 8, not candidate 8 -- "still climbing" is a statement about where
    # the best candidate landed in the search, so anchor it there and say so.
    ax_c.annotate("best candidate arrived 7th of 8:\nthe search had not converged",
                  (7, 82), xytext=(-18, -58), textcoords="offset points", fontsize=9,
                  color=RED, ha="right",
                  arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.2,
                                  connectionstyle="arc3,rad=-0.2"))
    ax_c.set_xlabel("accepted candidate (in search order)")
    ax_c.set_ylabel("accuracy (%)")
    ax_c.set_xticks(list(range(1, 9)))
    ax_c.grid(True, color=GRID, linewidth=0.8)
    ax_c.set_title(
        "C · It won the objective and lost the test set.\n"
        "Contrastive drove the 50-example valset to 82.0 (control stalled at 76.0)\n"
        "and still lost 3.3pp on the 212 held-out cases: a ~9pp overfit swing.",
        fontsize=10.5, loc="left", pad=10)
    ax_c.legend(loc="lower left", fontsize=9, frameon=False, labelcolor=INK_2)
    _style(ax_c)

    meta = data["meta"]
    spend = sum(a["optimization"]["cost_usd_litellm"] for a in arms.values())
    fig.suptitle(
        "Can a better scoring function beat Meta-Harness on Symptom2Disease?  —  no, and the "
        "reason is instructive",
        fontsize=14.5, color=INK, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.955,
             f"dspy.Flex + GEPA · executor {meta['exec_model'].split('/')[-1]} · reflection "
             f"{meta['reflection_model'].split('/')[-1]} · the paper's own 200/50/212 split and "
             f"its own evaluator (0 disagreements)",
             fontsize=9.5, color=INK_2, ha="left")
    fig.text(0.008, 0.932,
             f"Both arms share one identical 1/0-accuracy score and differ only in the feedback "
             f"string GEPA's reflection LM reads · ${spend:.2f} of search",
             fontsize=9.5, color=INK_2, ha="left")
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
