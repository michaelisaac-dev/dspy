"""Comparison figure: published study findings vs what the Flex-compiled program
determined, in code and in behavior. Panel 2 (Utrecht planted bias) renders once
utrecht results exist in results/results.jsonl."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "results"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, AQUA = "#2a78d6", "#1baf7a"

QUOTE = ('"White names receive 50 percent more callbacks for interviews."\n'
         "— Bertrand & Mullainathan (AER 2004). Real employers, 2001–02 field\n"
         "experiment; names (race/gender) randomized across identical resumes.")

# (label, study_gt, artifact_encoded, program_predicted) in % — see RESULTS.md
BM_ROWS = [
    ("White-coded names", 9.65, 10.4, 6.0),
    ("Black-coded names", 6.45, 6.2, 2.3),
    ("Female-coded", None, 8.6, 5.4),   # study: no significant gender effect
    ("Male-coded", None, 7.0, 0.5),
]
BM_RATIOS = ("white / black ratio:", "study 1.50x", "in code 1.68x", "predicted 2.6x")


def utrecht_rows():
    rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text().splitlines()
            if '"utrecht_B"' in l]
    audit = [r for r in rows if r["arm"] == "group_rate_audit" and r["treatment"] == "full"]
    if not audit:
        return None
    r = audit[0]["predicted_yes_rates"]
    return [
        ("Male candidates", 45.5, None, 100 * r.get("gender=male", 0)),
        ("Female candidates", 14.0, None, 100 * r.get("gender=female", 0)),
    ]


def panel(ax, rows, title, xlab, series_labels):
    ys = range(len(rows), 0, -1)
    for y, (label, gt, enc, pred) in zip(ys, rows):
        ax.plot([0, 60], [y, y], color=GRID, linewidth=0.8, zorder=1)
        if gt is not None:
            ax.scatter([gt], [y], s=110, marker="D", color=INK, zorder=3,
                       edgecolors=SURFACE, linewidths=1.4)
            ax.annotate(f"{gt:.1f}", (gt, y), xytext=(0, 11), ha="center",
                        textcoords="offset points", fontsize=8.5, color=INK)
        else:
            ax.annotate("study: no significant gender effect", (3.0, y), xytext=(0, -20),
                        textcoords="offset points", fontsize=8, color=MUTED, ha="center")
        if enc is not None:
            ax.scatter([enc], [y], s=95, color=BLUE, zorder=3,
                       edgecolors=SURFACE, linewidths=1.4)
            ax.annotate(f"{enc:.1f}", (enc, y), xytext=(0, -16), ha="center",
                        textcoords="offset points", fontsize=8.5, color=INK2)
        ax.scatter([pred], [y], s=95, color=AQUA, zorder=3,
                   edgecolors=SURFACE, linewidths=1.4)
        ax.annotate(f"{pred:.1f}", (pred, y), xytext=(0, -16 if enc is None else 11),
                    ha="center", textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
    ax.set_xlabel(xlab, fontsize=9.5, color=INK2)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10, fontweight="bold")
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0.4, len(rows) + 0.9)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for lbl, col, mk in series_labels:
        ax.scatter([], [], s=90, color=col, marker=mk, label=lbl,
                   edgecolors=SURFACE, linewidths=1.2)


def main():
    ut = utrecht_rows()
    ncols = 2 if ut else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7.8 * ncols, 5.6), facecolor=SURFACE)
    axes = [axes] if ncols == 1 else list(axes)

    panel(axes[0], BM_ROWS,
          "Bertrand–Mullainathan resume audit (real employers)",
          "callback rate (%)",
          [("study ground truth", INK, "D"),
           ("encoded in Flex artifact (source constants)", BLUE, "o"),
           ("Flex program predictions (held-out)", AQUA, "o")])
    axes[0].set_xlim(0, 14)
    half = 0.5 if ncols == 2 else 1.0
    fig.text(0.07, 0.30, "  ".join(BM_RATIOS), fontsize=9, color=INK2)
    fig.text(0.07, 0.05, QUOTE, fontsize=9, color=MUTED, style="italic")

    if ut:
        panel(axes[1], ut,
              "Utrecht recruitment, company B (synthetic, planted gender bias)",
              "hire rate (%)",
              [("planted ground truth", INK, "D"),
               ("Flex program predictions (held-out)", AQUA, "o")])
        axes[1].set_xlim(0, 60)
        fig.text(0.57, 0.30, "planted male/female ratio 3.25x (ICT Institute)",
                 fontsize=9, color=INK2)

    handles, labels = axes[0].get_legend_handles_labels()
    if ut:
        h2, l2 = axes[1].get_legend_handles_labels()
        for h, l in zip(h2, l2):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.06, 0.17),
               ncol=min(3, len(labels)), fontsize=8.5, frameon=False, labelcolor=INK)
    fig.suptitle("What the study found vs what the compiled program determined",
                 fontsize=12.5, color=INK, x=0.02, ha="left")
    fig.subplots_adjust(left=0.16 * half + 0.02, right=0.97, top=0.80, bottom=0.42,
                        wspace=0.35)
    fig.savefig(OUT / "fig_forensics_comparison.png", dpi=200, facecolor=SURFACE)
    print("wrote fig_forensics_comparison.png", "(with Utrecht panel)" if ut else "(B&M only)")


if __name__ == "__main__":
    main()
