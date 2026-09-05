"""Aggregate pilot results and plot against the LangProBe paper's own gpt-4o-mini matrix.

Reads results/results.jsonl (ours) + langprobe/experiment_data/20250305/gpt4omini_0305.csv
(theirs, shipped in their repo). Produces:
  results/fig_accuracy_vs_matrix.png  - per-task dot plot: our arms vs every paper cell
  results/fig_cost_accuracy.png       - test-time cost vs accuracy (heart)
  results/summary.md                  - aggregated tables

Palette: validated light-mode categorical slots (blue/orange/aqua) + neutral inks.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "results"
PAPER_CSV = HERE.parent / "langprobe/experiment_data/20250305/gpt4omini_0305.csv"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_C = "#c3c2b7"
BLUE = "#2a78d6"    # gepa_flex
ORANGE = "#eb6834"  # gepa_cot
AQUA = "#1baf7a"    # oneshot codegen
BENCH_LABEL = {"heart": "HeartDisease", "iris": "Iris (official split)",
               "iris_fixed": "Iris (fixed split)", "scone": "Scone"}
PAPER_BENCH = {"HeartDisease": "heart", "Iris": "iris", "Scone": "scone"}

ARMS = [
    ("predict_base", "Predict (unoptimized)", MUTED),
    ("cot_base", "CoT (unoptimized)", MUTED),
    ("flex_base", "Flex identity (unoptimized)", MUTED),
    ("gepa_cot", "Prompt space: GEPA (CoT)", ORANGE),
    ("gepa_flex", "Flex: GEPA-compiled", BLUE),
]


def load_ours():
    rows = []
    for line in (OUT / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["arm"].startswith("smoke"):
            continue
        r.setdefault("bench", "heart")  # early rows predate the bench field
        r.setdefault("seed", 0)
        rows.append(r)
    # keep last occurrence per (bench, seed, arm) — reruns overwrite
    dedup = {}
    for r in rows:
        dedup[(r["bench"], r["seed"], r["arm"])] = r
    return list(dedup.values())


LIST_PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)}  # $/M in, $/M out


def _cell_cost(row, model):
    """Recorded cost, or imputed from recorded token counts at OpenAI list prices.

    The paper's recorded costs reproduce exactly from tokens x list prices (verified on
    gpt-4o-mini Scone Predict: $0.0182), so imputation for cost==0 rows is their own formula."""
    c = float(row["cost"] or 0)
    if c > 0:
        return c, False
    pin, pout = LIST_PRICES[model]
    c = float(row["input_tokens"] or 0) / 1e6 * pin + float(row["output_tokens"] or 0) / 1e6 * pout
    return c, c > 0


def load_paper(model="gpt-4o-mini", fname="gpt4omini_0305.csv"):
    cells = defaultdict(list)  # bench -> [(program, optimizer, score, cost, imputed)]
    with open(PAPER_CSV.parent / fname) as f:
        for row in csv.DictReader(f):
            b = PAPER_BENCH.get(row["benchmark"])
            if b:
                cost, imputed = _cell_cost(row, model)
                cells[b].append((row["program"], row["optimizer"],
                                 float(row["score"]), cost, imputed))
    return cells


def fig_accuracy(ours, paper):
    benches = [b for b in ("heart", "iris", "iris_fixed", "scone")
               if any(r["bench"] == b for r in ours)]
    fig, axes = plt.subplots(1, len(benches), figsize=(5.4 * len(benches), 4.6),
                             facecolor=SURFACE)
    if len(benches) == 1:
        axes = [axes]
    for ax, bench in zip(axes, benches):
        ax.set_facecolor(SURFACE)
        scores = [c[2] for c in paper.get(bench, [])]
        best = max(scores) if scores else None
        # paper matrix cells: one gray dot per program x optimizer cell
        ax.scatter(scores, [0.35] * len(scores), s=42, color=BASELINE_C, zorder=2,
                   edgecolors=SURFACE, linewidths=1.2, clip_on=False)
        if best is not None:
            ax.scatter([best], [0.35], s=60, color=INK2, zorder=3,
                       edgecolors=SURFACE, linewidths=1.5)
            ax.annotate(f"paper best {best:.1f}", (best, 0.35), xytext=(0, 11),
                        textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
        ys, labels = [], []
        for i, (arm, label, color) in enumerate(ARMS):
            vals = [r["score"] for r in ours if r["bench"] == bench and r["arm"] == arm]
            if not vals:
                continue
            y = 1.0 + i * 0.55
            ax.scatter(vals, [y] * len(vals), s=64, color=color, zorder=3,
                       edgecolors=SURFACE, linewidths=1.5)
            m = mean(vals)
            note = f"{m:.1f}" + (f" ±{stdev(vals):.1f}" if len(vals) > 1 else "")
            ax.annotate(note, (max(vals), y), xytext=(8, -3),
                        textcoords="offset points", fontsize=9, color=INK)
            ys.append(y)
            labels.append(label)
        ax.set_yticks([0.35] + ys)
        ax.set_yticklabels(["LangProBe matrix\n(all cells, gpt-4o-mini)"] + labels,
                           fontsize=9, color=INK)
        ax.set_xlabel("accuracy (%)", fontsize=9.5, color=INK2)
        ax.set_title(BENCH_LABEL[bench], fontsize=12, color=INK, pad=10, loc="left",
                     fontweight="bold")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE_C)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_ylim(-0.1, 1.0 + len(ARMS) * 0.55)
    fig.suptitle("Flex + GEPA vs the LangProBe program x optimizer matrix (task model: gpt-4o-mini)",
                 fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_accuracy_vs_matrix.png", dpi=200, facecolor=SURFACE)
    print("wrote fig_accuracy_vs_matrix.png")


def fig_cost(ours, paper, bench="heart"):
    n_test = next(r["n_test"] for r in ours if r["bench"] == bench)
    fig, ax = plt.subplots(figsize=(7.4, 5.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    px = [c[3] / n_test * 1000 for c in paper.get(bench, [])]
    py = [c[2] for c in paper.get(bench, [])]
    ax.scatter(px, py, s=42, color=BASELINE_C, edgecolors=SURFACE, linewidths=1.2,
               zorder=2, label="LangProBe matrix cells")
    base_rows = [r for r in ours if r["bench"] == bench
                 and r["arm"] in ("predict_base", "cot_base", "flex_base")]
    ax.scatter([r["task_lm_cost_usd"] / r["n_test"] * 1000 for r in base_rows],
               [r["score"] for r in base_rows], s=90, color=INK2, zorder=3,
               edgecolors=SURFACE, linewidths=1.5,
               label="Unoptimized (Predict / CoT / Flex)")
    for arm, label, color in ARMS:
        if color == MUTED:
            continue
        rows = [r for r in ours if r["bench"] == bench and r["arm"] == arm]
        if not rows:
            continue
        xs = [r["task_lm_cost_usd"] / r["n_test"] * 1000 for r in rows]
        ys = [r["score"] for r in rows]
        ax.scatter(xs, ys, s=90, color=color, zorder=3,
                   edgecolors=SURFACE, linewidths=1.5, label=label)
    ax.set_xlabel("test-time LM cost per 1,000 records ($)", fontsize=10, color=INK2)
    ax.set_ylabel("accuracy (%)", fontsize=10, color=INK2)
    ax.set_title(f"{BENCH_LABEL[bench]}: accuracy vs inference cost — compiled code answers for $0",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE_C)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cost_accuracy.png", dpi=200, facecolor=SURFACE)
    print("wrote fig_cost_accuracy.png")


def summary_tables(ours, paper):
    lines = ["# Aggregated results\n"]
    for bench in ("heart", "iris", "iris_fixed", "scone"):
        rows = [r for r in ours if r["bench"] == bench]
        if not rows:
            continue
        cells = paper.get(bench, [])
        best = max(cells, key=lambda c: c[2]) if cells else None
        lines.append(f"\n## {BENCH_LABEL[bench]} (n_test={rows[0]['n_test']})\n")
        if best:
            lines.append(f"Paper matrix: {len(cells)} cells; best = {best[2]:.1f} "
                         f"({best[0]} + {best[1]}); baseline Predict = "
                         f"{next((c[2] for c in cells if c[0] == 'Predict' and c[1] == 'Baseline'), '?')}\n")
        lines.append("| arm | seeds | accuracy | test LM calls | test cost/1k |")
        lines.append("|---|---|---|---|---|")
        for arm, label, _ in ARMS:
            vals = [r for r in rows if r["arm"] == arm]
            if not vals:
                continue
            accs = [r["score"] for r in vals]
            acc = f"{mean(accs):.1f}" + (f" ± {stdev(accs):.1f}" if len(accs) > 1 else "")
            calls = round(mean(r["task_lm_calls"] for r in vals))
            cost = mean(r["task_lm_cost_usd"] / r["n_test"] * 1000 for r in vals)
            lines.append(f"| {label} | {len(vals)} | {acc} | {calls} | ${cost:.2f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("wrote summary.md")
    print("\n".join(lines))





def _pareto(points):
    """Upper-left Pareto set over (cost, score): cheapest-first, strictly rising score."""
    front, best = [], -1.0
    for c, s in sorted(points, key=lambda p: (p[0], -p[1])):
        if s > best:
            front.append((c, s))
            best = s
    return front


def fig_pareto(ours, paper):
    """The paper's cost/score chart (log-cost, Pareto hull), with our runs overlaid.

    Zero-cost compiled programs can't sit on a log axis, so they get a dedicated
    '$0' column at the left edge, separated by an axis break."""
    benches = [b for b in ("heart", "scone") if paper.get(b)]
    fig, axes = plt.subplots(1, len(benches), figsize=(5.6 * len(benches), 4.8),
                             facecolor=SURFACE)
    for ax, bench in zip(axes, benches):
        ax.set_facecolor(SURFACE)
        # paper cells with cost==0 have MISSING cost data (not free runs): keep them off
        # a cost axis entirely, and say so.
        n_imputed = sum(1 for c in paper[bench] if len(c) > 4 and c[4])
        cells = [(c[3], c[2]) for c in paper[bench] if c[3] > 0]
        our_rows = [r for r in ours if r["bench"] == bench
                    and (r["arm"] == "gepa_cot" or r["arm"].startswith("gepa_flex"))]
        our_pts = [(r["task_lm_cost_usd"], r["score"]) for r in our_rows]
        pos = [c for c, _ in cells + our_pts if c > 0]
        zero_x = min(pos) / 4  # the '$0' column position on the log axis

        def X(c):
            return max(c, zero_x)

        ax.set_xscale("log")
        # paper cells + paper-only frontier
        ax.scatter([X(c) for c, _ in cells], [s for _, s in cells], s=40,
                   color=BASELINE_C, edgecolors=SURFACE, linewidths=1.2, zorder=2,
                   label="LangProBe cells (program x optimizer)")
        xmax = max(pos) * 1.7
        pf = _pareto([(X(c), s) for c, s in cells])
        pf.append((xmax, pf[-1][1]))
        ax.plot([c for c, _ in pf], [s for _, s in pf], drawstyle="steps-post",
                color=INK2, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2,
                label="paper Pareto frontier")
        # our arms
        cot_rows = [r for r in our_rows if r["arm"] == "gepa_cot"]
        if cot_rows:
            ax.scatter([X(r["task_lm_cost_usd"]) for r in cot_rows],
                       [r["score"] for r in cot_rows], s=95, color=ORANGE, zorder=4,
                       edgecolors=SURFACE, linewidths=1.5, label="Prompt space: GEPA (CoT)")
        flex_rows = [r for r in our_rows if r["arm"].startswith("gepa_flex")]
        if flex_rows:
            ax.scatter([X(r["task_lm_cost_usd"]) for r in flex_rows],
                       [r["score"] for r in flex_rows], s=95, color=BLUE, zorder=4,
                       edgecolors=SURFACE, linewidths=1.5,
                       label="Flex + GEPA (incl. \u03bb sweep)")
        # combined frontier
        cf = _pareto([(X(c), s) for c, s in cells + our_pts])
        cf.append((xmax, cf[-1][1]))
        ax.plot([c for c, _ in cf], [s for _, s in cf], drawstyle="steps-post",
                color=INK, linewidth=2.0, zorder=3, label="frontier incl. Flex arms")
        if n_imputed:
            note = (f"{n_imputed} paper cell cost{'s' if n_imputed != 1 else ''} imputed "
                    "from recorded tokens at list prices")
            ax.annotate(note,
                        (0.98, 0.02), xycoords="axes fraction", ha="right",
                        fontsize=7.5, color=MUTED)
        # axis break for the $0 column
        ax.axvline(zero_x * 1.9, color=BASELINE_C, linewidth=0.9, linestyle=(0, (1, 2)))
        ticks = [t for t in (0.01, 0.03, 0.1, 0.3, 1.0)
                 if min(pos) / 1.5 <= t <= max(pos) * 1.5]
        ax.set_xticks([zero_x] + ticks)
        ax.set_xticklabels(["$0"] + [f"${t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(zero_x / 1.4, xmax)
        ax.set_xlabel("test-run inference cost ($, log scale)", fontsize=9.5, color=INK2)
        ax.set_ylabel("score (%)", fontsize=9.5, color=INK2)
        ax.set_title(BENCH_LABEL[bench], fontsize=12, color=INK, loc="left", pad=10,
                     fontweight="bold")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(BASELINE_C)
        ax.tick_params(colors=INK2, labelsize=9)
        if bench == benches[0]:
            ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle("Cost-score Pareto frontiers: LangProBe gpt-4o-mini matrix vs Flex arms "
                 "(zero-cost points are compiled programs)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig_pareto_frontier.png", dpi=200, facecolor=SURFACE)
    print("wrote fig_pareto_frontier.png")




PAPER_CSVS = {"gpt-4o-mini": "gpt4omini_0305.csv", "gpt-4o": "gpt4o_0305.csv"}
CONFIG_COLORS = {  # the paper's four configuration classes, in its color language
    "Model": "#eda100",
    "Model+Program": "#2a78d6",
    "Model+Optimizer": "#1baf7a",
    "Model+Program+Optimizer": "#e87ba4",
}
OVERLAY_BENCHES = ("heart", "scone")  # Iris excluded on BOTH sides (broken split, F4)


def load_paper_models():
    return {model: load_paper(model, fname) for model, fname in PAPER_CSVS.items()}


def _config_cell(cells, config):
    """Best-scoring cell (with recorded cost) of a benchmark for one paper config class."""
    def match(prog, opt):
        base_p, base_o = prog == "Predict", opt == "Baseline"
        return {"Model": base_p and base_o,
                "Model+Program": not base_p and base_o,
                "Model+Optimizer": base_p and not base_o,
                "Model+Program+Optimizer": not base_p and not base_o}[config]
    pool = [c for c in cells if match(c[0], c[1]) and c[3] > 0]
    return max(pool, key=lambda c: c[2]) if pool else None


def fig_paper_overlay(ours):
    """The paper's Figure 2 (aggregate cost-performance Pareto curves), reconstructed from
    their released per-cell data restricted to the shared benchmarks, with Flex+GEPA on top.

    Aggregation mirrors the paper: y = mean score across benchmarks, x = total inference
    cost (log scale). Best-scoring cell per (model, config, benchmark); Iris excluded on
    both sides (broken official split)."""
    paper_models = load_paper_models()
    fig, ax = plt.subplots(figsize=(8.2, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    markers = {"gpt-4o-mini": ("o", 70), "gpt-4o": ("o", 160)}
    for config, color in CONFIG_COLORS.items():
        pts = []
        for model, cells in paper_models.items():
            picked = [_config_cell(cells.get(b, []), config) for b in OVERLAY_BENCHES]
            if any(c is None for c in picked):
                continue
            score = mean(c[2] for c in picked)
            cost = sum(c[3] for c in picked)
            pts.append((cost, score, model))
        pts.sort()
        ax.plot([c for c, _, _ in pts], [s for _, s, _ in pts], color=color,
                linewidth=1.6, zorder=2, label=config)
        for cost, score, model in pts:
            m, size = markers[model]
            ax.scatter([cost], [score], s=size, color=color, marker=m, zorder=3,
                       edgecolors=SURFACE, linewidths=1.4)
    # Flex + GEPA (gpt-4o-mini): heart mean across seeds + each scone lambda variant
    heart = [r for r in ours if r["bench"] == "heart" and r["arm"] == "gepa_flex"]
    h_score = mean(r["score"] for r in heart)
    h_cost = mean(r["task_lm_cost_usd"] for r in heart)
    by_arm = defaultdict(list)
    for r in ours:
        if r["bench"] == "scone" and r["arm"].startswith("gepa_flex"):
            by_arm[r["arm"]].append(r)
    flex_pts = []
    for arm, rows in by_arm.items():
        lam = arm.removeprefix("gepa_flex_lam") if "lam" in arm else "0"
        lam = lam.replace("_warm", " warm")
        flex_pts.append((h_cost + mean(r["task_lm_cost_usd"] for r in rows),
                         mean([h_score, mean(r["score"] for r in rows)]), lam))
    flex_pts.sort()
    all_costs = [c for c, _, _ in flex_pts if c > 0]
    for config, color in CONFIG_COLORS.items():
        pass  # paper costs are all positive
    zero_x = min(all_costs) / 4
    def FX(c):
        return max(c, zero_x)
    ax.plot([FX(c) for c, _, _ in flex_pts], [s for _, s, _ in flex_pts], color=INK,
            linewidth=2.2, zorder=4, label="Flex + GEPA, gpt-4o-mini (λ sweep)")
    ax.scatter([FX(c) for c, _, _ in flex_pts], [s for _, s, _ in flex_pts], s=170,
               color=INK, marker="*", zorder=5, edgecolors=SURFACE, linewidths=1.2)
    for cost, score, lam in flex_pts:
        ax.annotate(f"λ={lam}", (FX(cost), score), xytext=(6, 7),
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_title("Aggregate cost-performance Pareto curves (paper Fig. 2, reconstructed on the\n"
                 "shared-benchmark subset from the paper's released cells) + Flex + GEPA overlay",
                 fontsize=11, color=INK, loc="left", pad=12)
    ax.set_xlabel("Total Inference Cost ($ in log-scale)", fontsize=10, color=INK2)
    ax.set_ylabel("LangProBe Score (HeartDisease + Scone subset)", fontsize=10, color=INK2)
    ax.annotate("small marker = gpt-4o-mini, large = gpt-4o · paper side: best cell per config per benchmark\n"
                "(oracle per-task selection — favors the paper) · Iris excluded on both sides (broken split)",
                (0.0, -0.19), xycoords="axes fraction", fontsize=8, color=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASELINE_C)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_paper_fig2_overlay.png", dpi=200, facecolor=SURFACE,
                bbox_inches="tight")
    print("wrote fig_paper_fig2_overlay.png")


if __name__ == "__main__":
    ours = load_ours()
    paper = load_paper()
    summary_tables(ours, paper)
    fig_accuracy(ours, paper)
    fig_cost(ours, paper)
    fig_pareto(ours, paper)
    fig_paper_overlay(ours)
