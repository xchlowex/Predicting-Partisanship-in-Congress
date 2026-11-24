"""Plot the Week 3 matched temporal text-versus-chamber comparison."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/week3/error_analysis/comparison.json"
DEST = ROOT / "results/week3/error_analysis/model_comparison.png"


def main():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = {(r["partition"], r["model"]): r for r in summary["comparison"]}
    phases = ("dev", "test")
    colors = {"text": "#536775", "chamber": "#087E83"}
    labels = {"text": "Text only", "chamber": "Text + chamber"}

    fig, ax = plt.subplots(figsize=(8.8, 5.2), facecolor="white")
    ax.set_facecolor("white")
    positions = {"dev": 1, "test": 0}
    for phase in phases:
        y = positions[phase]
        ax.text(-0.015, y, phase.upper(), transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontsize=12, fontweight="bold", color="#182A3A")
        for offset, model in ((-0.16, "text"), (0.16, "chamber")):
            row = rows[phase, model]
            lo = row["macro_f1_ci_low"]
            value = row["macro_f1"]
            hi = row["macro_f1_ci_high"]
            ax.plot([lo, hi], [y + offset, y + offset], color=colors[model], linewidth=2)
            ax.plot([lo, lo], [y + offset - 0.06, y + offset + 0.06], color=colors[model], linewidth=1)
            ax.plot([hi, hi], [y + offset - 0.06, y + offset + 0.06], color=colors[model], linewidth=1)
            ax.scatter([value], [y + offset], s=58, color=colors[model],
                       edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(value, y + offset + 0.09, f"{value:.3f}", ha="center",
                    va="bottom", fontsize=10, color="#182A3A")

    ax.set_xlim(.58, .76)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Macro-F1", fontsize=11, color="#182A3A")
    ax.grid(axis="x", color="#DEE5E8", linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#AEBBC0")
    ax.tick_params(axis="x", colors="#536775")
    ax.legend([plt.Line2D([0], [0], color=colors[m], marker="o", linestyle="-",
                          markeredgecolor="white", label=labels[m]) for m in ("text", "chamber")],
              [labels[m] for m in ("text", "chamber")],
              loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=2, frameon=False)
    fig.text(.06, .96, "Matched temporal-model comparison", fontsize=18,
             fontweight="bold", color="#182A3A")
    fig.text(.06, .915, "Test macro-F1 with 95% speaker-bootstrap intervals",
             fontsize=11, color="#536775")
    fig.tight_layout(rect=(.03, .04, .99, .88))
    fig.savefig(DEST, dpi=180, facecolor="white")
    plt.close(fig)
    print(DEST)


if __name__ == "__main__":
    main()
