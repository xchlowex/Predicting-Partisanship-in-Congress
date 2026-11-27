"""Figures for the RQ3 slides: lexical cues, class recall, and slice drops."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CUES = ROOT / "results/week3/error_analysis/lexical_cues.csv"
DEST = ROOT / "results/week3/robustness"
INK = "#182A3A"
MUTED = "#536775"
GRID = "#DEE5E8"
DEMOCRAT = "#2C5F8A"
REPUBLICAN = "#8C4A3A"


def style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#AEBBC0")
    ax.spines["bottom"].set_color("#AEBBC0")
    ax.tick_params(colors=MUTED)


def lexical():
    rows = []
    with CUES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["protocol"] == "temporal" and int(row["rank"]) <= 8:
                rows.append(row)
    democrat = [row for row in rows if row["direction"] == "D"]
    republican = [row for row in rows if row["direction"] == "R"]
    democrat.sort(key=lambda row: abs(float(row["coefficient"])))
    republican.sort(key=lambda row: abs(float(row["coefficient"])))
    labels = [row["cue"] for row in democrat] + [row["cue"] for row in republican]
    values = [-abs(float(row["coefficient"])) for row in democrat] + [
        abs(float(row["coefficient"])) for row in republican
    ]
    colors = [DEMOCRAT] * len(democrat) + [REPUBLICAN] * len(republican)
    fig, ax = plt.subplots(figsize=(8.6, 6.2), facecolor="white")
    style(ax)
    positions = np.arange(len(labels))
    ax.barh(positions, values, color=colors, height=0.72)
    ax.set_yticks(positions, labels)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("TF-IDF logistic-regression coefficient", color=INK)
    ax.tick_params(axis="y", colors=INK)
    fig.text(0.06, 0.96, "Lexical cues in the temporal baseline", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.06, 0.915, "Top eight cues each way. Left points to Democrat, right to Republican.", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.88))
    path = DEST / "lexical_cues.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    print(path)


def recall():
    labels = ["Development", "Test"]
    democrat = [0.851, 0.850]
    republican = [0.562, 0.439]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.8), facecolor="white")
    style(ax)
    ax.bar(x - width / 2, democrat, width, color=DEMOCRAT, label="Democrat recall")
    ax.bar(x + width / 2, republican, width, color=REPUBLICAN, label="Republican recall")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall", color=INK)
    ax.tick_params(axis="x", colors=INK)
    for offset, series in ((-width / 2, democrat), (width / 2, republican)):
        for i, value in enumerate(series):
            ax.text(i + offset, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.legend(frameon=False, loc="upper right")
    fig.text(0.06, 0.96, "The temporal drop is one class", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.06, 0.905, "Text model, seed 4011. Local 32-bit rescore.", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.86))
    path = DEST / "class_recall.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    print(path)


def slices():
    labels = ["Short", "Senate", "All", "No party name", "Party name", "Long", "House"]
    drops = [0.171, 0.141, 0.081, 0.081, 0.085, 0.057, 0.048]
    order = np.argsort(drops)
    labels = [labels[i] for i in order]
    drops = [drops[i] for i in order]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor="white")
    style(ax)
    ax.barh(np.arange(len(labels)), drops, color="#087E83", height=0.68)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Drop in text macro-F1 from Congress 113 to 114", color=INK)
    ax.tick_params(axis="y", colors=INK)
    for i, value in enumerate(drops):
        ax.text(value + 0.003, i, f"{value:.3f}", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, 0.22)
    fig.text(0.06, 0.96, "Where the next Congress gets harder", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.06, 0.905, "Development cuts, applied unchanged to test. Text model.", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.86))
    path = DEST / "slice_drops.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    lexical()
    recall()
    slices()
