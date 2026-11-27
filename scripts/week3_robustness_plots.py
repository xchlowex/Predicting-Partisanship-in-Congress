"""Plot the frozen-edit scores, label flips, and development slices."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/week3/robustness"
TEXT = "#536775"
CHAMBER = "#087E83"
INK = "#182A3A"
MUTED = "#536775"
GRID = "#DEE5E8"

EDITS = ["Original", "Party-name\nmask", "Heading\nremoved", "First sentence\nremoved", "Last sentence\nremoved"]
F1 = {
    "Development": {
        "Text only": [0.702, 0.697, 0.704, 0.701, 0.701],
        "Text + chamber": [0.723, 0.708, 0.722, 0.720, 0.720],
    },
    "Test": {
        "Text only": [0.621, 0.613, 0.629, 0.626, 0.619],
        "Text + chamber": [0.657, 0.636, 0.654, 0.647, 0.656],
    },
}
FLIPS = {
    "Development": {
        "Text only": [0, 70, 50, 114, 140],
        "Text + chamber": [0, 89, 43, 139, 173],
    },
    "Test": {
        "Text only": [0, 59, 64, 97, 156],
        "Text + chamber": [0, 105, 66, 126, 225],
    },
}
SLICES = [
    "Short quarter",
    "Long quarter",
    "House",
    "Senate",
    "Party name present",
    "No party name",
    "Close call",
    "Sure guess",
]
SLICE_F1 = {
    "Text only": [0.615, 0.781, 0.675, 0.750, 0.772, 0.680, 0.564, 0.832],
    "Text + chamber": [0.651, 0.804, 0.692, 0.777, 0.796, 0.699, 0.588, 0.850],
}


def style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#AEBBC0")
    ax.spines["bottom"].set_color("#AEBBC0")
    ax.tick_params(colors=MUTED)


def grouped(ax, labels, series, ymin, ymax, ylabel):
    x = np.arange(len(labels))
    width = 0.36
    colors = {"Text only": TEXT, "Text + chamber": CHAMBER}
    for offset, (name, values) in zip((-width / 2, width / 2), series.items()):
        bars = ax.bar(x + offset, values, width=width, color=colors[name], label=name)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (ymax - ymin) * 0.015,
                    f"{value:.3f}" if ymax <= 1 else f"{value:.0f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel, color=INK)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), facecolor="white", sharey=True)
    for ax, phase in zip(axes, ("Development", "Test")):
        grouped(ax, EDITS, F1[phase], 0.55, 0.82, "Macro-F1")
        ax.set_title(phase, color=INK, fontsize=13, pad=8)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Macro-F1 after each frozen edit", fontsize=16, fontweight="bold", color=INK, x=0.06, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(DEST / "edit_macro_f1.png", dpi=180, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), facecolor="white", sharey=True)
    for ax, phase in zip(axes, ("Development", "Test")):
        grouped(ax, EDITS, FLIPS[phase], 0, 280, "Speeches whose prediction changed")
        ax.set_title(phase, color=INK, fontsize=13, pad=8)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Predictions that change from the original text", fontsize=16, fontweight="bold", color=INK, x=0.06, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(DEST / "edit_flips.png", dpi=180, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.6), facecolor="white")
    y = np.arange(len(SLICES))
    height = 0.36
    ax.barh(y + height / 2, SLICE_F1["Text only"], height=height, color=TEXT, label="Text only")
    ax.barh(y - height / 2, SLICE_F1["Text + chamber"], height=height, color=CHAMBER, label="Text + chamber")
    ax.set_yticks(y, SLICES)
    ax.set_xlim(0.50, 0.95)
    ax.set_xlabel("Macro-F1", color=INK)
    ax.invert_yaxis()
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98), ncol=2)
    fig.suptitle("Development macro-F1 by slice", fontsize=16, fontweight="bold", color=INK, x=0.08, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(DEST / "dev_slices.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(DEST)


if __name__ == "__main__":
    main()
