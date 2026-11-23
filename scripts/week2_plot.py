"""Plot primary-seed test macro-F1 and speaker bootstrap intervals from week2_compare.py."""

import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


SUMMARY = Path(__file__).resolve().parents[1] / "results/week2/formal_summary"
MODELS = {"text": "Text only", "year": "Text + year", "chamber": "Text + chamber",
          "both": "Text + year + chamber", "tfidf_lr": "TF-IDF + logistic regression"}
PANELS = {"speaker": "Unseen legislators", "temporal": "Later Congress"}
SEED = 4011
NAVY = "#182A3A"
TEAL = "#087E83"
GRAY = "#536775"


def primary_test_rows(summary):
    marks = {}
    for row in summary["metrics"]:
        model, protocol = row["model"], row["protocol"]
        if (model not in MODELS or protocol not in PANELS or row["partition"] != "test"
                or row["subgroup"] != "all" or row["seed"] != (None if model == "tfidf_lr" else SEED)):
            continue
        values = [row[k] for k in ("macro_f1_ci_low", "macro_f1", "macro_f1_ci_high")]
        if (protocol, model) in marks or not all(0 <= v <= 1 for v in values) or values[0] > values[2]:
            raise ValueError(f"Duplicate or invalid interval for {protocol}/{model}")
        marks[protocol, model] = row
    missing = [(p, m) for p in PANELS for m in MODELS if (p, m) not in marks]
    if missing:
        raise ValueError(f"Primary comparison is incomplete. Missing {missing}.")
    for protocol in PANELS:
        if len({(marks[protocol, m]["speeches"], marks[protocol, m]["speakers"]) for m in MODELS}) != 1:
            raise ValueError(f"Models have different evaluation populations for {protocol}")
    return marks


def make_figure(summary):
    marks = primary_test_rows(summary)
    low = min(min(r["macro_f1"], r["macro_f1_ci_low"]) for r in marks.values())
    high = max(max(r["macro_f1"], r["macro_f1_ci_high"]) for r in marks.values())
    limits = (max(0, math.floor((low - 0.012) * 20) / 20),
              min(1, math.ceil((high + 0.012) * 20) / 20))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.35), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.235, right=0.925, top=0.695, bottom=0.205, wspace=0.31)
    fig.patch.set_facecolor("white")
    fig.text(0.055, 0.897, "Party prediction on held-out speeches", fontsize=24,
             fontweight="bold", color=NAVY)
    fig.text(0.055, 0.841, "Test macro-F1 for RoBERTa-base and the TF-IDF benchmark",
             fontsize=13, color=GRAY)
    positions = [4, 3, 2, 1, -0.25]

    for ax, (protocol, title) in zip(axes, PANELS.items(), strict=True):
        ref = marks[protocol, "text"]
        ax.set_title(title, loc="left", fontsize=16, color=NAVY, weight="bold", pad=23)
        ax.text(0, 1.012, f"{ref['speeches']:,} speeches · {ref['speakers']:,} legislators",
                transform=ax.transAxes, va="bottom", color=GRAY, fontsize=10)
        ax.set_xlim(*limits)
        ax.set_ylim(-0.85, 4.7)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="x", linewidth=0.8, color="#DEE5E8")
        ax.axhline(0.36, color="#C7D1D4", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.spines["bottom"].set_color("#AEBBC0")
        ax.tick_params(axis="both", length=0, pad=8, labelcolor=GRAY, labelsize=10)
        ax.set_yticks(positions, list(MODELS.values()), fontsize=12, color=NAVY)
        coord = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for y, model in zip(positions, MODELS, strict=True):
            row = marks[protocol, model]
            lo, value, hi = (row[k] for k in ("macro_f1_ci_low", "macro_f1", "macro_f1_ci_high"))
            color = GRAY if model == "tfidf_lr" else TEAL
            ax.hlines(y, lo, hi, colors=color, linewidth=2.0)
            ax.vlines([lo, hi], y - 0.085, y + 0.085, colors=color, linewidth=1.3)
            ax.scatter(value, y, marker="s" if model == "tfidf_lr" else "o",
                       s=57, color=color, edgecolors="white", linewidths=0.8, zorder=3)
            ax.text(1.025, y, f"{value:.3f}", ha="left", va="center", color=NAVY,
                    fontsize=11, transform=coord)
        ax.set_xlabel("Macro-F1", labelpad=7, color=NAVY, fontsize=11)

    n = summary["bootstrap"]["replicates"]
    fig.text(0.055, 0.09,
             f"Lines show 95% intervals from {n:,} bootstrap samples of legislators for each fitted model.",
             fontsize=10.5, color=GRAY)
    fig.text(0.055, 0.048, f"RoBERTa training seed {SEED}. Across-seed variation is reported separately.",
             fontsize=10.5, color=GRAY)
    return fig


def main():
    src = SUMMARY / "comparison.json"
    dest = SUMMARY / "model_comparison.png"
    fig = make_figure(json.loads(src.read_text(encoding="utf-8")))
    with NamedTemporaryFile(dir=dest.parent, suffix=".png", delete=False) as temp:
        staged = Path(temp.name)
    try:
        fig.savefig(staged, dpi=180, facecolor="white")
        staged.replace(dest)
    finally:
        plt.close(fig)
        staged.unlink(missing_ok=True)
    print(dest)


if __name__ == "__main__":
    main()
