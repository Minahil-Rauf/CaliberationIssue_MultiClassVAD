"""Paper figures for the ICEET 2026 submission, built only from ../outputs/tables/*.csv.

    python paper/make_figures.py

Writes figures/*.pdf (for LaTeX) and figures/*.png (for checking), plus
figures/key_numbers.txt with every number quoted in the text.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "outputs" / "tables"
OUT = HERE / "figures"

LOSSES = ["ce", "weighted_ce", "focal", "label_smoothing"]
NAMES = {"ce": "Cross-entropy", "weighted_ce": "Weighted CE", "focal": "Focal loss",
         "label_smoothing": "Label smoothing"}
# Okabe-Ito colour-blind-safe palette, plus distinct markers.
COLORS = {"ce": "#0072B2", "weighted_ce": "#E69F00", "focal": "#D55E00", "label_smoothing": "#009E73"}
MARKERS = {"ce": "o", "weighted_ce": "s", "focal": "^", "label_smoothing": "D"}
SPLITS = {"test_in": "In-domain test", "test_shift": "Unseen-source test"}
COL_W, FULL_W = 3.5, 7.16  # IEEE column and page widths in inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote figures/{name}.pdf/.png")


# ---------------------------------------------------------------- Fig. 1: method
def _box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=color, ec="#333333", lw=0.7))
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top", fontsize=7.4, weight="bold")
    ax.text(x + w / 2, y + h - 0.175, "\n".join(lines), ha="center", va="top", fontsize=6.2,
            linespacing=1.3)


def _arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8,
                                 lw=0.9, color="#333333"))


def fig_method() -> None:
    fig, ax = plt.subplots(figsize=(FULL_W, 1.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, h, y, gap = 0.142, 0.86, 0.06, 0.0256
    boxes = [
        ("Sources", ["MUSAN: train, val,", "in-domain test", "(70/10/20 % by file)", "",
                     "LibriSpeech, GTZAN,", "ESC-50: unseen-", "source test"], "#E8F1FA"),
        ("Mixing", ["10-s clips of", "1-4 s events", "SMR $-$5 to 15 dB", "exact 10-ms labels"],
         "#E8F1FA"),
        ("CRNN", ["log-mel, 64 bands", "3 conv + 1 GRU layer", "86 K parameters",
                  "4 classes per frame"], "#FDF1DC"),
        ("Training loss", ["cross-entropy", "weighted CE", "focal ($\\gamma$ = 2)",
                           "label smoothing", "($\\epsilon$ = 0.1)", "$\\times$ 3 seeds each"], "#FDF1DC"),
        ("Calibration", ["fitted on validation", "set after training:", "temperature ($T$)",
                         "vector ($\\mathbf{w}, \\mathbf{b}$)"], "#E6F4EE"),
        ("Evaluation", ["macro-F1, class F1", "ECE, classwise-ECE", "fixed thresholds",
                        "clip-level bootstrap"], "#E6F4EE"),
    ]
    xs = [0.009 + i * (w + gap) for i in range(len(boxes))]
    for x, (title, lines, color) in zip(xs, boxes):
        _box(ax, x, y, w, h, title, lines, color)
    for x in xs[:-1]:
        _arrow(ax, x + w + 0.001, x + w + gap - 0.001, y + h / 2)
    # label strip inside the mixing box: what one clip's frame labels look like
    strip = [(0.00, 0.18, "#BBBBBB"), (0.18, 0.47, "#0072B2"), (0.47, 0.55, "#CC79A7"),
             (0.55, 0.80, "#E69F00"), (0.80, 1.00, "#0072B2")]
    sx, sw, sy = xs[1] + 0.014, w - 0.028, y + 0.17
    for a, b, c in strip:
        ax.add_patch(Rectangle((sx + a * sw, sy), (b - a) * sw, 0.09, fc=c, ec="none"))
    legend = [("#BBBBBB", "bg"), ("#0072B2", "sp"), ("#E69F00", "mu"), ("#CC79A7", "sp+mu")]
    for i, (c, t) in enumerate(legend):
        lx = sx + i * sw / 4
        ax.add_patch(Rectangle((lx, sy - 0.085), 0.006, 0.05, fc=c, ec="none"))
        ax.text(lx + 0.009, sy - 0.06, t, fontsize=5.6, va="center")
    save(fig, "fig1_method")


# ---------------------------------------------------------------- Fig. 2: reliability
def fig_reliability(min_share: float = 0.001) -> None:
    rel = pd.read_csv(TABLES / "reliability_bins.csv")
    rel = rel[rel["calib"] == "raw"].copy()
    rel["acc_sum"] = rel["accuracy"].fillna(0) * rel["count"]
    rel["conf_sum"] = rel["confidence"].fillna(0) * rel["count"]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.95), sharey=True)
    for ax, split in zip(axes, SPLITS):
        ax.fill_between([0.3, 1], [0.3, 1], [1, 1], color="#D55E00", alpha=0.06, lw=0)
        ax.plot([0.3, 1], [0.3, 1], ls="--", color="#555555", lw=0.8)
        for loss in LOSSES:
            r = rel[(rel["split"] == split) & (rel["loss"] == loss)]
            g = r.groupby("bin")[["count", "acc_sum", "conf_sum"]].sum()
            g = g[g["count"] >= min_share * g["count"].sum()]
            ax.plot(g["conf_sum"] / g["count"], g["acc_sum"] / g["count"], marker=MARKERS[loss], ms=2.6,
                    color=COLORS[loss], label=NAMES[loss], lw=1.0)
        ax.set(xlim=(0.3, 1.0), ylim=(0.25, 1.0), xlabel="Confidence", title=SPLITS[split])
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.text(0.33, 0.95, "under-confident", fontsize=6, color="#A04000", va="top")
        ax.text(0.97, 0.36, "over-confident", fontsize=6, color="#555555", ha="right")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_yticks([0.4, 0.6, 0.8, 1.0])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06),
               handlelength=1.4, columnspacing=0.8, fontsize=6.4)
    fig.subplots_adjust(bottom=0.27, wspace=0.12)
    save(fig, "fig2_reliability")


# ---------------------------------------------------------------- Fig. 3: thresholds
def fig_coverage(split: str = "test_in") -> None:
    e5 = pd.read_csv(TABLES / "e5a_selective.csv")
    e5 = e5[e5["split"] == split].groupby(["loss", "calib", "tau"])["coverage"].agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.95), sharey=True)
    for ax, calib, title in zip(axes, ("raw", "ts"), ("Uncalibrated", "After temperature scaling")):
        for loss in LOSSES:
            d = e5[(e5["loss"] == loss) & (e5["calib"] == calib)]
            ax.errorbar(d["tau"], 100 * d["mean"], yerr=100 * d["std"].fillna(0), marker=MARKERS[loss], ms=2.6,
                        color=COLORS[loss], label=NAMES[loss], lw=1.0, capsize=1.2, elinewidth=0.6)
        ax.axvline(0.9, color="#999999", lw=0.6, ls=":")
        ax.set(title=title, xlabel=r"Threshold $\tau$", xlim=(0.48, 0.97), ylim=(0, 102))
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.grid(axis="y", lw=0.3, color="#DDDDDD")
    axes[0].set_ylabel(r"Frames accepted, conf. $\geq\tau$ (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06),
               handlelength=1.4, columnspacing=0.8, fontsize=6.4)
    fig.subplots_adjust(bottom=0.27, wspace=0.12)
    save(fig, "fig3_coverage")


# ---------------------------------------------------------------- Fig. 4: ECE before/after
def fig_ece() -> None:
    s = pd.read_csv(TABLES / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.75), sharey=True)
    x = np.arange(len(LOSSES))
    calibs = [("raw", "Uncalibrated", "#7F7F7F"), ("ts", "Temp. scaling", "#0072B2"), ("vs", "Vector scaling", "#56B4E9")]
    for ax, split in zip(axes, SPLITS):
        for j, (calib, label, color) in enumerate(calibs):
            d = s[(s["split"] == split) & (s["calib"] == calib)].set_index("loss").reindex(LOSSES)
            ax.bar(x + (j - 1) * 0.27, d["ece_mean"], 0.27, yerr=d["ece_std"], color=color, label=label,
                   capsize=1.2, error_kw={"elinewidth": 0.6})
        ax.set_xticks(x, ["CE", "W-CE", "Focal", "LS"])
        ax.set_title(SPLITS[split])
        ax.grid(axis="y", lw=0.3, color="#DDDDDD")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("ECE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07),
               fontsize=6.4)
    fig.subplots_adjust(bottom=0.27, wspace=0.1)
    save(fig, "fig4_ece")


# ---------------------------------------------------------------- numbers quoted in the text
def key_numbers() -> None:
    s = pd.read_csv(TABLES / "summary.csv").set_index(["loss", "split", "calib"])
    cal = pd.read_csv(TABLES / "calibrators.csv").groupby("loss")["temperature"].agg(["mean", "std"])
    e5 = pd.read_csv(TABLES / "e5a_selective.csv").groupby(["loss", "split", "calib", "tau"])["coverage"].mean()
    lines = []
    for split in SPLITS:
        for calib in ("raw", "ts", "vs"):
            for loss in LOSSES:
                r = s.loc[(loss, split, calib)]
                lines.append(f"{split:10s} {calib:3s} {loss:16s} macroF1={r.macro_f1_mean:.3f}±{r.macro_f1_std:.3f} "
                             f"spmuF1={r.f1_speech_music_mean:.3f}±{r.f1_speech_music_std:.3f} "
                             f"ECE={r.ece_mean:.4f}±{r.ece_std:.4f} cwECE={r.cw_ece_mean:.4f}±{r.cw_ece_std:.4f}")
    lines.append("")
    for loss in LOSSES:
        lines.append(f"T {loss:16s} {cal.loc[loss, 'mean']:.3f}±{cal.loc[loss, 'std']:.3f}")
    lines.append("")
    for split in SPLITS:
        for calib in ("raw", "ts"):
            for tau in (0.9, 0.95):
                lines.append(f"coverage {split:10s} {calib:3s} tau={tau} " + "  ".join(
                    f"{loss}={100 * e5[(loss, split, calib, tau)]:.1f}%" for loss in LOSSES))
    (OUT / "key_numbers.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    fig_method()
    fig_reliability()
    fig_coverage()
    fig_ece()
    key_numbers()
