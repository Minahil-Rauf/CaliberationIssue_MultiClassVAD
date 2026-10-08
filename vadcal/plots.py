"""Paper figures and LaTeX tables from outputs/tables/*.csv."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SPLIT_NAMES = {"test_in": "In-domain test", "test_shift": "Shifted test"}
CALIB_NAMES = {"raw": "Uncalibrated", "ts": "Temperature scaling", "vs": "Vector scaling"}
LOSS_NAMES = {"ce": "CE", "weighted_ce": "Weighted CE", "focal": "Focal", "label_smoothing": "Label smoothing"}
COLORS = {"raw": "#c0392b", "ts": "#2471a3", "vs": "#7d3c98"}
COLUMN_WIDTH = 3.5  # inches, one IEEE column

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "savefig.dpi": 300})


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.png/.pdf")


def reliability_figure(tables: Path, out: Path, loss: str = "ce") -> None:
    """Fig. 1: reliability diagrams, bins pooled over seeds."""
    rel = pd.read_csv(tables / "reliability_bins.csv")
    summary = pd.read_csv(tables / "summary.csv")
    rel = rel[rel["loss"] == loss]
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH * 2, 2.6), sharey=True)
    for ax, split in zip(axes, SPLIT_NAMES):
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8, label="Perfect calibration")
        for calib in ("raw", "ts"):
            r = rel[(rel["split"] == split) & (rel["calib"] == calib)].copy()
            r["acc_sum"] = r["accuracy"].fillna(0) * r["count"]
            r["conf_sum"] = r["confidence"].fillna(0) * r["count"]
            g = r.groupby("bin")[["count", "acc_sum", "conf_sum"]].sum()
            g = g[g["count"] > 0]
            ece = summary[(summary["loss"] == loss) & (summary["split"] == split)
                          & (summary["calib"] == calib)]["ece_mean"]
            label = f"{CALIB_NAMES[calib]} (ECE {ece.iloc[0]:.3f})" if len(ece) else CALIB_NAMES[calib]
            ax.plot(g["conf_sum"] / g["count"], g["acc_sum"] / g["count"], marker="o", ms=3,
                    color=COLORS[calib], label=label)
        ax.set(title=SPLIT_NAMES[split], xlabel="Confidence", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Accuracy")
    _save(fig, out, "fig1_reliability")


def imbalance_figure(tables: Path, out: Path) -> None:
    """E2: ECE and speech+music F1 per training loss (uncalibrated), mean +/- std over seeds."""
    s = pd.read_csv(tables / "summary.csv")
    s = s[s["calib"] == "raw"]
    losses = [l for l in LOSS_NAMES if l in set(s["loss"])]
    x = np.arange(len(losses))
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH * 2, 2.4))
    for ax, metric, title in zip(axes, ("ece", "f1_speech_music"), ("ECE (lower is better)", "Speech+music F1")):
        for j, split in enumerate(SPLIT_NAMES):
            d = s[s["split"] == split].set_index("loss").reindex(losses)
            ax.bar(x + (j - 0.5) * 0.38, d[f"{metric}_mean"], 0.38, yerr=d[f"{metric}_std"].fillna(0),
                   capsize=2, label=SPLIT_NAMES[split])
        ax.set_xticks(x, [LOSS_NAMES[l] for l in losses])
        ax.set_title(title)
    axes[0].legend()
    _save(fig, out, "fig_e2_imbalance")


def smoothing_figure(tables: Path, out: Path, loss: str = "ce") -> None:
    """Fig. 2: ECE and macro-F1 against the median-filter window."""
    e4 = pd.read_csv(tables / "e4_smoothing.csv")
    e4 = e4[e4["loss"] == loss].groupby(["split", "calib", "window"])[["ece", "macro_f1"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH * 2, 2.4))
    for ax, metric, title in zip(axes, ("ece", "macro_f1"), ("ECE", "Macro-F1")):
        for split, ls in zip(SPLIT_NAMES, ("-", "--")):
            for calib in ("raw", "ts"):
                d = e4[(e4["split"] == split) & (e4["calib"] == calib)]
                ax.plot(d["window"], d[metric], ls=ls, marker="o", ms=3, color=COLORS[calib],
                        label=f"{SPLIT_NAMES[split]}, {CALIB_NAMES[calib]}")
        ax.set(title=title, xlabel="Median filter window (frames of 10 ms)")
    axes[0].legend()
    _save(fig, out, "fig2_smoothing")


def threshold_figure(tables: Path, out: Path, loss: str = "ce") -> None:
    """Fig. 3: frames accepted at confidence >= tau. Dashed: what the model claims; solid: actual accuracy."""
    e5 = pd.read_csv(tables / "e5a_selective.csv")
    e5 = (e5[e5["loss"] == loss].groupby(["split", "calib", "tau"])
          [["accuracy_accepted", "mean_conf_accepted"]].mean().reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH * 2, 2.4), sharey=True)
    for ax, split in zip(axes, SPLIT_NAMES):
        for calib in ("raw", "ts"):
            d = e5[(e5["split"] == split) & (e5["calib"] == calib)]
            ax.plot(d["tau"], d["accuracy_accepted"], marker="o", ms=3, color=COLORS[calib],
                    label=f"{CALIB_NAMES[calib]}: actual accuracy")
            ax.plot(d["tau"], d["mean_conf_accepted"], ls="--", color=COLORS[calib],
                    label=f"{CALIB_NAMES[calib]}: claimed confidence")
        ax.set(title=SPLIT_NAMES[split], xlabel="Confidence threshold $\\tau$")
    axes[0].set_ylabel("Accepted frames")
    axes[0].legend()
    _save(fig, out, "fig3_thresholds")


def _fmt(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean:.3f}" if pd.isna(std) else f"{mean:.3f} $\\pm$ {std:.3f}"


def latex_tables(tables: Path, out: Path) -> None:
    """Table 1 (E2) and Table 2 (E3) as LaTeX tabulars."""
    s = pd.read_csv(tables / "summary.csv")
    out.mkdir(parents=True, exist_ok=True)
    losses = [l for l in LOSS_NAMES if l in set(s["loss"])]

    cols = [("macro_f1", "Macro-F1"), ("f1_speech_music", "F1 sp+mu"), ("ece", "ECE"), ("cw_ece", "cw-ECE")]
    lines = ["\\begin{tabular}{l" + "c" * len(cols) * 2 + "}", "\\hline",
             " & \\multicolumn{%d}{c}{In-domain} & \\multicolumn{%d}{c}{Shifted} \\\\" % (len(cols), len(cols)),
             "Loss & " + " & ".join(n for _, n in cols * 2) + " \\\\", "\\hline"]
    for loss in losses:
        cells = []
        for split in SPLIT_NAMES:
            r = s[(s["loss"] == loss) & (s["split"] == split) & (s["calib"] == "raw")]
            cells += [_fmt(r[f"{m}_mean"].iloc[0], r[f"{m}_std"].iloc[0]) if len(r) else "--" for m, _ in cols]
        lines.append(f"{LOSS_NAMES[loss]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    (out / "table1_e2.tex").write_text("\n".join(lines))

    lines = ["\\begin{tabular}{llcccc}", "\\hline",
             " & & \\multicolumn{2}{c}{In-domain} & \\multicolumn{2}{c}{Shifted} \\\\",
             "Loss & Calibration & ECE & Macro-F1 & ECE & Macro-F1 \\\\", "\\hline"]
    for loss in losses:
        for calib in CALIB_NAMES:
            cells = []
            for split in SPLIT_NAMES:
                r = s[(s["loss"] == loss) & (s["split"] == split) & (s["calib"] == calib)]
                cells += [_fmt(r[f"{m}_mean"].iloc[0], r[f"{m}_std"].iloc[0]) if len(r) else "--"
                          for m in ("ece", "macro_f1")]
            lines.append(f"{LOSS_NAMES[loss]} & {CALIB_NAMES[calib]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    (out / "table2_e3.tex").write_text("\n".join(lines))
    print(f"wrote {out / 'table1_e2.tex'} and {out / 'table2_e3.tex'}")
