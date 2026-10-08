"""E1-E5 from saved logits. Writes every table to outputs/tables/.

E1  baseline calibration (ce runs, raw)            -> metrics_per_run.csv, per_class.csv, reliability_bins.csv
E2  imbalance fixes (all losses, raw)              -> summary.csv, paired_vs_ce.csv
E3  temperature / vector scaling fitted on val     -> summary.csv, calibrators.csv, paired_calibration.csv
E4  median smoothing windows                       -> e4_smoothing.csv
E5  fixed, untuned thresholds                      -> e5a_selective.csv, e5b_speech_rule.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .bootstrap import bootstrap_metrics, clip_stats, metrics_from_stats, paired_difference, reliability
from .calibration import apply_temperature, apply_vector_scaling, fit_temperature, fit_vector_scaling
from .config import CLASSES, EVAL_SPLITS, N_CLASSES
from .metrics import softmax
from .paths import Paths
from .provenance import git_commit
from .smoothing import median_smooth

K = N_CLASSES
CI_METRICS = ("macro_f1", "f1_speech_music", "ece", "cw_ece", "nll", "brier")
SUMMARY_METRICS = ("macro_f1", "accuracy", "f1_speech", "f1_music", "f1_speech_music",
                   "ece", "cw_ece", "nll", "brier")


def finished_runs(paths: Paths) -> list[dict]:
    runs = []
    for d in sorted(paths.runs.glob("*_seed*")):
        if (d / "DONE").exists():
            info = json.loads((d / "run_info.json").read_text())
            runs.append({"dir": d, "loss": info["loss"], "seed": int(info["seed"])})
    return runs


def _logits(run_dir, split) -> np.ndarray:
    return np.load(run_dir / f"logits_{split}.npy").astype(np.float64)


def evaluate(cfg: dict, paths: Paths) -> None:
    ev = cfg["evaluation"]
    nb, B, seed = ev["ece_bins"], ev["bootstrap_resamples"], cfg["seed"]
    runs = finished_runs(paths)
    if not runs:
        raise RuntimeError(f"no finished runs in {paths.runs}. Run scripts/05_train.py first.")
    labels = {s: np.load(paths.labels(s)).astype(np.int64) for s in ("val",) + EVAL_SPLITS}
    tag = {"smoke": bool(cfg["smoke"]), "git_commit": git_commit()}

    metric_rows, class_rows, rel_rows, cal_rows = [], [], [], []
    e4_rows, e5a_rows, e5b_rows = [], [], []
    stats = {}

    for run in runs:
        rid = {"loss": run["loss"], "seed": run["seed"]}
        print(f"evaluating {run['dir'].name}")
        zv, yv = _logits(run["dir"], "val").reshape(-1, K), labels["val"].ravel()
        T = fit_temperature(zv, yv)
        w, b = fit_vector_scaling(zv, yv)
        cal_rows.append({**rid, "temperature": T, **{f"w_{c}": w[j] for j, c in enumerate(CLASSES)},
                         **{f"b_{c}": b[j] for j, c in enumerate(CLASSES)}})

        for split in EVAL_SPLITS:
            z, y = _logits(run["dir"], split), labels[split]
            variants = {"raw": softmax(z), "ts": apply_temperature(z, T), "vs": apply_vector_scaling(z, w, b)}
            for calib, P in variants.items():
                key = {**rid, "split": split, "calib": calib}
                S = clip_stats(P, y, nb)
                stats[(run["loss"], run["seed"], split, calib)] = S
                point, lo, hi = bootstrap_metrics(S, nb, K, B, seed)
                metric_rows.append({**key, **point, **{f"{m}_lo": lo.get(m) for m in CI_METRICS},
                                    **{f"{m}_hi": hi.get(m) for m in CI_METRICS}})
                for c in CLASSES:
                    class_rows.append({**key, "class": c, "share": float((y == CLASSES.index(c)).mean()),
                                       "f1": point[f"f1_{c}"], "classwise_ece": point[f"cwece_{c}"]})
                rel = reliability(S.sum(0), nb, K)
                for i in range(nb):
                    rel_rows.append({**key, **{k: float(v[i]) for k, v in rel.items()}})

                # E5a: accept a frame only if confidence >= tau. Does tau mean what it says?
                conf = P.max(-1).ravel()
                correct = (P.argmax(-1) == y).ravel()
                for tau in ev["accept_thresholds"]:
                    acc = conf >= tau
                    e5a_rows.append({**key, "tau": tau, "coverage": float(acc.mean()),
                                     "accuracy_accepted": float(correct[acc].mean()) if acc.any() else np.nan,
                                     "mean_conf_accepted": float(conf[acc].mean()) if acc.any() else np.nan})
                # E5b: the default speech rule p(speech) + p(speech+music) > threshold.
                score = (P[..., 1] + P[..., 3]).ravel()
                is_speech = np.isin(y.ravel(), (1, 3))
                flag = score > ev["speech_threshold"]
                e5b_rows.append({**key, "threshold": ev["speech_threshold"],
                                 "false_alarm_rate": float(flag[~is_speech].mean()),
                                 "miss_rate": float((~flag[is_speech]).mean())})

            # E4: smoothing on raw and temperature-scaled posteriors.
            for calib in ("raw", "ts"):
                for win in ev["smoothing_windows"]:
                    m = metrics_from_stats(clip_stats(median_smooth(variants[calib], win), y, nb).sum(0), nb, K)
                    e4_rows.append({**rid, "split": split, "calib": calib, "window": win,
                                    **{k: float(m[k][0]) for k in ("macro_f1", "ece", "cw_ece", "nll")}})

    paired_rows = []
    pair_metrics = ("ece", "cw_ece", "macro_f1", "f1_speech_music", "nll")
    for (loss, s, split, calib), S in stats.items():
        if calib == "raw" and loss != "ce" and ("ce", s, split, "raw") in stats:
            for r in paired_difference(S, stats[("ce", s, split, "raw")], nb, K, B, seed, pair_metrics):
                paired_rows.append({"comparison": f"{loss} - ce (raw)", "loss": loss, "seed": s, "split": split, **r})
        if calib in ("ts", "vs"):
            for r in paired_difference(S, stats[(loss, s, split, "raw")], nb, K, B, seed, pair_metrics):
                paired_rows.append({"comparison": f"{calib} - raw", "loss": loss, "seed": s, "split": split, **r})

    metrics = pd.DataFrame(metric_rows)
    out = paths.tables
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "metrics_per_run.csv": metrics,
        "summary.csv": summarise(metrics),
        "per_class.csv": pd.DataFrame(class_rows),
        "reliability_bins.csv": pd.DataFrame(rel_rows),
        "calibrators.csv": pd.DataFrame(cal_rows),
        "paired_differences.csv": pd.DataFrame(paired_rows),
        "e4_smoothing.csv": pd.DataFrame(e4_rows),
        "e5a_selective.csv": pd.DataFrame(e5a_rows),
        "e5b_speech_rule.csv": pd.DataFrame(e5b_rows),
    }
    for name, df in tables.items():
        for k, v in tag.items():
            df[k] = v
        df.to_csv(out / name, index=False)
        print(f"wrote {out / name} ({len(df)} rows)")


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds for each loss / split / calibration."""
    g = metrics.groupby(["loss", "split", "calib"])[list(SUMMARY_METRICS)]
    mean, std, n = g.mean(), g.std(ddof=1), g.size().rename("n_seeds")
    out = mean.add_suffix("_mean").join(std.add_suffix("_std")).join(n).reset_index()
    return out
