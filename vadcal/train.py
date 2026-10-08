"""Train one (loss, seed) run, then save raw logits for val, test_in and test_shift.

A crashed run resumes from `last.pt`; a finished run (DONE file) is skipped.
"""
from __future__ import annotations

import json
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import N_CLASSES
from .dataset import ClipDataset
from .losses import class_weights, make_loss
from .metrics import quick_metrics
from .model import build_model, count_params
from .paths import Paths
from .provenance import provenance

PREDICT_SPLITS = ("val", "test_in", "test_shift")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_logits(model, X: np.ndarray, mean: np.ndarray, std: np.ndarray, device, batch: int = 16) -> np.ndarray:
    model.eval()
    out = np.zeros((X.shape[0], X.shape[1], N_CLASSES), dtype=np.float32)
    for i in range(0, X.shape[0], batch):
        x = (np.asarray(X[i:i + batch], dtype=np.float32) - mean) / std
        out[i:i + batch] = model(torch.from_numpy(x).to(device)).float().cpu().numpy()
    return out


def train_run(cfg: dict, paths: Paths, loss_name: str, seed: int, force: bool = False) -> None:
    run_dir = paths.run_dir(loss_name, seed)
    if (run_dir / "DONE").exists() and not force:
        print(f"[{loss_name} seed {seed}] finished already, skipping")
        return
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tcfg = cfg["training"]
    set_seed(seed)
    if tcfg.get("torch_threads"):
        torch.set_num_threads(int(tcfg["torch_threads"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats = np.load(paths.norm_stats)
    mean, std = stats["mean"].astype(np.float32), stats["std"].astype(np.float32)
    X_tr, Y_tr = np.load(paths.feats("train"), mmap_mode="r"), np.load(paths.labels("train"))
    X_va, Y_va = np.load(paths.feats("val"), mmap_mode="r"), np.load(paths.labels("val"))

    crop = int(round(tcfg["crop_seconds"] * 1000 / cfg["audio"]["hop_ms"]))
    ds = ClipDataset(X_tr, Y_tr, mean, std, crop_frames=crop, seed=seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True,
                        num_workers=tcfg["num_workers"], generator=gen)

    model = build_model(cfg, N_CLASSES).to(device)
    weights = None
    if loss_name == "weighted_ce":
        weights = torch.tensor(class_weights(Y_tr, N_CLASSES), device=device)
        print(f"class weights: {weights.cpu().numpy().round(3).tolist()}")
    criterion = make_loss(loss_name, tcfg, weights)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    print(f"[{loss_name} seed {seed}] device={device} params={count_params(model):,} "
          f"train clips={len(ds)} crop={crop} frames")

    state = {"epoch": -1, "best_f1": -1.0, "best_epoch": -1, "bad": 0, "log": []}
    last = run_dir / "last.pt"
    if last.exists():
        # Loaded on CPU: RNG states must be CPU tensors; the optimizer moves its state to the params' device.
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        gen.set_state(ck["gen"])
        ds.rng.bit_generator.state = ck["ds_rng"]
        torch.set_rng_state(ck["torch_rng"])
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        print(f"resuming after epoch {state['epoch']}")

    for epoch in range(state["epoch"] + 1, tcfg["max_epochs"]):
        if state["bad"] >= tcfg["patience"]:
            print("early stopping")
            break
        model.train()
        t0, total, n_batches = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, N_CLASSES), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            total += loss.item()
            n_batches += 1
        val = quick_metrics(predict_logits(model, X_va, mean, std, device), Y_va)
        row = {"epoch": epoch, "train_loss": total / max(n_batches, 1), **{f"val_{k}": v for k, v in val.items()},
               "seconds": round(time.time() - t0, 1)}
        state["log"].append(row)
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if val["macro_f1"] > state["best_f1"]:
            state.update(best_f1=val["macro_f1"], best_epoch=epoch, bad=0)
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            state["bad"] += 1
        state["epoch"] = epoch
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                    "gen": gen.get_state(), "ds_rng": ds.rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, last)

    pd.DataFrame(state["log"]).to_csv(run_dir / "train_log.csv", index=False)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    for split in PREDICT_SPLITS:
        X = np.load(paths.feats(split), mmap_mode="r")
        np.save(run_dir / f"logits_{split}.npy", predict_logits(model, X, mean, std, device).astype(np.float16))
    info = {"loss": loss_name, "seed": seed, "best_epoch": state["best_epoch"],
            "best_val_macro_f1": state["best_f1"], "epochs_run": state["epoch"] + 1,
            "n_params": count_params(model), "device": str(device), "smoke": bool(cfg["smoke"]),
            **provenance()}
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    (run_dir / "DONE").write_text("ok")
    print(f"[{loss_name} seed {seed}] done: best epoch {state['best_epoch']}, val macro-F1 {state['best_f1']:.4f}")
