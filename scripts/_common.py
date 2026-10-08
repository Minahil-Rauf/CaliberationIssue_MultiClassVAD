"""Shared setup for the numbered scripts: import path and the --config argument."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vadcal.config import load_config  # noqa: E402
from vadcal.paths import Paths  # noqa: E402


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml",
                   help="protocol file (configs/smoke.yaml for the quick test)")
    return p


def setup(args) -> tuple[dict, Paths]:
    cfg = load_config(args.config)
    if cfg["smoke"]:
        print("*** SMOKE MODE: synthetic data, outputs are NOT for the paper ***")
    return cfg, Paths(cfg)
