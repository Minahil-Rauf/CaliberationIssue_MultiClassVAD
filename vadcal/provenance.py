"""Git commit and package versions, stored with every run and result table."""
from __future__ import annotations

import platform
import subprocess

from .config import ROOT


def git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "vadcal", "configs", "scripts"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "torch", "librosa", "soundfile", "pandas"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = "missing"
    return out


def provenance() -> dict:
    return {"git_commit": git_commit(), **{f"v_{k}": v for k, v in versions().items()}}
