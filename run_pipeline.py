"""Run the whole pipeline in order, stopping at the first failure.

    python run_pipeline.py --smoke                 # synthetic quick test (do this first)
    python run_pipeline.py                         # the real experiment
    python run_pipeline.py --from-stage train      # continue from a stage
    python run_pipeline.py --only-stage evaluate   # one stage

Each stage skips work that is already finished, so re-running after a crash is safe.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("download", ["scripts/01_download_data.py"]),
    ("index", ["scripts/02_index_sources.py"]),
    ("mix", ["scripts/03_build_mixtures.py"]),
    ("features", ["scripts/04_extract_features.py"]),
    ("train", ["scripts/05_train.py", "--all"]),
    ("evaluate", ["scripts/06_evaluate.py"]),
    ("figures", ["scripts/07_make_figures.py"]),
]
NAMES = [name for name, _ in STAGES]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="use configs/smoke.yaml (synthetic data)")
    p.add_argument("--config", help="custom config (overrides --smoke)")
    p.add_argument("--from-stage", choices=NAMES, default=NAMES[0])
    p.add_argument("--only-stage", choices=NAMES)
    args = p.parse_args()
    config = args.config or ("configs/smoke.yaml" if args.smoke else "configs/default.yaml")

    stages = [s for s in STAGES if s[0] == args.only_stage] if args.only_stage \
        else STAGES[NAMES.index(args.from_stage):]
    for name, cmd in stages:
        print(f"\n=== stage: {name} ({config}) ===", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, *cmd, "--config", config], cwd=ROOT)
        if result.returncode != 0:
            again = f"--config {args.config} " if args.config else ("--smoke " if args.smoke else "")
            sys.exit(f"stage '{name}' failed (exit code {result.returncode}). Fix it, then run:\n"
                     f"  python run_pipeline.py {again}--from-stage {name}")
        print(f"=== {name} finished in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    print("\nAll stages finished.")


if __name__ == "__main__":
    main()
