"""Stage 5: train CRNN runs and save their logits.

    python scripts/05_train.py --all                    # every loss x seed in the config
    python scripts/05_train.py --loss focal --seed 1    # one run

Finished runs are skipped; a crashed run resumes from its last epoch.
"""
from _common import parser, setup

from vadcal.losses import LOSSES
from vadcal.train import train_run


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--loss", choices=LOSSES)
    p.add_argument("--seed", type=int)
    p.add_argument("--all", action="store_true", help="run every loss and seed listed in the config")
    p.add_argument("--force", action="store_true", help="delete and retrain finished runs")
    args = p.parse_args()
    cfg, paths = setup(args)
    if args.all:
        jobs = [(l, s) for l in cfg["training"]["losses"] for s in cfg["training"]["seeds"]]
    elif args.loss is not None and args.seed is not None:
        jobs = [(args.loss, args.seed)]
    else:
        p.error("give --all, or both --loss and --seed")
    for loss, seed in jobs:
        train_run(cfg, paths, loss, seed, force=args.force)


if __name__ == "__main__":
    main()
