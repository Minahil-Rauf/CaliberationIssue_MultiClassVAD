"""Stage 3: build the labelled 10 s mixtures for train, val, test_in and test_shift.

Prints the achieved frame share of each class; expect roughly 25/50/20/5 % for
background/speech/music/speech+music (pauses inside speech move some frames to background).
"""
import pandas as pd
from _common import parser, setup

from vadcal.config import CLASSES, SPLITS
from vadcal.mixing import build_split


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    p.add_argument("--force", action="store_true", help="rebuild splits that already exist")
    args = p.parse_args()
    cfg, paths = setup(args)
    index = pd.read_csv(paths.index_csv)
    rows = []
    for split in args.splits:
        if args.force:  # rebuilt audio makes old features stale: force stage 4 to redo this split
            paths.feats_done(split).unlink(missing_ok=True)
        shares = build_split(index, paths.raw, paths.mix_dir(split), split, cfg, force=args.force)
        rows.append({"split": split, **{c: round(float(s), 4) for c, s in zip(CLASSES, shares)}})
    table = pd.DataFrame(rows)
    table.to_csv(paths.mixtures / "class_shares.csv", index=False)
    print("\nframe share per class:\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
