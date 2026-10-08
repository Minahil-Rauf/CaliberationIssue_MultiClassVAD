"""Stage 4: 64-band log-mel features for every clip, plus train-set normalisation stats."""
from _common import parser, setup

from vadcal.config import SPLITS
from vadcal.features import compute_norm_stats, extract_split


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cfg, paths = setup(args)
    for split in SPLITS:
        extract_split(paths, split, cfg, force=args.force)
    compute_norm_stats(paths)
    print(f"wrote features and {paths.norm_stats}")


if __name__ == "__main__":
    main()
