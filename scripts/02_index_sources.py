"""Stage 2: list every usable source file and assign the file-level splits.

Writes data/index/sources.csv and prints files and hours per split, kind and dataset.
"""
from _common import parser, setup

from vadcal.sources import build_index, summary


def main() -> None:
    args = parser(__doc__).parse_args()
    cfg, paths = setup(args)
    df = build_index(paths.raw, cfg)
    paths.index_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.index_csv, index=False)
    print(summary(df).to_string(index=False))
    print(f"wrote {paths.index_csv} ({len(df)} files)")


if __name__ == "__main__":
    main()
