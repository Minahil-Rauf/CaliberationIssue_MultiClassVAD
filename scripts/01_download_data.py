"""Stage 1: download and extract MUSAN, LibriSpeech test-clean, ESC-50 and GTZAN.

Smoke mode writes synthetic stand-ins instead (no download).
Interrupted downloads resume where they stopped; finished datasets are skipped.
"""
from _common import parser, setup

from vadcal.download import DATASETS, fetch
from vadcal.fake_sources import generate_all


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--keep-archives", action="store_true", help="keep the .tar.gz/.zip after extraction")
    args = p.parse_args()
    cfg, paths = setup(args)
    if cfg["smoke"]:
        generate_all(paths.raw, seed=cfg["seed"])
        return
    for name in args.datasets:
        fetch(name, paths.raw, paths.downloads, keep_archive=args.keep_archives)


if __name__ == "__main__":
    main()
