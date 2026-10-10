"""Stage 1: download and extract MUSAN, LibriSpeech test-clean, ESC-50 and GTZAN.

By default each .tar.gz is unpacked while it downloads, so archives are never stored
(one copy of the data on disk). If the connection drops, that dataset restarts from zero.
On an unreliable connection use --no-stream: downloads resume, but each archive is
stored until it has been extracted (MUSAN needs ~11 GB extra for a while).

Smoke mode writes synthetic stand-ins instead (no download). Finished datasets are skipped.
"""
from _common import parser, setup

from vadcal.download import DATASETS, fetch
from vadcal.fake_sources import generate_all


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--no-stream", action="store_true",
                   help="save each archive first (resumable downloads, needs extra disk)")
    p.add_argument("--keep-archives", action="store_true",
                   help="keep the archives after extraction (implies --no-stream)")
    args = p.parse_args()
    cfg, paths = setup(args)
    if cfg["smoke"]:
        generate_all(paths.raw, seed=cfg["seed"])
        return
    stream = not (args.no_stream or args.keep_archives)
    for name in args.datasets:
        fetch(name, paths.raw, paths.downloads, stream=stream, keep_archive=args.keep_archives)


if __name__ == "__main__":
    main()
