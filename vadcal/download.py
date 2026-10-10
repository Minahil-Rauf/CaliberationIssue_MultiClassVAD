"""Dataset downloads with mirror fallback and safe extraction.

Default (streaming): a .tar.gz is unpacked while it downloads, so the archive is never
stored and only one copy of each dataset is on disk. The cost: an interrupted download
starts that dataset again from zero (each mirror is tried several times automatically).

--no-stream: the archive is saved first (resumable via HTTP Range), then extracted and
deleted. Needs the archive's size in extra disk space while extracting.

A .zip (ESC-50) cannot be unpacked mid-download, so it always uses the second mode.
An archive already in data/downloads/ (e.g. downloaded by hand) is always used directly.
"""
from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATASETS = {
    "musan": {
        "urls": [
            "https://openslr.trmal.net/resources/17/musan.tar.gz",
            "https://openslr.elda.org/resources/17/musan.tar.gz",
            "https://openslr.magicdatatech.com/resources/17/musan.tar.gz",
            "https://www.openslr.org/resources/17/musan.tar.gz",
        ],
        "archive": "musan.tar.gz",
        "check": "musan/speech",
        "size": "11 GB",
    },
    "librispeech_test_clean": {
        "urls": [
            "https://openslr.trmal.net/resources/12/test-clean.tar.gz",
            "https://openslr.elda.org/resources/12/test-clean.tar.gz",
            "https://openslr.magicdatatech.com/resources/12/test-clean.tar.gz",
            "https://www.openslr.org/resources/12/test-clean.tar.gz",
        ],
        "archive": "test-clean.tar.gz",
        "check": "LibriSpeech/test-clean",
        "size": "350 MB",
    },
    "esc50": {
        "urls": ["https://github.com/karolpiczak/ESC-50/archive/master.zip"],
        "archive": "ESC-50-master.zip",
        "check": "ESC-50-master/meta/esc50.csv",
        "size": "600 MB",
    },
    "gtzan": {
        # The original opihi.cs.uvic.ca link is dead; this is the Hugging Face copy of genres.tar.gz.
        "urls": ["https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"],
        "archive": "genres.tar.gz",
        "check": "genres",
        "size": "1.2 GB",
    },
}


def _download(url: str, dest: Path, chunk: int = 1 << 20, timeout: int = 60) -> None:
    part = dest.with_name(dest.name + ".part")
    pos = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, stream=True, headers=headers, timeout=timeout, allow_redirects=True) as r:
        if r.status_code == 416:          # nothing left to fetch: the .part file is complete
            part.replace(dest)
            return
        r.raise_for_status()
        if pos and r.status_code != 206:  # server ignored the Range header: start again
            pos = 0
        length = r.headers.get("Content-Length")
        total = int(length) + pos if length else None
        with open(part, "ab" if pos else "wb") as f, \
                tqdm(total=total, initial=pos, unit="B", unit_scale=True, desc=dest.name) as bar:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
                bar.update(len(block))
    if total is not None and part.stat().st_size != total:
        raise IOError(f"incomplete download: {part.stat().st_size} of {total} bytes")
    part.replace(dest)


class _Progress(io.RawIOBase):
    """Read-only wrapper that reports bytes read to a progress bar."""

    def __init__(self, raw, bar):
        self._raw, self._bar = raw, bar

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        data = self._raw.read(len(buffer))
        n = len(data)
        buffer[:n] = data
        self._bar.update(n)
        return n


def extract_tar_stream(fileobj, dest: Path) -> None:
    """Extract a (compressed) tar read sequentially from fileobj; nothing is buffered on disk."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=fileobj, mode="r|*") as tf:
        tf.extractall(dest, filter="data")  # blocks path traversal (Python >= 3.11.4)


def _stream_extract(url: str, dest: Path, timeout: int = 60) -> None:
    with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        r.raw.decode_content = True  # undo any transport compression; leaves the .tar.gz bytes
        length = r.headers.get("Content-Length")
        with tqdm(total=int(length) if length else None, unit="B", unit_scale=True,
                  desc=f"download+extract {Path(url).name}") as bar:
            extract_tar_stream(_Progress(r.raw, bar), dest)


def _try_mirrors(name: str, spec: dict, action, attempts: int = 3) -> None:
    errors = []
    for url in spec["urls"]:
        for attempt in range(1, attempts + 1):
            print(f"[{name}] {spec['size']} from {url} (attempt {attempt}/{attempts})")
            try:
                action(url)
                return
            except Exception as e:  # retry, then try the next mirror
                errors.append(f"{url} (attempt {attempt}): {e}")
                print(f"[{name}] failed: {e}")
    raise RuntimeError(f"[{name}] every mirror failed:\n  " + "\n  ".join(errors))


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        root = dest.resolve()
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                target = (dest / name).resolve()
                if root not in target.parents and target != root:
                    raise RuntimeError(f"unsafe path in archive: {name}")
            zf.extractall(dest)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")  # blocks path traversal (Python >= 3.11.4)


def fetch(name: str, raw: Path, downloads: Path, stream: bool = True, keep_archive: bool = False) -> None:
    spec = DATASETS[name]
    marker = raw / f".done_{name}"
    if marker.exists():
        print(f"[{name}] already downloaded and extracted, skipping")
        return
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / spec["archive"]
    manual = (f"\nDownload {spec['archive']} by hand into {downloads} and run this script again "
              f"(an archive found there is extracted directly).")
    streamable = spec["archive"].endswith((".tar.gz", ".tgz"))
    try:
        if archive.exists():
            print(f"[{name}] extracting {archive.name} ...")
            _extract(archive, raw)
        elif stream and streamable:
            (downloads / (spec["archive"] + ".part")).unlink(missing_ok=True)  # stale --no-stream leftover
            _try_mirrors(name, spec, lambda url: _stream_extract(url, raw))
        else:
            _try_mirrors(name, spec, lambda url: _download(url, archive))
            print(f"[{name}] extracting {archive.name} ...")
            _extract(archive, raw)
    except RuntimeError as e:
        raise RuntimeError(str(e) + manual) from None
    if not (raw / spec["check"]).exists():
        raise RuntimeError(f"[{name}] expected {raw / spec['check']} after extraction. "
                           f"Delete it and any {archive.name} in {downloads}, then try again.")
    marker.write_text("ok")
    if archive.exists() and not keep_archive:
        archive.unlink()
    print(f"[{name}] done")
