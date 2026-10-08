"""Resumable dataset downloads (HTTP Range) with mirror fallback, then safe extraction."""
from __future__ import annotations

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


def fetch(name: str, raw: Path, downloads: Path, keep_archive: bool = False) -> None:
    spec = DATASETS[name]
    marker = raw / f".done_{name}"
    if marker.exists():
        print(f"[{name}] already downloaded and extracted, skipping")
        return
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / spec["archive"]
    if not archive.exists():
        errors = []
        for url in spec["urls"]:
            print(f"[{name}] downloading {spec['size']} from {url}")
            try:
                _download(url, archive)
                break
            except Exception as e:  # try the next mirror
                errors.append(f"{url}: {e}")
                print(f"[{name}] failed: {e}")
        else:
            raise RuntimeError(
                f"[{name}] every mirror failed:\n  " + "\n  ".join(errors)
                + f"\nDownload {spec['archive']} by hand into {downloads} and run this script again.")
    print(f"[{name}] extracting {archive.name} ...")
    _extract(archive, raw)
    if not (raw / spec["check"]).exists():
        raise RuntimeError(f"[{name}] expected {raw / spec['check']} after extraction. "
                           f"Delete {archive} and try again.")
    marker.write_text("ok")
    if not keep_archive:
        archive.unlink()
    print(f"[{name}] done")
