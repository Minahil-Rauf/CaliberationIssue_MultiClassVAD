"""Streaming extraction, tested on an in-memory archive (no network)."""
import io
import tarfile

from vadcal.download import _Progress, extract_tar_stream


class _Bar:
    def __init__(self):
        self.n = 0

    def update(self, n):
        self.n += n


def _tar_gz(files: dict[str, bytes]) -> io.BytesIO:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def test_stream_extraction_writes_files(tmp_path):
    files = {"musan/speech/a.wav": b"speech" * 1000, "musan/noise/b.wav": b"noise" * 1000}
    bar = _Bar()
    extract_tar_stream(_Progress(_tar_gz(files), bar), tmp_path)
    for name, data in files.items():
        assert (tmp_path / name).read_bytes() == data
    assert bar.n > 0
