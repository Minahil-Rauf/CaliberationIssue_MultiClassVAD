import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vadcal.config import load_config  # noqa: E402
from vadcal.fake_sources import generate_all  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def fake_raw(tmp_path_factory):
    """A small synthetic copy of the four datasets in their real folder layout."""
    raw = tmp_path_factory.mktemp("raw")
    generate_all(raw, seed=0, n_musan=5, musan_seconds=8.0, n_libri=3, n_esc=6, n_gtzan=2)
    return raw
