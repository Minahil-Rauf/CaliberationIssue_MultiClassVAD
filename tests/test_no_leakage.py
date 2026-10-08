"""No source file may appear in two splits, and the shifted test must use only unseen datasets.

Runs on the synthetic sources always, and on the real index once stage 2 has written it.
"""
import pandas as pd
import pytest

from vadcal.config import ROOT
from vadcal.sources import build_index

REAL_INDEX = ROOT / "data" / "index" / "sources.csv"


def _check(df: pd.DataFrame, cfg) -> None:
    assert df["path"].is_unique, "a file is listed twice"
    musan = df[df["dataset"] == "musan"]
    assert set(musan["split"]) <= {"train", "val", "test_in"}
    assert set(df[df["dataset"] != "musan"]["split"]) == {"test_shift"}
    assert not df[df["dataset"] == "esc50"]["group"].isin(cfg["sources"]["esc50_exclude"]).any()
    assert not df[df["dataset"] == "gtzan"]["group"].isin(cfg["sources"]["gtzan_exclude"]).any()
    for split in ("train", "val", "test_in", "test_shift"):
        for kind in ("speech", "music", "noise"):
            assert ((df["split"] == split) & (df["kind"] == kind)).any(), f"no {kind} in {split}"


def test_synthetic_index(cfg, fake_raw):
    _check(build_index(fake_raw, cfg), cfg)


def test_split_is_deterministic(cfg, fake_raw):
    a, b = build_index(fake_raw, cfg), build_index(fake_raw, cfg)
    assert a.equals(b)


@pytest.mark.skipif(not REAL_INDEX.exists(), reason="real index not built yet (stage 2)")
def test_real_index(cfg):
    df = pd.read_csv(REAL_INDEX)
    _check(df, cfg)
    for kind in ("speech", "music", "noise"):
        m = df[(df["dataset"] == "musan") & (df["kind"] == kind)]
        share = (m["split"] == "train").mean()
        assert 0.6 < share < 0.8, f"MUSAN {kind} train share {share:.2f} is off"
