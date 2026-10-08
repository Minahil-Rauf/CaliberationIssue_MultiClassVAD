import numpy as np

from vadcal.config import audio_dims
from vadcal.features import logmel
from vadcal.labels import close_gaps, combine, remove_islands, speech_activity
from vadcal.mixing import build_clip, load_pools
from vadcal.model import build_model, count_params
from vadcal.sources import build_index


def test_combine_mapping():
    s = np.array([0, 1, 0, 1], dtype=bool)
    m = np.array([0, 0, 1, 1], dtype=bool)
    assert combine(s, m).tolist() == [0, 1, 2, 3]  # background, speech, music, speech+music


def test_close_gaps_and_islands():
    mask = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
    assert close_gaps(mask, 2).tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 1]
    assert remove_islands(mask, 2).tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]


def test_speech_activity_follows_the_signal(cfg):
    sr, hop = 16000, 160
    x = np.zeros(sr, dtype=np.float32)
    x[4000:12000] = 0.1 * np.sin(2 * np.pi * 200 * np.arange(8000) / sr)
    act = speech_activity(x, hop, sr, cfg["labels"])
    assert act[30:70].all() and not act[:20].any() and not act[80:].any()


def test_build_clip_shapes_and_labels(cfg, fake_raw):
    index = build_index(fake_raw, cfg)
    pools = load_pools(index, fake_raw, "train")
    sr, _, n_samples, n_frames = audio_dims(cfg)
    for i in range(3):
        audio, labels, meta = build_clip(np.random.default_rng([0, 0, i]), pools, cfg)
        assert audio.shape == (n_samples,) and audio.dtype == np.float32
        assert np.abs(audio).max() <= 1.0
        assert labels.shape == (n_frames,) and set(np.unique(labels)) <= {0, 1, 2, 3}
        assert meta["n_events"] >= 3


def test_build_clip_is_reproducible(cfg, fake_raw):
    index = build_index(fake_raw, cfg)
    pools = load_pools(index, fake_raw, "test_shift")
    a = build_clip(np.random.default_rng([1, 3, 7]), pools, cfg)
    b = build_clip(np.random.default_rng([1, 3, 7]), pools, cfg)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_features_and_model_shapes(cfg):
    import torch

    _, _, n_samples, n_frames = audio_dims(cfg)
    feats = logmel(np.random.default_rng(0).normal(size=n_samples).astype(np.float32) * 0.01, cfg)
    assert feats.shape == (n_frames, cfg["features"]["n_mels"])
    model = build_model(cfg)
    out = model(torch.from_numpy(feats[None]))
    assert out.shape == (1, n_frames, 4)
    assert count_params(model) < 200_000
