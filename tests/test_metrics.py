import numpy as np

from vadcal.bootstrap import bootstrap_metrics, clip_stats, metrics_from_stats, paired_difference
from vadcal.calibration import apply_temperature, fit_temperature, fit_vector_scaling
from vadcal.metrics import brier, classwise_ece, confusion, ece, f1_from_confusion, nll, softmax
from vadcal.smoothing import median_smooth

K, BINS = 4, 15


def _sample(rng, n, scale=2.0):
    """Logits z and labels drawn from softmax(z): softmax(z) is calibrated by construction."""
    z = rng.normal(size=(n, K)) * scale
    p = softmax(z)
    y = (rng.random(n)[:, None] > np.cumsum(p, 1)).sum(1)
    return z, np.minimum(y, K - 1)


def test_calibrated_model_has_small_ece():
    z, y = _sample(np.random.default_rng(0), 200_000)
    assert ece(softmax(z), y, BINS) < 0.01


def test_overconfident_model_has_large_ece():
    z, y = _sample(np.random.default_rng(1), 200_000)
    assert ece(softmax(3 * z), y, BINS) > 0.05


def test_temperature_scaling_recovers_true_temperature():
    z, y = _sample(np.random.default_rng(2), 100_000)
    T = fit_temperature(3 * z, y)
    assert abs(T - 3) / 3 < 0.1
    assert ece(apply_temperature(3 * z, T), y, BINS) < 0.01


def test_temperature_scaling_keeps_predictions():
    z, _ = _sample(np.random.default_rng(3), 1000)
    assert (apply_temperature(z, 2.5).argmax(1) == z.argmax(1)).all()


def test_vector_scaling_improves_nll():
    z, y = _sample(np.random.default_rng(4), 50_000)
    w, b = fit_vector_scaling(3 * z, y)
    assert nll(softmax(3 * z * w + b), y) < nll(softmax(3 * z), y)


def test_f1_matches_hand_computation():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    f1, macro = f1_from_confusion(confusion(y, pred, 3))
    assert np.allclose(f1, [0.5, 0.8, 2 / 3])
    assert np.isclose(macro, np.mean([0.5, 0.8, 2 / 3]))


def test_clip_stats_equal_direct_metrics():
    rng = np.random.default_rng(5)
    z, y = _sample(rng, 30 * 50)
    P, Y = softmax(2 * z).reshape(30, 50, K), y.reshape(30, 50)
    m = metrics_from_stats(clip_stats(P, Y, BINS).sum(0), BINS, K)
    flatP, flatY = P.reshape(-1, K), Y.ravel()
    assert np.isclose(m["ece"][0], ece(flatP, flatY, BINS))
    assert np.isclose(m["cw_ece"][0], classwise_ece(flatP, flatY, BINS)[0])
    assert np.isclose(m["nll"][0], nll(flatP, flatY))
    assert np.isclose(m["brier"][0], brier(flatP, flatY))
    _, macro = f1_from_confusion(confusion(flatY, flatP.argmax(1), K))
    assert np.isclose(m["macro_f1"][0], macro)


def test_bootstrap_interval_contains_point():
    rng = np.random.default_rng(6)
    z, y = _sample(rng, 40 * 50)
    S = clip_stats(softmax(z).reshape(40, 50, K), y.reshape(40, 50), BINS)
    point, lo, hi = bootstrap_metrics(S, BINS, K, 200, seed=0)
    # NLL and macro-F1 are (nearly) unbiased; ECE is biased upward on small samples, so its
    # percentile interval can sit entirely above the point estimate and is not tested here.
    for m in ("nll", "macro_f1"):
        assert lo[m] <= point[m] <= hi[m]
    assert lo["ece"] <= hi["ece"]


def test_paired_difference_of_identical_inputs_is_zero():
    rng = np.random.default_rng(7)
    z, y = _sample(rng, 20 * 50)
    S = clip_stats(softmax(z).reshape(20, 50, K), y.reshape(20, 50), BINS)
    row = paired_difference(S, S, BINS, K, 100, 0, ("ece",))[0]
    assert row["diff"] == 0 and row["diff_lo"] == 0 and row["diff_hi"] == 0


def test_median_smoothing():
    P = softmax(np.random.default_rng(8).normal(size=(3, 40, K)))
    assert median_smooth(P, 1) is P
    S = median_smooth(P, 5)
    assert S.shape == P.shape and np.allclose(S.sum(-1), 1)
