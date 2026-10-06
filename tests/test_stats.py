import numpy as np
from scipy import stats

from drugresponse.stats import (
    adjusted_corr,
    noise_scale,
    normalized_kernel,
    partial_rho,
    ridge_predict,
    subtype_group,
    zscore_columns,
)


def grouped_data(seed, within_signal):
    rng = np.random.default_rng(seed)
    codes = np.repeat(np.arange(6), 60)
    offsets = np.repeat(np.linspace(-3, 3, 6), 60)
    x = offsets + rng.normal(size=codes.size)
    y = offsets + (within_signal * (x - offsets) if within_signal else 0) + rng.normal(size=codes.size)
    return x, y, codes


def test_subtype_group_rules():
    assert subtype_group("basal_A TNBC") == "TNBC"
    assert subtype_group("luminal TNBC") == "TNBC"
    assert subtype_group("HER2+") == "HER2+ (ER-)"
    assert subtype_group("ER-/PR-/HER2+") == "HER2+ (ER-)"
    assert subtype_group("luminal ER, PR+") == "ER+"
    assert subtype_group("ER+, PR+, HER2+") == "ER+"
    assert subtype_group(float("nan")) == "unknown"


def test_adjusted_corr_ignores_between_group_confounding():
    x, y, codes = grouped_data(1, within_signal=0)
    raw = stats.spearmanr(x, y)[0]
    adj = adjusted_corr(x[None, :], y, codes)[0]
    assert raw > 0.5
    assert abs(adj) < 0.15


def test_adjusted_corr_finds_within_group_signal():
    x, y, codes = grouped_data(2, within_signal=1.0)
    assert adjusted_corr(x[None, :], y, codes)[0] > 0.4


def test_partial_rho_removes_a_shared_cause():
    rng = np.random.default_rng(3)
    z = (rng.random(400) < 0.3).astype(float)
    x = 1.5 * z + rng.normal(size=400)
    y = 1.5 * z + rng.normal(size=400)
    ones = np.ones((400, 1))
    raw, _ = partial_rho(x, y, ones)
    adj, _ = partial_rho(x, y, np.column_stack([ones, z]))
    assert raw > 0.2
    assert abs(adj) < 0.12


def test_noise_scale_matches_known_spread():
    values = np.random.default_rng(4).normal(0, 0.02, 20000)
    assert abs(noise_scale(values) - 0.02) < 0.002
    assert np.isnan(noise_scale([1.0, 2.0]))


def test_kernel_helpers():
    x = np.random.default_rng(5).normal(size=(30, 8))
    x[:, 3] = 1.0
    z = zscore_columns(x)
    assert z.shape[1] == 7
    assert np.allclose(z.std(axis=0), 1.0)
    assert np.isclose(np.mean(np.diag(normalized_kernel(z))), 1.0)


def test_ridge_learns_signal_but_not_noise():
    rng = np.random.default_rng(6)
    x = zscore_columns(rng.normal(size=(240, 40)))
    kernel = normalized_kernel(x)
    train, test = np.arange(180), np.arange(180, 240)
    signal = x[:, 0] + 0.5 * rng.normal(size=240)
    noise = rng.normal(size=240)
    for y, low, high in [(signal, 0.5, 1.0), (noise, -0.35, 0.35)]:
        pred = ridge_predict(kernel[np.ix_(train, train)], kernel[np.ix_(test, train)], y[train])
        r = stats.spearmanr(pred, y[test])[0]
        assert low < r < high
