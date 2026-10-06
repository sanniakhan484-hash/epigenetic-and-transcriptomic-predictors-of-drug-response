"""Shared statistical helpers used by the analysis scripts and the app."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

DEFAULT_LAMBDAS = np.logspace(-1, 3, 9)


def subtype_group(label) -> str:
    """Map a DepMap ModelSubtypeFeatures label to TNBC, ER+, HER2+ (ER-) or unknown."""
    s = str(label)
    if "TNBC" in s:
        return "TNBC"
    if s in ("HER2+", "ER-/PR-/HER2+"):
        return "HER2+ (ER-)"
    if "ER" in s:
        return "ER+"
    return "unknown"


def demean_by_group(x, codes):
    """Subtract the per-group mean (groups given by integer codes) along the last axis."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    codes = np.asarray(codes)
    onehot = np.eye(int(codes.max()) + 1)[codes]
    counts = onehot.sum(axis=0)
    means = (x @ onehot) / np.maximum(counts, 1)
    return x - means @ onehot.T


def adjusted_corr(x_rows, y, codes):
    """Group-adjusted Spearman correlation of each row of x_rows with y."""
    xa = demean_by_group(sps.rankdata(np.atleast_2d(x_rows), axis=1), codes)
    ya = demean_by_group(sps.rankdata(y), codes)[0]
    den = np.sqrt((xa**2).sum(axis=1) * (ya**2).sum())
    return (xa * ya).sum(axis=1) / den


def partial_rho(x, y, design):
    """Rank correlation of x and y after regressing out the columns of design.

    Returns (rho, p). The design should include the group indicators (or an intercept).
    """

    def residual(v):
        beta, *_ = np.linalg.lstsq(design, v, rcond=None)
        return v - design @ beta

    rx = residual(sps.rankdata(x))
    ry = residual(sps.rankdata(y))
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return np.nan, np.nan
    r = float(np.clip(np.corrcoef(rx, ry)[0, 1], -0.999999, 0.999999))
    df = len(x) - int(np.linalg.matrix_rank(design)) - 1
    t = r * np.sqrt(df / (1 - r**2))
    return r, float(2 * sps.t.sf(abs(t), df))


def noise_scale(values) -> float:
    """Noise spread estimated from the negative side only (robust to a positive tail)."""
    values = np.asarray(values, dtype=float)
    neg = values[values < 0]
    return 1.4826 * float(np.median(np.abs(neg))) if len(neg) else float("nan")


def zscore_columns(x):
    """Standardize columns, dropping columns with no variation."""
    x = np.asarray(x, dtype=float)
    sd = x.std(axis=0)
    keep = sd > 1e-9
    return (x[:, keep] - x[:, keep].mean(axis=0)) / sd[keep]


def normalized_kernel(x):
    """Linear kernel scaled so that its mean diagonal is 1."""
    k = x @ x.T
    return k / np.mean(np.diag(k))


def ridge_predict(k_train, k_test, y_train, lambdas=DEFAULT_LAMBDAS):
    """Kernel ridge prediction with the penalty chosen by closed-form leave-one-out error."""
    mu = y_train.mean()
    yc = y_train - mu
    s, u = np.linalg.eigh(k_train)
    s = np.clip(s, 0, None)
    uty = u.T @ yc
    best_err, best_alpha = None, None
    for lam in lambdas:
        w = 1.0 / (s + lam)
        alpha = u @ (w * uty)
        hdiag = (u**2) @ w
        err = float(np.mean((alpha / hdiag) ** 2))
        if best_err is None or err < best_err:
            best_err, best_alpha = err, alpha
    return k_test @ best_alpha + mu
