"""Exact update of a Gaussian prior with one Laplace observation.

The posterior of z given a scalar x, for the prior N(z; m, P) and the likelihood
Laplace(x; c z, r), is a mixture of two truncated Gaussians. Its mean, covariance and
the evidence log p(x) have closed forms, written here with the scaled complementary
error function erfcx so that they do not overflow for outliers.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erfcx


def log_erfcx(x: np.ndarray | float) -> np.ndarray:
    """Log of the scaled complementary error function, without overflow.

    erfcx(x) overflows for x below about -26.6. There the identity
    erfcx(x) = 2 exp(x^2) - erfcx(-x) is used in log form.

    Args:
        x: Input, any shape.

    Returns:
        log(erfcx(x)), same shape as x.
    """
    x = np.asarray(x, dtype=float)
    big = x < -20
    with np.errstate(over="ignore", divide="ignore"):
        y = np.log(erfcx(x))
        xb = np.where(big, x, 0.0)
        y_big = xb**2 + np.log(2 - np.exp(-(xb**2)) * erfcx(-xb))
    return np.where(big, y_big, y)


def _check_variance(s: float) -> None:
    if not (s > 0 and math.isfinite(s)):
        raise ValueError(
            f"Predicted output variance c P c' = {s:g} is not positive and finite. "
            "The model is numerically unstable (e.g. A has eigenvalues far outside "
            "the unit circle)."
        )


def _laplace_terms(x_tilde: float, s: float, r: float) -> tuple[float, float, float]:
    """Mean shift, variance change and log evidence of the exact update.

    Args:
        x_tilde: Residual x - c m.
        s: Predicted output variance c P c'.
        r: Laplace scale.

    Returns:
        delta: Posterior mean of the output is c m + s delta / r.
        Delta: Posterior output variance is s + s^2 Delta / r^2 (Delta <= 0).
        log_px: log p(x).
    """
    e1 = (s / r - x_tilde) / math.sqrt(2 * s)
    e2 = (s / r + x_tilde) / math.sqrt(2 * s)
    phi_m = float(erfcx(e1))
    phi_p = float(erfcx(e2))

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        tot = np.float64(phi_m) + phi_p
        delta = float((phi_m - phi_p) / tot)
        Delta = float(
            (4 * (np.float64(phi_m) * phi_p) / tot - 2 * r / math.sqrt(math.pi / 2 * s)) / tot
        )

    # Far in a tail one erfcx term overflows: the posterior is then a single
    # truncated Gaussian pushed against x, with no change in variance.
    if math.isinf(phi_m):
        delta, Delta = 1.0, 0.0
    if math.isinf(phi_p):
        delta, Delta = -1.0, 0.0
    if not math.isfinite(delta):
        delta = -1.0 if phi_p > phi_m else 1.0
    if not math.isfinite(Delta):
        Delta = 0.0

    l1 = float(log_erfcx(e1))
    l2 = float(log_erfcx(e2))
    lmax = max(l1, l2)
    log_px = (
        -math.log(4 * r)
        - x_tilde**2 / (2 * s)
        + lmax
        + math.log(math.exp(l1 - lmax) + math.exp(l2 - lmax))
    )
    return delta, Delta, log_px


def exact_moments(
    m: np.ndarray, P: np.ndarray, x: float, c: np.ndarray, r: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact posterior moments for one Laplace observation.

    Args:
        m: Prior mean (Dz,).
        P: Prior covariance (Dz, Dz).
        x: Scalar observation.
        c: Observation row (Dz,).
        r: Laplace scale.

    Returns:
        mu: Posterior mean (Dz,).
        V: Posterior covariance (Dz, Dz).
        log_px: Exact log p(x).
    """
    s = float(c @ P @ c)
    _check_variance(s)
    delta, Delta, log_px = _laplace_terms(float(x) - float(c @ m), s, r)
    k = P @ c / r
    mu = m + k * delta
    V = P + np.outer(k, k) * Delta
    return mu, (V + V.T) / 2, log_px


def exact_moments_natural(
    m: np.ndarray, P: np.ndarray, x: float, c: np.ndarray, r: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """The exact update as natural parameters of a Gaussian site (for EP).

    The posterior is N(mu, V) with V^-1 = P^-1 + Lam and V^-1 mu = P^-1 m + ell.

    Args:
        m: Prior (cavity) mean (Dz,).
        P: Prior (cavity) covariance (Dz, Dz).
        x: Scalar observation.
        c: Observation row (Dz,).
        r: Laplace scale.

    Returns:
        ell: Site precision-mean (Dz,).
        Lam: Site precision (Dz, Dz), rank one.
        log_px: Exact log p(x).
    """
    s = float(c @ P @ c)
    _check_variance(s)
    x_hat = float(c @ m)
    delta, Delta, log_px = _laplace_terms(float(x) - x_hat, s, r)
    nu = r**2 + s * Delta
    ell = -c * (x_hat * Delta - r * delta) / nu
    Lam = -np.outer(c, c) * Delta / nu
    return ell, Lam, log_px
