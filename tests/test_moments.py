import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx

from lssm.moments import exact_moments, exact_moments_natural, log_erfcx


def quadrature_moments(m, P, x, c, r):
    """Posterior of z for prior N(m, P) and Laplace(x; c z, r), by 1-D quadrature over s = c z."""
    s0, S = c @ m, c @ P @ c

    def w(s):
        return (
            np.exp(-((s - s0) ** 2) / (2 * S))
            / np.sqrt(2 * np.pi * S)
            * np.exp(-abs(x - s) / r)
            / (2 * r)
        )

    lo = min(s0, x) - 40 * np.sqrt(S) - 40 * r
    hi = max(s0, x) + 40 * np.sqrt(S) + 40 * r

    def q(f):
        kw = dict(epsabs=1e-14, epsrel=1e-12, limit=200)
        return quad(f, lo, x, **kw)[0] + quad(f, x, hi, **kw)[0]

    Z = q(w)
    Es = q(lambda s: s * w(s)) / Z
    Vs = q(lambda s: (s - Es) ** 2 * w(s)) / Z
    k = P @ c / S
    return m + k * (Es - s0), P - np.outer(k, k) * S + np.outer(k, k) * Vs, np.log(Z)


def test_exact_moments_match_quadrature():
    m, P = np.array([0.3, -0.2]), np.array([[0.5, 0.1], [0.1, 0.3]])
    c, r = np.array([1.0, -0.5]), 0.2
    for x in [-1.5, -0.1, 0.0, 0.4, 2.0]:
        mu, V, log_px = exact_moments(m, P, x, c, r)
        mu_q, V_q, log_q = quadrature_moments(m, P, x, c, r)
        np.testing.assert_allclose(mu, mu_q, atol=1e-7)
        np.testing.assert_allclose(V, V_q, atol=1e-7)
        np.testing.assert_allclose(log_px, log_q, atol=1e-7)


def test_log_likelihood_no_overflow():
    # Far from the prior mean, p(x) ~ Laplace(x; c m, r) exp(S / (2 r^2)).
    m, P, c, r = np.zeros(1), np.array([[1e-4]]), np.ones(1), 1e-3
    S = 1e-4
    for x in [-50.0, 50.0, 1e3]:
        _, _, log_px = exact_moments(m, P, x, c, r)
        approx = -np.log(2 * r) - abs(x) / r + S / (2 * r**2)
        assert np.isfinite(log_px)
        np.testing.assert_allclose(log_px, approx, rtol=1e-9)


def test_log_erfcx():
    x = np.array([-30.0, -25.0, -5.0, 0.0, 3.0, 100.0])
    ok = x > -26
    np.testing.assert_allclose(log_erfcx(x)[ok], np.log(erfcx(x[ok])), rtol=1e-12)
    # erfcx(x) ~ 2 exp(x^2) for x -> -inf
    np.testing.assert_allclose(log_erfcx(-30.0), 900 + np.log(2), rtol=1e-12)


def test_natural_matches_moments():
    rng = np.random.default_rng(0)
    L = rng.standard_normal((3, 3))
    m, P = np.array([0.1, 0.4, -0.3]), L @ L.T + 0.1 * np.eye(3)
    c, r = np.array([0.5, -1.0, 0.2]), 0.3
    for x in [-1.0, 0.2, 1.5]:
        mu, V, lp1 = exact_moments(m, P, x, c, r)
        ell, Lam, lp2 = exact_moments_natural(m, P, x, c, r)
        V2 = np.linalg.inv(np.linalg.inv(P) + Lam)
        mu2 = V2 @ (np.linalg.solve(P, m) + ell)
        np.testing.assert_allclose(mu2, mu, atol=1e-9)
        np.testing.assert_allclose(V2, V, atol=1e-9)
        np.testing.assert_allclose(lp2, lp1, atol=1e-12)
