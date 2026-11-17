import numpy as np
from scipy.stats import multivariate_normal

from lssm.data import phasor
from lssm.gaussian import kalman_filter
from lssm.inference import forecast, smoother, update
from lssm.params import Params


def test_ep_univariate_matches_exact():
    m, P = np.array([0.2, -0.1]), np.array([[0.4, 0.05], [0.05, 0.2]])
    C, R, x = np.array([[1.0, 0.3]]), np.array([0.25]), np.array([0.9])
    mu1, V1, p1 = update(m, P, x, C, R)
    # A second row with a huge scale carries almost no information.
    C2 = np.array([[1.0, 0.3], [0.0, 1e-6]])
    mu2, V2, p2 = update(m, P, np.array([0.9, 0.0]), C2, np.array([0.25, 1e6]), max_iter=10)
    np.testing.assert_allclose(mu2, mu1, atol=1e-6)
    np.testing.assert_allclose(V2, V1, atol=1e-6)
    np.testing.assert_allclose(p2 + np.log(2e6), p1, atol=1e-6)


def test_kalman_matches_batch_inference():
    """Kalman filter and RTS smoother agree with exact inference in the joint Gaussian."""
    rng = np.random.default_rng(1)
    dz, dx, N = 2, 2, 5
    p = Params(
        A=0.95 * phasor(0.4),
        Q=np.array([[0.05, 0.01], [0.01, 0.03]]),
        C=np.array([[1.0, 0.5], [-0.3, 0.8]]),
        R=np.array([0.2, 0.1]),
        m0=np.array([0.5, -0.2]),
        P0=np.array([[0.4, 0.1], [0.1, 0.3]]),
        tau=np.ones(dz),
    )
    x = rng.standard_normal((N, dx))
    mu_f, V_f, log_px = kalman_filter(x, p)
    mu_s, V_s, _ = smoother(mu_f, V_f, p.A, p.Q)

    Ap = [np.linalg.matrix_power(p.A, k) for k in range(N)]
    mz = np.concatenate([Ap[n] @ p.m0 for n in range(N)])
    Sz = np.zeros((dz * N, dz * N))
    for n in range(N):
        for k in range(N):
            c = Ap[n] @ p.P0 @ Ap[k].T
            for j in range(1, min(n, k) + 1):
                c += Ap[n - j] @ p.Q @ Ap[k - j].T
            Sz[n * dz : (n + 1) * dz, k * dz : (k + 1) * dz] = c
    Cb = np.kron(np.eye(N), p.C)
    Sx = Cb @ Sz @ Cb.T + np.kron(np.eye(N), np.diag(p.R))
    xv, mx = x.ravel(), Cb @ mz
    np.testing.assert_allclose(log_px, multivariate_normal.logpdf(xv, mx, Sx), atol=1e-9)

    G = np.linalg.solve(Sx, Cb @ Sz).T
    m_post = mz + G @ (xv - mx)
    S_post = Sz - G @ Cb @ Sz
    last, first = slice(dz * (N - 1), dz * N), slice(0, dz)
    np.testing.assert_allclose(mu_f[-1], m_post[last], atol=1e-9)
    np.testing.assert_allclose(V_f[-1], S_post[last, last], atol=1e-9)
    np.testing.assert_allclose(mu_s[0], m_post[first], atol=1e-9)
    np.testing.assert_allclose(V_s[0], S_post[first, first], atol=1e-9)


def test_forecast_runs_dynamics():
    A, Q = 0.9 * phasor(0.3), 0.01 * np.eye(2)
    mu, V = np.array([[1.0, 0.0]]), np.zeros((1, 2, 2))
    mu_f, V_f = forecast(mu, V, A, Q, 3)
    np.testing.assert_allclose(mu_f[3], np.linalg.matrix_power(A, 3) @ mu[0])
    np.testing.assert_allclose(V_f[1], Q)
