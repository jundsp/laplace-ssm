import numpy as np
from scipy.stats import multivariate_normal

from lssm.data import phasor
from lssm.gaussian import gaussian_m_step
from lssm.learning import log_joint, m_step
from lssm.params import Params


def test_log_joint_point_mass():
    """With no posterior uncertainty, the expected log joint is the log joint at the means."""
    rng = np.random.default_rng(0)
    dz, dx, N = 2, 2, 6
    p = Params(
        A=0.9 * phasor(0.3),
        Q=0.05 * np.eye(dz),
        C=np.array([[1.0, 0.5], [-0.3, 0.8]]),
        R=np.array([0.2, 0.4]),
        m0=np.array([0.5, 0.0]),
        P0=0.3 * np.eye(dz),
        tau=np.array([2.0, 3.0]),
    )
    mu = rng.standard_normal((N, dz))
    x = mu @ p.C.T + 0.3 * rng.standard_normal((N, dx))
    V = np.broadcast_to(1e-14 * np.eye(dz), (N, dz, dz)).copy()
    V12 = np.zeros((N - 1, dz, dz))

    ref = multivariate_normal.logpdf(mu[0], p.m0, p.P0)
    ref += sum(multivariate_normal.logpdf(mu[n], p.A @ mu[n - 1], p.Q) for n in range(1, N))
    ref += (-np.log(2 * p.R) - np.abs(x - mu @ p.C.T) / p.R).sum()
    ref += sum(
        multivariate_normal.logpdf(p.C[:, j], np.zeros(dx), np.diag(p.R) / p.tau[j])
        for j in range(dz)
    )
    np.testing.assert_allclose(log_joint(x, mu, V, V12, p), ref, atol=1e-5)


def test_learn_R_per_dimension():
    """Identical data rows with identical rows of C must give identical noise scales."""
    rng = np.random.default_rng(1)
    dz, N = 2, 40
    mu = rng.standard_normal((N, dz))
    V = np.broadcast_to(0.01 * np.eye(dz), (N, dz, dz)).copy()
    V12 = np.broadcast_to(0.001 * np.eye(dz), (N - 1, dz, dz)).copy()
    row = rng.standard_normal(N)
    x = np.stack([row, row], 1)
    C = np.array([[0.7, -0.2], [0.7, -0.2]])
    p = Params(np.eye(dz), np.eye(dz), C, np.full(2, 0.1), np.zeros(dz), np.eye(dz), np.ones(dz))
    q = m_step(x, mu, V, V12, p)
    np.testing.assert_allclose(q.C[0], q.C[1], atol=1e-12)
    np.testing.assert_allclose(q.R[0], q.R[1], atol=1e-12)


def test_gaussian_m_step_is_optimal():
    """The Gaussian M-step for C and R maximizes the expected log joint."""
    rng = np.random.default_rng(2)
    dz, dx, N = 3, 2, 30
    mu = rng.standard_normal((N, dz))
    V = np.broadcast_to(0.05 * np.eye(dz), (N, dz, dz)).copy()
    V12 = np.broadcast_to(0.01 * np.eye(dz), (N - 1, dz, dz)).copy()
    x = mu @ rng.standard_normal((dx, dz)).T + 0.3 * rng.standard_normal((N, dx))
    tau = np.array([0.5, 2.0, 10.0])
    p = Params(
        np.eye(dz), np.eye(dz), np.zeros((dx, dz)), np.ones(dx), np.zeros(dz), np.eye(dz), tau
    )
    q = gaussian_m_step(x, mu, V, V12, p)

    zz = mu.T @ mu + V.sum(0)
    xz = x.T @ mu

    def objective(C, R):
        res = (x**2).sum(0) - 2 * (C * xz).sum(1) + np.einsum("ij,jk,ik->i", C, zz, C)
        return (-(N + dz) / 2 * np.log(R) - (res + (C**2) @ tau) / (2 * R)).sum()

    J0 = objective(q.C, q.R)
    for _ in range(20):
        assert objective(q.C + 1e-4 * rng.standard_normal(q.C.shape), q.R) < J0
        assert objective(q.C, q.R + 1e-4 * rng.standard_normal(dx)) < J0
