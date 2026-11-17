"""M-step (MAP parameter estimates) and the EM objective for the Laplace model."""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from lssm.params import Params, symmetrize

# Small constant for the vague priors and to keep variances positive.
SALT = 1e-9


def state_stats(
    mu: np.ndarray, V: np.ndarray, V12: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sufficient statistics of the smoothed states.

    Returns:
        zz: sum_n E[z_n z_n^T] (Dz, Dz).
        z1z2: sum_{n<N} E[z_n z_{n+1}^T] (Dz, Dz).
        z1z1: sum_{n<N} E[z_n z_n^T] (Dz, Dz).
        z2z2: sum_{n>1} E[z_n z_n^T] (Dz, Dz).
    """
    zz = mu.T @ mu + V.sum(0)
    z1z2 = mu[:-1].T @ mu[1:] + V12.sum(0)
    z1z1 = symmetrize(zz - (np.outer(mu[-1], mu[-1]) + V[-1]))
    z2z2 = symmetrize(zz - (np.outer(mu[0], mu[0]) + V[0]))
    return zz, z1z2, z1z1, z2z2


def learn_dynamics(
    mu: np.ndarray, V: np.ndarray, V12: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """M-step for the state parameters, shared by the Laplace and Gaussian models.

    A is the maximum likelihood estimate and Q the MAP estimate under a vague
    inverse-Wishart prior.

    Args:
        mu: Smoothed means (N, Dz).
        V: Smoothed covariances (N, Dz, Dz).
        V12: Smoothed cross-covariances (N-1, Dz, Dz).

    Returns:
        A (Dz, Dz), Q (Dz, Dz), m0 (Dz,), P0 (Dz, Dz).
    """
    N, dim_z = mu.shape
    _, z1z2, z1z1, z2z2 = state_stats(mu, V, V12)
    W0 = SALT * np.eye(dim_z) / dim_z
    A = np.linalg.solve(z1z1, z1z2).T
    W = W0 + z2z2 - (A @ z1z2 + z1z2.T @ A.T) + A @ z1z1 @ A.T
    Q = symmetrize(W / (dim_z + N - 1))
    return A, Q, mu[0].copy(), symmetrize(V[0])


def _output_stats(
    x: np.ndarray, mu: np.ndarray, V: np.ndarray, C: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Residuals x - C mu (N, Dx) and output variances c_i V_n c_i^T (N, Dx)."""
    S = np.einsum("ij,njk,ik->ni", C, V, C)
    return x - mu @ C.T, np.where(S > 0, S, SALT)


def _newton_row(
    x: np.ndarray, mu: np.ndarray, V: np.ndarray, c: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    """Newton iterations for one row of C.

    Maximizes the expected Laplace log-likelihood of one data dimension (with
    unit scale, which does not move the maximum) plus the ARD prior.

    Args:
        x: One data dimension (N,).
        mu: Smoothed means (N, Dz).
        V: Smoothed covariances (N, Dz, Dz).
        c: Starting row (Dz,).
        tau: ARD precisions (Dz,).

    Returns:
        The updated row (Dz,).
    """
    c = c.copy()
    for _ in range(20):
        CV = V @ c  # (N, Dz): c V_n, V_n symmetric
        S = CV @ c
        S = np.where(S > 0, S, SALT)
        x_tilde = x - mu @ c
        lv = x_tilde / np.sqrt(2 * S)
        f = np.exp(-(lv**2)) / np.sqrt(np.pi / 2 * S)
        grad = erf(lv) @ mu - f @ CV - c * tau
        ell = mu + CV * (x_tilde / S)[:, None]
        H = (
            np.einsum("n,ni,nj->ij", f / S, CV, CV)
            - np.einsum("n,ni,nj->ij", f, ell, ell)
            - np.einsum("n,nij->ij", f, V)
            - np.diag(tau)
        )
        v = -np.linalg.solve(H, grad)
        done = np.sqrt(abs(v @ H @ v)) < 1e-6
        # Limit the step to |dc| <= 1.
        c += v / max(1.0, np.abs(v).max())
        if done:
            break
    return c


def m_step(x: np.ndarray, mu: np.ndarray, V: np.ndarray, V12: np.ndarray, p: Params) -> Params:
    """M-step: MAP estimates of the Laplace state space model parameters.

    m0, P0, A and Q have closed forms (learn_dynamics). Each row of C is found
    by Newton's method on the expected Laplace log-likelihood plus the prior
    C[:, j] ~ N(0, diag(R) / tau[j]). R (one Laplace scale per data dimension)
    has a closed form, and the ARD precisions tau switch off the columns of C,
    and so the state dimensions, that are not needed.

    Args:
        x: Observations (N, Dx).
        mu: Smoothed means (N, Dz).
        V: Smoothed covariances (N, Dz, Dz).
        V12: Smoothed cross-covariances (N-1, Dz, Dz).
        p: Current parameters; C, R and tau are the starting points.

    Returns:
        The new parameters.
    """
    N, dim_z = mu.shape
    A, Q, m0, P0 = learn_dynamics(mu, V, V12)

    C = np.stack([_newton_row(x[:, i], mu, V, p.C[i], p.tau) for i in range(p.dim_x)])

    x_tilde, S = _output_stats(x, mu, V, C)
    b = (
        np.sqrt(2 / np.pi * S) * np.exp(-(x_tilde**2) / (2 * S))
        + x_tilde * erf(x_tilde / np.sqrt(2 * S))
    ).sum(0)
    R = (SALT + b + 0.5 * (C**2) @ p.tau) / (SALT + N + dim_z / 2)

    tau = (SALT + p.dim_x / 2) / (SALT + 0.5 * (C**2 / R[:, None]).sum(0))
    return Params(A=A, Q=Q, C=C, R=R, m0=m0, P0=P0, tau=tau)


def _gauss_state_terms(mu: np.ndarray, V: np.ndarray, V12: np.ndarray, p: Params) -> float:
    """Expected log densities of z_1 and of the transitions."""
    N = mu.shape[0]
    _, z1z2, z1z1, z2z2 = state_stats(mu, V, V12)
    d0 = mu[0] - p.m0
    z0 = np.outer(d0, d0) + V[0]
    out = -0.5 * np.linalg.slogdet(2 * np.pi * p.P0)[1]
    out -= 0.5 * np.trace(np.linalg.solve(p.P0, z0))
    Az1z2 = p.A @ z1z2
    out -= (N - 1) / 2 * np.linalg.slogdet(2 * np.pi * p.Q)[1]
    out -= 0.5 * np.trace(np.linalg.solve(p.Q, z2z2 - (Az1z2 + Az1z2.T) + p.A @ z1z1 @ p.A.T))
    return float(out)


def log_joint(x: np.ndarray, mu: np.ndarray, V: np.ndarray, V12: np.ndarray, p: Params) -> float:
    """Expected log joint E[log p(x, z, C)] under the smoothed state posterior.

    This is the objective of the M-step. The E-step is approximate, so EM is not
    guaranteed to increase it, and the ARD prior makes it grow without bound as
    unused columns of C shrink to zero. The filter's log p(x) is the better
    measure of fit.

    Args:
        x: Observations (N, Dx).
        mu: Smoothed means (N, Dz).
        V: Smoothed covariances (N, Dz, Dz).
        V12: Smoothed cross-covariances (N-1, Dz, Dz).
        p: Parameters.

    Returns:
        The expected log joint.
    """
    R, tau = p.R, p.tau
    log_prior = -0.5 * np.log(2 * np.pi * R[:, None] / tau[None, :]).sum()
    log_prior -= 0.5 * (tau * (p.C**2 / R[:, None]).sum(0)).sum()

    S = np.einsum("ij,njk,ik->ni", p.C, V, p.C)
    x_tilde = x - mu @ p.C.T
    lv = x_tilde / np.sqrt(2 * S)
    f = np.exp(-(lv**2)) / np.sqrt(2 * np.pi * S)
    # E|x - c z| for c z ~ N(c mu, S).
    abs_res = 2 * f * S + x_tilde * erf(lv)
    log_lik = -(abs_res / R + np.log(2 * R)).sum()

    return float(log_lik + _gauss_state_terms(mu, V, V12, p) + log_prior)
