"""Gaussian state space model (Kalman filter and M-step), the baseline.

It uses the same priors as the Laplace model and shares the dynamics update and
the smoother, so the two models differ only in the observation noise. Here R holds
noise variances, not Laplace scales.
"""

from __future__ import annotations

import numpy as np

from lssm.inference import predict
from lssm.learning import SALT, learn_dynamics, state_stats
from lssm.params import Params, symmetrize


def kalman_filter(x: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray, float]:
    """Kalman filter.

    Args:
        x: Observations (N, Dx).
        p: Model parameters (R holds noise variances).

    Returns:
        mu: Filtered means (N, Dz).
        V: Filtered covariances (N, Dz, Dz).
        log_px: Exact log p(x_{1:N}).
    """
    N, dim_x = x.shape
    mu = np.zeros((N, p.dim_z))
    V = np.zeros((N, p.dim_z, p.dim_z))
    log_px = 0.0
    m, P = p.m0, p.P0
    C = p.C
    for n in range(N):
        S = symmetrize(C @ P @ C.T + np.diag(p.R))
        K = np.linalg.solve(S, C @ P).T
        e = x[n] - C @ m
        mu[n] = m + K @ e
        V[n] = symmetrize(P - K @ S @ K.T)
        log_px -= 0.5 * (
            dim_x * np.log(2 * np.pi) + np.linalg.slogdet(S)[1] + e @ np.linalg.solve(S, e)
        )
        m, P = predict(mu[n], V[n], p.A, p.Q)
    return mu, V, float(log_px)


def gaussian_m_step(
    x: np.ndarray, mu: np.ndarray, V: np.ndarray, V12: np.ndarray, p: Params
) -> Params:
    """M-step of the Gaussian model: MAP estimates with the same ARD prior on C.

    Args:
        x: Observations (N, Dx).
        mu: Smoothed means (N, Dz).
        V: Smoothed covariances (N, Dz, Dz).
        V12: Smoothed cross-covariances (N-1, Dz, Dz).
        p: Current parameters (only tau is used).

    Returns:
        The new parameters.
    """
    N, dim_z = mu.shape
    A, Q, m0, P0 = learn_dynamics(mu, V, V12)
    zz = state_stats(mu, V, V12)[0]
    xz = x.T @ mu
    C = np.linalg.solve(zz + np.diag(p.tau), xz.T).T
    res = (x**2).sum(0) - 2 * (C * xz).sum(1) + np.einsum("ij,jk,ik->i", C, zz, C)
    R = (2 * SALT + res + (C**2) @ p.tau) / (2 * SALT + N + dim_z)
    tau = (SALT + p.dim_x / 2) / (SALT + 0.5 * (C**2 / R[:, None]).sum(0))
    return Params(A=A, Q=Q, C=C, R=R, m0=m0, P0=P0, tau=tau)
