"""Filtering and smoothing for the Laplace state space model."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.special import kve
from scipy.stats import multivariate_normal

from lssm.moments import exact_moments, exact_moments_natural
from lssm.params import Params, symmetrize

Method = Literal["ep", "vb"]


def predict(
    mu: np.ndarray, V: np.ndarray, A: np.ndarray, Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Prediction step: moments of A z + w for z ~ N(mu, V), w ~ N(0, Q).

    Args:
        mu: Mean (Dz,).
        V: Covariance (Dz, Dz).
        A: Transition matrix (Dz, Dz).
        Q: State noise covariance (Dz, Dz).

    Returns:
        m: Predicted mean (Dz,).
        P: Predicted covariance (Dz, Dz).
    """
    # Keep P symmetric: round-off asymmetry grows when A is unstable.
    return A @ mu, symmetrize(A @ V @ A.T + Q)


def update(
    m: np.ndarray,
    P: np.ndarray,
    x: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
    max_iter: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Update step with Laplace observations: exact moments or EP.

    For univariate x the posterior moments are exact. For multivariate x,
    expectation propagation uses one site per data dimension, with exact site
    updates, and runs at most min(max_iter, Dx) sweeps.

    Args:
        m: Predicted mean (Dz,).
        P: Predicted covariance (Dz, Dz).
        x: Observation (Dx,).
        C: Observation matrix (Dx, Dz).
        R: Laplace scales (Dx,).
        max_iter: Maximum number of EP sweeps (default Dx).

    Returns:
        mu: Posterior mean (Dz,).
        V: Posterior covariance (Dz, Dz).
        log_px: log p(x) (the EP approximation for multivariate x).
    """
    dim_x = x.shape[0]
    if dim_x == 1:
        return exact_moments(m, P, x[0], C[0], R[0])

    dim_z = m.shape[0]
    n_iter = min(max_iter or dim_x, dim_x)
    ell = np.zeros((dim_x, dim_z))
    Lam = np.broadcast_to(np.eye(dim_z), (dim_x, dim_z, dim_z)).copy()
    log_px = np.zeros(dim_x)

    Lam_hat = Lam.sum(0) + np.linalg.inv(P)
    ell_hat = ell.sum(0) + np.linalg.solve(P, m)
    ell_last = np.full(dim_z, np.inf)
    for _ in range(n_iter):
        for i in range(dim_x):
            Lam_cav = Lam_hat - Lam[i]
            ell_cav = ell_hat - ell[i]
            P_cav = np.linalg.inv(Lam_cav)
            ell[i], Lam[i], log_px[i] = exact_moments_natural(
                P_cav @ ell_cav, P_cav, x[i], C[i], R[i]
            )
            ell_hat = ell_cav + ell[i]
            Lam_hat = Lam_cav + Lam[i]
        if np.abs(ell_hat - ell_last).sum() < 1e-7:
            break
        ell_last = ell_hat

    V = symmetrize(np.linalg.inv(Lam_hat))
    return V @ ell_hat, V, float(log_px.sum())


def update_vb(
    m: np.ndarray, P: np.ndarray, x: np.ndarray, C: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Update step with Laplace observations by variational Bayes.

    The multivariate Laplace likelihood is written as a Gaussian scale mixture,
    and the state and the scale are updated in turn.

    Args:
        m: Predicted mean (Dz,).
        P: Predicted covariance (Dz, Dz).
        x: Observation (Dx,).
        C: Observation matrix (Dx, Dz).
        R: Laplace scales (Dx,).

    Returns:
        mu: Posterior mean (Dz,).
        V: Posterior covariance (Dz, Dz).
        log_px: log N(x; C m, S) with the final effective output covariance S.
    """
    dim_x = x.shape[0]
    mu, V = m, P
    mu_last = np.full_like(m, np.inf)
    CPC = C @ P @ C.T
    for _ in range(50):
        xi = float(((x - C @ mu) ** 2 + np.einsum("ij,jk,ik->i", C, V, C)) @ (1 / R))
        s = np.sqrt(2 * xi)
        # Ratio of modified Bessel functions; the exponentially scaled kve keeps
        # it finite where kv underflows.
        z_mean = np.sqrt(2 / xi) * kve(dim_x / 2, s) / kve(dim_x / 2 - 1, s)
        S = np.diag(R / z_mean) + CPC
        K = np.linalg.solve(S, C @ P).T
        V = symmetrize(P - K @ C @ P)
        mu = m + K @ (x - C @ m)
        if np.abs(mu - mu_last).sum() < 1e-9:
            break
        mu_last = mu
    log_px = float(multivariate_normal.logpdf(x, C @ m, S, allow_singular=True))
    return mu, V, log_px


def filter(
    x: np.ndarray, p: Params, method: Method = "ep", max_iter: int | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Forward filter of the Laplace state space model.

    Args:
        x: Observations (N, Dx).
        p: Model parameters (R holds Laplace scales).
        method: "ep" (exact moments / expectation propagation) or "vb".
        max_iter: Maximum EP sweeps per time step (default Dx).

    Returns:
        mu: Filtered means (N, Dz).
        V: Filtered covariances (N, Dz, Dz).
        log_px: Approximate log p(x_{1:N}).
    """
    method = method.lower()
    if method not in ("ep", "vb"):
        raise ValueError(f"method must be 'ep' or 'vb', not {method!r}")
    N = x.shape[0]
    mu = np.zeros((N, p.dim_z))
    V = np.zeros((N, p.dim_z, p.dim_z))
    log_px = 0.0
    m, P = p.m0, p.P0
    for n in range(N):
        if method == "vb":
            mu[n], V[n], lp = update_vb(m, P, x[n], p.C, p.R)
        else:
            mu[n], V[n], lp = update(m, P, x[n], p.C, p.R, max_iter)
        m, P = predict(mu[n], V[n], p.A, p.Q)
        log_px += lp
    return mu, V, log_px


def smooth_step(
    m: np.ndarray,
    P: np.ndarray,
    mu: np.ndarray,
    V: np.ndarray,
    mu_next: np.ndarray,
    V_next: np.ndarray,
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One Rauch-Tung-Striebel smoother step.

    Args:
        m: Predicted mean of z_{n+1} from the filtered z_n (Dz,).
        P: Predicted covariance (Dz, Dz).
        mu: Filtered mean of z_n (Dz,).
        V: Filtered covariance of z_n (Dz, Dz).
        mu_next: Smoothed mean of z_{n+1} (Dz,).
        V_next: Smoothed covariance of z_{n+1} (Dz, Dz).
        A: Transition matrix (Dz, Dz).

    Returns:
        mu: Smoothed mean of z_n (Dz,).
        V: Smoothed covariance of z_n (Dz, Dz).
        V12: Smoothed cross-covariance Cov[z_n, z_{n+1}] (Dz, Dz).
    """
    J = np.linalg.solve(P, A @ V).T  # V A^T P^-1; P and V are symmetric
    mu_s = mu + J @ (mu_next - m)
    V_s = symmetrize(V + J @ (V_next - P) @ J.T)
    return mu_s, V_s, J @ V_next


def smoother(
    mu: np.ndarray, V: np.ndarray, A: np.ndarray, Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel smoother over a sequence of filtered moments.

    Args:
        mu: Filtered means (N, Dz).
        V: Filtered covariances (N, Dz, Dz).
        A: Transition matrix (Dz, Dz).
        Q: State noise covariance (Dz, Dz).

    Returns:
        mu: Smoothed means (N, Dz).
        V: Smoothed covariances (N, Dz, Dz).
        V12: Smoothed lag-one cross-covariances Cov[z_n, z_{n+1}] (N-1, Dz, Dz).
    """
    N, dim_z = mu.shape
    mu, V = mu.copy(), V.copy()
    V12 = np.zeros((N - 1, dim_z, dim_z))
    for n in range(N - 2, -1, -1):
        m, P = predict(mu[n], V[n], A, Q)
        mu[n], V[n], V12[n] = smooth_step(m, P, mu[n], V[n], mu[n + 1], V[n + 1], A)
    return mu, V, V12


def forecast(
    mu: np.ndarray, V: np.ndarray, A: np.ndarray, Q: np.ndarray, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extend a sequence of state moments by running the dynamics forward.

    Args:
        mu: State means (N, Dz).
        V: State covariances (N, Dz, Dz).
        A: Transition matrix (Dz, Dz).
        Q: State noise covariance (Dz, Dz).
        steps: Number of steps to forecast.

    Returns:
        mu: Means (N + steps, Dz), the input followed by the forecast.
        V: Covariances (N + steps, Dz, Dz).
    """
    N, dim_z = mu.shape
    mu_f = np.concatenate([mu, np.zeros((steps, dim_z))])
    V_f = np.concatenate([V, np.zeros((steps, dim_z, dim_z))])
    for n in range(N, N + steps):
        mu_f[n], V_f[n] = predict(mu_f[n - 1], V_f[n - 1], A, Q)
    return mu_f, V_f
