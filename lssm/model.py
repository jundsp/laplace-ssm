"""High-level models: initialize, fit by EM, filter, smooth and forecast."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lssm.data import random_block_diagonal
from lssm.gaussian import gaussian_m_step, kalman_filter
from lssm.inference import Method, filter, forecast, smoother
from lssm.learning import log_joint, m_step
from lssm.params import Params

Callback = Callable[[int, Params, np.ndarray, np.ndarray, float], None]


def init_params(dim_z: int, dim_x: int, rng: np.random.Generator) -> Params:
    """Random initial parameters for EM.

    A is a random orthogonal matrix masked to its 2 x 2 diagonal blocks
    (random_block_diagonal), C is uniform in [-1, 1],
    and the ARD precisions start vague, so every state dimension starts active.

    Args:
        dim_z: State dimension, even. It can be overestimated: ARD prunes extras.
        dim_x: Observation dimension.
        rng: Random generator.

    Returns:
        The initial parameters.
    """
    return Params(
        A=random_block_diagonal(dim_z, rng),
        Q=0.01 * np.eye(dim_z),
        C=rng.random((dim_x, dim_z)) * 2 - 1,
        R=np.full(dim_x, 0.01),
        m0=np.zeros(dim_z),
        P0=np.eye(dim_z),
        tau=np.full(dim_z, 1e-9),
    )


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


class LaplaceSSM:
    """Linear state space model with Laplace observation noise, learned by EM.

    Example:
        >>> model = LaplaceSSM(dim_z=10, seed=0).fit(x, epochs=200)
        >>> mu, V = model.smooth(x)
        >>> mean, std = model.forecast(x, steps=50)

    Attributes:
        params: Current parameters (None until fit or set by hand).
        log_likelihood: log p(x) from the filter at each EM iteration.
    """

    noise = "laplace"

    def __init__(
        self,
        dim_z: int = 10,
        method: Method = "ep",
        seed: int | None = None,
        params: Params | None = None,
    ) -> None:
        """Create a model.

        Args:
            dim_z: State dimension, even. Extra dimensions are pruned by ARD.
            method: Filter update, "ep" (exact moments / EP) or "vb".
            seed: Seed of the random initialization.
            params: Initial parameters; random if None.
        """
        self.dim_z = dim_z
        self.method = method
        self.rng = np.random.default_rng(seed)
        self.params = params
        self.log_likelihood: list[float] = []

    def _filter(self, x: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray, float]:
        return filter(x, p, self.method)

    def _m_step(
        self, x: np.ndarray, mu: np.ndarray, V: np.ndarray, V12: np.ndarray, p: Params
    ) -> Params:
        return m_step(x, mu, V, V12, p)

    def e_step(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Filter and smooth with the current parameters.

        Args:
            x: Observations (N, Dx) or (N,).

        Returns:
            mu (N, Dz), V (N, Dz, Dz), V12 (N-1, Dz, Dz) and log p(x).
        """
        x = _as_2d(x)
        mu, V, log_px = self._filter(x, self.params)
        mu, V, V12 = smoother(mu, V, self.params.A, self.params.Q)
        return mu, V, V12, log_px

    def fit(self, x: np.ndarray, epochs: int = 100, callback: Callback | None = None) -> LaplaceSSM:
        """Learn the parameters by EM.

        Args:
            x: Observations (N, Dx) or (N,).
            epochs: Number of EM iterations.
            callback: Called after each E-step as callback(epoch, params, mu, V,
                log_px), with the smoothed moments of those params.

        Returns:
            self.
        """
        x = _as_2d(x)
        if self.params is None:
            self.params = init_params(self.dim_z, x.shape[1], self.rng)
        for epoch in range(epochs):
            mu, V, V12, log_px = self.e_step(x)
            self.log_likelihood.append(log_px)
            if callback is not None:
                callback(epoch, self.params, mu, V, log_px)
            self.params = self._m_step(x, mu, V, V12, self.params)
        return self

    def log_joint(self, x: np.ndarray) -> float:
        """Expected log joint (the M-step objective) at the current parameters."""
        x = _as_2d(x)
        mu, V, V12, _ = self.e_step(x)
        return log_joint(x, mu, V, V12, self.params)

    def smooth(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Smoothed state means (N, Dz) and covariances (N, Dz, Dz)."""
        mu, V, _, _ = self.e_step(x)
        return mu, V

    def noise_variance(self) -> np.ndarray:
        """Observation noise variances (Dx,)."""
        return 2 * self.params.R**2

    def forecast(self, x: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Smoothed estimate of the data followed by a forecast.

        Args:
            x: Observations (N, Dx) or (N,).
            steps: Number of steps to forecast past the end of x.

        Returns:
            mean: Mean of x (N + steps, Dx).
            std: Standard deviation of x, including the noise (N + steps, Dx).
        """
        p = self.params
        mu, V = forecast(*self.smooth(x), p.A, p.Q, steps)
        var = np.einsum("ij,njk,ik->ni", p.C, V, p.C) + self.noise_variance()
        return mu @ p.C.T, np.sqrt(var)

    def active_dims(self, threshold: float = 1e-3) -> np.ndarray:
        """State dimensions whose column of C is kept by ARD (L1 norm above threshold x max)."""
        norms = np.abs(self.params.C).sum(0)
        return np.flatnonzero(norms > threshold * norms.max())


class GaussianSSM(LaplaceSSM):
    """The Gaussian baseline: a Kalman filter and the same priors and EM.

    R holds noise variances. The method argument is ignored.
    """

    noise = "gaussian"

    def _filter(self, x: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray, float]:
        return kalman_filter(x, p)

    def _m_step(
        self, x: np.ndarray, mu: np.ndarray, V: np.ndarray, V12: np.ndarray, p: Params
    ) -> Params:
        return gaussian_m_step(x, mu, V, V12, p)

    def noise_variance(self) -> np.ndarray:
        """Observation noise variances (Dx,)."""
        return self.params.R.copy()

    def log_joint(self, x: np.ndarray) -> float:
        """Not implemented for the Gaussian baseline."""
        raise NotImplementedError("log_joint is only implemented for the Laplace model")
