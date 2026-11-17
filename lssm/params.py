"""Parameters of a linear state space model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class Params:
    """Parameters of z_1 ~ N(m0, P0), z_n = A z_{n-1} + w_n, x_n = C z_n + v_n.

    Attributes:
        A: State transition matrix (Dz, Dz).
        Q: State noise covariance (Dz, Dz).
        C: Observation matrix (Dx, Dz).
        R: Observation noise, diagonal (Dx,). Laplace scales for the Laplace model
            (noise variance 2 R^2), variances for the Gaussian model.
        m0: Mean of the initial state (Dz,).
        P0: Covariance of the initial state (Dz, Dz).
        tau: ARD precisions of the columns of C (Dz,). The prior is
            C[:, j] ~ N(0, diag(R) / tau[j]).
    """

    A: np.ndarray
    Q: np.ndarray
    C: np.ndarray
    R: np.ndarray
    m0: np.ndarray
    P0: np.ndarray
    tau: np.ndarray

    @property
    def dim_z(self) -> int:
        """State dimension Dz."""
        return self.A.shape[0]

    @property
    def dim_x(self) -> int:
        """Observation dimension Dx."""
        return self.C.shape[0]

    def copy(self) -> Params:
        """Deep copy."""
        return replace(self, **{k: np.array(v, copy=True) for k, v in vars(self).items()})


def symmetrize(M: np.ndarray) -> np.ndarray:
    """(M + M^T) / 2 over the last two axes, against round-off asymmetry."""
    return (M + np.swapaxes(M, -1, -2)) / 2
