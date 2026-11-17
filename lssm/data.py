"""Synthetic data: test sequences, sampling and random initial dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag
from scipy.stats import wishart

from lssm.params import Params, symmetrize


def phasor(omega: float) -> np.ndarray:
    """2-D rotation by omega radians per step, an undamped oscillator (2, 2)."""
    return np.array([[np.cos(omega), -np.sin(omega)], [np.sin(omega), np.cos(omega)]])


def rotrnd(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthogonal matrix: a Householder reflection of the anti-identity.

    Args:
        dim: Size.
        rng: Random generator.

    Returns:
        (I - 2 v v^T) J for a random unit vector v and J the anti-identity (dim, dim).
    """
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    J = np.fliplr(np.eye(dim))
    return J - 2 * np.outer(v, v) @ J


def random_block_diagonal(dim_z: int, rng: np.random.Generator) -> np.ndarray:
    """A random orthogonal matrix (rotrnd) masked to its 2 x 2 diagonal blocks.

    The blocks are small and unstructured, so the eigenvalues start well inside the
    unit circle and away from any oscillation; EM has to find the dynamics.

    Args:
        dim_z: State dimension, even.
        rng: Random generator.

    Returns:
        A (dim_z, dim_z).
    """
    if dim_z % 2:
        raise ValueError(f"dim_z must be even, not {dim_z}")
    return rotrnd(dim_z, rng) * block_diag(*[np.ones((2, 2))] * (dim_z // 2))


def sample(N: int, p: Params, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample a sequence from a Laplace state space model.

    Args:
        N: Number of time steps.
        p: Parameters (R holds Laplace scales; tau is not used).
        rng: Random generator.

    Returns:
        x: Observations (N, Dx).
        z: States (N, Dz).
    """
    z = np.zeros((N, p.dim_z))
    z[0] = rng.multivariate_normal(p.m0, symmetrize(p.P0))
    w = rng.multivariate_normal(np.zeros(p.dim_z), symmetrize(p.Q), size=N - 1)
    for n in range(1, N):
        z[n] = p.A @ z[n - 1] + w[n - 1]
    x = rng.laplace(z @ p.C.T, p.R)
    return x, z


@dataclass
class TestSequence:
    """A synthetic sequence and the parameters that generated it.

    Attributes:
        x: Observations (N, Dx).
        z: True states (N, 2).
        params: Generating parameters (R: noise scales; tau: ones, unused).
    """

    x: np.ndarray
    z: np.ndarray
    params: Params


# (N, Dx, P0, Qs, R) for each test set.
_SETTINGS = {
    1: (100, 1, 1e-10, 1e-3, 1e-2),
    2: (100, 1, 1e-3, 1e-3, 1e-4),
    3: (100, 1, 1e-10, 1e-2, 0.3),
    4: (50, 4, 1e-10, 1e-4, 0.1),
    5: (100, 1, 1e-10, 1e-2, 0.1),
}

TEST_SETS = {
    1: "Laplace noise, replaced by much louder Gaussian noise for 30 <= n <= 50",
    2: "Laplace noise with random sensor failures (about 20% of samples set to ~0)",
    3: "Univariate data with Laplace noise",
    4: "Multivariate (4-D) data with Laplace noise",
    5: "Cauchy noise",
}


def make_test_set(test_set: int, rng: np.random.Generator) -> TestSequence:
    """Generate one of the five synthetic test sequences.

    Each is a 2-D latent oscillator observed through a random C normalized to unit
    L1 norm (rows; columns for test set 4). The parameters are drawn at random on
    each call. Time n below counts from 1.

    1. Laplace noise (scale 0.01), except for 30 <= n <= 50, where the data is
       pure Gaussian noise of variance 1.5. N = 100, Dx = 1.
    2. Laplace noise (scale 1e-4); each sample after the first has a 20% chance of a
       sensor failure that replaces it by (almost) zero. N = 100, Dx = 1.
    3. Laplace noise (scale 0.3). N = 100, Dx = 1.
    4. Laplace noise (scale 0.1), multivariate. N = 50, Dx = 4.
    5. Cauchy noise (scale 0.1). N = 100, Dx = 1.

    Args:
        test_set: 1 to 5.
        rng: Random generator.

    Returns:
        The sequence and its generating parameters.
    """
    if test_set not in _SETTINGS:
        raise ValueError(f"test_set must be 1-5, not {test_set}")
    N, dim_x, p0, qs, r = _SETTINGS[test_set]
    dim_z = 2

    Q = wishart(df=dim_z, scale=qs * np.eye(dim_z) / dim_z).rvs(random_state=rng)
    phase = 2 * np.pi * rng.random()
    m0 = np.array([np.cos(phase), np.sin(phase)])
    match test_set:
        case 1:
            A = 0.999 * phasor(2 * np.pi * 4.5 / N)
        case 2:
            A = phasor(2 * np.pi * 3 / N)
        case 3:
            A = 0.99 * phasor(2 * np.pi * (6 * rng.random() + 2) / N)
        case 4:
            A = phasor(2 * np.pi * (3 * rng.random() + 1) / N)
        case _:
            A = np.exp(-2 / N) * phasor(2 * np.pi * (6 * rng.random() + 3) / N)
            m0 = 1.5 * m0

    C = rng.standard_normal((dim_x, dim_z))
    C /= np.abs(C).sum(0 if test_set == 4 else 1, keepdims=True)

    p = Params(
        A=A,
        Q=Q,
        C=C,
        R=np.full(dim_x, r),
        m0=m0,
        P0=p0 * np.eye(dim_z),
        tau=np.ones(dim_z),
    )
    x, z = sample(N, p, rng)
    n = np.arange(1, N + 1)
    if test_set == 1:
        burst = (n >= 30) & (n <= 50)
        x[burst] = np.sqrt(1.5) * rng.standard_normal((burst.sum(), dim_x))
    elif test_set == 2:
        failed = (n > 1) & (rng.random(N) < 0.2)
        x[failed] = 1e-3 * rng.standard_normal((failed.sum(), dim_x))
    elif test_set == 5:
        x = z @ C.T + r * rng.standard_cauchy((N, dim_x))
    return TestSequence(x=x, z=z, params=p)
