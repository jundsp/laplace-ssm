import numpy as np
import pytest

from lssm.data import make_test_set, random_block_diagonal, rotrnd, sample
from lssm.params import Params


@pytest.mark.parametrize("ts", [1, 2, 3, 4, 5])
def test_make_test_set_shapes(ts):
    seq = make_test_set(ts, np.random.default_rng(ts))
    N, dx = (50, 4) if ts == 4 else (100, 1)
    assert seq.x.shape == (N, dx)
    assert seq.z.shape == (N, 2)
    assert np.all(np.isfinite(seq.x))


def test_rotrnd_is_orthogonal():
    A = rotrnd(7, np.random.default_rng(0))
    np.testing.assert_allclose(A @ A.T, np.eye(7), atol=1e-12)


def test_random_block_diagonal():
    A = random_block_diagonal(6, np.random.default_rng(0))
    mask = np.kron(np.eye(3), np.ones((2, 2)))
    assert np.all(A[mask == 0] == 0)
    assert np.all(np.abs(np.linalg.eigvals(A)) <= 1 + 1e-12)
    with pytest.raises(ValueError):
        random_block_diagonal(5, np.random.default_rng(0))


def test_sample_noise_variance():
    """With a static state, the observation noise has variance 2 R^2."""
    p = Params(
        A=np.eye(2),
        Q=1e-12 * np.eye(2),
        C=np.array([[1.0, 0.0]]),
        R=np.array([0.5]),
        m0=np.array([1.0, 0.0]),
        P0=1e-12 * np.eye(2),
        tau=np.ones(2),
    )
    x, _ = sample(100_000, p, np.random.default_rng(0))
    np.testing.assert_allclose(x.mean(), 1.0, atol=0.01)
    np.testing.assert_allclose(x.var(), 2 * 0.5**2, rtol=0.02)
