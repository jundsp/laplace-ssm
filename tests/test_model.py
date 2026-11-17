import numpy as np
import pytest

from lssm.data import make_test_set
from lssm.model import GaussianSSM, LaplaceSSM


@pytest.mark.parametrize("ts", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("kind", ["ep", "vb", "gaussian"])
def test_short_em_is_finite(ts, kind):
    x = make_test_set(ts, np.random.default_rng(ts)).x
    if kind == "gaussian":
        model = GaussianSSM(dim_z=4, seed=ts)
    else:
        model = LaplaceSSM(dim_z=4, method=kind, seed=ts)
    model.fit(x, epochs=5)
    p = model.params
    assert np.all(np.isfinite(model.log_likelihood))
    for a in (p.A, p.Q, p.C, p.R, p.tau):
        assert np.all(np.isfinite(a))
    if kind != "gaussian":
        assert np.isfinite(model.log_joint(x))


def test_fit_learns_oscillator():
    """On noise-burst data, EM keeps few state dimensions and the fit improves."""
    x = make_test_set(1, np.random.default_rng(0)).x
    model = LaplaceSSM(dim_z=6, seed=0).fit(x, epochs=40)
    assert model.log_likelihood[-1] > model.log_likelihood[0]
    assert 1 <= len(model.active_dims()) <= 4
    mean, std = model.forecast(x[:, 0], steps=10)
    assert mean.shape == (110, 1) and std.shape == (110, 1)
