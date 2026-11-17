"""Checks against reference outputs of the original implementation of the paper.

tests/fixtures/reference.mat holds, for each of the five test sets, the data, a random
4-D initialization and the outputs of each step (EP and VB filters, smoother, M-step,
expected log joint, Kalman filter and Gaussian M-step) and of 5 EM iterations of each
model. The Python code gets the same inputs and must agree to a tolerance.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from lssm.gaussian import gaussian_m_step, kalman_filter
from lssm.inference import filter, smoother
from lssm.learning import log_joint, m_step
from lssm.params import Params

REF = loadmat(Path(__file__).parent / "fixtures" / "reference.mat", simplify_cells=True)


def _vec(a):
    return np.atleast_1d(np.asarray(a, dtype=float)).ravel()


def _mat(a, d):
    return np.asarray(a, dtype=float).reshape(d, -1)


def _params(s, dim_x, dim_z):
    """Params from a stored struct: R is a diagonal matrix there."""
    return Params(
        A=_mat(s["A"], dim_z),
        Q=_mat(s["Q"], dim_z),
        C=_mat(s["C"], dim_x),
        R=np.diag(_mat(s["R"], dim_x)).copy(),
        m0=_vec(s["m0"]),
        P0=_mat(s["P0"], dim_z),
        tau=_vec(s["tau"]),
    )


def _seq(a, dim_z):
    """(Dz, N) means to (N, Dz)."""
    return np.asarray(a, dtype=float).reshape(dim_z, -1).T


def _covs(a, dim_z):
    """(Dz, Dz, N) covariances to (N, Dz, Dz)."""
    return np.moveaxis(np.asarray(a, dtype=float).reshape(dim_z, dim_z, -1), -1, 0)


def _case(ts):
    s = REF[f"ts{ts}"]
    x = np.atleast_2d(np.asarray(s["x"], dtype=float))
    dim_x = x.shape[0]
    x = x.T
    dim_z = np.asarray(s["A"]).shape[0]
    return s, x, _params(s, dim_x, dim_z), dim_x, dim_z


def assert_params_close(p, q, rtol, atol):
    for k in ("A", "Q", "C", "R", "m0", "P0", "tau"):
        np.testing.assert_allclose(getattr(p, k), getattr(q, k), rtol=rtol, atol=atol, err_msg=k)


@pytest.mark.parametrize("ts", [1, 2, 3, 4, 5])
def test_filters_and_smoother(ts):
    s, x, p, _, dim_z = _case(ts)
    for method in ("ep", "vb"):
        mu, V, log_px = filter(x, p, method)
        np.testing.assert_allclose(mu, _seq(s[f"{method}_mu"], dim_z), rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(V, _covs(s[f"{method}_V"], dim_z), rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(log_px, s[f"{method}_pX"], rtol=1e-8)

    mu, V, V12 = smoother(*filter(x, p, "ep")[:2], p.A, p.Q)
    np.testing.assert_allclose(mu, _seq(s["sm_mu"], dim_z), rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(V, _covs(s["sm_V"], dim_z), rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(V12, _covs(s["sm_V12"], dim_z), rtol=1e-7, atol=1e-10)

    mu, V, log_px = kalman_filter(x, p)
    np.testing.assert_allclose(mu, _seq(s["kf_mu"], dim_z), rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(V, _covs(s["kf_V"], dim_z), rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(log_px, s["kf_pX"], rtol=1e-8)


@pytest.mark.parametrize("ts", [1, 2, 3, 4, 5])
def test_m_steps_and_log_joint(ts):
    s, x, p, dim_x, dim_z = _case(ts)
    mu, V, V12 = smoother(_seq(s["ep_mu"], dim_z), _covs(s["ep_V"], dim_z), p.A, p.Q)
    q = m_step(x, mu, V, V12, p)
    assert_params_close(q, _params(s["ms"], dim_x, dim_z), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(log_joint(x, mu, V, V12, q), s["lj"], rtol=1e-7)

    mu, V, V12 = smoother(_seq(s["kf_mu"], dim_z), _covs(s["kf_V"], dim_z), p.A, p.Q)
    q = gaussian_m_step(x, mu, V, V12, p)
    assert_params_close(q, _params(s["gms"], dim_x, dim_z), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("ts", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("model", ["lssm", "gssm"])
def test_em_iterations(ts, model):
    s, x, p, dim_x, dim_z = _case(ts)
    history = []
    for _ in range(5):
        if model == "lssm":
            mu, V, log_px = filter(x, p, "ep")
        else:
            mu, V, log_px = kalman_filter(x, p)
        history.append(log_px)
        mu, V, V12 = smoother(mu, V, p.A, p.Q)
        p = (m_step if model == "lssm" else gaussian_m_step)(x, mu, V, V12, p)
    ref = s[f"em_{model}"]
    np.testing.assert_allclose(history, _vec(ref["pX"]), rtol=1e-5)
    assert_params_close(p, _params(ref, dim_x, dim_z), rtol=1e-4, atol=1e-7)
