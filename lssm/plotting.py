"""Figures: data with the smoothed estimate and forecast, and the learned parameters."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from lssm.inference import forecast
from lssm.params import Params

# Parameters shown on a log10 scale of their magnitude: Q shrinks by orders of
# magnitude as the dynamics are learned.
LOG_PARAMS = ("Q", "R", "P0")
PARAMS = ("A", "C", "Q", "R", "P0", "m0")

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_white_red", [(0.13, 0.40, 0.67), (1.0, 1.0, 1.0), (0.70, 0.09, 0.17)]
)


def data_ylim(x: np.ndarray) -> tuple[float, float]:
    """Y limits that depend only on the data: its range padded by 25% on each side."""
    lo, hi = float(x.min()), float(x.max())
    pad = 0.25 * max(hi - lo, np.finfo(float).eps)
    return lo - pad, hi + pad


def _as_matrix(p: Params, name: str) -> np.ndarray:
    M = getattr(p, name)
    if name == "R":
        return np.diag(M)
    return M[:, None] if M.ndim == 1 else M


def param_clims(snapshots: list[Params]) -> dict[str, tuple[float, float]]:
    """Shared color limits over EM snapshots, so their figures use the same colors.

    Args:
        snapshots: Parameters at several EM iterations.

    Returns:
        For each parameter in PARAMS, (lo, hi): hi is the largest magnitude; for
        LOG_PARAMS, lo is the smallest diagonal magnitude (0 for the others).
    """
    clims = {}
    for name in PARAMS:
        mats = [_as_matrix(p, name) for p in snapshots]
        hi = max(float(np.abs(M).max()) for M in mats)
        lo = min(float(np.abs(np.diag(M)).min()) for M in mats) if name in LOG_PARAMS else 0.0
        clims[name] = (lo, hi)
    return clims


def _show_matrix(
    ax: Axes, title: str, M: np.ndarray, clim: tuple[float, float] | None, log: bool
) -> None:
    fig = ax.get_figure()
    if log:
        lo, hi = clim if clim else (float(np.abs(np.diag(M)).min()), float(np.abs(M).max()))
        lo, hi = np.log10(max(lo, 1e-300)), np.log10(max(hi, 1e-300))
        if not hi > lo:
            lo = hi - 1
        with np.errstate(divide="ignore"):
            img = np.maximum(np.log10(np.abs(M)), lo)
        im = ax.imshow(img, cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax)
        e = np.arange(np.ceil(lo), np.floor(hi) + 1)
        if len(e) >= 2:
            # Powers of ten, at most about four of them.
            e = e[::-1][:: max(1, int(np.ceil(len(e) / 4)))][::-1]
            cb.set_ticks(e, labels=[f"$10^{{{int(k)}}}$" for k in e])
        else:
            decades = np.arange(np.floor(lo), np.ceil(hi) + 1)
            v = (np.array([1, 2, 5])[:, None] * 10.0**decades).ravel()
            v = np.sort(v[(np.log10(v) >= lo) & (np.log10(v) <= hi)])
            cb.set_ticks(np.log10(v), labels=[f"{t:g}" for t in v])
    else:
        L = clim[1] if clim else float(np.abs(M).max())
        L = L if L > 0 else 1.0
        im = ax.imshow(M, cmap=DIVERGING, vmin=-L, vmax=L, interpolation="nearest")
        fig.colorbar(im, ax=ax)
    ax.set_title(title)
    if M.shape[0] != M.shape[1]:
        ax.set_aspect("auto")
    rows, cols = M.shape
    ax.set_xticks(*_ticks(cols))
    ax.set_yticks(*_ticks(rows))


def _ticks(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Tick positions and 1-based labels, every other one for more than six."""
    step = 2 if n > 6 else 1
    return np.arange(n)[::step], np.arange(1, n + 1)[::step]


def plot_results(
    x: np.ndarray,
    mu: np.ndarray,
    V: np.ndarray,
    p: Params,
    noise_var: np.ndarray,
    steps: int = 50,
    clims: dict[str, tuple[float, float]] | None = None,
    title: str | None = None,
) -> Figure:
    """Data, smoothed estimate with ±1 std, a forecast, and the parameters.

    Args:
        x: Observations (N, Dx).
        mu: Smoothed means (N, Dz).
        V: Smoothed covariances (N, Dz, Dz).
        p: Parameters.
        noise_var: Observation noise variances (Dx,): 2 R^2 for the Laplace model,
            R for the Gaussian model.
        steps: Forecast length.
        clims: Shared color limits from param_clims.
        title: Figure title.

    Returns:
        The figure.
    """
    N, dim_x = x.shape
    mu_f, V_f = forecast(mu, V, p.A, p.Q, steps)
    mean = mu_f @ p.C.T
    std = np.sqrt(np.einsum("ij,njk,ik->ni", p.C, V_f, p.C) + noise_var)
    t = np.arange(1, N + steps + 1)

    fig = plt.figure(figsize=(6.8, 8.0), layout="constrained")
    gs = fig.add_gridspec(5, 2)
    ax = fig.add_subplot(gs[:2, :])
    for i in range(dim_x):
        ax.fill_between(
            t,
            mean[:, i] - std[:, i],
            mean[:, i] + std[:, i],
            color="r",
            alpha=0.2,
            lw=0,
            label=r"$\mu_x \pm \sigma_x$" if i == 0 else None,
        )
        ax.plot(t[:N], x[:, i], "k.-", ms=4, lw=0.8, label="Data x" if i == 0 else None)
        ax.plot(t, mean[:, i], "b-", lw=1, label=r"$\mu_x$" if i == 0 else None)
    ax.axvline(N, color="k", lw=0.8)
    ax.text(N + 1, 0.97, "forecast", transform=ax.get_xaxis_transform(), va="top")
    ax.set(xlabel="Time", xlim=(1, N + steps), ylim=data_ylim(x))
    ax.legend(loc="lower right")
    if title:
        ax.set_title(title)

    clims = clims or {}
    labels = {"A": "A", "C": "C", "Q": "|Q|", "R": "|R|", "P0": "$|P_0|$", "m0": "$m_0$"}
    for k, name in enumerate(PARAMS):
        a = fig.add_subplot(gs[2 + k // 2, k % 2])
        _show_matrix(a, labels[name], _as_matrix(p, name), clims.get(name), name in LOG_PARAMS)
    return fig
