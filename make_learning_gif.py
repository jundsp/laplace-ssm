"""Animation of EM learning of a Gaussian and a Laplace state space model.

    uv run make_learning_gif.py

Both models start from the same random 10-D initialization (init_params) on
test set 1 and differ only in the observation noise. Each frame shows the data, the
smoothed estimate with ±1 std and a forecast, the column norms of C (ARD), and the
eigenvalues of A with the true ones. Writes docs/figures/learning.gif.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from PIL import Image  # noqa: E402

from lssm.data import make_test_set  # noqa: E402
from lssm.gaussian import gaussian_m_step, kalman_filter  # noqa: E402
from lssm.inference import filter, forecast, smoother  # noqa: E402
from lssm.learning import m_step  # noqa: E402
from lssm.model import init_params  # noqa: E402
from lssm.params import Params  # noqa: E402
from lssm.plotting import data_ylim  # noqa: E402

COLORS = {"GSSM": (0.90, 0.25, 0.10), "LSSM": (0.0, 0.35, 0.95)}


def draw_model(
    ax: list[Axes],
    name: str,
    x: np.ndarray,
    mu: np.ndarray,
    V: np.ndarray,
    p: Params,
    log_px: float,
    steps: int,
    A_true: np.ndarray,
) -> None:
    """Draw one model's column: data and forecast, ARD column norms, eigenvalues of A."""
    color = COLORS[name]
    N = x.shape[0]
    noise_var = 2 * p.R[0] ** 2 if name == "LSSM" else p.R[0]
    mu_f, V_f = forecast(mu, V, p.A, p.Q, steps)
    c = p.C[0]
    mean = mu_f @ c
    std = np.sqrt(np.einsum("j,njk,k->n", c, V_f, c) + noise_var)
    t = np.arange(1, N + steps + 1)

    a = ax[0]
    a.cla()
    a.fill_between(t, mean - std, mean + std, color=color, alpha=0.15, lw=0, label="±1 std")
    a.plot(t[:N], x[:, 0], "k.", ms=5, label="data")
    a.plot(t, mean, "-", color=color, lw=1.5, label=name)
    a.axvline(N, ls="--", color=(0.4, 0.4, 0.4), lw=1)
    a.text(N + 1, 0.97, "forecast", transform=a.get_xaxis_transform(), va="top", color="0.3")
    a.set(xlim=(1, N + steps), ylim=data_ylim(x))
    a.set_title(f"{name}     ln p(X) = {log_px:.1f}", fontsize=12)
    a.legend(loc="lower right", ncols=3, frameon=False, fontsize=9)

    dim_z = p.dim_z
    a = ax[1]
    a.cla()
    a.bar(np.arange(1, dim_z + 1), np.linalg.norm(p.C, axis=0), 0.6, color=color)
    a.set(xlim=(0.4, dim_z + 0.6), ylim=(0, 2.6), xticks=np.arange(1, dim_z + 1))
    a.set_xlabel("state dimension j", fontsize=9)
    a.set_title("column norms of C (ARD)", fontsize=10)
    a.tick_params(labelsize=9)

    a = ax[2]
    a.cla()
    w = np.linspace(0, 2 * np.pi, 200)
    a.plot(np.cos(w), np.sin(w), "-", color=(0.7, 0.7, 0.7), lw=1)
    ev_true = np.linalg.eigvals(A_true)
    a.plot(ev_true.real, ev_true.imag, "ko", mfc="none", ms=7)
    ev = np.linalg.eigvals(p.A)
    a.plot(ev.real, ev.imag, "x", color=color, ms=7, mew=1.5)
    a.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), xticks=[], yticks=[], aspect="equal")
    a.set_title("eigenvalues of A", fontsize=10)


def main() -> None:
    """Run EM for both models and write the GIF."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-set", type=int, default=1)
    ap.add_argument("--dim-z", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/learning.gif"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    seq = make_test_set(args.test_set, rng)
    x = seq.x
    init = init_params(args.dim_z, x.shape[1], rng)
    params = {"GSSM": init.copy(), "LSSM": init.copy()}
    frames = sorted(
        set(range(31))
        | set(range(32, 51, 2))
        | set(range(55, 101, 5))
        | set(range(110, 201, 10))
        | set(range(220, args.epochs + 1, 20))
        | {args.epochs}
    )
    frames = [f for f in frames if f <= args.epochs]

    fig = plt.figure(figsize=(9.6, 6.0), dpi=100)
    axes = {}
    for k, name in enumerate(params):
        x0 = 0.06 + k * 0.49
        axes[name] = [
            fig.add_axes((x0, 0.45, 0.42, 0.43)),
            fig.add_axes((x0, 0.08, 0.24, 0.25)),
            fig.add_axes((x0 + 0.27, 0.06, 0.15, 0.29)),
        ]
    heading = fig.text(0.5, 0.955, "", ha="center", fontsize=14)

    images = []
    # Iteration e uses the parameters after e M-steps (0: the initialization).
    for epoch in range(args.epochs + 1):
        for name, p in params.items():
            mu, V, log_px = filter(x, p) if name == "LSSM" else kalman_filter(x, p)
            mu, V, V12 = smoother(mu, V, p.A, p.Q)
            if epoch in frames:
                draw_model(axes[name], name, x, mu, V, p, log_px, 50, seq.params.A)
            if epoch < args.epochs:
                step = m_step if name == "LSSM" else gaussian_m_step
                params[name] = step(x, mu, V, V12, p)
        if epoch in frames:
            heading.set_text(f"EM iteration {epoch}")
            fig.canvas.draw()
            rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            images.append(rgb.quantize(64, dither=Image.Dither.NONE))
            print(f"frame {len(images)} / {len(frames)}: EM iteration {epoch}")

    durations = [1000] + [120] * (len(images) - 2) + [3000]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(args.out, save_all=True, append_images=images[1:], duration=durations, loop=0)
    print(f"Wrote {args.out} ({len(images)} frames)")


if __name__ == "__main__":
    main()
