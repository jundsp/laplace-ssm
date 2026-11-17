"""EM demo: learn a Laplace (or Gaussian) state space model on a synthetic test set.

    uv run demo.py --test-set 1 --model lssm --epochs 200

Prints log p(X) (and, for the LSSM, the expected log joint) at each EM iteration,
saves a figure to results/ every 20 iterations and shows the last one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lssm.data import TEST_SETS, make_test_set
from lssm.gaussian import gaussian_m_step, kalman_filter
from lssm.inference import filter, smoother
from lssm.learning import log_joint, m_step
from lssm.model import init_params
from lssm.plotting import param_clims, plot_results


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--test-set",
        type=int,
        default=1,
        choices=sorted(TEST_SETS),
        help="\n".join(f"{k}: {v}" for k, v in TEST_SETS.items()),
    )
    ap.add_argument(
        "--dim-z", type=int, default=10, help="state dimension (even); extras are pruned"
    )
    ap.add_argument("--epochs", type=int, default=200, help="number of EM iterations")
    ap.add_argument(
        "--model",
        choices=["lssm", "gssm"],
        default="lssm",
        help="Laplace (lssm) or Gaussian (gssm) observation noise",
    )
    ap.add_argument(
        "--method",
        choices=["ep", "vb"],
        default="ep",
        help="LSSM filter update: expectation propagation or variational Bayes",
    )
    ap.add_argument("--seed", type=int, default=0, help="seed for the data and the initialization")
    ap.add_argument("--plot-every", type=int, default=20, help="save a figure every k iterations")
    ap.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    ap.add_argument("--no-plots", action="store_true", help="do not save or show figures")
    return ap.parse_args()


def main() -> None:
    """Run the demo."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    x = make_test_set(args.test_set, rng).x
    p = init_params(args.dim_z, x.shape[1], rng)
    lssm = args.model == "lssm"

    plot_epochs = set(range(0, args.epochs + 1, args.plot_every)) | {args.epochs}
    snaps = []
    # Iteration e uses the parameters after e M-steps (0: the initialization).
    for epoch in range(args.epochs + 1):
        mu, V, log_px = filter(x, p, args.method) if lssm else kalman_filter(x, p)
        mu, V, V12 = smoother(mu, V, p.A, p.Q)
        line = f"Epoch {epoch:3d} / {args.epochs}   ln p(X) = {log_px:8.2f}"
        if epoch in plot_epochs:
            snaps.append((epoch, p.copy(), mu, V))
        if epoch < args.epochs:
            if lssm:
                p = m_step(x, mu, V, V12, p)
                line += f"   E[ln p(X,Z)] = {log_joint(x, mu, V, V12, p):8.2f}"
            else:
                p = gaussian_m_step(x, mu, V, V12, p)
        print(line)

    norms = np.abs(p.C).sum(0)
    active = np.flatnonzero(norms > 1e-3 * norms.max()) + 1
    print(
        f"Active state dimensions (columns of C kept by ARD): {active.tolist()}, "
        f"{len(active)} of {args.dim_z}"
    )

    if args.no_plots:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    clims = param_clims([s[1] for s in snaps])
    for epoch, q, mu, V in snaps:
        noise_var = 2 * q.R**2 if lssm else q.R
        fig = plot_results(
            x, mu, V, q, noise_var, clims=clims, title=f"{args.model.upper()}, EM iteration {epoch}"
        )
        fig.savefig(args.out / f"{args.model}_epoch_{epoch}.png", dpi=120)
        if epoch < args.epochs:
            plt.close(fig)
    print(f"Figures saved to {args.out}/")
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
