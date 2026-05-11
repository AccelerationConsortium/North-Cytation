"""
Visualize the 4D synthetic landscape via 2D slices.

For each output (turbidity, ratio), shows a 4x4 grid:
  rows = which axes are held fixed
  cols = different fixed values for the held axes

Run:
  python -m recommenders.synthetic_nd_visualize --d 4
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recommenders.synthetic_nd import build_landscape


OUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs",
                       "synth_nd_diag")


def slice_2d(landscape, fn_name, fixed_axes, fixed_vals, n=80):
    """Return a 2D field by varying the 2 axes NOT in fixed_axes,
    holding the others at fixed_vals. fixed_axes/fixed_vals are
    parallel lists of length d-2.
    """
    d = landscape.d
    free = [i for i in range(d) if i not in fixed_axes]
    g = np.linspace(0, 1, n)
    A, B = np.meshgrid(g, g, indexing="ij")
    pts = np.zeros((n * n, d))
    pts[:, free[0]] = A.ravel()
    pts[:, free[1]] = B.ravel()
    for ax, v in zip(fixed_axes, fixed_vals):
        pts[:, ax] = v
    fn = getattr(landscape, fn_name)
    return fn(pts).reshape(n, n)


def render_4d(landscape, out_dir):
    """For d=4, plot 2x2 grid of slices for each output:
       rows = (x3,x4) fixed at (0.2,0.2), (0.5,0.5), (0.8,0.8), (1.0,1.0)
       cols = (x1,x2) fixed similarly
       Actually simpler: 4 panels showing the (x1,x2) plane at 4 (x3,x4) values.
    """
    d = landscape.d
    if d != 4:
        print(f"render_4d only supports d=4, got {d}")
        return
    fixed_pairs = [(0.1, 0.1), (0.5, 0.5), (0.7, 0.7), (0.9, 0.9)]
    for fn_name, vmin, vmax, label in [
            ("f1", 0.04, 1.00, "turbidity (f1)"),
            ("f2", 0.70, 0.90, "ratio (f2)")]:
        fig, axs = plt.subplots(1, len(fixed_pairs),
                                figsize=(4.0 * len(fixed_pairs), 4.0),
                                squeeze=False)
        for col, (v3, v4) in enumerate(fixed_pairs):
            F = slice_2d(landscape, fn_name, [2, 3], [v3, v4])
            ax = axs[0][col]
            im = ax.imshow(F.T, origin="lower", extent=[0, 1, 0, 1],
                           vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(f"x3={v3}, x4={v4}")
            ax.set_xlabel("x1"); ax.set_ylabel("x2")
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"4D synthetic: {label} (slice in x1-x2 plane)")
        fig.tight_layout()
        path = os.path.join(out_dir, f"4d_{fn_name}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"Wrote {path}")

    # also: marginal value distribution (uniform random sample)
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (200000, d))
    Y = landscape.evaluate(X)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    axs[0].hist(Y[:, 0], bins=80, color="tab:blue", alpha=0.8)
    axs[0].set_title(f"4D turbidity values (n=200k uniform), >0.5: {(Y[:,0]>0.5).mean()*100:.2f}%")
    axs[0].set_xlabel("turbidity")
    axs[1].hist(Y[:, 1], bins=80, color="tab:orange", alpha=0.8)
    axs[1].set_title(f"4D ratio values, >0.85: {(Y[:,1]>0.85).mean()*100:.2f}%")
    axs[1].set_xlabel("ratio")
    fig.tight_layout()
    path = os.path.join(out_dir, "4d_value_dist.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=4)
    args = p.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    landscape = build_landscape(args.d)
    if args.d == 4:
        render_4d(landscape, OUT_DIR)
    else:
        print(f"Only d=4 implemented in this script (got {args.d}).")


if __name__ == "__main__":
    main()
