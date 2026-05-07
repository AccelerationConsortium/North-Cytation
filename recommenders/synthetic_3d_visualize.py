"""
Visualize a proposed 3D synthetic ground truth before running any
recommender against it. No picks, no algorithms, no metrics.

Two outputs in [0,1]^3:
  f1 (turbidity-like): SHARP spike emerging from corner (0,0,0) and
      pointing toward the center / opposite corner. Plateaus at 0.04 far
      away, rises to 1.00 inside the spike. Sharp transition.
  f2 (ratio-like): SMOOTH transition across a rounded inner cube
      centered at (0.5, 0.5, 0.5). Plateaus at 0.7 outside, 0.9 inside.
      Wider transition.

Outputs:
  recommenders/test_outputs/synth3d_diag/<name>.png
  Per output: 3 mid-plane heatmaps (f value) + 3 mid-plane masks
  (top X% of |grad f|), plus a 3D scatter of boundary cells.

Run:
  python -m recommenders.synthetic_3d_visualize
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


N_GRID = 60                 # 60^3 = 216k cells
TOP_FRAC_F1 = 0.10          # turbidity boundary mask
TOP_FRAC_F2 = 0.30          # ratio boundary mask

# Turbidity spike (f1) parameters
# Cone is BROAD at the (1,1,1) corner (max of all three concentrations)
# and tapers to a point at distance SPIKE_LENGTH along the inward
# diagonal axis. Value is highest at the corner and decays along the
# axis toward the center.
F1_LOW, F1_HIGH = 0.04, 1.00
F1_TIP = 0.35               # value at the spike tip (inside, far end)
SPIKE_R0 = 0.35             # cone half-radius at the corner
SPIKE_CORNER = np.array([1.0, 1.0, 1.0])
SPIKE_AXIS = np.array([-1.0, -1.0, -1.0]) / np.sqrt(3.0)   # corner->center
SPIKE_LENGTH = 0.95         # cone extends this far from corner along the axis
EPS_SHARP = 0.03            # lateral sharpness

# Ratio half-cube wedge (f2) parameters
# High in the (0,0,0) octant (low concentration of all three components,
# OPPOSITE the turbidity peak). Three transition faces meet at
# WEDGE_CORNER and face the (0,0,0) corner. The opposite three faces
# (toward (1,1,1)) have no transition.
F2_LOW, F2_HIGH = 0.70, 0.90
WEDGE_CORNER = np.array([0.7, 0.7, 0.7])
EPS_SMOOTH = 0.05


OUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs",
                       "synth3d_diag")


# -----------------------------------------------------------------
# Field definitions
# -----------------------------------------------------------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def signed_dist_wedge(X):
    """Signed distance to the (0,0,0)-facing wedge with corner at
    WEDGE_CORNER. Negative inside (all xi <= corner_i), positive outside.
    The 3 active boundary faces are x1=0.5, x2=0.5, x3=0.5; the back
    faces (toward (1,1,1)) are absent.
    """
    d = X - WEDGE_CORNER                  # positive component => outside
    return np.max(d, axis=1)              # max-norm signed distance


def f1_turbidity(X):
    """Cone-shaped turbid region: broad at corner (1,1,1) (max of all
    three concentrations), tapering to a point at SPIKE_LENGTH along the
    inward diagonal. Value is HIGHEST at the corner (F1_HIGH) and decays
    linearly along the axis toward F1_TIP at the tip. Outside the cone
    the value is F1_LOW (baseline).
    """
    Xs = X - SPIKE_CORNER                                    # shift origin to (1,1,1)
    s = Xs @ SPIKE_AXIS                                      # arc length
    proj = np.outer(s, SPIKE_AXIS)
    perp = np.linalg.norm(Xs - proj, axis=1)
    # Cone half-radius: R0 at corner, taper linearly to 0 at SPIKE_LENGTH.
    r_at_s = SPIKE_R0 * np.clip(1.0 - s / SPIKE_LENGTH, 0.0, 1.0)
    # Inside-ness via smooth sigmoid on (r_at_s - perp) / eps.
    inside = _sigmoid((r_at_s - perp) / EPS_SHARP)
    # Behind the corner (s<0) suppress; past the tip (s>length) suppress.
    inside = np.where(s < -EPS_SHARP, 0.0, inside)
    inside = np.where(s > SPIKE_LENGTH + EPS_SHARP, 0.0, inside)
    # Axial value: F1_HIGH at corner, F1_TIP at tip and beyond.
    t = np.clip(s / SPIKE_LENGTH, 0.0, 1.0)
    v_inside = F1_HIGH + (F1_TIP - F1_HIGH) * t
    return F1_LOW + (v_inside - F1_LOW) * inside


def f2_ratio(X):
    h = signed_dist_wedge(X)
    # h<0 inside wedge => high ratio (0.9). sigmoid(-h/eps).
    return F2_LOW + (F2_HIGH - F2_LOW) * _sigmoid(-h / EPS_SMOOTH)


# -----------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------

def grid():
    g = np.linspace(0.0, 1.0, N_GRID)
    X1, X2, X3 = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1)
    return g, pts, (X1, X2, X3)


def grad_mag_3d(F):
    g1, g2, g3 = np.gradient(F)
    return np.sqrt(g1 ** 2 + g2 ** 2 + g3 ** 2)


def render_panel(name, F, top_frac, lo, hi, out_path):
    g, _, _ = grid()
    G = grad_mag_3d(F)
    thresh = np.percentile(G, 100 * (1 - top_frac))
    mask = G >= thresh

    fig = plt.figure(figsize=(16, 10))
    mid = N_GRID // 2

    # Row 1: mid-plane heatmaps of F
    slices_F = [
        (F[mid, :, :], "x1=0.5", ("x2", "x3")),
        (F[:, mid, :], "x2=0.5", ("x1", "x3")),
        (F[:, :, mid], "x3=0.5", ("x1", "x2")),
    ]
    for k, (S, slabel, axlabels) in enumerate(slices_F):
        ax = fig.add_subplot(2, 4, k + 1)
        im = ax.pcolormesh(g, g, S.T, cmap="viridis", shading="auto",
                           vmin=lo, vmax=hi)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{slabel}: {name} value\n[{lo:.2f}..{hi:.2f}]",
                     fontsize=9)
        ax.set_xlabel(axlabels[0]); ax.set_ylabel(axlabels[1])

    # Row 1, col 4: 3D scatter of boundary cells
    ax3d = fig.add_subplot(2, 4, 4, projection="3d")
    # subsample for plot speed
    bnd_idx = np.argwhere(mask)
    if len(bnd_idx) > 4000:
        sel = np.random.default_rng(0).choice(len(bnd_idx), 4000,
                                              replace=False)
        bnd_idx = bnd_idx[sel]
    ax3d.scatter(g[bnd_idx[:, 0]], g[bnd_idx[:, 1]], g[bnd_idx[:, 2]],
                 s=2, c="red", alpha=0.4)
    ax3d.set_title(f"{name} boundary cells\n(top {int(top_frac*100)}% |grad|, "
                   f"n={mask.sum()})", fontsize=9)
    ax3d.set_xlim(0, 1); ax3d.set_ylim(0, 1); ax3d.set_zlim(0, 1)
    ax3d.set_xlabel("x1"); ax3d.set_ylabel("x2"); ax3d.set_zlabel("x3")

    # Row 2: same mid-plane slices showing |grad F| with mask overlay
    slices_G = [
        (G[mid, :, :], mask[mid, :, :], "x1=0.5", ("x2", "x3")),
        (G[:, mid, :], mask[:, mid, :], "x2=0.5", ("x1", "x3")),
        (G[:, :, mid], mask[:, :, mid], "x3=0.5", ("x1", "x2")),
    ]
    for k, (Gs, Ms, slabel, axlabels) in enumerate(slices_G):
        ax = fig.add_subplot(2, 4, 5 + k)
        im = ax.pcolormesh(g, g, Gs.T, cmap="inferno", shading="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ys, xs = np.where(Ms)   # row=axis1, col=axis2
        ax.scatter(g[xs], g[ys], s=4, c="cyan", alpha=0.6,
                   edgecolors="none")
        ax.set_title(f"{slabel}: |grad {name}| + mask", fontsize=9)
        ax.set_xlabel(axlabels[0]); ax.set_ylabel(axlabels[1])

    # Row 2, col 4: histogram of F values + chosen plateau bands
    ax = fig.add_subplot(2, 4, 8)
    ax.hist(F.ravel(), bins=50, color="steelblue")
    ax.axvline(lo, ls="--", c="k", label=f"low={lo}")
    ax.axvline(hi, ls="--", c="k", label=f"high={hi}")
    ax.set_title(f"{name} value distribution", fontsize=9)
    ax.set_xlabel("value"); ax.set_ylabel("count"); ax.legend(fontsize=7)

    fig.suptitle(f"3D synthetic diagnostic: {name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    g, pts, _ = grid()
    print(f"Evaluating {len(pts)} grid points...")
    F1 = f1_turbidity(pts).reshape(N_GRID, N_GRID, N_GRID)
    F2 = f2_ratio(pts).reshape(N_GRID, N_GRID, N_GRID)
    print(f"  f1 (turbidity) range: {F1.min():.3f} .. {F1.max():.3f}")
    print(f"  f2 (ratio)     range: {F2.min():.3f} .. {F2.max():.3f}")
    render_panel("turbidity (f1)", F1, TOP_FRAC_F1, F1_LOW, F1_HIGH,
                 os.path.join(OUT_DIR, "turbidity.png"))
    render_panel("ratio (f2)", F2, TOP_FRAC_F2, F2_LOW, F2_HIGH,
                 os.path.join(OUT_DIR, "ratio.png"))


if __name__ == "__main__":
    main()
