"""
Phase 4 test harness for the gradient-based transition recommender.

Runs head-to-head comparisons between:
  - BayesianTransitionRecommender (existing local-contrast acquisition)
  - GradientTransitionRecommender (new per-dim gradient UCB)

on three 2D problems (in mandatory order):
  step2d       - smooth line boundary x[0] = 0.5
  circle2d     - smooth circular boundary radius 0.3 around (0.5, 0.5)
  surfactant2d - real simulate_surfactant_measurements() (copied inline to
                 avoid pulling in workflow/hardware imports)

Also provides a `--unit` mode that runs Phase-3 sanity checks on
_grad_mu and _grad_var (1D GP fit to y = sin(x)).

Outputs go to recommenders/test_outputs/<testname>/:
  all_data_<recname>.csv
  metrics_<recname>.csv
  2d_exploration_<recname>.png       (true boundary contour + picks colored by iter)
  gradient_map_<recname>.png         (||grad mu|| heatmap; gradient recommender only)
  comparison_hd.png                  (boundary-hit fraction vs iteration, both)

Run:
  python -m recommenders.test_gradient_transition_recommender --test step2d
  python -m recommenders.test_gradient_transition_recommender --test circle2d
  python -m recommenders.test_gradient_transition_recommender --test surfactant2d
  python -m recommenders.test_gradient_transition_recommender --test all
  python -m recommenders.test_gradient_transition_recommender --unit
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # never try to open a display from a test run
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from botorch.utils.sampling import draw_sobol_samples

from recommenders.bayesian_transition_recommender import BayesianTransitionRecommender
from recommenders.gradient_transition_recommender import GradientTransitionRecommender


OUT_ROOT = os.path.join(os.path.dirname(__file__), "test_outputs")

Q_BATCH = 8
N_ITERATIONS = 21       # 21 * 8 = 168 picks; +25 init = 193 total ~= 192
N_INIT_GRID = 5         # 5x5 = 25 initial points (replaces n=10 Sobol)
SEED = 0


# =================================================================
# Synthetic 2D problems (operate in normalized [0, 1]^2 space)
# =================================================================

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def step2d_outputs(x):
    """Smooth line boundary: h(x) = x[0] - 0.5."""
    h = x[0] - 0.5
    p = _sigmoid(h / 0.05)
    # Two outputs that both transition across the line (for 2-output infra).
    return float(p), float(0.2 + 0.6 * p)


def step2d_boundary(x):
    return float(x[0] - 0.5)


def circle2d_outputs(x):
    """Smooth circle boundary radius 0.3 around (0.5, 0.5)."""
    r = float(np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))
    h = r - 0.3
    p = _sigmoid(h / 0.05)
    return float(p), float(0.2 + 0.6 * (1.0 - p))


def circle2d_boundary(x):
    r = float(np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))
    return float(r - 0.3)


# -----------------------------------------------------------------
# Surfactant 2D simulator (copied verbatim from
# workflows/surfactant_grid_adaptive_concentrations.py to keep this
# test file free of robot/workflow imports). Inputs are mM
# concentrations; outputs are 'ratio' and 'turbidity_600'.
# -----------------------------------------------------------------

MIN_CONC = 0.01
MAX_CONC = 25.0


def simulate_surfactant_measurements(surf_a_conc, surf_b_conc, add_noise=True):
    log_a = np.log10(surf_a_conc)
    log_b = np.log10(surf_b_conc)
    log_min = np.log10(MIN_CONC)
    log_max_estimate = np.log10(25.0)
    log_range = log_max_estimate - log_min

    norm_a = (log_a - log_min) / log_range
    norm_b = (log_b - log_min) / log_range
    norm_a = max(0, min(1, norm_a))
    norm_b = max(0, min(1, norm_b))

    edge_a_effect = np.tanh(8.0 * (norm_a - 0.7))
    edge_b_effect = np.tanh(8.0 * (norm_b - 0.8))
    edge_combined = np.maximum(edge_a_effect, edge_b_effect)
    center_distance = np.sqrt((norm_a - 0.5) ** 2 + (norm_b - 0.5) ** 2)
    middle_extension = 0.4 * np.exp(-2.0 * center_distance)
    ratio_factor = 0.9 - 0.4 * (edge_combined + middle_extension)

    ratio_baseline = 0.70
    ratio_elevated = 0.85
    simulated_ratio = ratio_baseline + (ratio_elevated - ratio_baseline) * ratio_factor

    direction_a, direction_b = -0.5, -0.5
    relative_a = norm_a - 1.0
    relative_b = norm_b - 1.0
    projection_length = (relative_a * direction_a + relative_b * direction_b) / 0.707
    perp_distance = abs(relative_a * direction_b - relative_b * direction_a) / 0.707

    band_width = 0.15
    band_length = 0.7
    in_band = (perp_distance < band_width) and (0 < projection_length < band_length)
    if in_band:
        length_factor = 1.0 - (projection_length / band_length)
        width_factor = 1.0 - (perp_distance / band_width)
        turb_factor = 0.8 * length_factor * width_factor
    else:
        turb_factor = 0.05
    turb_factor = max(0, min(1, turb_factor))

    turbidity_baseline = 0.04
    turbidity_elevated = 3.0
    simulated_turbidity = turbidity_baseline + (turbidity_elevated - turbidity_baseline) * turb_factor

    if add_noise:
        rng = np.random.default_rng()
        simulated_ratio *= 1.0 + rng.normal(0, 0.015)
        simulated_turbidity *= 1.0 + rng.normal(0, 0.015 * 1.5)

    simulated_ratio = max(0.70, min(0.95, simulated_ratio))
    simulated_turbidity = max(0.02, min(1.0, simulated_turbidity))
    return float(simulated_ratio), float(simulated_turbidity)


def surfactant2d_outputs(x_mM):
    """x_mM is in original (mM) space, NOT normalized."""
    return simulate_surfactant_measurements(float(x_mM[0]), float(x_mM[1]),
                                            add_noise=False)


def surfactant2d_boundary(x_norm01):
    """Signed proxy for 'how far from a transition' in [0,1]^2 normalized
    log-space. Negative = inside the high-ratio-edge region (norm_a>0.7 OR
    norm_b>0.8); positive = outside. The h=0 contour is the L-shaped union
    of the two ratio transition edges.
    """
    a, b = float(x_norm01[0]), float(x_norm01[1])
    # Distance to the union of the two half-planes {a>=0.7} U {b>=0.8}.
    # A point is "inside" if a>=0.7 OR b>=0.8.
    inside = (a >= 0.7) or (b >= 0.8)
    if inside:
        return float(-min(a - 0.7, b - 0.8))
    return float(min(0.7 - a, 0.8 - b))


def _surfactant_turbidity_boundary(x_norm01):
    """Signed distance to the edge of the turbidity band, in [0,1]^2 normalized
    log-space. The turbidity simulator emits a high-turbidity band of half-width
    0.15 along the diagonal from corner (1,1) toward center, with band_length
    0.7 along that diagonal. Returns 0 on the band edge, negative inside.
    """
    a, b = float(x_norm01[0]), float(x_norm01[1])
    relative_a = a - 1.0
    relative_b = b - 1.0
    # Project onto unit vector from (1,1) toward (0.5,0.5): (-1/sqrt2, -1/sqrt2)
    proj = (-relative_a - relative_b) / np.sqrt(2.0)        # along band
    perp = abs(-relative_a + relative_b) / np.sqrt(2.0)     # |across band|
    band_width = 0.15
    band_length = 0.7
    # Inside band <=> 0 < proj < band_length AND perp < band_width.
    # Distance to nearest band edge (signed: negative inside, positive outside):
    if 0.0 < proj < band_length and perp < band_width:
        d_perp = band_width - perp                # >0 inside
        d_end = min(proj, band_length - proj)     # >0 inside
        return float(-min(d_perp, d_end))
    # Outside: positive distance to nearest edge segment
    d_perp_out = max(perp - band_width, 0.0)
    if proj < 0:
        d_end_out = -proj
    elif proj > band_length:
        d_end_out = proj - band_length
    else:
        d_end_out = 0.0
    return float(np.hypot(d_perp_out, d_end_out))


# =================================================================
# Driver
# =================================================================

def _make_initial_dataset(boundary_kind, input_columns,
                          n_init_grid=N_INIT_GRID, seed=SEED):
    """Create initial design as a regular n_init_grid x n_init_grid grid in
    [0, 1]^2 and evaluate the chosen problem on it. Returns a DataFrame with
    input_columns + ['ratio', 'turbidity_600', 'well_type'='experiment'].
    Coordinates are in the ORIGINAL space expected by the recommender (mM
    for surfactant, [0,1] for synthetic).
    """
    torch.manual_seed(seed)
    g = np.linspace(0.0, 1.0, n_init_grid)
    G0, G1 = np.meshgrid(g, g)
    X01 = np.stack([G0.ravel(), G1.ravel()], axis=1)  # (n_init_grid**2, 2)

    rows = []
    for x in X01:
        if boundary_kind == "step2d":
            r, t = step2d_outputs(x)
            rows.append({input_columns[0]: x[0], input_columns[1]: x[1],
                         "ratio": r, "turbidity_600": t,
                         "well_type": "experiment"})
        elif boundary_kind == "circle2d":
            r, t = circle2d_outputs(x)
            rows.append({input_columns[0]: x[0], input_columns[1]: x[1],
                         "ratio": r, "turbidity_600": t,
                         "well_type": "experiment"})
        elif boundary_kind == "surfactant2d":
            # Map [0,1]^2 -> log10([MIN, MAX]) -> mM
            log_min, log_max = np.log10(MIN_CONC), np.log10(MAX_CONC)
            mM = 10 ** (log_min + x * (log_max - log_min))
            r, t = simulate_surfactant_measurements(mM[0], mM[1], add_noise=False)
            rows.append({input_columns[0]: float(mM[0]),
                         input_columns[1]: float(mM[1]),
                         "ratio": r, "turbidity_600": t,
                         "well_type": "experiment"})
        else:
            raise ValueError(f"Unknown boundary kind: {boundary_kind}")
    return pd.DataFrame(rows)


def _evaluate_recommendations(rec_df, boundary_kind, input_columns):
    """Evaluate the simulator on a recommendations DataFrame, append
    ratio + turbidity_600 columns and well_type='experiment'."""
    rs, ts = [], []
    for _, row in rec_df.iterrows():
        x = np.array([row[input_columns[0]], row[input_columns[1]]])
        if boundary_kind == "step2d":
            r, t = step2d_outputs(x)
        elif boundary_kind == "circle2d":
            r, t = circle2d_outputs(x)
        elif boundary_kind == "surfactant2d":
            r, t = simulate_surfactant_measurements(x[0], x[1], add_noise=False)
        else:
            raise ValueError(boundary_kind)
        rs.append(r); ts.append(t)
    out = rec_df.copy()
    out["ratio"] = rs
    out["turbidity_600"] = ts
    out["well_type"] = "experiment"
    return out


def _run_recommender(rec, boundary_kind, input_columns, boundary_func):
    """Run N_ITERATIONS rounds against the chosen boundary."""
    data = _make_initial_dataset(boundary_kind, input_columns,
                                 n_init_grid=N_INIT_GRID)
    data["iteration"] = 0
    for it in range(1, N_ITERATIONS + 1):
        print(f"\n=== {rec.__class__.__name__}: iteration {it}/{N_ITERATIONS} ===")
        recs = rec.get_recommendations(
            data, n_points=Q_BATCH, iteration=it,
            boundary_func=boundary_func,
        )
        new_rows = _evaluate_recommendations(recs, boundary_kind, input_columns)
        new_rows["iteration"] = it
        data = pd.concat([data, new_rows], ignore_index=True)
    return data, rec.get_metrics_df()


# =================================================================
# Plot helpers
# =================================================================

def _normalize_mM(xa, xb):
    log_min, log_max = np.log10(MIN_CONC), np.log10(MAX_CONC)
    na = (np.log10(xa) - log_min) / (log_max - log_min)
    nb = (np.log10(xb) - log_min) / (log_max - log_min)
    return na, nb


def _plot_2d_exploration(data, boundary_kind, input_columns, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Background: true boundary contour on a 200x200 grid in [0,1]^2
    n_grid = 200
    g = np.linspace(0.0, 1.0, n_grid)
    G0, G1 = np.meshgrid(g, g)

    if boundary_kind == "surfactant2d":
        # Two true boundaries: ratio L-shape AND turbidity diagonal band.
        # Plot both as separate contours.
        H_ratio = np.zeros_like(G0)
        H_turb = np.zeros_like(G0)
        for i in range(n_grid):
            for j in range(n_grid):
                xn = np.array([G0[i, j], G1[i, j]])
                H_ratio[i, j] = surfactant2d_boundary(xn)
                H_turb[i, j] = _surfactant_turbidity_boundary(xn)
        ax.contour(G0, G1, H_ratio, levels=[0.0], colors="k",
                   linewidths=2, linestyles="--")
        ax.contour(G0, G1, H_turb, levels=[0.0], colors="red",
                   linewidths=2, linestyles=":")
        # Legend handles
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], color="k", linestyle="--", label="ratio boundary"),
            Line2D([0], [0], color="red", linestyle=":", label="turbidity band edge"),
        ], loc="lower left")
    else:
        H = np.zeros_like(G0)
        for i in range(n_grid):
            for j in range(n_grid):
                xn = np.array([G0[i, j], G1[i, j]])
                if boundary_kind == "step2d":
                    H[i, j] = step2d_boundary(xn)
                elif boundary_kind == "circle2d":
                    H[i, j] = circle2d_boundary(xn)
        cs = ax.contour(G0, G1, H, levels=[0.0], colors="k", linewidths=2,
                        linestyles="--")
        ax.clabel(cs, fmt={0.0: "boundary"}, fontsize=9)

    # Picks (in normalized space)
    if boundary_kind == "surfactant2d":
        nx, ny = _normalize_mM(data[input_columns[0]].values,
                               data[input_columns[1]].values)
    else:
        nx = data[input_columns[0]].values
        ny = data[input_columns[1]].values
    sc = ax.scatter(nx, ny, c=data["iteration"].values, cmap="viridis",
                    s=40, edgecolor="white", linewidth=0.5)
    plt.colorbar(sc, ax=ax, label="iteration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"{input_columns[0]} (normalized)")
    ax.set_ylabel(f"{input_columns[1]} (normalized)")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def _plot_gradient_map(rec, last_data, boundary_kind, input_columns, out_path):
    """Heatmap of ||grad mu|| (summed across outputs, summed across dims)
    for the gradient recommender, using its current models."""
    # Re-fit on full data so we can evaluate the GP gradients
    print(f"  Re-fitting GP on full dataset ({len(last_data)} pts) for gradient map...")
    experiment = rec._prepare_data(last_data)
    _, X = rec._process_inputs(experiment)
    _, Y = rec._process_outputs(experiment)
    models = rec._fit_models(X, Y)

    n_grid = 60
    g = np.linspace(0.02, 0.98, n_grid)
    G0, G1 = np.meshgrid(g, g)
    pts = torch.tensor(np.stack([G0.ravel(), G1.ravel()], axis=1),
                       dtype=rec.dtype, device=rec.device)
    grad_mu = rec._grad_mu(models, pts)        # (N, n_out, d)
    grad_var = rec._grad_var(models, pts)      # (N, n_out, d)
    score = (torch.abs(grad_mu) + rec.beta * torch.sqrt(grad_var)).sum(
        dim=(1, 2)).detach().cpu().numpy().reshape(n_grid, n_grid)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(score, origin="lower", extent=[0, 1, 0, 1],
                   cmap="magma", aspect="equal")
    plt.colorbar(im, ax=ax, label="sum_j sum_k (|grad mu| + beta*sigma_grad)")

    # Overlay picks (normalized)
    if boundary_kind == "surfactant2d":
        nx, ny = _normalize_mM(last_data[input_columns[0]].values,
                               last_data[input_columns[1]].values)
    else:
        nx = last_data[input_columns[0]].values
        ny = last_data[input_columns[1]].values
    ax.scatter(nx, ny, c="cyan", s=20, edgecolor="black", linewidth=0.4)
    ax.set_xlabel(f"{input_columns[0]} (norm)")
    ax.set_ylabel(f"{input_columns[1]} (norm)")
    ax.set_title("Gradient acquisition heatmap (final iteration)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def _plot_comparison_hd(metrics_by_name, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, df in metrics_by_name.items():
        ax.plot(df["iteration"], df["frac_near_cumulative"],
                marker="o", label=f"{name} (near boundary)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative fraction within boundary band")
    ax.set_title("Boundary-hit fraction vs iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_path}")


# =================================================================
# Test runner
# =================================================================

TESTS = {
    "step2d":       dict(input_columns=["x0", "x1"],
                         boundary_func=step2d_boundary,
                         log_transform=False),
    "circle2d":     dict(input_columns=["x0", "x1"],
                         boundary_func=circle2d_boundary,
                         log_transform=False),
    "surfactant2d": dict(input_columns=["surf_A_mm", "surf_B_mm"],
                         boundary_func=surfactant2d_boundary,
                         log_transform=True),
}


def run_test(name):
    if name not in TESTS:
        raise ValueError(f"Unknown test {name!r}. Choices: {list(TESTS)}")
    cfg = TESTS[name]
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'#' * 70}\n# Test: {name}\n# Output dir: {out_dir}\n{'#' * 70}")

    output_columns = ["ratio", "turbidity_600"]
    log_xform = cfg["log_transform"]

    # Build both recommenders fresh (same seed).
    torch.manual_seed(SEED); np.random.seed(SEED)
    bayes = BayesianTransitionRecommender(
        input_columns=cfg["input_columns"],
        output_columns=output_columns,
        log_transform_inputs=log_xform,
    )
    grad = GradientTransitionRecommender(
        input_columns=cfg["input_columns"],
        output_columns=output_columns,
        log_transform_inputs=log_xform,
    )

    metrics_by_name = {}
    last_data_by_name = {}
    for rec, label in [(bayes, "BayesianContrast"), (grad, "GradientUCB")]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        data, metrics = _run_recommender(
            rec, name, cfg["input_columns"], cfg["boundary_func"])
        data_path = os.path.join(out_dir, f"all_data_{label}.csv")
        met_path = os.path.join(out_dir, f"metrics_{label}.csv")
        data.to_csv(data_path, index=False)
        metrics.to_csv(met_path, index=False)
        print(f"  Wrote {data_path}\n  Wrote {met_path}")
        _plot_2d_exploration(
            data, name, cfg["input_columns"],
            title=f"{name}: {label}",
            out_path=os.path.join(out_dir, f"2d_exploration_{label}.png"),
        )
        metrics_by_name[label] = metrics
        last_data_by_name[label] = data

    # Gradient-only diagnostic heatmap
    try:
        _plot_gradient_map(
            grad, last_data_by_name["GradientUCB"], name,
            cfg["input_columns"],
            out_path=os.path.join(out_dir, "gradient_map_GradientUCB.png"),
        )
    except Exception as e:
        print(f"  WARNING: gradient map plot failed: {e}")

    _plot_comparison_hd(metrics_by_name,
                        out_path=os.path.join(out_dir, "comparison_hd.png"))


# =================================================================
# Phase 3 unit checks (kept here so they're reproducible from CLI)
# =================================================================

def run_unit_checks():
    """Sanity-check _grad_mu and _grad_var on a 1D GP fit to y = sin(x)."""
    print("\n--- Phase 3 unit checks ---")
    import gpytorch
    from botorch.models import SingleTaskGP, ModelListGP
    from botorch.fit import fit_gpytorch_mll

    n = 25
    x_train = torch.linspace(0, 2 * np.pi, n, dtype=torch.double).unsqueeze(-1)
    y_train = torch.sin(x_train)
    model_a = SingleTaskGP(x_train, y_train)
    model_b = SingleTaskGP(x_train, y_train.clone())  # dummy 2nd output
    for m in (model_a, model_b):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(m.likelihood, m)
        fit_gpytorch_mll(mll)
    models = ModelListGP(model_a, model_b)

    rec = GradientTransitionRecommender(
        input_columns=["x0", "x1"],     # not actually used for this 1D check
        output_columns=["a", "b"],
        log_transform_inputs=False,
    )
    rec.n_inputs = 1
    rec.n_outputs = 2

    grad_mu = rec._grad_mu(models, x_train).squeeze().numpy()  # (n, 2)
    grad_var = rec._grad_var(models, x_train).squeeze().numpy()  # (n, 2)

    cos_ref = np.cos(x_train.numpy().flatten())
    err = np.max(np.abs(grad_mu[:, 0] - cos_ref))
    print(f"  _grad_mu vs cos(x):  max abs err = {err:.4f} "
          f"(should be small, < ~0.2)")

    var_at_train = float(np.median(grad_var[:, 0]))
    print(f"  _grad_var median at training points: {var_at_train:.4e} "
          f"(should be near zero)")

    # Test off-training points: variance should grow
    x_test = torch.linspace(-1.0, 7.5, 50, dtype=torch.double).unsqueeze(-1)
    grad_var_test = rec._grad_var(models, x_test).squeeze().numpy()
    print(f"  _grad_var at off-training pts: "
          f"median={np.median(grad_var_test[:, 0]):.4e}, "
          f"max={np.max(grad_var_test[:, 0]):.4e}")

    # 2D exclusion sanity check
    rec.n_inputs = 2
    rec.exclusion_regions = []
    center = torch.tensor([0.5, 0.5], dtype=torch.double)
    radii = torch.tensor([0.05, 0.20], dtype=torch.double)
    rec.exclusion_regions.append({"center": center, "radii": radii})
    pool = torch.tensor([
        [0.50, 0.50],   # center: excluded
        [0.50, 0.69],   # within y radius: excluded
        [0.50, 0.71],   # just outside y radius: NOT excluded
        [0.56, 0.50],   # outside x radius: NOT excluded
        [0.54, 0.50],   # within x radius: excluded
    ], dtype=torch.double)
    mask = rec._is_excluded(pool).numpy()
    expected = np.array([True, True, False, False, True])
    print(f"  _is_excluded mask: {mask.tolist()}")
    print(f"  expected:          {expected.tolist()}")
    assert np.array_equal(mask, expected), "Exclusion mask mismatch!"
    print("  Exclusion shape check PASSED.")


# =================================================================
# CLI
# =================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=list(TESTS) + ["all"], default=None)
    parser.add_argument("--unit", action="store_true")
    args = parser.parse_args()

    if args.unit:
        run_unit_checks()
        return

    if args.test is None:
        parser.print_help()
        sys.exit(1)

    if args.test == "all":
        for name in TESTS:
            run_test(name)
    else:
        run_test(args.test)


if __name__ == "__main__":
    main()
