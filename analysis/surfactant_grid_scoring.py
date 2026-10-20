"""Local-fit 2D ratio/turbidity scoring for surfactant grid studies.

The score maps follow scoring_v2's sigmoid, turbidity transform, and below-CMC
mask. Global fallback values are a snapshot of scoring_v2's
analysis/1D CMC results/cmc_parameter_summary.csv.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def _valid_fit(row):
    values = np.array([row[k] for k in ("A1", "A2", "x0", "dx", "cmc_mm", "r_squared")], dtype=float)
    return (row["fit_status"] == "success" and np.isfinite(values).all()
            and row["n_points"] >= 5 and row["r_squared"] >= 0.8
            and row["A1"] <= 1 and row["A1"] > row["A2"]
            and row["dx"] > 0 and row["cmc_mm"] > 0)


def score_surfactant_grid_2d(results_csv, fit_folder, output_folder, surfactant_a, surfactant_b,
                             global_parameters_csv=None):
    """Save calibration, score grids, summary, and diagnostic figure.

    Returns a one-row summary even when calibration or grid data is insufficient.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    if global_parameters_csv is None:
        global_parameters_csv = Path(__file__).with_name("surfactant_grid_global_cmc_parameters.csv")
    global_parameters = pd.read_csv(global_parameters_csv).set_index("surfactant")
    df = pd.read_csv(results_csv)
    fits = []
    for surfactant in (surfactant_a, surfactant_b):
        safe_name = str(surfactant).replace(" ", "_")
        path = Path(fit_folder) / f"study_1d_{safe_name}_fit.csv"
        fits.append(pd.read_csv(path).iloc[0] if path.exists() else None)

    row = {
        "surfactant_A": surfactant_a, "surfactant_B": surfactant_b,
        "status": "missing_calibration", "n_experiment_points": 0,
    }
    selected = []
    for slot, surfactant, fit in zip(("A", "B"), (surfactant_a, surfactant_b), fits):
        row[f"fit_{slot}_status"] = "missing" if fit is None else fit["fit_status"]
        row[f"fit_{slot}_accepted"] = fit is not None and _valid_fit(fit)
        for key in ("A1", "A2", "cmc_mm", "r_squared"):
            row[f"fit_{slot}_{key}"] = np.nan if fit is None else fit[key]
        if row[f"fit_{slot}_accepted"]:
            cmc = (float(global_parameters.loc[surfactant, "cmc_mean_mm"])
                   if surfactant in global_parameters.index else float(fit["cmc_mm"]))
            values = (float(fit["A1"]), float(fit["A2"]), cmc)
            source = "local"
        elif surfactant in global_parameters.index:
            global_row = global_parameters.loc[surfactant]
            values = tuple(float(global_row[key]) for key in ("A1_mean", "A2_mean", "cmc_mean_mm"))
            source = "global"
        else:
            values = (np.nan, np.nan, np.nan)
            source = "missing"
        row[f"calibration_{slot}_source"] = source
        row[f"selected_{slot}_A1"], row[f"selected_{slot}_A2"], row[f"selected_{slot}_cmc_mm"] = values
        selected.append(values)

    summary_path = output_folder / "score_summary.csv"
    if not all(np.isfinite(values).all() and values[0] > values[1] and values[2] > 0
               for values in selected):
        pd.DataFrame([row]).to_csv(summary_path, index=False)
        return row

    wells = df.loc[df["well_type"] == "experiment", [
        "surf_A_conc_mm", "surf_B_conc_mm", "ratio", "turbidity_600"
    ]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    wells = wells[(wells.surf_A_conc_mm > 0) & (wells.surf_B_conc_mm > 0)]
    row["n_experiment_points"] = len(wells)
    if len(wells) < 4:
        row["status"] = "insufficient_grid_points"
        pd.DataFrame([row]).to_csv(summary_path, index=False)
        return row

    points = np.log10(wells[["surf_A_conc_mm", "surf_B_conc_mm"]].to_numpy())
    if np.unique(points, axis=0).shape[0] < 4 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        row["status"] = "insufficient_grid_geometry"
        pd.DataFrame([row]).to_csv(summary_path, index=False)
        return row

    x_axis = np.linspace(points[:, 0].min(), points[:, 0].max(), 50)
    y_axis = np.linspace(points[:, 1].min(), points[:, 1].max(), 50)
    xx, yy = np.meshgrid(x_axis, y_axis)

    def interpolate(column):
        values = wells[column].to_numpy()
        linear = griddata(points, values, (xx, yy), method="linear")
        nearest = griddata(points, values, (xx, yy), method="nearest")
        return np.where(np.isfinite(linear), linear, nearest)

    ratio = interpolate("ratio")
    turbidity = interpolate("turbidity_600")
    a1 = float(np.mean([values[0] for values in selected]))
    a2 = float(np.mean([values[1] for values in selected]))
    all_global = all(row[f"calibration_{slot}_source"] == "global" for slot in ("A", "B"))
    midpoint = (float(np.mean(global_parameters.loc[[surfactant_a, surfactant_b], "midpoint_mean"]))
                if all_global else (a1 + a2) / 2)
    k = (a2 - a1) / 8
    # Same orientation as scoring_v2: lower pyrene ratio means more micellar.
    from scipy.special import expit
    ratio_score = expit((ratio - midpoint) / k)
    clarity = np.minimum(0.5 ** ((np.maximum(turbidity, 0.04) - 0.04) / 0.04), 1.0)
    turbidity_penalty = 1 - clarity
    cmc_mask = (10 ** xx < selected[0][2]) & (10 ** yy < selected[1][2])
    overall = ratio_score * clarity * cmc_mask
    cell_area = (x_axis[1] - x_axis[0]) * (y_axis[1] - y_axis[0])
    row.update(
        status="success", avg_A1=a1, avg_A2=a2, midpoint=midpoint, k=k,
        calibration_source=("local" if all(row[f"calibration_{slot}_source"] == "local"
                                            for slot in ("A", "B")) else
                            "global_fallback" if all(row[f"calibration_{slot}_source"] == "global"
                                                     for slot in ("A", "B")) else "mixed_fallback"),
        below_cmc_area_dec2=float(cmc_mask.sum() * cell_area),
        micelle_score_dec2=float((ratio_score * cmc_mask).sum() * cell_area),
        turbidity_penalty_dec2=float((turbidity_penalty * cmc_mask).sum() * cell_area),
        overall_score_dec2=float(overall.sum() * cell_area),
        high_score_area_dec2=float((overall > 0.5).sum() * cell_area),
        mean_score_in_mask=float(overall[cmc_mask].mean()) if cmc_mask.any() else np.nan,
    )
    if cmc_mask.any():
        row["pct_ratio_usable_raw"] = float(((ratio_score > 0.5) & cmc_mask).sum() / cmc_mask.sum())
        row["pct_turbid_usable_raw"] = float(((turbidity_penalty > 0.5) & cmc_mask).sum() / cmc_mask.sum())
        row["pct_overall_raw"] = row["pct_ratio_usable_raw"] - row["pct_turbid_usable_raw"]
    else:
        row.update(pct_ratio_usable_raw=np.nan, pct_turbid_usable_raw=np.nan,
                   pct_overall_raw=np.nan)
    row["overall_score_dec"] = np.sqrt(row["overall_score_dec2"])
    pd.DataFrame([row]).to_csv(summary_path, index=False)

    pd.DataFrame({
        "surf_A_conc_mm": (10 ** xx).ravel(), "surf_B_conc_mm": (10 ** yy).ravel(),
        "ratio": ratio.ravel(), "turbidity_600": turbidity.ravel(),
        "ratio_score": ratio_score.ravel(), "clarity_score": clarity.ravel(),
        "turbidity_penalty": turbidity_penalty.ravel(),
        "below_cmc": cmc_mask.ravel(),
        "masked_ratio_score": (ratio_score * cmc_mask).ravel(),
        "overall_score": overall.ravel(),
    }).to_csv(output_folder / "score_grid.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]
    panels = (
        (ratio, "Pyrene ratio", None), (turbidity, "Turbidity 600 nm", None),
        (ratio_score, "Micelle ratio score", (0, 1)),
        (turbidity_penalty, "Turbidity penalty", (0, 1)),
        (cmc_mask.astype(float), "Below both CMCs", (0, 1)),
        (overall, "Overall score", (0, 1)),
    )
    for ax, (values, title, limits) in zip(axes.flat, panels):
        image = ax.imshow(values, origin="lower", extent=extent, aspect="auto",
                          vmin=None if limits is None else limits[0],
                          vmax=None if limits is None else limits[1])
        ax.axvline(np.log10(selected[0][2]), color="white", linestyle="--", linewidth=1)
        ax.axhline(np.log10(selected[1][2]), color="white", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel=f"log10 {surfactant_a} (mM)",
               ylabel=f"log10 {surfactant_b} (mM)")
        fig.colorbar(image, ax=ax)
    fig.suptitle(f"{surfactant_a} + {surfactant_b}: overall = {row['overall_score_dec2']:.3f} decades²"
                 f" ({row['calibration_source']})")
    fig.savefig(output_folder / "score_maps.png", dpi=180)
    plt.close(fig)
    return row
