"""Analysis for workflows/surfactant_dye_cmc.py runs.

Reads a run folder's results.csv (one row per well per read) and config.yaml, and
writes to <run folder>/analysis/:

1. CMC and K (cmc_fit.csv, cmc_fit.png): binding-model fit of the CMC block at
   each read time. Fluorescence = F_free + (F_bound - F_free) * K[M] / (1 + K[M]),
   with [M] the micellised surfactant, a smooth version of max(0, S - CMC) whose
   width is a fraction of the CMC (Pineiro 2013; Al-Soufi 2012).
2. Dye x surfactant (dye_matrix.csv, dye_matrix.png): background-subtracted
   fluorescence vs dye at each surfactant level, the slope over the low dye levels,
   and how far the top dye level falls from that line (quenching, inner filter or
   saturation). Descriptive only: not a capacity or encapsulation measure.
3. Stock check (stock_check.csv, stock_check.png): absorbance and fluorescence vs
   nominal dye in solvent. With a molar absorptivity, the absorbance slope gives
   the real stock concentration (path length from the well volume and radius).
4. Time course (time_course.png): CMC, K and the dye-matrix slopes at every read.

Usage:
    python analysis/surfactant_dye_cmc_analysis.py <run folder> --fluorescence fluorescence_485_530
        [--absorbance absorbance_460 --epsilon 54000 --path-cm 0.7]
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import curve_fit

_WELLPLATES_FILE = Path(__file__).resolve().parents[1] / "robot_state" / "wellplates.yaml"


# ============================================================================
# 1. CMC and K
# ============================================================================

def micellised(s, cmc, width):
    """Micellised surfactant: max(0, S - CMC) smoothed over width * CMC."""
    scale = width * cmc
    return scale * np.logaddexp(0.0, (np.asarray(s, float) - cmc) / scale)


def binding_model(s, f_free, f_bound, cmc, k, width):
    m = micellised(s, cmc, width)
    return f_free + (f_bound - f_free) * k * m / (1 + k * m)


def fit_cmc(surfactant_mM, signal, cmc_guess):
    """Fit binding_model. Returns dict of values and standard errors (NaN if it fails)."""
    s, y = np.asarray(surfactant_mM, float), np.asarray(signal, float)
    f_free = float(np.mean(y[s == s.min()]))
    top = float(np.max(y))
    p0 = [f_free, f_free + 2 * (top - f_free), cmc_guess, 1.0, 0.05]
    lower = [-np.inf, -np.inf, 0.1 * cmc_guess, 1e-4, 1e-3]
    upper = [np.inf, np.inf, 10 * cmc_guess, 1e4, 0.5]
    names = ["f_free", "f_bound", "cmc_mM", "k_per_mM", "width"]
    try:
        values, cov = curve_fit(binding_model, s, y, p0=p0, bounds=(lower, upper), maxfev=20000)
        errors = np.sqrt(np.diag(cov))
    except (RuntimeError, ValueError):
        return {**{n: np.nan for n in names}, **{f"{n}_se": np.nan for n in names}, "r2": np.nan}
    fitted = binding_model(s, *values)
    r2 = 1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    return {**dict(zip(names, values)), **{f"{n}_se": e for n, e in zip(names, errors)}, "r2": r2}


def cmc_by_time(results, fluorescence, cmc_guess, block="cmc"):
    rows = []
    data = results[results.block == block]
    for t, group in data.groupby("timepoint_min"):
        fit = fit_cmc(group.surfactant_mM, group[fluorescence], cmc_guess)
        rows.append({"timepoint_min": t, "wells": len(group), **fit})
    return pd.DataFrame(rows)


def plot_cmc(results, fits, fluorescence, out, block="cmc"):
    data = results[results.block == block]
    fig, ax = plt.subplots(figsize=(7, 5))
    times = sorted(data.timepoint_min.unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(times)))
    grid = np.linspace(0, data.surfactant_mM.max(), 400)
    for t, color in zip(times, colors):
        group = data[data.timepoint_min == t]
        ax.scatter(group.surfactant_mM, group[fluorescence], s=12, color=color, alpha=0.7)
        fit = fits.set_index("timepoint_min").loc[t]
        if np.isfinite(fit.cmc_mM):
            params = fit[["f_free", "f_bound", "cmc_mM", "k_per_mM", "width"]].to_numpy(float)
            ax.plot(grid, binding_model(grid, *params), color=color,
                    label=f"{t:g} min: CMC {fit.cmc_mM:.2f} mM, K {fit.k_per_mM:.2g}/mM")
    ax.set_xlabel("Surfactant (mM)")
    ax.set_ylabel(fluorescence)
    ax.set_title("CMC block: binding-model fit per read")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "cmc_fit.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 2. Dye x surfactant
# ============================================================================

def dye_matrix(results, fluorescence, block="dye_matrix"):
    """Mean +/- SD per (time, surfactant, dye), background-subtracted with the
    dye-0 wells at the same surfactant. Includes other water wells (e.g. the CMC
    block) at the matrix's own levels, since those pairs were not repeated."""
    matrix = results[results.block == block]
    if matrix.empty:
        return pd.DataFrame(), pd.DataFrame()
    levels_s, levels_d = set(matrix.surfactant_mM), set(matrix.dye_uM)
    wells = results[(results.matrix == "water") & results.surfactant_mM.isin(levels_s)
                    & results.dye_uM.isin(levels_d)]
    stats = (wells.groupby(["timepoint_min", "surfactant_mM", "dye_uM"])[fluorescence]
             .agg(mean="mean", sd="std", n="size").reset_index())
    background = stats[stats.dye_uM == 0].set_index(["timepoint_min", "surfactant_mM"])["mean"]
    key = list(zip(stats.timepoint_min, stats.surfactant_mM))
    stats["background"] = [background.get(k, np.nan) for k in key]
    stats["signal"] = stats["mean"] - stats["background"]
    # Slope over the low dye levels (all but the top), then the top level vs that line.
    rows = []
    top = max(levels_d)
    for (t, s), g in stats.groupby(["timepoint_min", "surfactant_mM"]):
        low = g[(g.dye_uM > 0) & (g.dye_uM < top)]
        slope = (np.sum(low.dye_uM * low.signal) / np.sum(low.dye_uM ** 2)) if len(low) else np.nan
        at_top = g.loc[g.dye_uM == top, "signal"]
        ratio = float(at_top.iloc[0] / (slope * top)) if len(at_top) and slope else np.nan
        rows.append({"timepoint_min": t, "surfactant_mM": s, "slope_per_uM": slope,
                     "top_dye_uM": top, "top_vs_line": ratio})
    return stats, pd.DataFrame(rows)


def plot_dye_matrix(stats, out):
    last = stats.timepoint_min.max()
    data = stats[stats.timepoint_min == last]
    fig, ax = plt.subplots(figsize=(7, 5))
    for s, g in data.groupby("surfactant_mM"):
        g = g.sort_values("dye_uM")
        ax.errorbar(g.dye_uM, g.signal, yerr=g.sd, marker="o", capsize=3, label=f"{s:g} mM")
    ax.set_xlabel("Dye (uM)")
    ax.set_ylabel("Fluorescence - dye-free background")
    ax.set_title(f"Dye x surfactant at {last:g} min")
    ax.legend(title="Surfactant", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "dye_matrix.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 3. Stock check
# ============================================================================

def path_length_cm(volume_ul, plate_type):
    """Liquid height in a flat-bottomed well: volume / (pi r^2)."""
    with open(_WELLPLATES_FILE) as f:
        radius_mm = float(yaml.safe_load(f)[plate_type]["well_radius"])
    return volume_ul / (math.pi * radius_mm ** 2) / 10


def stock_check(results, fluorescence, absorbance=None, epsilon=None, path_cm=None,
                nominal_stock_uM=None, block="stock_check"):
    """Linear fits vs nominal dye (first read only: solvent evaporates)."""
    data = results[(results.block == block) & (results.timepoint_min == results.timepoint_min.min())]
    if data.empty:
        return {}, data
    summary = {}
    for label, column in (("fluorescence", fluorescence), ("absorbance", absorbance)):
        if column is None or column not in data:
            continue
        blank = data.loc[data.dye_uM == 0, column].mean()
        x, y = data.dye_uM.to_numpy(float), (data[column] - blank).to_numpy(float)
        slope = float(np.sum(x * y) / np.sum(x ** 2))
        r2 = 1 - np.sum((y - slope * x) ** 2) / np.sum((y - y.mean()) ** 2)
        summary[f"{label}_slope_per_uM"] = slope
        summary[f"{label}_r2"] = r2
    if "absorbance_slope_per_uM" in summary and epsilon and path_cm:
        expected = epsilon * path_cm * 1e-6  # absorbance per uM
        summary["path_cm"] = path_cm
        summary["actual_over_nominal"] = summary["absorbance_slope_per_uM"] / expected
        if nominal_stock_uM:
            summary["measured_stock_uM"] = nominal_stock_uM * summary["actual_over_nominal"]
    return summary, data


def plot_stock_check(data, fluorescence, absorbance, out):
    columns = [c for c in (fluorescence, absorbance) if c is not None and c in data]
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4), squeeze=False)
    for ax, column in zip(axes[0], columns):
        ax.plot(data.dye_uM, data[column], "o")
        ax.set_xlabel("Nominal dye in solvent (uM)")
        ax.set_ylabel(column)
    fig.suptitle("Stock check (first read)")
    fig.tight_layout()
    fig.savefig(out / "stock_check.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 4. Time course and entry point
# ============================================================================

def plot_time_course(fits, slopes, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    hours = fits.timepoint_min / 60
    axes[0].errorbar(hours, fits.cmc_mM, yerr=fits.cmc_mM_se, marker="o", capsize=3)
    axes[0].set_ylabel("CMC (mM)")
    axes[1].errorbar(hours, fits.k_per_mM, yerr=fits.k_per_mM_se, marker="o", capsize=3)
    axes[1].set_ylabel("K (1/mM)")
    for s, g in slopes.groupby("surfactant_mM"):
        axes[2].plot(g.timepoint_min / 60, g.slope_per_uM, marker="o", label=f"{s:g} mM")
    axes[2].set_ylabel("Fluorescence per uM dye")
    axes[2].legend(title="Surfactant", fontsize=7)
    for ax in axes:
        ax.set_xlabel("Time after dispensing (h)")
    fig.tight_layout()
    fig.savefig(out / "time_course.png", dpi=150)
    plt.close(fig)


def analyze(run_dir, fluorescence, absorbance=None, epsilon=None, path_cm=None, logger=None):
    """Run all four analyses on a run folder. Returns dict of result tables."""
    run_dir = Path(run_dir)
    results = pd.read_csv(run_dir / "results.csv")
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    if fluorescence not in results:
        found = [c for c in results.columns if c.startswith("fluorescence_")]
        raise ValueError(f"No column {fluorescence!r} in results.csv; fluorescence columns: {found}")
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)

    fits = cmc_by_time(results, fluorescence, config["SURFACTANT_CMC_MM"])
    fits.to_csv(out / "cmc_fit.csv", index=False)
    if not fits.empty:
        plot_cmc(results, fits, fluorescence, out)

    stats, slopes = dye_matrix(results, fluorescence)
    if not stats.empty:
        stats.to_csv(out / "dye_matrix.csv", index=False)
        slopes.to_csv(out / "dye_matrix_slopes.csv", index=False)
        plot_dye_matrix(stats, out)

    if path_cm is None and absorbance is not None:
        path_cm = path_length_cm(config["TOTAL_VOLUME_UL"], config["WELLPLATE_TYPE"])
    summary, check = stock_check(results, fluorescence, absorbance, epsilon, path_cm,
                                 config["DYE_STOCK_UM"])
    if summary:
        pd.DataFrame([summary]).to_csv(out / "stock_check.csv", index=False)
        plot_stock_check(check, fluorescence, absorbance, out)

    if not fits.empty and not slopes.empty:
        plot_time_course(fits, slopes, out)
    if logger:
        logger.info(f"Analysis saved in {out}")
    return dict(cmc_fit=fits, dye_matrix=stats, dye_matrix_slopes=slopes, stock_check=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--fluorescence", required=True, help="results.csv column, e.g. fluorescence_485_530")
    parser.add_argument("--absorbance", help="results.csv column at the dye peak, e.g. absorbance_460")
    parser.add_argument("--epsilon", type=float, help="molar absorptivity at that wavelength (1/M/cm)")
    parser.add_argument("--path-cm", type=float, help="path length (default: from well volume and radius)")
    args = parser.parse_args()
    tables = analyze(args.run_dir, args.fluorescence, args.absorbance, args.epsilon, args.path_cm)
    print(tables["cmc_fit"][["timepoint_min", "cmc_mM", "cmc_mM_se", "k_per_mM", "k_per_mM_se", "r2"]]
          .round(3).to_string(index=False))
    if tables["stock_check"]:
        print({k: round(v, 4) for k, v in tables["stock_check"].items()})


if __name__ == "__main__":
    main()
