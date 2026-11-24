"""Analysis for workflows/surfactant_dye_cmc.py runs.

Reads a run folder's results.csv (one row per well per read), spectra.csv and
config.yaml, and writes to <run folder>/analysis/ (numbered by plate block):

1. Ethanol series (1_ethanol_series.png, _summary.csv, _wells.csv): dye in solvent at
   the first read, 2x2: absorbance and fluorescence spectra, fixed-wavelength
   fluorescence vs its low-dye line, and the stock concentration from absorbance
   (epsilon; path ASSUMED, see ASSUMED_PATH_CM). Inner-filter test
   (1_ethanol_inner_filter.png, .csv): fluorescence raw and corrected with each well's
   absorbance at the excitation and emission wavelengths.
2. CMC (2_cmc_fits.csv, 2_cmc_binding_model.png, 2_cmc_sigmoid.png, 2_cmc_over_time.png,
   2_cmc_how_calculated.png): binding-model fit of the CMC block at each read.
   Fluorescence = F_free + (F_bound - F_free) * K[M] / (1 + K[M]), with [M] the
   micellised surfactant, a smooth version of max(0, S - CMC) whose width is a
   fraction of the CMC (Pineiro 2013; Al-Soufi 2012). Also, for comparison with
   earlier studies, a Boltzmann sigmoid on log10(S), CMC = midpoint. CMC and K at
   every read (2_cmc_over_time.png).
3. Dye x SDS (3_dye_x_sds_*): background-subtracted fluorescence vs dye at each
   surfactant level (one read as png, every read as a 3D html with its shape change between
   reads in 3_dye_x_sds_fluorescence_3d_shape_change.png; table in csv);
   absorbance at the dye peak (% of the ethanol series) and at 750 nm, per well
   and over time (3_dye_x_sds_absorbance.png); normalised absorbance and emission
   spectra vs the ethanol series (3_dye_x_sds_*_spectra.png); the 1 uM dye signal
   over time (3_dye_x_sds_1uM_over_time.png); low-dye slopes (csv).
4. No-SDS controls (4_no_sds_controls.png): dye in water without surfactant over time.
5. Replicates (5_replicates.png, .csv): CMC-block plateau wells (3 replicates), each well's
   deviation from its replicates over reads, against the spread expected from pipetting
   precision (latest validation run before this run, planned volumes).

Usage:
    python analysis/surfactant_dye_cmc_analysis.py <run folder> --fluorescence fluorescence_485_530
        --absorbance absorbance_460 --epsilon 54000 [--path-cm 0.5]
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


def boltzmann(x, a1, a2, x0, dx):
    """Boltzmann sigmoid, as in analysis/control_cmc_analysis.py (earlier CMC studies)."""
    return a2 + (a1 - a2) / (1 + np.exp((x - x0) / dx))


def fit_sigmoid(surfactant_mM, signal):
    """Boltzmann fit on log10(surfactant), as in the earlier CMC studies: CMC = 10**x0.

    0 mM wells cannot go on a log axis and are left out. SE of the CMC from the SE of x0.
    """
    s, y = np.asarray(surfactant_mM, float), np.asarray(signal, float)
    keep = s > 0
    x, y = np.log10(s[keep]), y[keep]
    p0 = [y[x == x.min()].mean(), y[x == x.max()].mean(), float(np.median(x)), (x.max() - x.min()) / 10]
    names = ["sigmoid_a1", "sigmoid_a2", "sigmoid_x0", "sigmoid_dx"]
    try:
        values, cov = curve_fit(boltzmann, x, y, p0=p0, maxfev=20000)
        errors = np.sqrt(np.diag(cov))
    except (RuntimeError, ValueError):
        return {**{n: np.nan for n in names}, "sigmoid_cmc_mM": np.nan, "sigmoid_cmc_mM_se": np.nan,
                "sigmoid_r2": np.nan}
    r2 = 1 - np.sum((y - boltzmann(x, *values)) ** 2) / np.sum((y - y.mean()) ** 2)
    cmc = 10 ** values[2]
    return {**dict(zip(names, values)), "sigmoid_cmc_mM": cmc,
            "sigmoid_cmc_mM_se": cmc * np.log(10) * errors[2], "sigmoid_r2": r2}


def cmc_by_time(results, fluorescence, cmc_guess, block="cmc"):
    """Binding-model and Boltzmann-sigmoid fits of the CMC block at each read time."""
    rows = []
    data = results[results.block == block]
    for t, group in data.groupby("timepoint_min"):
        fit = fit_cmc(group.surfactant_mM, group[fluorescence], cmc_guess)
        sigmoid = fit_sigmoid(group.surfactant_mM, group[fluorescence])
        rows.append({"timepoint_min": t, "wells": len(group), **fit, **sigmoid})
    return pd.DataFrame(rows)


def plot_cmc_sigmoid(results, fits, fluorescence, out, block="cmc"):
    """Boltzmann fits on a log surfactant axis (0 mM wells left out)."""
    data = results[(results.block == block) & (results.surfactant_mM > 0)]
    fig, ax = plt.subplots(figsize=(7, 5))
    times = sorted(data.timepoint_min.unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(times)))
    grid = np.linspace(np.log10(data.surfactant_mM.min()), np.log10(data.surfactant_mM.max()), 400)
    for t, color in zip(times, colors):
        group = data[data.timepoint_min == t]
        ax.scatter(group.surfactant_mM, group[fluorescence], s=12, color=color, alpha=0.7)
        fit = fits.set_index("timepoint_min").loc[t]
        if np.isfinite(fit.sigmoid_cmc_mM):
            params = fit[["sigmoid_a1", "sigmoid_a2", "sigmoid_x0", "sigmoid_dx"]].to_numpy(float)
            ax.plot(10 ** grid, boltzmann(grid, *params), color=color,
                    label=f"{t:g} min: CMC {fit.sigmoid_cmc_mM:.2f} mM")
    ax.set_xscale("log")
    ax.set_xlabel("Surfactant (mM, log)")
    ax.set_ylabel(fluorescence)
    ax.set_title("CMC block: Boltzmann sigmoid on log10(S) per read (CMC = midpoint)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "2_cmc_sigmoid.png", dpi=150)
    plt.close(fig)


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
    fig.savefig(out / "2_cmc_binding_model.png", dpi=150)
    plt.close(fig)


def plot_cmc_method(results, fits, fluorescence, out, block="cmc"):
    """How the CMC is calculated, step by step, using the first read's real fit."""
    fit = fits[np.isfinite(fits.cmc_mM)].iloc[0]
    t = fit.timepoint_min
    data = results[(results.block == block) & (results.timepoint_min == t)]
    f_free, f_bound, cmc, k, width = fit[["f_free", "f_bound", "cmc_mM", "k_per_mM", "width"]].to_numpy(float)
    s = np.linspace(0, data.surfactant_mM.max(), 600)
    m = micellised(s, cmc, width)
    bound = k * m / (1 + k * m)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ax = axes[0]
    ax.plot(s, m, lw=2)
    ax.plot(s, np.maximum(0, s - cmc), color="0.5", lw=1, ls="--", label="max(0, S - CMC)")
    ax.set(xlabel="total SDS, S (mM)", ylabel="micellised SDS, [M] (mM)",
           title="1. Micelles form above the CMC\n[M] = S - CMC (corner smoothed)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(s, bound, lw=2)
    ax.set_ylim(0, 1.05)
    ax.set(xlabel="total SDS, S (mM)", ylabel="fraction of dye in micelles",
           title=f"2. Dye binds micelles (K = {k:.3g} /mM)\nbound = K[M] / (1 + K[M])")

    ax = axes[2]
    ax.scatter(data.surfactant_mM, data[fluorescence], s=18, color="0.3", zorder=3, label="wells")
    ax.plot(s, f_free + (f_bound - f_free) * bound, lw=2, label="fit")
    ax.axhline(f_free, color="0.5", lw=1, ls=":")
    ax.axhline(f_bound, color="0.5", lw=1, ls=":")
    ax.text(s.max(), f_free, " F_free", va="bottom", ha="right", fontsize=8, color="0.3")
    ax.text(s.max(), f_bound, " F_bound", va="bottom", ha="right", fontsize=8, color="0.3")
    ax.set(xlabel="total SDS, S (mM)", ylabel=fluorescence,
           title="3. Signal = F_free + (F_bound - F_free) x bound\nfitted to the wells")
    ax.legend(fontsize=8, loc="center right")

    for ax in axes:
        ax.axvline(cmc, color="#eb6834", lw=1.2, ls="--")
    axes[0].text(cmc, axes[0].get_ylim()[1] * 0.95, f" CMC {cmc:.2f} mM", color="#eb6834", fontsize=9, va="top")
    fig.suptitle(f"How the CMC is calculated (read at {t:g} min). Fitted together: F_free, F_bound, CMC, K, "
                 f"corner width. CMC = {cmc:.2f} +/- {fit.cmc_mM_se:.2g} mM (fit SE)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "2_cmc_how_calculated.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 2. Dye x surfactant
# ============================================================================

def water_blank(results, column):
    """Mean of the no-dye, no-surfactant water wells, per read (Series by timepoint_min)."""
    blank = results[(results.matrix == "water") & (results.dye_uM == 0) & (results.surfactant_mM == 0)]
    if blank.empty:
        raise ValueError("No water blank well (0 mM surfactant, 0 uM dye)")
    return blank.groupby("timepoint_min")[column].mean()


def dye_matrix(results, fluorescence, block="dye_matrix"):
    """Mean +/- SD per (time, surfactant, dye), background-subtracted with the
    no-dye, no-surfactant water wells at the same read (SDS does not change the
    blank: 0-25 mM read the same in the 2026-11-21/22 runs). Includes every water
    well at the matrix's surfactant levels (e.g. 1 uM from the CMC block)."""
    matrix = results[results.block == block]
    if matrix.empty:
        return pd.DataFrame(), pd.DataFrame()
    levels_s, levels_d = set(matrix.surfactant_mM), set(matrix.dye_uM)
    wells = results[(results.matrix == "water") & results.surfactant_mM.isin(levels_s)]
    stats = (wells.groupby(["timepoint_min", "surfactant_mM", "dye_uM"])[fluorescence]
             .agg(mean="mean", sd="std", n="size").reset_index())
    background = water_blank(results, fluorescence)
    stats["background"] = stats.timepoint_min.map(background)
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


DYE_MATRIX_PLOT_MIN = 480  # read shown in dye_matrix.png


def inner_filter_corrected(results, all_spectra, fluorescence, timepoint, matrix="water"):
    """Per (surfactant, dye): mean +/- SD of background-subtracted fluorescence corrected for
    inner filter, F x 10^((A_ex + A_em) / 2), with A_ex and A_em from each well's own
    absorbance spectrum at that read (minus the median no-dye well). Ex/em from the column
    name (fluorescence_<ex>_<em>). Approximate for a top-read plate."""
    ex_nm, em_nm = (float(v) for v in fluorescence.split("_")[-2:])
    wells = results[(results.matrix == matrix) & (results.timepoint_min == timepoint)].copy()
    absorb = all_spectra[(all_spectra.matrix == matrix) & (all_spectra.timepoint_min == timepoint)
                         & all_spectra.protocol.str.contains("Abs") & (all_spectra.value != SPECTRUM_OVER_RANGE)]
    if absorb.empty:
        raise ValueError(f"No {matrix} absorbance spectra at {timepoint} min for the inner-filter correction")
    blank = absorb[absorb.dye_uM == 0].groupby("wavelength_nm").value.median()
    absorb = absorb.assign(a=absorb.value - absorb.wavelength_nm.map(blank)).sort_values("wavelength_nm")
    by_well = {w: g for w, g in absorb.groupby("well_position")}
    a_ex = [np.interp(ex_nm, by_well[w].wavelength_nm, by_well[w].a) for w in wells.well_position]
    a_em = [np.interp(em_nm, by_well[w].wavelength_nm, by_well[w].a) for w in wells.well_position]
    wells["factor"] = 10 ** ((np.clip(a_ex, 0, None) + np.clip(a_em, 0, None)) / 2)
    background = water_blank(wells, fluorescence).loc[timepoint]
    wells["corrected"] = (wells[fluorescence] - background) * wells.factor
    return (wells.groupby(["surfactant_mM", "dye_uM"]).agg(corrected=("corrected", "mean"),
                                                          corrected_sd=("corrected", "std"),
                                                          factor=("factor", "mean")).reset_index())


def plot_dye_matrix(stats, out, corrected=None):
    last = DYE_MATRIX_PLOT_MIN
    if last not in set(stats.timepoint_min):
        raise ValueError(f"No {last} min read for dye_matrix.png; reads: {sorted(stats.timepoint_min.unique())}")
    data = stats[stats.timepoint_min == last]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for s, g in data.groupby("surfactant_mM"):
        g = g.sort_values("dye_uM")
        line = ax.errorbar(g.dye_uM, g.signal, yerr=g.sd, marker="o", capsize=3, label=f"{s:g} mM")
        if corrected is not None:
            c = corrected[corrected.surfactant_mM == s].sort_values("dye_uM")
            ax.errorbar(c.dye_uM, c.corrected, yerr=c.corrected_sd, marker="o", mfc="white", ls="--", capsize=3,
                        color=line[0].get_color())
    if corrected is not None:
        ax.plot([], [], color="0.4", ls="--", marker="o", mfc="white", label="inner-filter corrected")
        fig.text(0.01, 0.01, "Dashed: F x 10^((A_ex + A_em)/2), A from each well's absorbance spectrum at this read. "
                             "Approximate for a top-read plate.", fontsize=7, color="0.4")
    ax.set_xlabel("Dye (uM)")
    ax.set_ylabel("Fluorescence - dye-free background")
    ax.set_title(f"Dye x surfactant at {last:g} min")
    ax.legend(title="Surfactant", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / f"3_dye_x_sds_fluorescence_{DYE_MATRIX_PLOT_MIN:g}min.png", dpi=150)
    plt.close(fig)


def surface_shape_change(stats):
    """Shape change of the dye x surfactant surface (mean signal per cell, dye > 0) between
    consecutive reads. The previous surface is scaled by the single least-squares factor
    that best matches the current one, so a uniform fade does not count; shape change =
    root-sum-square residual / root-sum-square surface. Noise level = the same ratio
    expected from replicate scatter alone (SD / sqrt(n) per cell, both reads). Assumes
    nothing about the surface's shape."""
    stats = stats[stats.dye_uM > 0]
    grids = {t: g.set_index(["surfactant_mM", "dye_uM"]) for t, g in stats.groupby("timepoint_min")}
    times = sorted(grids)
    rows = []
    for prev, t in zip(times, times[1:]):
        a, b = grids[prev].align(grids[t], join="inner")
        scale = float((a.signal * b.signal).sum() / (a.signal ** 2).sum())
        resid = b.signal - scale * a.signal
        norm = float(np.sqrt((b.signal ** 2).sum()))
        noise = float(np.sqrt((b.sd.fillna(0) ** 2 / b.n + scale ** 2 * a.sd.fillna(0) ** 2 / a.n).sum()))
        biggest = resid.abs().idxmax()
        rows.append(dict(from_min=prev, to_min=t, scale=scale,
                         shape_change_pct=100 * float(np.sqrt((resid ** 2).sum())) / norm,
                         noise_pct=100 * noise / norm,
                         biggest_change_surfactant_mM=biggest[0], biggest_change_dye_uM=biggest[1]))
    return pd.DataFrame(rows)


def plot_surface_shape_change(change, out):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    hours = change.to_min / 60
    ax.plot(hours, change.shape_change_pct, marker="o", label="shape change vs previous read")
    ax.plot(hours, change.noise_pct, ls=":", color="0.4", label="expected from replicate scatter alone")
    for h, row in zip(hours, change.itertuples()):
        ax.annotate(f"{row.biggest_change_surfactant_mM:g} mM / {row.biggest_change_dye_uM:g} uM",
                    (h, row.shape_change_pct), textcoords="offset points", xytext=(5, 5), fontsize=6, color="0.4")
    ax.set(xlabel="read (h after dispensing); each point vs the previous read", ylabel="% of surface signal",
           title="Dye x SDS fluorescence surface: shape change between reads\n"
                 "(uniform fade removed; at the dotted line = no change beyond noise; label = cell that changed most)")
    ax.title.set_fontsize(9)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "3_dye_x_sds_fluorescence_3d_shape_change.png", dpi=150)
    plt.close(fig)


def plot_dye_matrix_html(stats, out):
    """Interactive 3D dye_matrix.html: signal over dye x surfactant, with a slider over read times."""
    import plotly.graph_objects as go
    times = sorted(stats.timepoint_min.unique())
    levels_s, levels_d = sorted(stats.surfactant_mM.unique()), sorted(stats.dye_uM.unique())
    zmax = float(stats.signal.max()) * 1.05
    fig = go.Figure()
    for t in times:
        g = stats[stats.timepoint_min == t]
        grid = g.pivot(index="surfactant_mM", columns="dye_uM", values="signal").reindex(
            index=levels_s, columns=levels_d)
        fig.add_trace(go.Surface(x=levels_d, y=levels_s, z=grid.to_numpy(float), colorscale="Viridis",
                                 cmin=0, cmax=zmax, opacity=0.85, visible=bool(t == times[0]),
                                 colorbar=dict(title="signal"), hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(
            x=g.dye_uM, y=g.surfactant_mM, z=g.signal, mode="markers", marker=dict(size=4, color="black"),
            visible=bool(t == times[0]), customdata=np.stack([g.sd, g.n], axis=1), showlegend=False,
            hovertemplate="dye %{x:g} uM<br>SDS %{y:g} mM<br>signal %{z:.3s} +/- %{customdata[0]:.2s}"
                          "<br>n=%{customdata[1]}<extra></extra>"))
    steps = [dict(method="update", label=f"{t:g} min",
                  args=[{"visible": [k // 2 == j for k in range(2 * len(times))]},
                        {"title": f"Dye x surfactant at {t:g} min (fluorescence minus dye-free background)"}])
             for j, t in enumerate(times)]
    fig.update_layout(title=f"Dye x surfactant at {times[0]:g} min (fluorescence minus dye-free background)",
                      scene=dict(xaxis_title="Dye (uM)", yaxis_title="SDS (mM)", zaxis_title="signal",
                                 zaxis_range=[min(0, float(stats.signal.min())), zmax]),
                      sliders=[dict(steps=steps, currentvalue=dict(prefix="Read: "))], height=750)
    fig.write_html(out / "3_dye_x_sds_fluorescence_3d.html", include_plotlyjs=True)


# ============================================================================
# 3. Stock check
# ============================================================================

# ASSUMED, not measured: 200 uL in a standard 96-well plate. Literature estimates span
# ~0.50-0.62 cm (7.15 mm well diameter -> 0.50 cm; GBO 175 uL = 0.5 cm -> ~0.57 cm).
ASSUMED_PATH_CM = 0.5


SPECTRUM_OVER_RANGE = -636.0  # value Gen5 writes to spectra.csv for an over-range point
FLUORESCENCE_LINEAR_MAX_UM = 4.0  # fixed-read fluorescence line is fitted to dye levels up to this


def _slope_through_origin(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope = float(np.sum(x * y) / np.sum(x ** 2))
    r2 = float(1 - np.sum((y - slope * x) ** 2) / np.sum((y - y.mean()) ** 2))
    return slope, r2


def stock_check(results, fluorescence, absorbance, epsilon, path_cm, nominal_stock_uM,
                block="stock_check"):
    """Dye in solvent, first read only (solvent evaporates). Blank = the 0 uM well.

    Absorbance: measured uM = (A - blank) / (epsilon * path); the slope of measured vs
    nominal (through the origin) times the nominal stock is the stock estimate.
    Fluorescence: slope over the low dye levels, and how far each well falls from it.
    """
    data = results[(results.block == block) & (results.timepoint_min == results.timepoint_min.min())]
    if data.empty:
        return {}, data
    data = data.sort_values("dye_uM").copy()
    blanks = data[data.dye_uM == 0]
    if blanks.empty:
        raise ValueError(f"Block {block!r} has no 0 uM (blank) well")
    data["absorbance_minus_blank"] = data[absorbance] - blanks[absorbance].mean()
    data["fluorescence_minus_blank"] = data[fluorescence] - blanks[fluorescence].mean()
    data["measured_uM"] = data.absorbance_minus_blank / (epsilon * 1e-6 * path_cm)
    dyed = data[data.dye_uM > 0]
    ratio, a_r2 = _slope_through_origin(dyed.dye_uM, dyed.measured_uM)
    low = dyed[dyed.dye_uM <= FLUORESCENCE_LINEAR_MAX_UM]
    f_slope, _ = _slope_through_origin(low.dye_uM, low.fluorescence_minus_blank)
    data["stock_estimate_uM"] = data.measured_uM / data.dye_uM.where(data.dye_uM > 0) * nominal_stock_uM
    data["fluorescence_vs_line"] = data.fluorescence_minus_blank / (f_slope * data.dye_uM.where(data.dye_uM > 0))
    summary = dict(epsilon_per_M_cm=epsilon, path_cm=path_cm, nominal_stock_uM=nominal_stock_uM,
                   actual_over_nominal=ratio, measured_stock_uM=ratio * nominal_stock_uM,
                   absorbance_r2=a_r2, fluorescence_slope_per_uM=f_slope,
                   fluorescence_linear_max_uM=FLUORESCENCE_LINEAR_MAX_UM)
    return summary, data


def plot_stock_check(data, spectra, summary, fluorescence, absorbance, dye, solvent, out):
    """2x2: absorbance spectra, fluorescence spectra, fixed fluorescence, stock estimate."""
    levels = sorted(data.dye_uM.unique())
    shade = {c: plt.cm.Blues(0.3 + 0.7 * i / max(1, len(levels) - 1)) for i, c in enumerate(levels)}
    fig, ((ax_as, ax_fs), (ax_f, ax_a)) = plt.subplots(2, 2, figsize=(13, 9.5))

    def spectrum_panel(ax, kind):
        rows = spectra[spectra.protocol.str.contains(kind)] if spectra is not None else spectra
        if rows is None or rows.empty:
            ax.text(0.5, 0.5, f"no {kind} spectrum read", ha="center", va="center", transform=ax.transAxes)
            return None
        protocol = rows.protocol.iloc[0]
        if rows.protocol.nunique() > 1:
            raise ValueError(f"More than one {kind} spectrum protocol: {sorted(rows.protocol.unique())}")
        blank = rows[rows.dye_uM == 0].groupby("wavelength_nm").value.mean()
        for c, g in rows.groupby("dye_uM"):
            g = g.sort_values("wavelength_nm")
            over = g.value == SPECTRUM_OVER_RANGE
            y = g.value.where(~over) - g.wavelength_nm.map(blank)
            ax.plot(g.wavelength_nm, y, color=shade[c], lw=2, label=f"{c:g} uM")
            if over.any():
                top = rows.value[rows.value != SPECTRUM_OVER_RANGE].max()
                ax.scatter(g.wavelength_nm[over], [top] * over.sum(), marker="x", color=shade[c], s=30, zorder=3)
        return protocol

    p = spectrum_panel(ax_as, "Abs")
    ax_as.axvline(float(absorbance.split("_")[-1]), color="0.4", lw=1, ls=":")
    ax_as.set(xlabel="wavelength (nm)", ylabel="absorbance minus blank", title=f"Absorbance spectra ({p})")
    ax_as.legend(title="nominal", fontsize=8)
    p = spectrum_panel(ax_fs, "Fluor")
    ax_fs.set(xlabel="emission (nm)", ylabel="RFU minus blank", title=f"Fluorescence spectra ({p}; x = over range)")

    dyed = data[data.dye_uM > 0]
    xs = np.linspace(0, dyed.dye_uM.max() * 1.05, 50)
    ax_f.plot(xs, summary["fluorescence_slope_per_uM"] * xs, color="0.3", lw=1, ls="--",
              label=f"line through wells <= {summary['fluorescence_linear_max_uM']:g} uM")
    ax_f.scatter(dyed.dye_uM, dyed.fluorescence_minus_blank, s=55, zorder=3, label="wells")
    for _, row in dyed.iterrows():
        ax_f.annotate(f"{row.fluorescence_vs_line:.0%}", (row.dye_uM, row.fluorescence_minus_blank),
                      textcoords="offset points", xytext=(6, -12), fontsize=8, color="0.3")
    ax_f.set(xlabel=f"nominal {dye} (uM)", ylabel=f"{fluorescence} minus blank",
             title="Fixed-wavelength fluorescence (label = % of line)")
    ax_f.legend(fontsize=8)

    top = dyed.dye_uM.max() * 1.1
    ratio = summary["actual_over_nominal"]
    ax_a.plot([0, top], [0, top], color="0.3", lw=1, ls="--", label="measured = nominal")
    ax_a.plot([0, top], [0, ratio * top], color="#eb6834", lw=1.5, label=f"fit, all wells: slope {ratio:.3f}")
    ax_a.scatter(dyed.dye_uM, dyed.measured_uM, s=55, zorder=3, label="wells")
    for _, row in dyed.iterrows():
        ax_a.annotate(f"{row.measured_uM:.1f} uM", (row.dye_uM, row.measured_uM),
                      textcoords="offset points", xytext=(6, -12), fontsize=8, color="0.3")
    ax_a.set(xlabel=f"nominal {dye} (uM)", ylabel=f"measured {dye} from {absorbance} (uM)")
    ax_a.set_title(f"Stock estimate {summary['measured_stock_uM']:.0f} uM (nominal {summary['nominal_stock_uM']:g})\n"
                   f"epsilon {summary['epsilon_per_M_cm']:,.0f} /M/cm, path ASSUMED {summary['path_cm']:.2f} cm",
                   fontsize=10)
    ax_a.legend(fontsize=8)
    fig.suptitle(f"{dye} in {solvent}, dilution series (first read)")
    fig.tight_layout()
    fig.savefig(out / "1_ethanol_series.png", dpi=150)
    plt.close(fig)


def plot_inner_filter(check, spectra, fluorescence, dye, solvent, out):
    """Inner-filter test on the solvent series: fixed-wavelength fluorescence, raw and
    corrected as F x 10^((A_ex + A_em) / 2) (Lakowicz), with A_ex and A_em read from each
    well's own absorbance spectrum. Ex/em wavelengths from the column name
    (fluorescence_<ex>_<em>). If the corrected points are linear, the bend is optical
    (inner filter), not quenching. Approximate for a top-read plate.
    """
    ex_nm, em_nm = (float(v) for v in fluorescence.split("_")[-2:])
    absorb = spectra[spectra.protocol.str.contains("Abs") & (spectra.value != SPECTRUM_OVER_RANGE)]
    blank = absorb[absorb.dye_uM == 0].set_index("wavelength_nm").value
    rows = []
    for d, g in absorb[absorb.dye_uM > 0].groupby("dye_uM"):
        g = (g.set_index("wavelength_nm").value - blank).dropna().sort_index()
        a_ex, a_em = np.interp([ex_nm, em_nm], g.index, g.values)
        f = float(check.loc[check.dye_uM == d, "fluorescence_minus_blank"].iloc[0])
        rows.append(dict(dye_uM=d, a_ex=a_ex, a_em=a_em, factor=10 ** ((a_ex + a_em) / 2), raw=f))
    t = pd.DataFrame(rows)
    t["corrected"] = t.raw * t.factor
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    xs = np.array([0, t.dye_uM.max() * 1.05])
    for ax, column, title in ((axes[0], "raw", "Raw: fluorescence minus blank"),
                              (axes[1], "corrected", f"Inner-filter corrected: F x 10^((A{ex_nm:g} + A{em_nm:g})/2)")):
        low = t[t.dye_uM <= FLUORESCENCE_LINEAR_MAX_UM]
        slope, _ = _slope_through_origin(low.dye_uM, low[column])
        ax.plot(xs, slope * xs, color="0.3", lw=1, ls="--", label=f"line through wells <= {FLUORESCENCE_LINEAR_MAX_UM:g} uM")
        ax.scatter(t.dye_uM, t[column], s=55, zorder=3, label="wells")
        for _, row in t.iterrows():
            ax.annotate(f"{row[column] / (slope * row.dye_uM):.0%}", (row.dye_uM, row[column]),
                        textcoords="offset points", xytext=(6, -12), fontsize=8, color="0.3")
        ax.set(xlabel=f"nominal {dye} in {solvent} (uM)", ylabel=fluorescence, title=f"{title}\n(label = % of line)")
        ax.legend(fontsize=8)
    axes[2].plot(t.dye_uM, t.factor, marker="o")
    axes[2].set(xlabel=f"nominal {dye} in {solvent} (uM)", ylabel="correction factor",
                title=f"Correction factor per well\n(A{ex_nm:g}, A{em_nm:g} from each well's absorbance spectrum)")
    for ax in axes:
        ax.title.set_fontsize(9)
    fig.suptitle(f"{dye} in {solvent}: is the fluorescence bend inner filter? Corrected linear = yes "
                 "(correction approximate for a top-read plate)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "1_ethanol_inner_filter.png", dpi=150)
    plt.close(fig)
    t.to_csv(out / "1_ethanol_inner_filter.csv", index=False)


# ============================================================================
# 4. Time course and entry point
# ============================================================================

MAX_RELATIVE_SE = 0.5  # a fitted value is "not resolved" (left off 2_cmc_over_time.png) if SE > this x value
TIME_COURSE_DYE_UM = 1.0  # dye level whose signal vs time is plotted (3_dye_x_sds_<dye>uM_over_time.png)


# Evaporation from the open wells (Fick's law, still air, whole plate). Order-of-magnitude estimate.
WELL_DEPTH_MM = 10.7          # ASSUMED standard 96-well depth; radius follows from ASSUMED_PATH_CM
D_WATER_M2_S = 2.5e-5         # water vapour in air, ~23 C
D_ETHANOL_M2_S = 1.2e-5       # ethanol vapour in air
ETHANOL_ACTIVITY_COEFF = 4.0  # ASSUMED, dilute ethanol in water (literature ~4-5)
GAS_R = 8.314


def evaporation_volumes(temp_c, rh, n_wells, water_ul, ethanol_ul, hours, dt_s=10.0):
    """Well volume (uL) at each time in hours. Water and ethanol diffuse through the empty well
    column and a still-air zone shared by all wells above the plate (a disc of the plate's
    area). Water activity ~ 1 - x_ethanol; ambient ethanol 0."""
    t_k = temp_c + 273.15
    area = (water_ul + ethanol_ul) * 1e-9 / (ASSUMED_PATH_CM * 1e-2)   # m2, liquid surface
    plate_radius = math.sqrt(0.0855 * 0.1278 / math.pi)
    c_sat_w = 611.2 * math.exp(17.62 * temp_c / (243.12 + temp_c)) * 0.018015 / (GAS_R * t_k)
    p_sat_e = 10 ** (8.20417 - 1642.89 / (230.3 + temp_c)) * 133.322
    m_w, m_e = water_ul * 1e-9 * 997.0, ethanol_ul * 1e-9 * 789.0
    end_s = float(np.max(hours)) * 3600 + dt_s
    times, volumes, t = [], [], 0.0

    def flux(c, d, column):
        return max(c, 0.0) / (column / (d * area) + n_wells / (4 * d * plate_radius))

    while t <= end_s:
        v = (m_w / 997.0 + m_e / 789.0) * 1e9
        times.append(t / 3600)
        volumes.append(v)
        column = WELL_DEPTH_MM * 1e-3 - v * 1e-9 / area
        x_e = (m_e / 0.04607) / (m_w / 0.018015 + m_e / 0.04607)
        m_w -= flux(c_sat_w * ((1 - x_e) - rh), D_WATER_M2_S, column) * dt_s
        m_e -= flux(x_e * ETHANOL_ACTIVITY_COEFF * p_sat_e * 0.04607 / (GAS_R * t_k), D_ETHANOL_M2_S, column) * dt_s
        t += dt_s
    return np.interp(np.asarray(hours, float), times, volumes)


def evaporation_factor(run_dir, config, results, hours, block="cmc"):
    """Real / nominal concentration at each time (nominal is over TOTAL_VOLUME_UL), and a note.

    Room temperature and humidity: the run's environmental_metadata (mean during the run).
    Ethanol on top: none if the dye went first with a pause after it in the block's batch
    (solvent assumed gone before water was added); otherwise the dye solution (DYE_VOLUME_UL)
    sits on top when DYE_SOLVENT_EVAPORATES, else it is inside the total.
    """
    env_file = Path(run_dir) / "environmental_metadata" / "summary.yaml"
    if not env_file.exists():
        raise ValueError(f"No {env_file} for the evaporation estimate")
    during = yaml.safe_load(env_file.read_text())["during_run"]
    temp_c, rh = float(during["temp_c"]["mean"]), float(during["humidity_pct"]["mean"]) / 100
    total, dye_ul = float(config["TOTAL_VOLUME_UL"]), float(config["DYE_VOLUME_UL"])
    batch = int(results.loc[results.block == block, "batch"].iloc[0])
    paused = any(p["batch"] == batch and p["after"] == "dye" for p in config["DISPENSE_PAUSES"])
    if config["DISPENSE_ORDER"][0] == "dye" and paused:
        ethanol_ul = 0.0
    elif config["DYE_SOLVENT_EVAPORATES"]:
        if config["DYE_SOLVENT"] != "ethanol":
            raise ValueError(f"No vapour data for solvent {config['DYE_SOLVENT']!r}")
        ethanol_ul = dye_ul
    else:
        ethanol_ul = 0.0
    n_wells = results.well_position.nunique()
    volumes = evaporation_volumes(temp_c, rh, n_wells, total, ethanol_ul, hours)
    note = (f"Dashed = evaporation-corrected: Fick's law, still air, {temp_c:.1f} C, {rh:.0%} RH, {n_wells} open "
            f"wells, {ethanol_ul:g} uL ethanol on top; concentration x {total:g} uL / V(t). Order-of-magnitude.")
    return total / volumes, note


REPLICATE_FROM_MIN = 240           # replicate deviations averaged over reads from here (after equilibration)
REPLICATE_MIN_SDS_OVER_CMC = 1.1   # CMC-block wells with SDS >= this x SURFACTANT_CMC_MM (plateau, steady signal)
_PIPET_TIPS_FILE = Path(__file__).resolve().parents[1] / "robot_state" / "pipet_tips.yaml"


def latest_validation_run(run_dir):
    """The last LIVE pipetting validation run (output/validate_all_calibrations) before this run,
    by the timestamp at the start of the folder names."""
    run_dir = Path(run_dir)
    root = run_dir.parents[1] / "validate_all_calibrations"
    stamp = run_dir.name[:15]
    earlier = sorted(p for p in root.glob("*_LIVE") if p.name[:15] < stamp)
    if not earlier:
        raise ValueError(f"No LIVE validation run in {root} before {stamp}")
    return earlier[-1]


def _tip_for(volume_ul):
    """Smallest tip whose range holds the volume (as North_Safe.select_pipet_tip)."""
    tips = yaml.safe_load(_PIPET_TIPS_FILE.read_text())
    ml = volume_ul / 1000
    largest = max(t["volume"] for t in tips.values())
    fits = [(t["min_suggested_volume"], name) for name, t in tips.items()
            if t["min_suggested_volume"] <= ml and (ml < t["volume"] or ml == t["volume"] == largest)]
    if not fits:
        raise ValueError(f"No tip for {volume_ul} uL in {_PIPET_TIPS_FILE}")
    return min(fits)[1]


def _volume_sd(validation, liquid, volume_ul):
    """SD (uL) at this volume from the validation run (linear between validated volumes).
    None if the volume is outside the validated range for that liquid and tip."""
    tip = _tip_for(volume_ul)
    files = list((validation / f"{liquid}_{tip}" / "calibration_validation").glob("*/volume_statistics.csv"))
    if len(files) != 1:
        raise ValueError(f"Expected one validation table for {liquid} {tip} in {validation}, found {len(files)}")
    v = pd.read_csv(files[0]).sort_values("target_volume_ul")
    if not v.target_volume_ul.min() <= volume_ul <= v.target_volume_ul.max():
        return None
    return float(np.interp(volume_ul, v.target_volume_ul, v.measured_std_ul))


def expected_replicate_spread(run_dir, config, wells):
    """Expected SD of a well's deviation from the mean of its replicates (%), from pipetting
    precision only: dye concentration = dye volume / aqueous volume (surfactant + water, plus
    solvent make-up unless the dye solvent evaporates); relative variances add; a deviation
    from the mean of n replicates has SD x sqrt((n - 1) / n). Volumes from planned_wells.csv,
    precision from the latest validation run before this run. NaN where a volume is outside
    the validated range (with the reason)."""
    validation = latest_validation_run(run_dir)
    planned = pd.read_csv(Path(run_dir) / "planned_wells.csv").set_index("well_position").round(2)  # float noise
    liquids = {"dye_ul": config["DYE_SOLVENT"], "surfactant_ul": config["SURFACTANT"], "water_ul": "water",
               "solvent_ul": config["DYE_SOLVENT"]}
    rows = []
    for s, g in wells.groupby("surfactant_mM"):
        p = planned.loc[g.well_position.iloc[0]]
        aqueous = ["surfactant_ul", "water_ul"] + ([] if config["DYE_SOLVENT_EVAPORATES"] else ["solvent_ul", "dye_ul"])
        sds, missing = {}, []
        for col in ["dye_ul"] + aqueous:
            if p[col] > 0:
                sd = _volume_sd(validation, liquids[col], float(p[col]))
                if sd is None:
                    missing.append(f"{p[col]:.0f} uL {liquids[col]}")
                sds[col] = sd
        if missing:
            rows.append(dict(surfactant_mM=s, expected_sd_pct=np.nan, note="not validated: " + ", ".join(missing)))
            continue
        v_aq = sum(float(p[c]) for c in aqueous)
        rel_var = (sds["dye_ul"] / float(p["dye_ul"])) ** 2 + sum(sds[c] ** 2 for c in aqueous if c in sds) / v_aq ** 2
        n = len(g)
        rows.append(dict(surfactant_mM=s, expected_sd_pct=100 * math.sqrt(rel_var) * math.sqrt((n - 1) / n), note=""))
    return pd.DataFrame(rows), validation


def replicate_deviations(results, fluorescence, expected_cmc_mM, block="cmc"):
    """Each plateau well's deviation from the mean of its replicates (%), per read from
    REPLICATE_FROM_MIN, summarised per well as mean and SD over reads."""
    c = results[(results.block == block) & (results.surfactant_mM >= REPLICATE_MIN_SDS_OVER_CMC * expected_cmc_mM)
                & (results.timepoint_min >= REPLICATE_FROM_MIN)].copy()
    mean = c.groupby(["surfactant_mM", "timepoint_min"])[fluorescence].transform("mean")
    c["deviation_pct"] = 100 * (c[fluorescence] - mean) / mean
    return (c.groupby(["surfactant_mM", "well_position"]).deviation_pct
            .agg(deviation_pct="mean", sd_over_reads_pct="std", reads="size").reset_index())


def plot_replicates(wells, expected, validation, out):
    levels = sorted(wells.surfactant_mM.unique())
    xpos = {s: i for i, s in enumerate(levels)}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for row in expected.itertuples():
        i = xpos[row.surfactant_mM]
        if np.isfinite(row.expected_sd_pct):
            ax.fill_between([i - 0.3, i + 0.3], -row.expected_sd_pct, row.expected_sd_pct, color="0.85", lw=0,
                            label="expected from pipetting (+/- 1 SD)" if i == 0 else None)
        else:
            ax.text(i, 0.02, row.note, ha="center", va="bottom", fontsize=6, color="0.4", transform=ax.get_xaxis_transform())
    x = wells.surfactant_mM.map(xpos)
    ax.errorbar(x, wells.deviation_pct, yerr=wells.sd_over_reads_pct, fmt="o", capsize=3, zorder=3)
    for xi, row in zip(x, wells.itertuples()):
        ax.annotate(row.well_position, (xi, row.deviation_pct), textcoords="offset points", xytext=(5, -3),
                    fontsize=7, color="0.3")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(range(len(levels)), [f"{s:g}" for s in levels])
    ax.set(xlabel="SDS (mM), CMC-block plateau wells", ylabel="well deviation from its replicates (%)",
           title=f"Replicate wells: each well's deviation, mean over reads from {REPLICATE_FROM_MIN / 60:g} h "
                 f"(error bar = SD over reads)\nspread of well deviations (SD) = {wells.deviation_pct.std():.1f}%")
    ax.title.set_fontsize(9)
    ax.legend(fontsize=8, loc="lower right")
    fig.text(0.01, 0.01, f"Grey band: dye, surfactant and water volume precision from {validation.name} "
                         "(SD at each planned volume and tip), combined for the dye concentration.",
             fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "5_replicates.png", dpi=150)
    plt.close(fig)


def plot_cmc_over_time(fits, out, factor=None, note=None):
    """CMC (binding model and sigmoid) and K at every read; unresolved fits left off."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    hours = fits.timepoint_min / 60
    axes[0].set_xlim(-0.5, hours.max() + 0.5)

    def resolved_errorbar(ax, value, se, marker, label, scale):
        ok = np.isfinite(fits[se]) & (fits[se] < MAX_RELATIVE_SE * fits[value].abs())
        dropped = int((~ok).sum())
        line = ax.errorbar(hours[ok], fits.loc[ok, value], yerr=fits.loc[ok, se], marker=marker, capsize=3,
                           label=label + (f" ({dropped} read(s) not resolved)" if dropped else ""))
        if scale is not None:
            ax.errorbar(hours[ok], (fits[value] * scale)[ok], yerr=(fits[se] * scale)[ok], marker=marker,
                        capsize=3, ls="--", mfc="white", color=line[0].get_color(),
                        label=f"{label}, evaporation-corrected")

    resolved_errorbar(axes[0], "cmc_mM", "cmc_mM_se", "o", "binding model", factor)
    resolved_errorbar(axes[0], "sigmoid_cmc_mM", "sigmoid_cmc_mM_se", "s", "sigmoid midpoint", factor)
    axes[0].set(ylabel="CMC (mM)", title="CMC per read")
    axes[0].legend(fontsize=7)
    resolved_errorbar(axes[1], "k_per_mM", "k_per_mM_se", "o", "binding model",
                      None if factor is None else 1 / factor)
    axes[1].set(ylabel="K (1/mM)", title="Dye-micelle binding constant K per read")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("Time after dispensing (h)")
        ax.title.set_fontsize(9)
    footer = (f"Error bars: fit standard error. Not resolved = SE > {MAX_RELATIVE_SE:.0%} of the value "
              "(too few points on the transition); those reads are left off.")
    if note:
        footer += "\n" + note
    fig.text(0.01, 0.01, footer, fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "2_cmc_over_time.png", dpi=150)
    plt.close(fig)


def plot_dye_signal_over_time(stats, out):
    """Signal of TIME_COURSE_DYE_UM dye at each surfactant level over time (mean +/- SD of wells)."""
    dye = stats[stats.dye_uM == TIME_COURSE_DYE_UM]
    if dye.empty:
        raise ValueError(f"No {TIME_COURSE_DYE_UM:g} uM dye wells for the dye signal over time plot")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, g in dye.groupby("surfactant_mM"):
        g = g.sort_values("timepoint_min")
        ax.errorbar(g.timepoint_min / 60, g.signal, yerr=g.sd, marker="o", capsize=3,
                    label=f"{s:g} mM (n={g.n.iloc[0]})")
    ax.set(xlabel="Time after dispensing (h)", ylabel="fluorescence minus dye-free background",
           title=f"{TIME_COURSE_DYE_UM:g} uM dye at each SDS level (mean +/- SD of wells)")
    ax.title.set_fontsize(9)
    ax.legend(title="SDS", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"3_dye_x_sds_{TIME_COURSE_DYE_UM:g}uM_over_time.png", dpi=150)
    plt.close(fig)


BASELINE_ABSORBANCE = "absorbance_750"  # off the dye peak (coumarin-6 in ethanol reads blank here)


def plot_absorbance(results, check, absorbance, out, block="dye_matrix"):
    """Absorbance of the water wells, 2x2. Row 1: absorbance at the dye peak minus the
    BASELINE_ABSORBANCE read, as % of the ethanol series' value for the same dye (same run,
    same substocks). Row 2: BASELINE_ABSORBANCE above the blank. Left: vs dye at
    DYE_MATRIX_PLOT_MIN per surfactant level; right: each well's first vs last read.
    Blank = median no-dye water well at each read.
    """
    dyed = check[check.dye_uM > 0]
    per_uM, _ = _slope_through_origin(dyed.dye_uM, dyed.absorbance_minus_blank)
    water = results[results.matrix == "water"]
    above = {}
    for column in (absorbance, BASELINE_ABSORBANCE):
        blank = water[water.dye_uM == 0].groupby("timepoint_min")[column].median()
        above[column] = water[column] - water.timepoint_min.map(blank)
    water = water.assign(
        peak_pct=100 * (above[absorbance] - above[BASELINE_ABSORBANCE]) / (per_uM * water.dye_uM.where(water.dye_uM > 0)),
        baseline_mOD=1000 * above[BASELINE_ABSORBANCE])
    levels_s = sorted(results.loc[results.block == block, "surfactant_mM"].unique())
    water = water[water.surfactant_mM.isin(levels_s)]
    colors = dict(zip(levels_s, plt.cm.viridis(np.linspace(0, 0.9, len(levels_s)))))
    first, last = water.timepoint_min.min(), water.timepoint_min.max()
    peak_nm, base_nm = absorbance.split("_")[-1], BASELINE_ABSORBANCE.split("_")[-1]
    # (column, wells, measured label, reference line, likely meaning at one read, likely meaning of change)
    rows = [("peak_pct", water[water.dye_uM > 0], f"A{peak_nm} - A{base_nm}, % of ethanol series", 100.0,
             "likely: dye in solution; 100% = all of it",
             "above the line: more dye in solution (e.g. taken up by micelles);\n"
             "below: dye leaving solution (e.g. precipitating or sticking to the plate)"),
            ("baseline_mOD", water, f"A{base_nm} minus blank (mOD)", 0.0,
             "likely: light scattering by particles (e.g. precipitate)",
             "above the line: more particles (e.g. precipitate forming);\nbelow: particles clearing")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for (column, data, label, reference, meaning, change), (ax_dye, ax_time) in zip(rows, axes):
        at = data[data.timepoint_min == DYE_MATRIX_PLOT_MIN]
        for s, g in at.groupby("surfactant_mM"):
            mean = g.groupby("dye_uM")[column].mean()
            ax_dye.scatter(g.dye_uM, g[column], s=18, color=colors[s], alpha=0.7)
            ax_dye.plot(mean.index, mean.values, color=colors[s], lw=1.5, label=f"{s:g} mM")
        ax_dye.axhline(reference, color="0.5", lw=1, ls=":")
        ax_dye.set(xlabel="Dye added (uM)", ylabel=label, title=f"{label} at {DYE_MATRIX_PLOT_MIN:g} min\n({meaning})")
        ax_dye.legend(title="SDS", fontsize=8)
        per_well = data.pivot_table(index=["well_position", "surfactant_mM", "dye_uM"], columns="timepoint_min",
                                    values=column).reset_index()
        sizes = {d: 15 + 12 * i for i, d in enumerate(sorted(per_well.dye_uM.unique()))}
        for s, g in per_well.groupby("surfactant_mM"):
            ax_time.scatter(g[first], g[last], s=g.dye_uM.map(sizes), color=colors[s], alpha=0.7,
                            edgecolor="white", label=f"{s:g} mM")
        values = per_well[[first, last]].to_numpy(float)
        low, high = min(0.0, float(np.nanmin(values))), max(reference * 1.1, float(np.nanmax(values)) * 1.05)
        ax_time.plot([low, high], [low, high], color="0.4", lw=1, ls="--", label="no change")
        for d, size in sizes.items():
            ax_time.scatter([], [], s=size, color="0.6", label=f"{d:g} uM dye")
        ax_time.set(xlabel=f"first read, {first:g} min", ylabel=f"last read, {last:g} min",
                    title=f"{label}: each well, first vs last read\n{change}")
        ax_time.legend(fontsize=7, ncol=2)
    for ax in axes.flat:
        ax.title.set_fontsize(9)
    fig.text(0.01, 0.005, f"Blank: median no-dye water well at each read. 100% = A{peak_nm} per uM of the ethanol "
                          f"series (same run). Coumarin-6 in ethanol does not absorb at {base_nm} nm.",
             fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / "3_dye_x_sds_absorbance.png", dpi=150)
    plt.close(fig)


# Spectrum band-shape plots: (protocol name contains, peak range nm, x range nm, y label, file, title, note)
SPECTRUM_MIN_FRACTION = 0.10  # water spectra with peak < this x the brightest at the same dye level are not drawn
SPECTRUM_SHAPES = [
    ("Abs", (420, 520), (400, 600), "absorbance minus blank, / peak", "3_dye_x_sds_absorbance_spectra.png",
     "Absorbance band shape", "Broader, extra band near 500 nm, or flat tail vs {solvent}: likely aggregates / particles"),
    ("Fluor", (490, 600), (490, 700), "emission minus blank, / peak", "3_dye_x_sds_fluorescence_spectra.png",
     "Emission band shape", "Shift vs {solvent}: different dye environment"),
]


def plot_spectrum_shapes(all_spectra, dye, solvent, out, block="dye_matrix"):
    """Normalised spectra of the water wells at DYE_MATRIX_PLOT_MIN, one panel per dye
    level, one line per surfactant level, with the solvent series at the same dye level.
    Each spectrum: mean of wells minus the median no-dye well of the same matrix and read,
    divided by its peak in the band. Spectra with an over-range point in the band are left
    out (their peak is unknown). One file per entry in SPECTRUM_SHAPES.
    """
    levels_s = sorted(all_spectra.loc[all_spectra.block == block, "surfactant_mM"].unique())
    colors = dict(zip(levels_s, plt.cm.viridis(np.linspace(0, 0.9, len(levels_s)))))
    for key, band_nm, x_nm, ylabel, filename, title, note in SPECTRUM_SHAPES:
        spectra = all_spectra[all_spectra.protocol.str.contains(key)]
        if spectra.protocol.nunique() != 1:
            raise ValueError(f"Expected one {key} spectrum protocol, found {sorted(spectra.protocol.unique())}")

        def normalised(rows, check_weak):
            over = rows[(rows.value == SPECTRUM_OVER_RANGE) & rows.wavelength_nm.between(*band_nm)]
            skip = set(zip(over.surfactant_mM, over.dye_uM))
            rows = rows[rows.value != SPECTRUM_OVER_RANGE]
            blank = rows[rows.dye_uM == 0].groupby("wavelength_nm").value.median()
            rows = rows[rows.dye_uM > 0].assign(v=rows.value - rows.wavelength_nm.map(blank))
            mean = rows.groupby(["surfactant_mM", "dye_uM", "wavelength_nm"]).v.mean().reset_index()
            mean = mean[[k not in skip for k in zip(mean.surfactant_mM, mean.dye_uM)]]
            band = mean[mean.wavelength_nm.between(*band_nm)]
            peak = band.groupby(["surfactant_mM", "dye_uM"]).v.max().rename("peak").reset_index()
            mean = mean.merge(peak)
            weak = set()
            if check_weak:
                low = peak.peak < SPECTRUM_MIN_FRACTION * peak.groupby("dye_uM").peak.transform("max")
                weak = set(zip(peak.surfactant_mM[low], peak.dye_uM[low]))
                mean = mean[[k not in weak for k in zip(mean.surfactant_mM, mean.dye_uM)]]
            return mean.assign(norm=mean.v / mean.peak), skip, weak

        water_rows = spectra[(spectra.matrix == "water") & (spectra.timepoint_min == DYE_MATRIX_PLOT_MIN)]
        if water_rows.empty:
            raise ValueError(f"No water-well {key} spectra at {DYE_MATRIX_PLOT_MIN} min")
        water, _, weak = normalised(water_rows, check_weak=True)
        ref, ref_skip, _ = normalised(spectra[(spectra.matrix == "solvent")
                                              & (spectra.timepoint_min == spectra.timepoint_min.min())],
                                      check_weak=False)
        water = water[water.surfactant_mM.isin(levels_s)]
        dyes = sorted(water.dye_uM.unique())
        fig, axes = plt.subplots(1, len(dyes), figsize=(4.2 * len(dyes), 4.3), sharey=True, squeeze=False)
        for ax, d in zip(axes[0], dyes):
            for s, g in water[water.dye_uM == d].groupby("surfactant_mM"):
                ax.plot(g.wavelength_nm, g.norm, color=colors[s], lw=1.8, label=f"{s:g} mM SDS")
            r = ref[ref.dye_uM == d]
            if not r.empty:
                ax.plot(r.wavelength_nm, r.norm, color="black", lw=1.5, ls="--", label=f"{solvent} (first read)")
            elif (0.0, d) in ref_skip:
                ax.text(0.98, 0.95, f"{solvent}: over range", transform=ax.transAxes, ha="right", va="top",
                        fontsize=7, color="0.3")
            hidden = sorted(s for s, dd in weak if dd == d and s in levels_s)
            if hidden:
                ax.text(0.98, 0.87, "too weak to show: " + ", ".join(f"{s:g}" for s in hidden) + " mM SDS",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7, color="0.3")
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xlim(*x_nm)
            ax.set_ylim(-0.2, 1.3)
            ax.set(xlabel="wavelength (nm)", title=f"{d:g} uM {dye}")
        axes[0, 0].set_ylabel(ylabel)
        axes[0, -1].legend(fontsize=7, loc="center right")
        fig.suptitle(f"{title} at {DYE_MATRIX_PLOT_MIN:g} min (each line divided by its own peak; lines with a peak under "
                     f"{SPECTRUM_MIN_FRACTION:.0%} of the brightest at that dye level not drawn). "
                     + note.format(solvent=solvent), fontsize=10)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)


def plot_water_controls(results, fluorescence, out):
    """Wells with no surfactant in water (any block), per dye level, over time."""
    data = results[(results.matrix == "water") & (results.surfactant_mM == 0)].copy()
    data["hours"] = data.timepoint_min / 60
    first = data.groupby("well_position")[fluorescence].transform(
        lambda v: v.loc[data.loc[v.index, "timepoint_min"].idxmin()])
    data["pct_of_first"] = 100 * data[fluorescence] / first
    # Reference: mean of the wells with the same dye at the highest surfactant level, same read.
    water = results[(results.matrix == "water") & (results.dye_uM > 0)]
    top = water[water.surfactant_mM == water.groupby("dye_uM").surfactant_mM.transform("max")]
    reference = top.groupby(["dye_uM", "timepoint_min"]).agg(
        ref=(fluorescence, "mean"), ref_mM=("surfactant_mM", "first"))
    data = data.join(reference, on=["dye_uM", "timepoint_min"])
    data["pct_of_micellar"] = 100 * data[fluorescence] / data.ref
    levels = sorted(data.dye_uM.unique())
    colors = dict(zip(levels, plt.cm.viridis(np.linspace(0, 0.9, len(levels)))))
    fig, (ax_abs, ax_rel, ax_mic) = plt.subplots(1, 3, figsize=(18, 4.8))
    for c, g in data.groupby("dye_uM"):
        mean = g.groupby("hours")[[fluorescence, "pct_of_first", "pct_of_micellar"]].mean()
        label = f"{c:g} uM dye ({g.well_position.nunique()} wells)"
        ax_abs.scatter(g.hours, g[fluorescence], s=12, color=colors[c], alpha=0.6)
        ax_abs.plot(mean.index, mean[fluorescence], color=colors[c], lw=1.5, label=label)
        if c > 0:
            ax_rel.scatter(g.hours, g.pct_of_first, s=12, color=colors[c], alpha=0.6)
            ax_rel.plot(mean.index, mean.pct_of_first, color=colors[c], lw=1.5, label=label)
            ax_mic.scatter(g.hours, g.pct_of_micellar, s=12, color=colors[c], alpha=0.6)
            ax_mic.plot(mean.index, mean.pct_of_micellar, color=colors[c], lw=1.5,
                        label=f"{c:g} uM dye (vs {g.ref_mM.iloc[0]:g} mM SDS)")
    ax_abs.set_yscale("log")
    ax_abs.set(xlabel="time after dispensing (h)", ylabel=f"{fluorescence} (log)",
               title="Dye in water, no surfactant")
    ax_abs.legend(fontsize=8)
    ax_rel.axhline(100, color="0.5", lw=1, ls=":")
    ax_rel.set(xlabel="time after dispensing (h)", ylabel="% of the well's first read",
               title="Same wells, relative to first read")
    ax_rel.legend(fontsize=8)
    ax_mic.set_yscale("log")
    ax_mic.set(xlabel="time after dispensing (h)", ylabel="% of micellar signal (log)",
               title="Same wells vs same dye at the highest SDS, same read")
    ax_mic.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "4_no_sds_controls.png", dpi=150)
    plt.close(fig)


def analyze(run_dir, fluorescence, absorbance=None, epsilon=None, path_cm=ASSUMED_PATH_CM, logger=None):
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
    fits.to_csv(out / "2_cmc_fits.csv", index=False)
    if not fits.empty:
        plot_cmc(results, fits, fluorescence, out)
        plot_cmc_sigmoid(results, fits, fluorescence, out)
        if np.isfinite(fits.cmc_mM).any():
            plot_cmc_method(results, fits, fluorescence, out)

    spectra_file = run_dir / "spectra.csv"
    all_spectra = pd.read_csv(spectra_file) if spectra_file.exists() else None

    stats, slopes = dye_matrix(results, fluorescence)
    if not stats.empty:
        stats.to_csv(out / "3_dye_x_sds_fluorescence.csv", index=False)
        slopes.to_csv(out / "3_dye_x_sds_slopes.csv", index=False)
        corrected = None
        if all_spectra is not None:
            corrected = inner_filter_corrected(results, all_spectra, fluorescence, DYE_MATRIX_PLOT_MIN)
        plot_dye_matrix(stats, out, corrected)
        plot_dye_matrix_html(stats, out)
        change = surface_shape_change(stats)
        if not change.empty:
            change.to_csv(out / "3_dye_x_sds_fluorescence_3d_shape_change.csv", index=False)
            plot_surface_shape_change(change, out)

    if (results.block == "stock_check").any():
        if absorbance is None or epsilon is None:
            raise ValueError("The stock check needs --absorbance and --epsilon")
        summary, check = stock_check(results, fluorescence, absorbance, epsilon, path_cm,
                                     config["DYE_STOCK_UM"])
        spectra = None
        if all_spectra is not None:
            spectra = all_spectra[(all_spectra.block == "stock_check")
                                  & (all_spectra.timepoint_min == all_spectra.timepoint_min.min())]
        pd.DataFrame([summary]).to_csv(out / "1_ethanol_series_summary.csv", index=False)
        check.to_csv(out / "1_ethanol_series_wells.csv", index=False)
        plot_stock_check(check, spectra, summary, fluorescence, absorbance,
                         config["DYE"], config["DYE_SOLVENT"], out)
        if spectra is not None:
            plot_inner_filter(check, spectra, fluorescence, config["DYE"], config["DYE_SOLVENT"], out)
        plot_absorbance(results, check, absorbance, out)
        if all_spectra is not None:
            plot_spectrum_shapes(all_spectra, config["DYE"], config["DYE_SOLVENT"], out)
    else:
        summary = {}

    if not fits.empty:
        factor, note = None, None
        if (run_dir / "environmental_metadata" / "summary.yaml").exists():
            factor, note = evaporation_factor(run_dir, config, results, fits.timepoint_min / 60)
            fits = fits.assign(evaporation_factor=factor, cmc_mM_evap_corrected=fits.cmc_mM * factor,
                               sigmoid_cmc_mM_evap_corrected=fits.sigmoid_cmc_mM * factor,
                               k_per_mM_evap_corrected=fits.k_per_mM / factor)
            fits.to_csv(out / "2_cmc_fits.csv", index=False)
        else:
            message = ("No environmental_metadata/summary.yaml (room temperature, humidity): "
                       "CMC not corrected for evaporation")
            logger.warning(message) if logger else print(f"WARNING: {message}")
        plot_cmc_over_time(fits, out, factor, note)
        replicates = replicate_deviations(results, fluorescence, config["SURFACTANT_CMC_MM"])
        if not replicates.empty:
            expected, validation = expected_replicate_spread(run_dir, config, replicates)
            replicates.merge(expected, on="surfactant_mM").to_csv(out / "5_replicates.csv", index=False)
            plot_replicates(replicates, expected, validation, out)
    if not stats.empty:
        plot_dye_signal_over_time(stats, out)
    plot_water_controls(results, fluorescence, out)
    if logger:
        logger.info(f"Analysis saved in {out}")
    return dict(cmc_fit=fits, dye_matrix=stats, dye_matrix_slopes=slopes, stock_check=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--fluorescence", required=True, help="results.csv column, e.g. fluorescence_485_530")
    parser.add_argument("--absorbance", help="results.csv column at the dye peak, e.g. absorbance_460")
    parser.add_argument("--epsilon", type=float, help="molar absorptivity at that wavelength (1/M/cm)")
    parser.add_argument("--path-cm", type=float, default=ASSUMED_PATH_CM, help=f"path length (default: assumed {ASSUMED_PATH_CM} cm)")
    args = parser.parse_args()
    tables = analyze(args.run_dir, args.fluorescence, args.absorbance, args.epsilon, args.path_cm)
    print(tables["cmc_fit"][["timepoint_min", "cmc_mM", "cmc_mM_se", "k_per_mM", "r2",
                             "sigmoid_cmc_mM", "sigmoid_cmc_mM_se", "sigmoid_r2"]]
          .round(3).to_string(index=False))
    if tables["stock_check"]:
        print({k: round(v, 4) for k, v in tables["stock_check"].items()})


if __name__ == "__main__":
    main()
