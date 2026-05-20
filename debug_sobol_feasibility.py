"""
Debug: visualise Sobol point coverage and actual vs. assumed feasibility boundaries.

Two plots, side by side (same Sobol points in [0.01, 45]^2 log-space):
  Left  — points coloured by the STOCK-based feasibility check (what is_feasible() uses)
  Right — points coloured by the ACTUAL-substock-based feasibility check
          (what filter_points_by_actual_volumes() uses)

No contour maps, just dots + boundary lines.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import Sobol

# ── Hardware constants (must match workflow config) ────────────────────────────
STOCK_CONC_SDS   = 50.0   # mM
STOCK_CONC_TTAB  = 50.0   # mM
WELL_VOLUME_UL   = 250.0
BUDGET_UL        = 225.0  # dispensing budget per well
MIN_PIPETTE_UL   = 10.0   # minimum pipettable volume
NUM_SUBSTOCKS    = 6      # substocks per surfactant
MIN_CONC_MM      = 0.01
N_SURFACTANTS    = 2      # match actual workflow (SDS, TTAB)

# Corrected simplex max: other (N-1) surfactants each need a minimum pipettable
# volume at MIN_CONC_MM, leaving less budget for the surfactant on each axis.
_MIN_VOL_PER_OTHER = WELL_VOLUME_UL / 10.0   # 25 uL (from lowest substock at 10*MIN_CONC_MM)
_SIMPLEX_BUDGET_FOR_ONE = max(BUDGET_UL - (N_SURFACTANTS - 1) * _MIN_VOL_PER_OTHER, 0.0)
MAX_CONC_MM = STOCK_CONC_SDS * (_SIMPLEX_BUDGET_FOR_ONE / WELL_VOLUME_UL)  # ~30 mM for 4 surfs
CUBE_MAX    = STOCK_CONC_SDS * ((BUDGET_UL / N_SURFACTANTS) / WELL_VOLUME_UL)  # ~11.25 mM

print(f"N_SURFACTANTS={N_SURFACTANTS}, simplex max={MAX_CONC_MM:.1f} mM, cube max={CUBE_MAX:.1f} mM")
print(f"Minimum vol reserved for other {N_SURFACTANTS-1} surfs: {(N_SURFACTANTS-1)*_MIN_VOL_PER_OTHER:.0f} uL")

N_POINTS = 512    # Sobol draws (must be power of 2)

# ── Substock series (mirrors calculate_systematic_dilution_series) ─────────────
def get_substock_series(stock_conc, min_target=MIN_CONC_MM, n=NUM_SUBSTOCKS):
    """Return sorted list of substock concentrations the planner will create."""
    series_max = stock_conc * 0.20          # 10 mM for 50 mM stock
    series_min = min_target * 10            # 0.10 mM
    log_pts = np.linspace(np.log10(series_max), np.log10(series_min), n)
    raw = 10.0 ** log_pts

    def round_nice(v):
        if v <= 0:
            return 0.0
        mag = 10 ** np.floor(np.log10(v))
        norm = v / mag
        for threshold, nice in [(1.2, 1.0), (1.8, 1.5), (2.5, 2.0), (3.5, 3.0),
                                 (4.5, 4.0), (6.0, 5.0), (8.0, 7.0), (11.0, 10.0)]:
            if norm <= threshold:
                return nice * mag
        return 10.0 * mag

    seen, out = set(), []
    for v in raw:
        r = round_nice(v)
        if r not in seen and r < stock_conc:
            seen.add(r)
            out.append(r)
    return sorted(out, reverse=True)   # highest first

SUBSTOCKS_SDS  = get_substock_series(STOCK_CONC_SDS)
SUBSTOCKS_TTAB = get_substock_series(STOCK_CONC_TTAB)
print(f"SDS  substocks: {[f'{c:.2g}' for c in SUBSTOCKS_SDS]}")
print(f"TTAB substocks: {[f'{c:.2g}' for c in SUBSTOCKS_TTAB]}")

# ── Volume per well: OLD greedy (highest vol per surf independently) ───────────
def actual_vol_ul(target_c, stock_conc, substocks):
    best_vol = None
    for src in substocks + [stock_conc]:
        vol = target_c * WELL_VOLUME_UL / src
        if MIN_PIPETTE_UL <= vol <= BUDGET_UL:
            if best_vol is None or vol > best_vol:
                best_vol = vol
    return best_vol

# ── Joint feasibility: NEW method (maximise min vol across surfs jointly) ──────
def joint_feasible(c_sds, c_ttab):
    """
    Enumerate all (source_sds, source_ttab) combinations.
    Keep only those where total <= BUDGET_UL and both vols >= MIN_PIPETTE_UL.
    Among those, pick the one maximising min(vol_sds, vol_ttab).
    Returns True if any valid combo exists.
    """
    import itertools
    all_sds  = SUBSTOCKS_SDS  + [STOCK_CONC_SDS]
    all_ttab = SUBSTOCKS_TTAB + [STOCK_CONC_TTAB]
    best_score = (-1.0, -1.0)
    found = False
    for src_sds, src_ttab in itertools.product(all_sds, all_ttab):
        v_sds  = c_sds  * WELL_VOLUME_UL / src_sds
        v_ttab = c_ttab * WELL_VOLUME_UL / src_ttab
        if v_sds < MIN_PIPETTE_UL or v_ttab < MIN_PIPETTE_UL:
            continue
        if v_sds > BUDGET_UL or v_ttab > BUDGET_UL:
            continue
        if v_sds + v_ttab > BUDGET_UL:
            continue
        score = (min(v_sds, v_ttab), v_sds + v_ttab)
        if score > best_score:
            best_score = score
            found = True
    return found

# ── Stock-based feasibility (is_feasible in the workflow) ─────────────────────
def stock_feasible(c_sds, c_ttab):
    vol = (c_sds / STOCK_CONC_SDS + c_ttab / STOCK_CONC_TTAB) * WELL_VOLUME_UL
    return vol <= BUDGET_UL + 1.0

# ── Actual-substock feasibility (filter_points_by_actual_volumes) ─────────────
def actual_feasible(c_sds, c_ttab):
    v_sds  = actual_vol_ul(c_sds,  STOCK_CONC_SDS,  SUBSTOCKS_SDS)
    v_ttab = actual_vol_ul(c_ttab, STOCK_CONC_TTAB, SUBSTOCKS_TTAB)
    if v_sds is None or v_ttab is None:
        return False   # no valid source → can't be dispensed at all
    return (v_sds + v_ttab) <= BUDGET_UL

# ── Generate Sobol in log-space ───────────────────────────────────────────────
raw = Sobol(d=2, scramble=True, seed=42).random(N_POINTS)
lo, hi = np.log10(MIN_CONC_MM), np.log10(MAX_CONC_MM)
pts = 10 ** (lo + raw * (hi - lo))

# ── Classify every point ──────────────────────────────────────────────────────
stock_ok  = np.array([stock_feasible(p[0], p[1])  for p in pts])
actual_ok = np.array([actual_feasible(p[0], p[1]) for p in pts])
joint_ok  = np.array([joint_feasible(p[0], p[1])  for p in pts])

both_ok              = stock_ok  &  actual_ok
stock_but_not_actual = stock_ok  & ~actual_ok
neither              = ~stock_ok & ~actual_ok

print(f"\nOut of {N_POINTS} points [0.01-45]:")
print(f"  Stock-based OK       : {stock_ok.sum()}")
print(f"  Old greedy OK        : {actual_ok.sum()}")
print(f"  NEW joint-select OK  : {joint_ok.sum()}")
print(f"  Old rejected (orange): {stock_but_not_actual.sum()}")
print(f"  Joint rejected       : {(stock_ok & ~joint_ok).sum()}")

# ── Boundary lines ─────────────────────────────────────────────────────────────
c_vals = np.logspace(np.log10(MIN_CONC_MM), np.log10(MAX_CONC_MM), 500)

# Stock boundary: c_sds + c_ttab = BUDGET/WELL * STOCK  (both stocks same)
stock_limit = BUDGET_UL / WELL_VOLUME_UL * STOCK_CONC_SDS   # = 45.0 mM
ttab_stock_bnd = stock_limit - c_vals
m_stock = (ttab_stock_bnd > MIN_CONC_MM) & (c_vals > MIN_CONC_MM)

# Actual boundary: scan along SDS axis and find max TTAB
def max_ttab_for_sds(c_sds):
    v_sds = actual_vol_ul(c_sds, STOCK_CONC_SDS, SUBSTOCKS_SDS)
    if v_sds is None or v_sds >= BUDGET_UL:
        return None
    remaining = BUDGET_UL - v_sds
    # Find max TTAB such that actual_vol_ul(c_ttab) <= remaining
    lo_t, hi_t = MIN_CONC_MM, MAX_CONC_MM
    for _ in range(60):
        mid = (lo_t + hi_t) / 2
        v = actual_vol_ul(mid, STOCK_CONC_TTAB, SUBSTOCKS_TTAB)
        if v is None or v > remaining:
            hi_t = mid
        else:
            lo_t = mid
    return lo_t

actual_bnd = np.array([max_ttab_for_sds(c) for c in c_vals], dtype=object)
m_actual = np.array([v is not None and v > MIN_CONC_MM for v in actual_bnd])
c_actual_bnd  = c_vals[m_actual]
ttab_actual_bnd = np.array(actual_bnd[m_actual], dtype=float)

# ── Cube-bounded Sobol (cube max per axis) ────────────────────────────────────
raw_cube = Sobol(d=2, scramble=True, seed=42).random(N_POINTS)
lo_c, hi_c = np.log10(MIN_CONC_MM), np.log10(CUBE_MAX)
pts_cube = 10 ** (lo_c + raw_cube * (hi_c - lo_c))

actual_ok_cube = np.array([actual_feasible(p[0], p[1]) for p in pts_cube])
bad_cube       = ~actual_ok_cube
print(f"\n22.5x22.5 actual substock check: {actual_ok_cube.sum()} feasible, {bad_cube.sum()} rejected")

# Boundary lines clipped to 22.5
c_cube = np.logspace(np.log10(MIN_CONC_MM), np.log10(CUBE_MAX), 500)
ttab_stock_cube = stock_limit - c_cube
m_stock_cube = (ttab_stock_cube > MIN_CONC_MM) & (c_cube > MIN_CONC_MM)
actual_bnd_cube = np.array([max_ttab_for_sds(c) for c in c_cube], dtype=object)
m_actual_cube = np.array([v is not None and v > MIN_CONC_MM for v in actual_bnd_cube])
c_act_c = c_cube[m_actual_cube]
ttab_act_c = np.array(actual_bnd_cube[m_actual_cube], dtype=float)

# ── Joint-select classifications ──────────────────────────────────────────────
joint_ok_cube = np.array([joint_feasible(p[0], p[1]) for p in pts_cube])
joint_bad_cube = ~joint_ok_cube
print(f"22.5x22.5 joint-select: {joint_ok_cube.sum()} feasible, {joint_bad_cube.sum()} rejected")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(26, 6))

# Panel 1: stock-based check, full range
ax = axes[0]
ax.scatter(pts[ stock_ok, 0], pts[ stock_ok, 1], c="steelblue", s=14, alpha=0.7,
           label=f"feasible by stock ({stock_ok.sum()})")
ax.scatter(pts[~stock_ok, 0], pts[~stock_ok, 1], c="tomato",    s=14, alpha=0.7,
           label=f"infeasible ({(~stock_ok).sum()})")
ax.plot(c_vals[m_stock], ttab_stock_bnd[m_stock], "k--", lw=1.5, label="stock boundary")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(MIN_CONC_MM*0.8, MAX_CONC_MM*1.2); ax.set_ylim(MIN_CONC_MM*0.8, MAX_CONC_MM*1.2)
ax.set_xlabel("SDS (mM)"); ax.set_ylabel("TTAB (mM)")
ax.set_title(f"1. is_feasible() — stock only\n[0.01-45] ({N_POINTS} pts)")
ax.legend(fontsize=8)

# Panel 2: old greedy check, full range
ax = axes[1]
ax.scatter(pts[both_ok, 0],              pts[both_ok, 1],              c="steelblue", s=14, alpha=0.7,
           label=f"feasible old greedy ({both_ok.sum()})")
ax.scatter(pts[stock_but_not_actual, 0], pts[stock_but_not_actual, 1], c="orange",    s=14, alpha=0.8,
           label=f"REJECTED old greedy ({stock_but_not_actual.sum()})")
ax.scatter(pts[neither, 0],              pts[neither, 1],              c="tomato",    s=14, alpha=0.7,
           label=f"infeasible both ({neither.sum()})")
ax.plot(c_vals[m_stock], ttab_stock_bnd[m_stock], "k--", lw=1.2, label="stock boundary")
ax.plot(c_actual_bnd, ttab_actual_bnd,             "r-",  lw=2.0, label="old greedy boundary")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(MIN_CONC_MM*0.8, MAX_CONC_MM*1.2); ax.set_ylim(MIN_CONC_MM*0.8, MAX_CONC_MM*1.2)
ax.set_xlabel("SDS (mM)"); ax.set_ylabel("TTAB (mM)")
ax.set_title(f"2. Old greedy (per-surf max vol)\n[0.01-45] ({N_POINTS} pts)")
ax.legend(fontsize=8)

# Panel 3: NEW joint-select check, full range
joint_bad_full = stock_ok & ~joint_ok
ax = axes[2]
ax.scatter(pts[joint_ok, 0],       pts[joint_ok, 1],       c="steelblue", s=14, alpha=0.7,
           label=f"feasible joint ({joint_ok.sum()})")
ax.scatter(pts[joint_bad_full, 0], pts[joint_bad_full, 1], c="orange",    s=14, alpha=0.8,
           label=f"REJECTED joint ({joint_bad_full.sum()})")
ax.scatter(pts[neither, 0],        pts[neither, 1],        c="tomato",    s=14, alpha=0.7,
           label=f"infeasible both ({neither.sum()})")
ax.plot(c_vals[m_stock], ttab_stock_bnd[m_stock], "k--", lw=1.2, label="stock boundary")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(MIN_CONC_MM*0.8, MAX_CONC_MM*1.2); ax.set_ylim(MIN_CONC_MM*0.8, MAX_CONC_MM*1.2)
ax.set_xlabel("SDS (mM)"); ax.set_ylabel("TTAB (mM)")
ax.set_title(f"3. NEW joint select\n[0.01-45] ({N_POINTS} pts)")
ax.legend(fontsize=8)

# Panel 4: NEW joint-select, cube bounds 22.5 x 22.5
ax = axes[3]
ax.scatter(pts_cube[joint_ok_cube, 0],  pts_cube[joint_ok_cube, 1],  c="steelblue", s=14, alpha=0.7,
           label=f"feasible joint ({joint_ok_cube.sum()})")
ax.scatter(pts_cube[joint_bad_cube, 0], pts_cube[joint_bad_cube, 1], c="orange",    s=14, alpha=0.8,
           label=f"REJECTED joint ({joint_bad_cube.sum()})")
ax.plot(c_cube[m_stock_cube], ttab_stock_cube[m_stock_cube], "k--", lw=1.2, label="stock boundary")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(MIN_CONC_MM*0.8, CUBE_MAX*1.2); ax.set_ylim(MIN_CONC_MM*0.8, CUBE_MAX*1.2)
ax.set_xlabel("SDS (mM)"); ax.set_ylabel("TTAB (mM)")
ax.set_title(f"4. NEW joint select\n[0.01-22.5] cube bounds ({N_POINTS} pts)")
ax.legend(fontsize=8)

plt.suptitle(
    "Old greedy vs NEW joint source selection\n"
    "Orange = passes stock check but gets rejected",
    fontsize=11
)
plt.tight_layout()
plt.savefig("debug_sobol_feasibility.png", dpi=150)
plt.show()
print("Saved: debug_sobol_feasibility.png")
