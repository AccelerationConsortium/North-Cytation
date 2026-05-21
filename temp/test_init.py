import sys; sys.path.insert(0, '.')
from workflows.surfactant_multidimensional_workflow import (
    SURFACTANTS, n_surfactants, INIT_STRATEGY,
    INIT_AXIS_PTS, INIT_FACE_PTS, INIT_INTERIOR_PTS,
    simplex_max_conc_mm, cube_max_conc_mm, MIN_CONC_MM,
    generate_simplex_init, generate_log_grid, is_feasible,
    SURFACTANT_BUDGET_UL, WELL_VOLUME_UL, SURFACTANT_LIBRARY,
)

def total_vol(pt):
    return sum(pt[s] * WELL_VOLUME_UL / SURFACTANT_LIBRARY[s]["stock_conc"] for s in SURFACTANTS)

print(f"SURFACTANTS ({n_surfactants}D): {SURFACTANTS}")
print(f"INIT_STRATEGY: {INIT_STRATEGY}")
print(f"Simplex vertex maxima: { {s: round(simplex_max_conc_mm[s], 2) for s in SURFACTANTS} }")
print(f"Cube maxima:           { {s: round(cube_max_conc_mm[s], 2) for s in SURFACTANTS} }")
print()

print("=== generate_simplex_init ===")
pts = generate_simplex_init()
print(f"Total points: {len(pts)}")
infeasible = [p for p in pts if not is_feasible(p)]
print(f"Infeasible points: {len(infeasible)}")
vols = [total_vol(p) for p in pts]
print(f"Max total vol: {max(vols):.3f} uL  (budget={SURFACTANT_BUDGET_UL} uL)")
print(f"Min total vol: {min(vols):.3f} uL")

# Show one axis-ray point and one interior point
print("\nSample axis ray pt (SDS varies):")
ax_pts = [p for p in pts if p[SURFACTANTS[1]] == MIN_CONC_MM and
          p[SURFACTANTS[2]] == MIN_CONC_MM and p[SURFACTANTS[3]] == MIN_CONC_MM]
for p in ax_pts[:2]:
    print(f"  {p}  vol={total_vol(p):.1f}")
