"""
Grid generation analysis WITHOUT the north package.

Strategy:
  1. Use FK constants from north_location.py + standard SCARA formula
  2. Fit link lengths L1, L2 from rack_pip ground-truth data
     (rack_pip is an 8-col x 6-row grid, column-major, known 26 mm pitch)
  3. Use fitted model to simulate both algorithms:
       A) north_location  Grid.make_grid()  - fixed IK solution, rotation-aware
       B) generate_grid_array()             - adaptive IK solution
  4. Compare outputs to rack_pip ground truth
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import math

# ── FK constants from north_location.py ────────────────────────────────────
ELBOW_OFFSET            = 21250
SHOULDER_OFFSET         = 33667
ELBOW_COUNTS_PER_REV    = 51000
SHOULDER_COUNTS_PER_REV = 101000

def cts_to_rad_sh(cts):
    return (cts - SHOULDER_OFFSET) * 2 * math.pi / SHOULDER_COUNTS_PER_REV

def cts_to_rad_el(cts):
    return (cts - ELBOW_OFFSET) * 2 * math.pi / ELBOW_COUNTS_PER_REV

def rad_to_cts_sh(rad):
    return int(rad * SHOULDER_COUNTS_PER_REV / (2 * math.pi) + SHOULDER_OFFSET)

def rad_to_cts_el(rad):
    return int(rad * ELBOW_COUNTS_PER_REV / (2 * math.pi) + ELBOW_OFFSET)

# ── SCARA FK ─────────────────────────────────────────────────────────────────
# Standard 2-link planar arm:
#   shoulder = base joint   (inner link L1)
#   elbow    = forearm joint (outer link L2, angle relative to L1)
def fk(elbow_cts, shoulder_cts, L1, L2):
    th1 = cts_to_rad_sh(shoulder_cts)   # base joint
    th2 = cts_to_rad_el(elbow_cts)      # elbow relative to upper arm
    x = L1 * math.cos(th1) + L2 * math.cos(th1 + th2)
    y = L1 * math.sin(th1) + L2 * math.sin(th1 + th2)
    return x, y

# ── IK (two solutions) ───────────────────────────────────────────────────────
def ik(x, y, L1, L2, shoulder_preference=0):
    """Returns (elbow_rad, shoulder_rad) in radians for both IK solutions.
    shoulder_preference: 0 = center (elbow_up), 1 = out (elbow_down)
    Returns None if unreachable.
    """
    r2 = x*x + y*y
    cos_th2 = (r2 - L1*L1 - L2*L2) / (2*L1*L2)
    if abs(cos_th2) > 1.0:
        return None   # out of reach

    # Two solutions: +/- acos
    sin_th2_a = math.sqrt(1 - cos_th2*cos_th2)   # elbow "up"
    sin_th2_b = -sin_th2_a                         # elbow "down"

    solutions = []
    for sin_th2 in [sin_th2_a, sin_th2_b]:
        th2 = math.atan2(sin_th2, cos_th2)         # elbow angle (relative)
        # base angle
        k1 = L1 + L2*cos_th2
        k2 = L2*sin_th2
        th1 = math.atan2(y, x) - math.atan2(k2, k1)
        solutions.append((th2, th1))  # (elbow_rad, shoulder_rad)

    # shoulder_preference 0 → solution with smaller |th1|  ("center")
    # shoulder_preference 1 → solution with larger  |th1|  ("out")
    solutions.sort(key=lambda s: abs(s[1]))
    return solutions[0] if shoulder_preference == 0 else solutions[1]

# ── rack_pip ground truth (column-major, 8 cols x 6 rows) ───────────────────
rack_pip = [
    [2184, 6798, 35307, 16281], [2106, 7484, 34689, 16281], [2029, 8313, 34375, 16281],
    [1951, 9292, 34359, 16281], [1873, 10437, 34648, 16281], [1792, 11787, 35268, 16281],
    [2187, 5539, 32865, 16281], [2105, 6251, 32222, 16281], [2026, 7100, 31907, 16281],
    [1949, 8084, 31902, 16281], [1872, 9213, 32194, 16281], [1794, 10512, 32789, 16281],
    [2175, 4373, 30276, 16281], [2091, 5128, 29646, 16281], [2011, 6012, 29377, 16281],
    [1934, 7020, 29430, 16281], [1859, 8156, 29776, 16281], [1783, 9438, 30407, 16281],
    [2148, 3294, 27454, 16281], [2063, 4108, 26902, 16281], [1983, 5042, 26742, 16281],
    [1908, 6088, 26910, 16281], [1835, 7247, 27363, 16281], [1762, 8531, 28079, 16281],
    [2101, 2305, 24304, 16281], [2017, 3196, 23935, 16281], [1940, 4192, 23971, 16281],
    [1868, 5286, 24324, 16281], [1799, 6477, 24939, 16281], [1731, 7776, 25788, 16281],
    [2029, 1423, 20730, 16281], [1951, 2405, 20709, 16281], [1881, 3473, 21058, 16281],
    [1816, 4620, 21677, 16281], [1752, 5847, 22511, 16281], [1689, 7167, 23537, 16281],
    [1927,  678, 16673, 16281], [1864, 1762, 17234, 16281], [1806, 2903, 18032, 16281],
    [1750, 4101, 18999, 16281], [1695, 5364, 20105, 16281], [1639, 6706, 21344, 16281],
    [1794,  121, 12212, 16281], [1756, 1300, 13607, 16281], [1716, 2504, 14975, 16281],
    [1673, 3744, 16353, 16281], [1628, 5036, 17767, 16281], [1580, 6396, 19242, 16281],
]

N_COLS = 8
N_ROWS = 6
assert len(rack_pip) == N_COLS * N_ROWS

# ── Step 1: Fit L1 and L2 ────────────────────────────────────────────────────
# For a perfect grid the distance between every adjacent pair = PITCH.
# We also want row-steps and col-steps to be orthogonal.

PITCH = 26.0   # mm - user specified

def fit_error(params):
    L1, L2 = params
    if L1 < 50 or L2 < 50:
        return 1e9
    total = 0.0
    xy = []
    for pos in rack_pip:
        x, y = fk(pos[1], pos[2], L1, L2)
        xy.append((x, y))

    # Distance error between row-adjacent pairs
    for col in range(N_COLS):
        for row in range(N_ROWS - 1):
            i0 = col * N_ROWS + row
            i1 = col * N_ROWS + row + 1
            dx = xy[i1][0] - xy[i0][0]
            dy = xy[i1][1] - xy[i0][1]
            total += (math.sqrt(dx*dx + dy*dy) - PITCH)**2

    # Distance error between col-adjacent pairs
    for col in range(N_COLS - 1):
        for row in range(N_ROWS):
            i0 = col * N_ROWS + row
            i1 = (col+1) * N_ROWS + row
            dx = xy[i1][0] - xy[i0][0]
            dy = xy[i1][1] - xy[i0][1]
            total += (math.sqrt(dx*dx + dy*dy) - PITCH)**2

    # Orthogonality: row-step and col-step at origin should be perpendicular
    i00 = 0; i01 = 1; i10 = N_ROWS
    row_dx = xy[i01][0] - xy[i00][0]
    row_dy = xy[i01][1] - xy[i00][1]
    col_dx = xy[i10][0] - xy[i00][0]
    col_dy = xy[i10][1] - xy[i00][1]
    dot = row_dx*col_dx + row_dy*col_dy
    total += dot**2 * 0.1   # soft orthogonality constraint weight

    return total

print("Fitting SCARA link lengths from rack_pip data...")
result = differential_evolution(fit_error, bounds=[(50, 600), (50, 600)],
                                 seed=42, maxiter=2000, tol=1e-8, polish=True)
L1_fit, L2_fit = result.x
print(f"  L1 = {L1_fit:.2f} mm,  L2 = {L2_fit:.2f} mm  (residual={result.fun:.4f})")

# ── Step 2: Show FK results for all rack_pip positions ────────────────────────
print("\n── FK of rack_pip positions (fitted model) ──")
xy_gt = []
for idx, pos in enumerate(rack_pip):
    col = idx // N_ROWS
    row = idx %  N_ROWS
    x, y = fk(pos[1], pos[2], L1_fit, L2_fit)
    xy_gt.append((x, y))
    if col < 2 and row < 3:   # show first few
        print(f"  rack_pip[{idx:2d}] col={col} row={row}:  ({x:8.2f}, {y:8.2f}) mm"
              f"  [el={pos[1]} sh={pos[2]}]")

# Verify pitch
print("\n── Pitch verification (should all be ~26 mm) ──")
row_dists = []
col_dists = []
for col in range(N_COLS):
    for row in range(N_ROWS - 1):
        i0 = col * N_ROWS + row
        i1 = col * N_ROWS + row + 1
        dx = xy_gt[i1][0] - xy_gt[i0][0]
        dy = xy_gt[i1][1] - xy_gt[i0][1]
        row_dists.append(math.sqrt(dx*dx + dy*dy))
for col in range(N_COLS - 1):
    for row in range(N_ROWS):
        i0 = col * N_ROWS + row
        i1 = (col+1) * N_ROWS + row
        dx = xy_gt[i1][0] - xy_gt[i0][0]
        dy = xy_gt[i1][1] - xy_gt[i0][1]
        col_dists.append(math.sqrt(dx*dx + dy*dy))

print(f"  Row-step distances: mean={np.mean(row_dists):.2f} mm, "
      f"std={np.std(row_dists):.3f}, min={min(row_dists):.2f}, max={max(row_dists):.2f}")
print(f"  Col-step distances: mean={np.mean(col_dists):.2f} mm, "
      f"std={np.std(col_dists):.3f}, min={min(col_dists):.2f}, max={max(col_dists):.2f}")

# ── Step 3: Simulate generate_grid_array() ───────────────────────────────────
print("\n── Algorithm B: generate_grid_array() (adaptive IK) ──")
origin_pos = rack_pip[0]    # [gripper, elbow, shoulder, z]
x0, y0 = fk(origin_pos[1], origin_pos[2], L1_fit, L2_fit)
print(f"  Origin (rack_pip[0]) FK:  ({x0:.2f}, {y0:.2f}) mm")

# Determine x_offset and y_offset from the grid data
# row steps  → y_offset direction
# col steps  → x_offset direction
row_dx = xy_gt[1][0] - xy_gt[0][0]
row_dy = xy_gt[1][1] - xy_gt[0][1]
col_dx = xy_gt[N_ROWS][0] - xy_gt[0][0]
col_dy = xy_gt[N_ROWS][1] - xy_gt[0][1]
row_angle_deg = math.degrees(math.atan2(row_dy, row_dx))
col_angle_deg = math.degrees(math.atan2(col_dy, col_dx))
print(f"  Row-step direction: {row_angle_deg:.1f} deg  ({row_dx:.2f}, {row_dy:.2f}) mm per step")
print(f"  Col-step direction: {col_angle_deg:.1f} deg  ({col_dx:.2f}, {col_dy:.2f}) mm per step")

# Decompose col-step into x/y offsets  (col_dx = x_offset, col_dy contributes to y)
# For generate_grid_array, x_offset and y_offset are AXIS-ALIGNED offsets in mm
# We'll use the ACTUAL measured directions to see how well axis-aligned offsets work
x_offset = round(col_dx)   # nearest mm, X component of one col step
y_offset = round(row_dy)   # nearest mm, Y component of one row step

print(f"\n  Using x_offset={x_offset} mm, y_offset={y_offset} mm")

# Simulate generate_grid_array logic
SHOULDER_CENTER = 0
SHOULDER_OUT    = 1

init_shoulder_rad = cts_to_rad_sh(origin_pos[2])

gen_grid = []
last_shoulder_rad = init_shoulder_rad
errors_B = []

for col in range(N_COLS):
    for row in range(N_ROWS):
        target_x = x0 + col * x_offset
        target_y = y0 + row * y_offset

        sol_c = ik(target_x, target_y, L1_fit, L2_fit, SHOULDER_CENTER)
        sol_o = ik(target_x, target_y, L1_fit, L2_fit, SHOULDER_OUT)

        if sol_c is None and sol_o is None:
            gen_grid.append(None)
            continue

        if sol_c is None:
            chosen = sol_o
        elif sol_o is None:
            chosen = sol_c
        else:
            chosen = sol_c if abs(sol_c[1] - last_shoulder_rad) <= abs(sol_o[1] - last_shoulder_rad) else sol_o

        last_shoulder_rad = chosen[1]   # chosen[1] = shoulder_rad

        el_cts  = rad_to_cts_el(chosen[0])
        sh_cts  = rad_to_cts_sh(chosen[1])
        gen_grid.append([el_cts, sh_cts])

        # Compare to rack_pip ground truth
        gt_idx = col * N_ROWS + row
        gt = rack_pip[gt_idx]
        el_err = el_cts - gt[1]
        sh_err = sh_cts - gt[2]
        errors_B.append((col, row, el_err, sh_err))

print(f"\n  generate_grid_array() - error vs rack_pip ground truth:")
print(f"  {'col':>3} {'row':>3}  {'el_err':>8}  {'sh_err':>8}  {'gen_el':>7}  {'gt_el':>7}  {'gen_sh':>7}  {'gt_sh':>7}")
for col, row, el_err, sh_err in errors_B:
    gt_idx = col * N_ROWS + row
    gt = rack_pip[gt_idx]
    g = gen_grid[gt_idx]
    if g is not None:
        print(f"  {col:3d} {row:3d}  {el_err:8d}  {sh_err:8d}  {g[0]:7d}  {gt[1]:7d}  {g[1]:7d}  {gt[2]:7d}")

el_errs = [abs(e[2]) for e in errors_B if gen_grid[e[0]*N_ROWS+e[1]] is not None]
sh_errs = [abs(e[3]) for e in errors_B if gen_grid[e[0]*N_ROWS+e[1]] is not None]
print(f"\n  Elbow   error: mean={np.mean(el_errs):.1f}, max={max(el_errs):.1f} counts")
print(f"  Shoulder error: mean={np.mean(sh_errs):.1f}, max={max(sh_errs):.1f} counts")

# ── Step 4: Simulate north_location Grid.make_grid() ─────────────────────────
print("\n── Algorithm A: north_location Grid.make_grid() (fixed IK solution) ──")
# Fixed IK solution = whichever solution matches the origin
origin_sol = ik(x0, y0, L1_fit, L2_fit, SHOULDER_CENTER)
origin_sol_sh = origin_sol[1]

# north_location uses rotation=0 (no rotation), so X/Y offsets are axis-aligned
# But the grid X/Y axes may not be aligned with robot X/Y.
# north_location DOES support rotation - assume rot=0 for this comparison.

nl_grid = []
errors_A = []

for col in range(N_COLS):
    for row in range(N_ROWS):
        # Same offsets as Algorithm B
        target_x = x0 + col * x_offset
        target_y = y0 + row * y_offset

        # Fixed IK solution throughout (SHOULDER_CENTER)
        sol = ik(target_x, target_y, L1_fit, L2_fit, SHOULDER_CENTER)
        if sol is None:
            nl_grid.append(None)
            continue

        el_cts  = rad_to_cts_el(sol[0])
        sh_cts  = rad_to_cts_sh(sol[1])
        nl_grid.append([el_cts, sh_cts])

        gt_idx = col * N_ROWS + row
        gt = rack_pip[gt_idx]
        el_err = el_cts - gt[1]
        sh_err = sh_cts - gt[2]
        errors_A.append((col, row, el_err, sh_err))

print(f"  north_location Grid - error vs rack_pip ground truth:")
print(f"  {'col':>3} {'row':>3}  {'el_err':>8}  {'sh_err':>8}")
for col, row, el_err, sh_err in errors_A:
    print(f"  {col:3d} {row:3d}  {el_err:8d}  {sh_err:8d}")

el_errs_A = [abs(e[2]) for e in errors_A]
sh_errs_A = [abs(e[3]) for e in errors_A]
print(f"\n  Elbow   error: mean={np.mean(el_errs_A):.1f}, max={max(el_errs_A):.1f} counts")
print(f"  Shoulder error: mean={np.mean(sh_errs_A):.1f}, max={max(sh_errs_A):.1f} counts")

# ── Step 5: Did the adaptive solution flip at any point? ─────────────────────
print("\n── IK solution flips in generate_grid_array ──")
last_sh = init_shoulder_rad
for col in range(N_COLS):
    for row in range(N_ROWS):
        target_x = x0 + col * x_offset
        target_y = y0 + row * y_offset
        sol_c = ik(target_x, target_y, L1_fit, L2_fit, SHOULDER_CENTER)
        sol_o = ik(target_x, target_y, L1_fit, L2_fit, SHOULDER_OUT)
        if sol_c is None or sol_o is None:
            continue
        dist_c = abs(sol_c[1] - last_sh)
        dist_o = abs(sol_o[1] - last_sh)
        chosen = 'CENTER' if dist_c <= dist_o else 'OUT'
        if chosen == 'OUT':
            print(f"  FLIP at col={col} row={row}: chose OUT  "
                  f"(dist_C={dist_c:.4f} dist_O={dist_o:.4f})")
        last_sh = sol_c[1] if chosen == 'CENTER' else sol_o[1]

print("  (no output above = no flips detected)")

# ── Step 6: What X/Y offsets does the ground truth actually imply? ───────────
print("\n── What x_offset/y_offset do the rack_pip positions actually imply? ──")
print("   (if these are not axis-aligned, generate_grid_array will drift)")
print(f"  Actual row-step vector: ({row_dx:.2f}, {row_dy:.2f}) mm")
print(f"  Actual col-step vector: ({col_dx:.2f}, {col_dy:.2f}) mm")
print(f"  generate_grid_array uses axis-aligned: x_offset={x_offset}, y_offset={y_offset}")
print(f"  Discarded row X-component = {row_dx:.2f} mm (treated as 0)")
print(f"  Discarded col Y-component = {col_dy:.2f} mm (treated as 0)")
