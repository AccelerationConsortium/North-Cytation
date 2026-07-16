"""Compare joint-linear vs Cartesian-linear-then-IK interpolation.

Both anchor at A (tip 0) and B (tip 47). Joint-linear failed physically at k=5, k=12.
This tests whether IK-in-Cartesian gives a physically straight line instead.
"""
from north import n9_kinematics as n9k

A = [668, 27786, 12526, 8620]   # tip 0
B = [986, 25530, 16028, 8620]   # tip 47 (diagonal opposing corner)

# FK both endpoints to get model (x, y)
xA, yA, thA = n9k.fk(A[0], A[1], A[2])
xB, yB, thB = n9k.fk(B[0], B[1], B[2])
print(f"Corner A (tip 0):  counts={A}")
print(f"  FK model coords: x={xA:.3f} y={yA:.3f} theta={thA:+.4f}rad")
print(f"Corner B (tip 47): counts={B}")
print(f"  FK model coords: x={xB:.3f} y={yB:.3f} theta={thB:+.4f}rad")
print(f"Model-frame delta: dx={xB-xA:+.3f}mm dy={yB-yA:+.3f}mm")
print(f"Model-frame distance: {((xB-xA)**2 + (yB-yA)**2)**0.5:.3f}mm")
print(f"Physical expected (2 cols x 9mm, 15 rows x 9mm): 136.202mm")
print()

# Interpolate in Cartesian, IK each point, pin gripper at A's value.
N = 16  # k = 0..15
print(f"{'k':>3} {'x_mm':>10} {'y_mm':>10} | {'g':>4} {'e':>7} {'s':>7} {'z':>5}")
print("-" * 60)

# Pick shoulder preference by comparing to A's shoulder to keep pose consistent
sA_rad = n9k.counts_to_rad(n9k.SHOULDER, A[2])

for k in range(N):
    t = k / (N - 1)
    tx = xA + t * (xB - xA)
    ty = yA + t * (yB - yA)

    sol_c = n9k.ik(tx, ty, shoulder_preference=n9k.SHOULDER_CENTER)
    sol_o = n9k.ik(tx, ty, shoulder_preference=n9k.SHOULDER_OUT)
    # Pick the one closer to A's shoulder (should stay CENTER throughout)
    chosen = sol_c if abs(sol_c[2] - sA_rad) <= abs(sol_o[2] - sA_rad) else sol_o

    g = int(A[0])  # PIN gripper to A's value
    e = int(n9k.rad_to_counts(n9k.ELBOW, chosen[1]))
    s = int(n9k.rad_to_counts(n9k.SHOULDER, chosen[2]))
    z = int(A[3])
    print(f"{k:>3} {tx:>10.3f} {ty:>10.3f} | {g:>4} {e:>7} {s:>7} {z:>5}")
