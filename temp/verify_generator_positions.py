"""Verify fixed _compute_grid_entries would reproduce user's expected counts.

User's expected column (position 0 = origin, then 4 rows of +9mm Y):
    [668, 27804, 12594, 8620]  pos 0
    [668, 27944, 13192, 8620]  pos 1
    [668, 28051, 13761, 8620]  pos 2
    [668, 28127, 14301, 8620]  pos 3
    [668, 28172, 14811, 8620]  pos 4
"""
from north import n9_kinematics as n9k

origin = [668, 27804, 12594, 8620]
g0, e0, s0, z0 = origin
x0, y0, theta0 = n9k.fk(g0, e0, s0)
print(f"Origin counts: {origin}")
print(f"Origin FK: x={x0:.3f}, y={y0:.3f}, theta={theta0:+.4f}rad")
print(f"Origin gripper (PRESERVED for all cells): {g0}")
print()

user_expected = [
    [668, 27804, 12594, 8620],
    [668, 27944, 13192, 8620],
    [668, 28051, 13761, 8620],
    [668, 28127, 14301, 8620],
    [668, 28172, 14811, 8620],
]

last_shoulder_rad = n9k.counts_to_rad(n9k.SHOULDER, s0)
y_offset = 9.0

header = "{:>3} {:>10} {:>10} | {:>6} {:>7} {:>7} | {:>6} {:>7} {:>7} | {:>5} {:>5}".format(
    "row", "target_x", "target_y", "gen_g", "gen_e", "gen_s", "exp_g", "exp_e", "exp_s", "de", "ds"
)
print(header)
print("-" * len(header))

for row in range(5):
    target_x = x0
    target_y = y0 + row * y_offset

    sol_c = n9k.ik(target_x, target_y, shoulder_preference=n9k.SHOULDER_CENTER)
    sol_o = n9k.ik(target_x, target_y, shoulder_preference=n9k.SHOULDER_OUT)
    chosen = sol_c if abs(sol_c[2] - last_shoulder_rad) <= abs(sol_o[2] - last_shoulder_rad) else sol_o
    last_shoulder_rad = chosen[2]

    gen_g = int(g0)
    gen_e = int(n9k.rad_to_counts(n9k.ELBOW, chosen[1]))
    gen_s = int(n9k.rad_to_counts(n9k.SHOULDER, chosen[2]))

    exp = user_expected[row]
    de = gen_e - exp[1]
    ds = gen_s - exp[2]
    print("{:>3} {:>10.3f} {:>10.3f} | {:>6} {:>7} {:>7} | {:>6} {:>7} {:>7} | {:>+5} {:>+5}".format(
        row, target_x, target_y, gen_g, gen_e, gen_s, exp[0], exp[1], exp[2], de, ds
    ))
