"""Diagnose the +Y jog: FK each of the counts the user got from sequential +Y jogs.

If the jog were correct in the model frame, we'd see x = const and y increasing by 9mm
each step. Any x-drift means the jog is physically diagonal in the model's coordinate
system.
"""
from north import n9_kinematics as n9k

jog_sequence = [
    ("pos 0 (origin)", [668, 27804, 12594, 8620]),
    ("pos 1 (+Y jog)", [668, 27944, 13192, 8620]),
    ("pos 2 (+Y jog)", [668, 28051, 13761, 8620]),
    ("pos 3 (+Y jog)", [668, 28127, 14301, 8620]),
    ("pos 4 (+Y jog)", [668, 28172, 14811, 8620]),
]

print("FK of the user's actual +Y jog sequence")
print("If jog were pure +Y in model frame: dx = 0, dy = +9 each step")
print()
header = "{:>16} | {:>10} {:>10} {:>10} | {:>7} {:>7}".format(
    "label", "x_mm", "y_mm", "theta_rad", "dx", "dy"
)
print(header)
print("-" * len(header))

prev = None
for label, counts in jog_sequence:
    g, e, s, z = counts
    x, y, theta = n9k.fk(g, e, s)
    if prev is None:
        dx_str, dy_str = "-", "-"
    else:
        dx = x - prev[0]
        dy = y - prev[1]
        dx_str = f"{dx:+.3f}"
        dy_str = f"{dy:+.3f}"
    print("{:>16} | {:>10.3f} {:>10.3f} {:>+10.4f} | {:>7} {:>7}".format(
        label, x, y, theta, dx_str, dy_str
    ))
    prev = (x, y, theta)
