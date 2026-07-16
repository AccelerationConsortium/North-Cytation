"""4-corner bilinear interpolation of pgrid_high_2.

Rack layout: 3 cols x 16 rows, column-major indexing: idx = col*16 + row.
"""

# User-jogged corner counts [g, e, s, z]
Q00 = [668, 27786, 12526, 8620]   # tip 0  (col=0, row=0)
Q20 = [753, 25447, 10066, 8620]   # tip 32 (col=2, row=0)
Q05 = [912, 27959, 18887, 8620]   # tip 15 (col=0, row=15)
Q25 = [986, 25705, 16213, 8620]   # tip 47 (col=2, row=15)

N_COLS = 3
N_ROWS = 16

def bilinear(col, row):
    a = col / (N_COLS - 1)   # 0..1 across columns
    b = row / (N_ROWS - 1)   # 0..1 across rows
    out = []
    for j in range(4):
        v = ((1 - a) * (1 - b) * Q00[j]
             + a * (1 - b) * Q20[j]
             + (1 - a) * b * Q05[j]
             + a * b * Q25[j])
        out.append(int(round(v)))
    return out

# Full grid, column-major
grid = []
for col in range(N_COLS):
    for row in range(N_ROWS):
        grid.append(bilinear(col, row))

# Print a few interior tips to test physically
test_tips = [
    (0, "tip 0  (col=0, row=0)  -- should match your jogged corner exactly"),
    (4, "tip 4  (col=0, row=4)  -- left column, 1/4 down"),
    (8, "tip 8  (col=0, row=8)  -- left column, middle"),
    (15, "tip 15 (col=0, row=15) -- should match jogged corner"),
    (16, "tip 16 (col=1, row=0)  -- middle column, top edge"),
    (20, "tip 20 (col=1, row=4)  -- middle column, 1/4 down"),
    (24, "tip 24 (col=1, row=8)  -- dead center of rack"),
    (28, "tip 28 (col=1, row=12) -- middle column, 3/4 down"),
    (31, "tip 31 (col=1, row=15) -- middle column, bottom"),
    (32, "tip 32 (col=2, row=0)  -- should match jogged corner"),
    (40, "tip 40 (col=2, row=8)  -- right column, middle"),
    (47, "tip 47 (col=2, row=15) -- should match jogged corner"),
]

print("Bilinear-interpolated pgrid_high_2 (test tips):")
print()
for idx, label in test_tips:
    print(f"  {idx:>3}: {grid[idx]}   # {label}")

print()
print("Full 48-position array:")
print(f"pgrid_high_2 = {grid}")
