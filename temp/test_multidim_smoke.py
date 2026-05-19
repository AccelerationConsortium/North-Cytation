"""Quick smoke test for the new N-D workflow's recommender + recipe path."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np, pandas as pd
import workflows.surfactant_multidimensional_workflow as m

np.random.seed(0)
pts = m.generate_log_grid(3)
print("Grid points:", len(pts))
print("INPUT_COLS:", m.INPUT_COLS)
print("MAX_CONC_MM:", m.MAX_CONC_MM)

rows = []
for i, p in enumerate(pts[:30]):
    sim = m.simulate_measurements_nd(p)
    row = {"wellplate_index": i, "well_type": "experiment"}
    for s in m.SURFACTANTS:
        row[f"{s}_conc_mm"] = p[s]
    row.update(sim)
    rows.append(row)
df = pd.DataFrame(rows)
print("Sim DF shape:", df.shape, "ratio range:", df["ratio"].min(), df["ratio"].max(),
      "turb range:", df["turbidity_600"].min(), df["turbidity_600"].max())

recs = m.get_next_batch(df, n_points=5)
print("REC count:", len(recs))
for r in recs:
    print("  ", {k: round(v, 4) for k, v in r.items()})
