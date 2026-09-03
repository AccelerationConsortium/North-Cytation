import pandas as pd
import numpy as np
import os

BASE = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign"

df200 = pd.read_csv(os.path.join(BASE, "200uL", "incremental_results.csv"))
df1000 = pd.read_csv(os.path.join(BASE, "1000uL", "incremental_results.csv"))

param_cols = ["aspirate_speed", "dispense_speed", "aspirate_wait_time", "dispense_wait_time",
              "pre_asp_air_vol_uL", "post_asp_air_vol_uL", "overaspirate_vol_uL",
              "blowout_vol_uL", "retract_speed", "post_retract_wait_time"]

ok200 = df200[df200["status"] == "ok"].set_index("row_index")[param_cols + ["volume_ul_target", "accuracy_pct"]]
ok1000 = df1000[df1000["status"] == "ok"].set_index("row_index")[param_cols + ["volume_ul_target", "accuracy_pct"]]

# Normalize each param to [0,1] within each campaign and compare
def normalize(df, cols):
    out = df[cols].copy().astype(float)
    for c in cols:
        mn, mx = out[c].min(), out[c].max()
        if mx > mn:
            out[c] = (out[c] - mn) / (mx - mn)
    return out

n200 = normalize(ok200, param_cols)
n1000 = normalize(ok1000, param_cols)

common = n200.index.intersection(n1000.index)
diff = (n200.loc[common] - n1000.loc[common]).abs()
print("Max normalized difference per param (0=identical Sobol sequence):")
print(diff.max().to_string())
print()
print("Mean normalized difference per param:")
print(diff.mean().round(4).to_string())
