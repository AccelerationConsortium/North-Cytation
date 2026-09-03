import pandas as pd, os

BASE = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign"
df200 = pd.read_csv(os.path.join(BASE, "200uL", "incremental_results.csv"))
df1000 = pd.read_csv(os.path.join(BASE, "1000uL", "incremental_results.csv"))

param_cols = ["aspirate_speed", "dispense_speed", "aspirate_wait_time", "dispense_wait_time",
              "pre_asp_air_vol_uL", "post_asp_air_vol_uL", "overaspirate_vol_uL",
              "blowout_vol_uL", "retract_speed", "post_retract_wait_time"]

ok200 = df200[df200["status"] == "ok"].set_index("row_index")
ok1000 = df1000[df1000["status"] == "ok"].set_index("row_index")

print("Row 0 params - 200uL:", ok200.loc[0, param_cols].tolist())
print("Row 0 params - 1000uL:", ok1000.loc[0, param_cols].tolist())
print()

common = ok200.index.intersection(ok1000.index)
match = (ok200.loc[common, param_cols].values == ok1000.loc[common, param_cols].values).all()
print("Same param values for same row_index across campaigns:", match)
print()

vol200 = ok200["volume_ul_target"]
vol1000 = ok1000["volume_ul_target"]
print(f"200uL volume range: {vol200.min():.1f} - {vol200.max():.1f} uL")
print(f"1000uL volume range: {vol1000.min():.1f} - {vol1000.max():.1f} uL")
