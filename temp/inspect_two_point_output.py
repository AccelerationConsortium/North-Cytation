import pandas as pd

det = pd.read_csv("output/two_point_series_demo_details_20260602_172336.csv")
summ = pd.read_csv("output/two_point_series_demo_summary_20260602_172336.csv")

print("=== DETAIL shape:", det.shape, "===")
print(det.head(12).to_string())
print("\n... (showing first 12 of", len(det), "rows)")

print("\n=== SUMMARY (all rows) ===")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 20)
print(summ.to_string())

print("\n=== SUMMARY key columns ===")
key_cols = ["label", "target_volume_uL", "point1_mean_uL", "point1_shortfall_uL",
            "spread_uL", "point2_direction", "point2_mean_uL", "optimal_overaspirate_uL"]
print(summ[key_cols].to_string())

print("\n=== Liquids covered:", summ["liquid_name"].unique().tolist())
print("=== Volumes covered:", sorted(summ["target_volume_uL"].unique().tolist()))
print("=== Total summary rows:", len(summ), "(expected 30 = 6 liquids x 5 volumes)")
print("=== Total detail rows:", len(det), "(expected 180 = 30 x 2 points x 3 replicates)")
