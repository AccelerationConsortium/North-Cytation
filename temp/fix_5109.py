import pandas as pd

path = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign\200uL\incremental_results.csv"
df = pd.read_csv(path)
df["row_index"] = df["row_index"].astype(int)

mask_5109 = df["row_index"] == 5109
idx_5109 = df[mask_5109].index.tolist()
print("DataFrame indices for row 5109:", idx_5109)
print("Timestamps:", df.loc[idx_5109, "timestamp"].tolist())

# Second occurrence has "redo" in filename -> relabel as redo
second = idx_5109[1]
df.loc[second, "run_type"] = "redo"

# First occurrence -> mark as failed (a redo was performed)
first = idx_5109[0]
df.loc[first, "status"] = "failed"

df.to_csv(path, index=False)

print("\nVerification:")
print(df[df["row_index"] == 5109][["row_index", "timestamp", "run_type", "status", "mass_data_file"]].to_string())

n_redo = len(df[df["run_type"] == "redo"])
expected = 5120 + n_redo
actual = len(df)
match = "OK" if expected == actual else f"MISMATCH (diff={actual - expected})"
print(f"\nrun_type counts: {df['run_type'].value_counts().to_dict()}")
print(f"status counts: {df['status'].value_counts().to_dict()}")
print(f"Total rows: {actual}")
print(f"Expected 5120 + {n_redo} redos = {expected}, actual = {actual} -> {match}")
