import pandas as pd

for c in ["200uL", "1000uL"]:
    path = rf"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign\{c}\incremental_results.csv"
    df = pd.read_csv(path)
    df["row_index"] = df["row_index"].astype(int)

    print(f"=== {c} ===")
    print(f"Total rows: {len(df)}")
    run_counts = df["run_type"].value_counts().to_dict()
    status_counts = df["status"].value_counts().to_dict()
    print(f"run_type counts: {run_counts}")
    print(f"status counts: {status_counts}")

    originals = df[df["run_type"] == "original"]
    redos = df[df["run_type"] == "redo"]
    print(f"Unique original indices: {originals['row_index'].nunique()}")
    print(f"Unique redo indices: {redos['row_index'].nunique()}")

    # Check for duplicate originals
    dup_orig = originals[originals.duplicated("row_index", keep=False)]
    if len(dup_orig):
        dup_idx = sorted(dup_orig["row_index"].unique())
        print(f"WARNING: {len(dup_idx)} indices have duplicate originals: {dup_idx}")
    else:
        print("No duplicate originals - good")

    # Coverage check
    all_idx = set(range(5120))
    orig_idx = set(originals["row_index"].unique())
    missing = all_idx - orig_idx
    extra = orig_idx - all_idx
    if missing:
        print(f"Missing indices (no original): {sorted(missing)}")
    else:
        print("All 5120 indices have an original entry")
    if extra:
        print(f"Extra indices beyond 0-5119: {sorted(extra)}")

    # Redo distribution
    if len(redos):
        redo_counts = redos.groupby("row_index").size().value_counts().sort_index().to_dict()
        print(f"Redo distribution (redos per index): {redo_counts}")

    # Expected total
    n_redo = len(redos)
    expected = 5120 + n_redo
    actual = len(df)
    match = "OK" if expected == actual else f"MISMATCH (diff={actual - expected})"
    print(f"Expected 5120 + {n_redo} redos = {expected}, actual = {actual} -> {match}")
    print()
