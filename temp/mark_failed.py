import pandas as pd

for campaign, thresh in [("200uL", None), ("1000uL", None)]:
    path = rf"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign\{campaign}\incremental_results.csv"
    df = pd.read_csv(path)

    # Find all indices that have at least one redo
    redo_indices = set(df[df["run_type"] == "redo"]["row_index"].astype(int).unique())

    # Mark the "original" entry as failed for those indices
    mask = (df["run_type"] == "original") & (df["row_index"].astype(int).isin(redo_indices))
    before = (df["status"] == "failed").sum()
    df.loc[mask, "status"] = "failed"
    after = (df["status"] == "failed").sum()

    df.to_csv(path, index=False)
    print(f"{campaign}: marked {after - before} originals as failed")
    print(f"  status counts: {df['status'].value_counts().to_dict()}")

    # Report indices with 3 entries
    counts = df["row_index"].value_counts()
    triple = sorted(counts[counts >= 3].index.tolist())
    if triple:
        print(f"  Indices with 3 entries (original + 2 redos): {triple}")
    print()
