import pandas as pd

THRESHOLD = 0.003

dfs = {}
for c in ["200uL", "1000uL"]:
    path = rf"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign\{c}\incremental_results.csv"
    df = pd.read_csv(path)
    ok = df[df["status"] == "ok"].copy()
    ok["date"] = ok["timestamp"].astype(str).str[:8]
    ok["flagged"] = (ok["pre_baseline_std"] > THRESHOLD) | (ok["post_baseline_std"] > THRESHOLD)
    dfs[c] = ok

# Get all dates that appear in either campaign
all_dates = sorted(set(dfs["200uL"]["date"].unique()) | set(dfs["1000uL"]["date"].unique()))

print(f"{'Date':<12} {'200uL flag':>12} {'200uL total':>12} {'200uL%':>8}  {'1000uL flag':>12} {'1000uL total':>12} {'1000uL%':>8}")
print("-" * 80)

for date in all_dates:
    row = {}
    for c in ["200uL", "1000uL"]:
        sub = dfs[c][dfs[c]["date"] == date]
        n_total = len(sub)
        n_flag = sub["flagged"].sum()
        row[c] = (n_flag, n_total)

    f200, t200 = row["200uL"]
    f1000, t1000 = row["1000uL"]

    pct200 = f"{100*f200/t200:.0f}%" if t200 else "N/A"
    pct1000 = f"{100*f1000/t1000:.0f}%" if t1000 else "N/A"

    t200_str = str(t200) if t200 else "not run"
    t1000_str = str(t1000) if t1000 else "not run"

    # Only print dates where either campaign has flags
    if f200 > 0 or f1000 > 0:
        print(f"{date:<12} {f200:>12} {t200_str:>12} {pct200:>8}  {f1000:>12} {t1000_str:>12} {pct1000:>8}")
