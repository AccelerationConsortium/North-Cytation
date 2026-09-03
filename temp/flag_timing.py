import pandas as pd
import os

BASE = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign"
THRESHOLD = 0.003

# Key dates where the campaigns showed different flag rates
KEY_DATES = ["20260506", "20260507", "20260508", "20260513", "20260514",
             "20260522", "20260608", "20260619", "20260709", "20260728"]

dfs = {}
for c in ["200uL", "1000uL"]:
    df = pd.read_csv(os.path.join(BASE, c, "incremental_results.csv"))
    ok = df[df["status"] == "ok"].copy()
    ok["date"] = ok["timestamp"].astype(str).str[:8]
    ok["time"] = ok["timestamp"].astype(str).str[9:]  # HHMMSS
    ok["flagged"] = (ok["pre_baseline_std"] > THRESHOLD) | (ok["post_baseline_std"] > THRESHOLD)
    dfs[c] = ok

print(f"{'Date':<12} {'Campaign':<10} {'First meas':>12} {'Last meas':>12} {'N rows':>8} {'N flagged':>10} {'Flag%':>7}")
print("-" * 70)

for date in KEY_DATES:
    for c in ["200uL", "1000uL"]:
        sub = dfs[c][dfs[c]["date"] == date]
        if len(sub) == 0:
            print(f"{date:<12} {c:<10} {'not run':>12}")
            continue
        first = sub["timestamp"].astype(str).min()[9:]  # HHMMSS
        last = sub["timestamp"].astype(str).max()[9:]
        n = len(sub)
        nf = sub["flagged"].sum()
        pct = 100 * nf / n
        print(f"{date:<12} {c:<10} {first:>12} {last:>12} {n:>8} {nf:>10} {pct:>6.0f}%")
    print()
