import pandas as pd

THRESHOLD = 0.003  # g — ~100x normal noise floor

for c in ["200uL", "1000uL"]:
    path = rf"C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign\{c}\incremental_results.csv"
    df = pd.read_csv(path)
    df["row_index"] = df["row_index"].astype(int)

    # Only consider currently-valid rows (don't re-flag already-failed ones)
    ok = df[df["status"] == "ok"].copy()
    ok["date"] = ok["timestamp"].astype(str).str[:8]  # YYYYMMDD

    flagged = ok[
        (ok["pre_baseline_std"] > THRESHOLD) | (ok["post_baseline_std"] > THRESHOLD)
    ]
    flagged_pre_only = ok[ok["pre_baseline_std"] > THRESHOLD]
    flagged_post_only = ok[ok["post_baseline_std"] > THRESHOLD]
    flagged_both = ok[
        (ok["pre_baseline_std"] > THRESHOLD) & (ok["post_baseline_std"] > THRESHOLD)
    ]

    print(f"=== {c} ===")
    print(f"Total ok rows: {len(ok)}")
    print(f"Flagged (pre OR post > {THRESHOLD}): {len(flagged)} ({100*len(flagged)/len(ok):.1f}%)")
    print(f"  pre only:  {len(flagged_pre_only) - len(flagged_both)}")
    print(f"  post only: {len(flagged_post_only) - len(flagged_both)}")
    print(f"  both:      {len(flagged_both)}")

    print(f"\nBy run_type:")
    for rt in ["original", "redo"]:
        sub = flagged[flagged["run_type"] == rt]
        total_rt = ok[ok["run_type"] == rt]
        print(f"  {rt}: {len(sub)} / {len(total_rt)} ({100*len(sub)/max(len(total_rt),1):.1f}%)")

    print(f"\nBy date (flagged counts, only dates with >=1 flag):")
    by_date = flagged.groupby("date").size().sort_index()
    ok_by_date = ok.groupby("date").size()
    for date, n in by_date.items():
        total = ok_by_date.get(date, 0)
        pct = 100 * n / total if total else 0
        print(f"  {date}: {n}/{total} ({pct:.0f}%)")
    print()
