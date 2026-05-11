import pandas as pd

dfs = []
for campaign in ['200uL', '1000uL']:
    path = f'output/glycerol_sobol_campaign/{campaign}/incremental_results.csv'
    try:
        df = pd.read_csv(path)
        df['campaign'] = campaign
        dfs.append(df)
        print(f'{campaign}: {len(df)} total rows')
    except Exception as e:
        print(f'{campaign}: {e}')

all_data = pd.concat(dfs, ignore_index=True)
all_data['ts'] = pd.to_datetime(all_data['timestamp'], format='%Y%m%d_%H%M%S')
all_data['date'] = all_data['ts'].dt.date

cutoff = pd.Timestamp('2026-04-22').date()
recent = all_data[all_data['date'] >= cutoff]

by_day = recent.groupby('date').size().reset_index(name='count')
print()
print(f"{'Date':<12} | Points | Breakdown")
print('-' * 55)
for _, r in by_day.iterrows():
    day_data = recent[recent['date'] == r['date']]
    parts = [f"{c}:{len(d)}" for c, d in day_data.groupby('campaign')]
    print(f"{str(r['date']):<12} |  {r['count']:4d}  | {', '.join(parts)}")

total_days = (pd.Timestamp('2026-05-06').date() - cutoff).days + 1
print()
print(f"Total days with data: {len(by_day)}")
print(f"Total points (Apr 22 - May 6): {by_day['count'].sum()}")
print(f"Avg per active day: {by_day['count'].mean():.1f}")
print(f"Avg per calendar day ({total_days} days): {by_day['count'].sum() / total_days:.1f}")
