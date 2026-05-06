import csv, glob, os, datetime

def analyze_dir(data_dir, label):
    pattern = os.path.join(data_dir, 'mass_data_row_*.csv')
    files = sorted(glob.glob(pattern))
    
    results = []
    for fp in files:
        row_num = int(os.path.basename(fp).replace('mass_data_row_','').replace('.csv',''))
        total = 0
        bad = 0
        first_ts = None
        with open(fp, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if first_ts is None:
                    first_ts = float(row['timestamp'])
                total += 1
                if float(row['mass_g']) <= 0.0:
                    bad += 1
        results.append({'row': row_num, 'bad': bad, 'total': total, 'ts': first_ts})
    
    print('\n=== ' + label + ' ===')
    print('Total files: ' + str(len(results)))
    bad_rows = [r for r in results if r['bad'] > 0]
    print('Rows with mass <= 0: ' + str(len(bad_rows)))
    
    if bad_rows:
        r = bad_rows[0]
        dt = datetime.datetime.fromtimestamp(r['ts'])
        print('First bad row: ' + str(r['row']) + ' at ' + str(dt) + ' (' + str(r['bad']) + '/' + str(r['total']) + ' pts)')
        r = bad_rows[-1]
        dt = datetime.datetime.fromtimestamp(r['ts'])
        print('Last bad row:  ' + str(r['row']) + ' at ' + str(dt) + ' (' + str(r['bad']) + '/' + str(r['total']) + ' pts)')
    
    for r in bad_rows:
        pct = r['bad'] / r['total'] * 100
        dt = datetime.datetime.fromtimestamp(r['ts'])
        print('  Row %03d: %4d/%d (%5.1f%%) - %s' % (r['row'], r['bad'], r['total'], pct, dt))
    
    if results:
        dt_first = datetime.datetime.fromtimestamp(results[0]['ts'])
        dt_last = datetime.datetime.fromtimestamp(results[-1]['ts'])
        print('Timeline: rows 1-' + str(len(results)) + ' span ' + str(dt_first) + ' to ' + str(dt_last))
    
    return results

uL200 = analyze_dir(r'output/glycerol_sobol_campaign/200uL/mass_time_data', '200uL Campaign')
uL1000 = analyze_dir(r'output/glycerol_sobol_campaign/1000uL/mass_time_data', '1000uL Campaign')

# Cross-campaign comparison
print('\n=== TIMELINE COMPARISON ===')
all_rows = []
for r in uL200:
    all_rows.append({'campaign': '200uL', 'row': r['row'], 'ts': r['ts'], 'bad_pct': r['bad']/r['total']*100})
for r in uL1000:
    all_rows.append({'campaign': '1000uL', 'row': r['row'], 'ts': r['ts'], 'bad_pct': r['bad']/r['total']*100})

all_rows.sort(key=lambda x: x['ts'])

# Find first bad timestamp across both
bad_all = [r for r in all_rows if r['bad_pct'] > 0]
if bad_all:
    first_bad_ts = bad_all[0]['ts']
    dt_first_bad = datetime.datetime.fromtimestamp(first_bad_ts)
    print('First bad measurement anywhere: ' + bad_all[0]['campaign'] + ' row ' + str(bad_all[0]['row']) + ' at ' + str(dt_first_bad))
    
    # How many 200uL rows came AFTER the first bad measurement
    uL200_after_first_bad = [r for r in uL200 if r['ts'] > first_bad_ts]
    uL200_bad_after = [r for r in uL200_after_first_bad if r['bad'] > 0]
    uL200_bad_before = [r for r in uL200 if r['ts'] <= first_bad_ts and r['bad'] > 0]
    
    print('200uL rows after first bad timestamp: ' + str(len(uL200_after_first_bad)))
    print('200uL bad rows BEFORE first bad event: ' + str(len(uL200_bad_before)))
    print('200uL bad rows AFTER first bad event:  ' + str(len(uL200_bad_after)))
    if uL200_bad_after:
        print('  -> These 200uL rows may have corrupted volume tracking:')
        for r in uL200_bad_after:
            pct = r['bad']/r['total']*100
            dt = datetime.datetime.fromtimestamp(r['ts'])
            print('     Row %03d: %5.1f%% bad - %s' % (r['row'], pct, dt))
