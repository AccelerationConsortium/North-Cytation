import pandas as pd

csv = r'c:\Users\Imaging Controller\Desktop\utoronto_demo\output\experimental_surfactant_grid\DSS_BZT_replay_new_min_conc_plus_1d_cmc_assay_20260512_125219\complete_experiment_results.csv'
df = pd.read_csv(csv)
exp = df[df['well_type'] == 'experiment']

water_wells = exp[exp['water_volume_ul'] > 0].sort_values('water_volume_ul', ascending=True)
mid = len(water_wells) // 2
w1 = water_wells.iloc[:mid]  # goes to 'water' vial
w2 = water_wells.iloc[mid:]  # goes to 'water_2' vial

print(f"Total experiment wells with water: {len(water_wells)}")
print(f"water vial (small half) : {len(w1)} wells, total = {w1['water_volume_ul'].sum()/1000:.3f} mL")
print(f"water_2 vial (large half): {len(w2)} wells, total = {w2['water_volume_ul'].sum()/1000:.3f} mL")
print()

# Simulate the chunk logic (REFILL_CHECK_CHUNK_SIZE=24) on water_2
CHUNK = 24
THRESHOLD = 4.0
start_vol = 8.0
vol = start_vol
w2_list = w2['water_volume_ul'].tolist()

print(f"Simulating water_2 with CHUNK={CHUNK}, THRESHOLD={THRESHOLD} mL:")
print(f"  Start: {vol:.3f} mL")
for chunk_start in range(0, len(w2_list), CHUNK):
    chunk = w2_list[chunk_start:chunk_start+CHUNK]
    chunk_total_ml = sum(chunk) / 1000
    print(f"  Pre-chunk check at {vol:.3f} mL -> {'REFILL' if vol < THRESHOLD else 'OK (no refill)'}")
    if vol < THRESHOLD:
        print(f"    -> Refill to 8.0 mL")
        vol = 8.0
    vol -= chunk_total_ml
    print(f"  After chunk ({len(chunk)} wells, {chunk_total_ml:.3f} mL consumed): {vol:.3f} mL")

print()
# Simulate with CHUNK=12
CHUNK = 12
vol = 8.0
print(f"Simulating water_2 with CHUNK={CHUNK}, THRESHOLD={THRESHOLD} mL:")
print(f"  Start: {vol:.3f} mL")
for chunk_start in range(0, len(w2_list), CHUNK):
    chunk = w2_list[chunk_start:chunk_start+CHUNK]
    chunk_total_ml = sum(chunk) / 1000
    print(f"  Pre-chunk check at {vol:.3f} mL -> {'REFILL' if vol < THRESHOLD else 'OK (no refill)'}")
    if vol < THRESHOLD:
        print(f"    -> Refill to 8.0 mL")
        vol = 8.0
    vol -= chunk_total_ml
    print(f"  After chunk ({len(chunk)} wells, {chunk_total_ml:.3f} mL consumed): {vol:.3f} mL")

print()
# What threshold is actually needed with CHUNK=24 to catch the problem?
vol = 8.0
w2_chunks = [w2_list[i:i+24] for i in range(0, len(w2_list), 24)]
consumed = [sum(c)/1000 for c in w2_chunks]
print("Chunk volumes for water_2 (CHUNK=24):")
for i, c in enumerate(consumed):
    print(f"  Chunk {i+1}: {len(w2_chunks[i])} wells = {c:.3f} mL")
print(f"  To catch chunk 2, threshold must be > {8.0 - consumed[0]:.3f} mL (volume after chunk 1)")
