import csv

# Read and analyze the current vial file
with open('c:\\Users\\owenm\\OneDrive\\Desktop\\North Robotics\\utoronto_demo\\utoronto_demo\\status\\peroxide_assay_vial_status_v3.csv', 'r') as f:
    reader = csv.DictReader(f)
    vials = list(reader)

print("=== VIAL ANALYSIS ===")
for vial in vials:
    print(f"'{vial['vial_name']}' -> location: '{vial['location']}', index: {vial['vial_index']}")

print("\n=== DUPLICATES CHECK ===")
names = [v['vial_name'] for v in vials]
duplicates = [name for name in set(names) if names.count(name) > 1]
if duplicates:
    print(f"DUPLICATE NAMES: {duplicates}")
else:
    print("No duplicate names in CSV file")

indices = [v['vial_index'] for v in vials]  
duplicate_indices = [idx for idx in set(indices) if indices.count(idx) > 1]
if duplicate_indices:
    print(f"DUPLICATE INDICES: {duplicate_indices}")
else:
    print("No duplicate indices in CSV file")