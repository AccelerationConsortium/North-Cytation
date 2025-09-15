import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
sys.path.append("../utoronto_demo")

# Load the data
data = pd.DataFrame([
    [0, 'SDS', 'main_8mL_rack', 0, 8.0, True, 'open', '8_mL', 'main_8mL_rack', 0],
    [1, 'NaDC', 'main_8mL_rack', 1, 8.0, True, 'open', '8_mL', 'main_8mL_rack', 1],
    [2, 'NaC', 'main_8mL_rack', 2, 8.0, True, 'open', '8_mL', 'main_8mL_rack', 2],
    [3, 'CTAB', 'main_8mL_rack', 3, 8.0, True, 'open', '8_mL', 'main_8mL_rack', 3],
    [4, 'DTAB', 'main_8mL_rack', 4, 8.0, True, 'open', '8_mL', 'main_8mL_rack', 4],
    [5, 'TTAB', 'main_8mL_rack', 5, 8.0, True, 'open', '8_mL', 'main_8mL_rack', 5],
    [6, 'P188', 'main_8mL_rack', 6, 8.0, True, 'open', '8_mL', 'main_8mL_rack', 6],
    [7, 'P407', 'main_8mL_rack', 7, 8.0, True, 'open', '8_mL', 'main_8mL_rack', 7],
    [8, 'CAPB', 'main_8mL_rack', 8, 8.0, True, 'open', '8_mL', 'main_8mL_rack', 8],
    [9, 'CHAPS', 'main_8mL_rack', 9, 8.0, True, 'open', '8_mL', 'main_8mL_rack', 9],
    [10, 'substock_1', 'main_8mL_rack', 10, 1.535, True, 'open', '8_mL', 'main_8mL_rack', 10],
    [11, 'substock_2', 'main_8mL_rack', 11, 0.0, True, 'open', '8_mL', 'main_8mL_rack', 11],
    [12, 'substock_3', 'main_8mL_rack', 12, 0.0, True, 'open', '8_mL', 'main_8mL_rack', 12],
    [13, 'substock_4', 'main_8mL_rack', 13, 0.0, True, 'open', '8_mL', 'main_8mL_rack', 13],
    [14, 'substock_5', 'main_8mL_rack', 14, 0.0, True, 'open', '8_mL', 'main_8mL_rack', 14],
    [15, 'substock_6', 'main_8mL_rack', 15, 0.0, True, 'open', '8_mL', 'main_8mL_rack', 15],
    [16, 'substock_7', 'main_8mL_rack', 16, 0.0, True, 'open', '8_mL', 'main_8mL_rack', 16],
    [17, 'substock_8', 'main_8mL_rack', 17, 0.0, True, 'open', '8_mL', 'main_8mL_rack', 17],
    [18, 'substock_9', 'main_8mL_rack', 18, 0.0, True, 'open', '8_mL', 'main_8mL_rack', 18],
    [19, 'substock_10', 'main_8mL_rack', 19, 0.0, True, 'open', '8_mL', 'main_8mL_rack', 19],
    [20, 'water', 'main_8mL_rack', 45, 0, True, 'open', '8_mL', 'main_8mL_rack', 45],
    [21, 'pyrene_DMSO', 'main_8mL_rack', 47, 0.6, False, 'none', '8_mL', 'main_8mL_rack', 47],
], columns=['vial_index', 'vial_name', 'location', 'location_index', 'vial_volume', 'capped', 'cap_type', 'vial_type', 'home_location', 'home_location_index'])



# Filter for main_8mL_rack
rack_data = data[data['location'] == 'main_8mL_rack']

# Generate a full set of location indices (0 to 47)
all_indices = set(range(48))
present_indices = set(rack_data['location_index'])
missing_indices = all_indices - present_indices


fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-0.5, 7.5)
ax.set_xticks(range(6))
ax.set_yticks(range(8))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True)

# Draw the vials in the dataset
for _, row in rack_data.iterrows():
    index = row['location_index']
    col = 5 - index // 8
    row_pos = index % 8 if col % 2 == 0 else 7 - (index % 8)

    # Determine fill color
    fill_color = 'lightgreen' if row['vial_volume'] > 0 else 'white'
    circle = patches.Circle((col, row_pos), 0.45, edgecolor='black', facecolor=fill_color)
    ax.add_patch(circle)

    # Determine cap status
    if not row['capped']:
        cap_status = "uncapped"
    elif row['cap_type'] == 'open':
        cap_status = "open cap"
    elif row['cap_type'] == 'closed':
        cap_status = "closed cap"
    else:
        cap_status = "unknown cap"

    # Add text
    text = f"{row['vial_name']}\n{row['vial_volume']:.2f} mL\n{cap_status}"
    ax.text(col, row_pos, text, ha='center', va='center', fontsize=8)

# Add dotted-line circles for missing vials
for index in missing_indices:
    col = 5 - index // 8
    row_pos = index % 8 if col % 2 == 0 else 7 - (index % 8)
    circle = patches.Circle((col, row_pos), 0.45, edgecolor='black', facecolor='none', linestyle='dotted')
    ax.add_patch(circle)

ax.set_title("main_8mL_rack Layout with Cap Status")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
