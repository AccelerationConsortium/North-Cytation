# --- cmc_pairings_workflow.py ---
import sys
sys.path.append("../utoronto_demo")
from cmc_shared import *
import analysis.cmc_exp_new as experimental_planner
import analysis.cmc_data_analysis as analyzer
from master_usdl_coordinator import Lash_E
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import time


LOGGING_FOLDER = "../utoronto_demo/logs/"
MEASUREMENT_PROTOCOL_FILE = [
    r"C:\Protocols\CMC_Fluorescence.prt",
    r"C:\Protocols\CMC_Absorbance.prt"
]
simulate = False
run = 6  # This determines which Run group you are running
INPUT_VIAL_STATUS_FILE = f"../utoronto_demo/status/CMC_double_input_{run}.csv"

#Experiment-1
#replacements = {'CTAB': ['CTAB_2', 5.0]} #Run specific, based on what vials we need to use
replacements = {}


# Load pairing data from CSV
data_in = pd.read_csv("../utoronto_demo/analysis/greedy_grouped_trials.csv")  # Add full path if needed
data_in = data_in[data_in['trial'] == run]

# Extract pairings and ratios for this trial
pairings_and_ratios = []
for _, row in data_in.iterrows():
    pairings_and_ratios.append(
        ([row['surfactant_1'], row['surfactant_2']], [row['surfactant_1_ratio'], row['surfactant_2_ratio']])
    )


# Show user what’s being run and pause
print(f"Run {run} — Pairings and Ratios:")
for (sfs, ratios) in pairings_and_ratios:
    print(f"  {sfs[0]}:{ratios[0]} + {sfs[1]}:{ratios[1]}")

lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
lash_e.nr_robot.check_input_file()
lash_e.nr_track.check_input_file()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

surfactants_used = set()
for pair, _ in pairings_and_ratios:
    surfactants_used.update(pair)
surfactants_used.update(["water_large", "pyrene_DMSO"])
initial_volumes = {}
for surf in surfactants_used:
    try:
        initial_volumes[surf] = lash_e.nr_robot.get_vial_info(surf, 'vial_volume')
    except Exception as e:
        print(f"Warning: couldn't get initial volume for {surf}: {e}")


if not simulate:
    folder = f'C:/Users/Imaging Controller/Desktop/CMC/{timestamp}/'
    os.makedirs(folder, exist_ok=True)
    print(f"Folder created at: {folder}")
else:
    folder = None

lash_e.nr_track.get_new_wellplate()
starting_wp_index = 0
lash_e.nr_robot.prime_reservoir_line(1, 'water', 0.5)

surfactants = ['SDS', 'NaDC', 'NaC', 'CTAB', 'DTAB', 'TTAB', 'CAPB', 'CHAPS']
item_to_index = {item: i for i, item in enumerate(surfactants)}
n_substocks = len(surfactants) * 2
substock_name_list = [f'substock_{i}' for i in range(1, n_substocks + 1)]

# pairings_and_ratios = [
#     (['SDS', 'CAPB'], [0.5, 0.5]),
#     (['CTAB', 'TTAB'], [0.5, 0.5]),
#     (['CAPB', 'CHAPS'], [0.5, 0.5]),
# ] #Note: These will be read in from a csv file for the actual experiment

padded_ratios = []
pairing_labels = []
for pair, ratios in pairings_and_ratios:
    vector = [0.0] * len(surfactants)
    for surf, ratio in zip(pair, ratios):
        vector[item_to_index[surf]] = ratio
    padded_ratios.append(vector)
    label = "_".join(f"{s}-{r:.1f}" for s, r in zip(pair, ratios))
    pairing_labels.append(label)

substock_count = 0
summary_data = []
if not simulate:
    summary_file = os.path.join(folder, f"CMC_summary_{timestamp}.csv")

summary_columns = [
    "Surfactant_1", "Surfactant_2", "conc_1", "conc_2",
    "cmc_rough", "r2_rough", "A1_rough", "A2_rough", "dx_rough",
    "cmc_fine", "r2_fine", "A1_fine", "A2_fine", "dx_fine",
    "replicates"
]
if not simulate:
    with open(summary_file, "w") as f:
        f.write(",".join(summary_columns) + "\n")

for i, ratio in enumerate(padded_ratios):
    rough_vial = substock_name_list[substock_count]
    fine_vial = substock_name_list[substock_count + 1]
    substock_count += 2
    label = pairing_labels[i]
    pair, ratios = pairings_and_ratios[i]

    # Rough search
    rough_exp, _ = experimental_planner.generate_exp_flexible(surfactants, ratio, rough_screen=True)
    sub_stock_vols = rough_exp['surfactant_sub_stock_vols']
    
    for old_name, (new_name, volume) in replacements.items():
        sub_stock_vols = change_stock_solution_vial(lash_e, old_name, new_name, volume, sub_stock_vols) 

    wellplate_data = rough_exp['df']
    samples_per_assay = wellplate_data.shape[0]

    mix_surfactants(lash_e, sub_stock_vols, rough_vial)
    fill_water_vial(lash_e)
    create_wellplate_samples(lash_e, wellplate_data, rough_vial, starting_wp_index)

    if not simulate:
        time.sleep(600)
        results = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, range(starting_wp_index, starting_wp_index + samples_per_assay), plate_type="48 WELL PLATE", repeats=3)
        results_concat = merge_absorbance_and_fluorescence(coalesce_replicates_long(results))
        metrics_rough = analyze_and_save_results(folder, label + '_rough', wellplate_data, results_concat, analyzer, 'rough', log=True)
        cmc_rough = metrics_rough["CMC"]
        print("Rough CMC:", cmc_rough)
    else:
        cmc_rough = None

    starting_wp_index += samples_per_assay
    

    # Fine search
    fine_exp, _ = experimental_planner.generate_exp_flexible(surfactants, ratio, rough_screen=False, estimated_CMC=cmc_rough)
    sub_stock_vols = fine_exp['surfactant_sub_stock_vols']

    for old_name, (new_name, volume) in replacements.items():
        sub_stock_vols = change_stock_solution_vial(lash_e, old_name, new_name, volume, sub_stock_vols) 

    wellplate_data = fine_exp['df']
    samples_per_assay = wellplate_data.shape[0]

    mix_surfactants(lash_e, sub_stock_vols, fine_vial)
    fill_water_vial(lash_e)
    create_wellplate_samples(lash_e, wellplate_data, fine_vial, starting_wp_index)

    if not simulate:
        time.sleep(600)
        results = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, range(starting_wp_index, starting_wp_index + samples_per_assay), plate_type="48 WELL PLATE", repeats=3)
        results_concat = merge_absorbance_and_fluorescence(coalesce_replicates_long(results))
        metrics_fine = analyze_and_save_results(folder, label + '_fine', wellplate_data, results_concat, analyzer, 'fine')
        cmc_fine = metrics_fine["CMC"]
        print("Refined CMC:", cmc_fine)

    starting_wp_index += samples_per_assay

    if not simulate:
        with open(summary_file, "a") as f:
            f.write(",".join(str(x) for x in [
                pair[0], pair[1],
                ratios[0], ratios[1],
                metrics_rough.get("CMC"),
                metrics_rough.get("r2"),
                metrics_rough.get("A1"),
                metrics_rough.get("A2"),
                metrics_rough.get("dx"),
                metrics_fine.get("CMC"),
                metrics_fine.get("r2"),
                metrics_fine.get("A1"),
                metrics_fine.get("A2"),
                metrics_fine.get("dx"),
                3  # replicates, make configurable if needed
            ]) + "\n")
        print(f"Saved summary for {pair[0]} + {pair[1]}")

    if starting_wp_index >= 48 and i < len(padded_ratios) - 1:
        lash_e.discard_used_wellplate()
        lash_e.grab_new_wellplate()
        starting_wp_index = 0

if lash_e.nr_track.NR_OCCUPIED:
    lash_e.discard_used_wellplate()
    print("Workflow complete and wellplate discarded")

if not simulate:
    df = pd.read_csv(summary_file)
    # Generate 8 high-contrast, visually distinct colors using seaborn's color palette
    surf_set = ['SDS', 'NaDC', 'NaC', 'CTAB', 'DTAB', 'TTAB', 'CAPB', 'CHAPS']
    solo_cmc = {
        'SDS':   8.5,  # <-- Replace with actual value
        'NaDC':  5.34,
        'NaC':   14,
        'CTAB':  1.07,
        'DTAB':  15.85,
        'TTAB':  3.985,
        'CAPB':  0.627,
        'CHAPS': 8,
    }
    pair_indices = {(s1, s2): (surf_set.index(s1), surf_set.index(s2))
                for s1 in surf_set for s2 in surf_set}
    palette = sns.color_palette("tab10", n_colors=8)  # "tab10" is optimized for contrast

    # Map each surfactant to one color
    base_colors = {surf: color for surf, color in zip(surf_set, palette)}

    # Restore the previous clean layout, but add the missing CMC labels on the diagonals
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(len(surf_set), len(surf_set), figure=fig, wspace=0.4, hspace=0.4)

    for (s1, s2), (i, j) in pair_indices.items():
        ax = fig.add_subplot(gs[i, j])

        if s1 == s2:
            ax.scatter([0.5], [solo_cmc[s1]], color=base_colors[s1], s=100)
            ax.set_title(f"{s1}", fontsize=8, color=base_colors[s1])
            ax.text(0.5, solo_cmc[s1], f"{solo_cmc[s1]:.2f}", fontsize=7,
                    ha='center', va='bottom', color=base_colors[s1])
        else:
            matching = df[((df['Surfactant_1'] == s1) & (df['Surfactant_2'] == s2)) |
                        ((df['Surfactant_1'] == s2) & (df['Surfactant_2'] == s1))]

            if not matching.empty:
                matching = matching.copy()
                matching['ratio_plot'] = matching.apply(
                    lambda row: row['conc_1'] if row['Surfactant_1'] == s1 else row['conc_2'],
                    axis=1
                )
                matching_sorted = matching.sort_values('ratio_plot')

                colors = []
                for r in matching_sorted['ratio_plot']:
                    color = tuple(
                        (1 - r) * base_colors[s1][k] + r * base_colors[s2][k]
                        for k in range(3)
                    )
                    colors.append(color)

                ax.scatter(matching_sorted['ratio_plot'], matching_sorted['cmc_fine'], c=colors, s=30)
                ax.plot(matching_sorted['ratio_plot'], matching_sorted['cmc_fine'], color='black', linewidth=0.5)

            ax.set_xticks([0.2, 0.5, 0.8])
            ax.set_yticks([])
            ax.set_xticklabels(['.2', '.5', '.8'], fontsize=6)
            ax.set_title(f'{s1}-{s2}', fontsize=7)
            ax.tick_params(axis='x', labelsize=6)
            ax.set_ylim(0.4, 2.6)
            ax.set_xlim(0, 1)

    fig.suptitle("CMC vs. Ratio with Color Interpolation and Solo CMC Markers", fontsize=16)
    fig.savefig(os.path.join(folder, "CMC_pairwise_plot.png"), dpi=300, bbox_inches='tight')
    #plt.show()

print("\n--- Volume Usage Summary ---")
for surf in surfactants_used:
    try:
        final_vol = lash_e.nr_robot.get_vial_info(surf, 'vial_volume')
        initial_vol = initial_volumes.get(surf, final_vol)
        used_vol = initial_vol - final_vol
        print(f"{surf}: used {used_vol:.2f} mL (from {initial_vol:.2f} -> {final_vol:.2f})")
    except Exception as e:
        print(f"{surf}: Error retrieving final volume — {e}")

