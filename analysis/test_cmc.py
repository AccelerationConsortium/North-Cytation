import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
summary_path = r"C:\Users\Imaging Controller\Desktop\CMC\20250702_1121\CMC_measurement_summary.csv"
output_dir = os.path.dirname(summary_path)

# --- LOAD DATA ---
summary_df = pd.read_csv(summary_path)

# Backwards-compatible handling of Time_min
if "Time_min" not in summary_df.columns and "Timing" in summary_df.columns:
    summary_df["Time_min"] = summary_df["Timing"].str.extract(r"(\d+)").astype(float)
else:
    summary_df["Time_min"] = pd.to_numeric(summary_df["Time_min"], errors="coerce")

summary_df["Assay"] = summary_df["Assay"].astype(str)

# --- PLOT: REPLICATES PER TRIAL (optional, from earlier block) ---
def plot_variation_clean(summary_df, surfactant_name, output_dir):
    df = summary_df[summary_df["Surfactant"] == surfactant_name].copy()
    metrics = ["CMC", "r2", "A1", "A2", "dx"]

    for metric in metrics:
        plt.figure()
        for trial in sorted(df["Assay"].unique()):
            sub = df[df["Assay"] == trial].sort_values("Time_min")
            plt.scatter(sub["Time_min"], sub[metric], alpha=0.6, label=f"Trial {trial}")

        plt.title(f"{metric} vs Time ({surfactant_name})")
        plt.xlabel("Time (minutes)")
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend(title="Trial")
        plt.tight_layout()

        fname = f"{surfactant_name}_{metric}_clean_variation.png".replace(" ", "_")
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

# Run replicate plots
for surfactant in summary_df["Surfactant"].unique():
    plot_variation_clean(summary_df, surfactant, output_dir)

# --- 1. Std Dev of Replicates (within Time x Assay) ---
replicate_std = summary_df.groupby(["Time_min", "Assay"])["CMC"].std().reset_index()
replicate_std.rename(columns={"CMC": "replicate_std"}, inplace=True)

# --- 2. Std Dev Across Assays (per Time, after averaging replicates) ---
rep_avg = summary_df.groupby(["Time_min", "Assay"])["CMC"].mean().reset_index()
assay_std = rep_avg.groupby("Time_min")["CMC"].std().reset_index()
assay_std.rename(columns={"CMC": "assay_std"}, inplace=True)

# --- 3. Std Dev Across Time (per Assay, after averaging replicates) ---
time_std = rep_avg.groupby("Assay")["CMC"].std().reset_index()
time_std.rename(columns={"CMC": "time_std"}, inplace=True)

# --- SAVE CSVs ---
replicate_std.to_csv(os.path.join(output_dir, "replicate_std.csv"), index=False)
assay_std.to_csv(os.path.join(output_dir, "assay_std.csv"), index=False)
time_std.to_csv(os.path.join(output_dir, "time_std.csv"), index=False)

# --- PLOTTING ---

# Plot 1: Std Dev of Replicates vs Time (per Assay)
plt.figure()
for assay in replicate_std["Assay"].unique():
    sub = replicate_std[replicate_std["Assay"] == assay]
    plt.plot(sub["Time_min"], sub["replicate_std"], marker='o', label=f"Trial {assay}")
plt.title("Std Dev of Replicates vs Time (per Trial)")
plt.xlabel("Time (min)")
plt.ylabel("Std Dev (CMC)")
plt.legend(title="Trial")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot_replicate_std.png"))
plt.close()

# Plot 2: Std Dev Across Trials vs Time
plt.figure()
plt.plot(assay_std["Time_min"], assay_std["assay_std"], marker='o')
plt.title("Std Dev Across Trials vs Time")
plt.xlabel("Time (min)")
plt.ylabel("Std Dev (CMC across trial averages)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot_assay_std.png"))
plt.close()

# Plot 3: Std Dev Across Timepoints per Trial
plt.figure()
plt.bar(time_std["Assay"], time_std["time_std"])
plt.title("Std Dev Across Timepoints per Trial")
plt.xlabel("Trial")
plt.ylabel("Std Dev (CMC over time)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot_time_std.png"))
plt.close()

print("✅ Analysis complete. Outputs saved to:", output_dir)
