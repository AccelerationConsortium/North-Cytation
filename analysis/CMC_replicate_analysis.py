import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from analysis.cmc_data_analysis import CMC_plot



def group_replicate_files(folder):
    pattern = re.compile(r"output_data_(.*)_rep(\d+)\.csv")
    file_dict = defaultdict(list)

    for fname in os.listdir(folder):
        match = pattern.match(fname)
        if match:
            base_key = match.group(1)
            rep = match.group(2)
            output_path = os.path.join(folder, fname)
            wellplate_path = os.path.join(folder, f"wellplate_data_{base_key}_rep{rep}.csv")

            if os.path.exists(wellplate_path):
                file_dict[base_key].append((output_path, wellplate_path))

    return file_dict


def analyze_combined_replicates(input_folder, output_folder):
    output_dir = output_folder

    file_dict = group_replicate_files(input_folder)
    param_records = []

    for key, file_pairs in file_dict.items():
        print(f"\nProcessing group: {key} with {len(file_pairs)} replicates")

        key_match = re.match(r"(.+?)_(\d+)min_([abc])$", key)
        if not key_match:
            print(f"Skipping unrecognized key format: {key}")
            continue

        surfactant = key_match.group(1).split("_")[-1]
        time_min = int(key_match.group(2))
        assay = key_match.group(3)

        all_ratios, all_concs, rep_ids = [], [], []

        for output_file, wellplate_file in sorted(file_pairs):
            rep_match = re.search(r"rep(\d+)", output_file)
            rep_num = int(rep_match.group(1)) if rep_match else None

            output_df = pd.read_csv(output_file)
            well_df = pd.read_csv(wellplate_file)

            if "ratio" not in output_df or "concentration" not in well_df:
                continue

            conc = well_df["concentration"].values
            ratio = output_df["ratio"].values

            all_concs.append(conc)
            all_ratios.append(ratio)
            rep_ids.append(rep_num)

        if not all_ratios or not all(np.allclose(all_concs[0], c) for c in all_concs):
            continue

        base_conc = all_concs[0]

        # Plot all replicate curves
        plt.figure(figsize=(8, 6))
        for i, rep in enumerate(all_ratios):
            plt.plot(base_conc, rep, marker='o', alpha=0.5, label=f"Rep {rep_ids[i]}")
        plt.title(f"I₁/I₃ Ratios ({surfactant}, {time_min}min, {assay})")
        plt.xlabel("Concentration (mM)")
        plt.ylabel("Ratio (I₁/I₃)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{key}_all_replicates.png"))
        plt.close()

        # --- Fit combined
        try:
            A1, A2, x0, dx, r2 = CMC_plot(
                np.concatenate(all_ratios),
                np.tile(base_conc, len(all_ratios)),
                os.path.join(output_dir, f"{key}_fit_combined.png")
            )
            param_records.append(dict(Surfactant=surfactant, Assay=assay, Time_min=time_min,
                                      FitType="combined", A1=A1, A2=A2, x0=x0, dx=dx, r_squared=r2))
        except Exception as e:
            print(f"Combined fit failed: {e}")

        # --- Fit average
        avg_ratio = np.mean(np.vstack(all_ratios), axis=0)
        try:
            A1, A2, x0, dx, r2 = CMC_plot(
                avg_ratio,
                base_conc,
                os.path.join(output_dir, f"{key}_fit_average.png")
            )
            param_records.append(dict(Surfactant=surfactant, Assay=assay, Time_min=time_min,
                                      FitType="average", A1=A1, A2=A2, x0=x0, dx=dx, r_squared=r2))
        except Exception as e:
            print(f"Averaged fit failed: {e}")

        # Save wide-format replicate data
        df_wide = pd.DataFrame({"Concentration": base_conc})
        for i, rep in enumerate(rep_ids):
            df_wide[f"Rep{rep}"] = all_ratios[i]
        df_wide.to_csv(os.path.join(output_dir, f"{key}_combined_ratios.csv"), index=False)

        # Save averaged data
        pd.DataFrame({
            "Concentration": base_conc,
            "Avg_Ratio": avg_ratio
        }).to_csv(os.path.join(output_dir, f"{key}_averaged_ratios.csv"), index=False)

    # Save all fit parameters
    df_params = pd.DataFrame(param_records)
    df_params.sort_values(by=["Surfactant", "Assay", "Time_min", "FitType"], inplace=True)
    df_params.to_csv(os.path.join(output_dir, "combined_replicates_fit_params.csv"), index=False)

    print("\n✅ Replicate analysis complete.")

def analyze_summary_variation(summary_df, output_folder):

    summary_df["Time_min"] = pd.to_numeric(summary_df["Time_min"], errors="coerce")
    summary_df["Assay"] = summary_df["Assay"].astype(str)

    metrics = ["CMC", "r2", "A1", "A2", "dx"]

    for surf in summary_df["Surfactant"].unique():
        df = summary_df[summary_df["Surfactant"] == surf].copy()

        for metric in metrics:
            plt.figure()
            for assay in sorted(df["Assay"].unique()):
                sub = df[df["Assay"] == assay].sort_values("Time_min")
                plt.scatter(sub["Time_min"], sub[metric], alpha=0.6, label=f"Trial {assay}")
            plt.title(f"{metric} vs Time ({surf})")
            plt.xlabel("Time (minutes)")
            plt.ylabel(metric)
            plt.grid(True)
            plt.legend(title="Trial")
            plt.tight_layout()
            fname = f"{surf}_{metric}_clean_variation.png".replace(" ", "_")
            plt.savefig(os.path.join(output_folder, fname))
            plt.close()

        # --- Std dev calculations ---
        rep_avg = df.groupby(["Time_min", "Assay"])["CMC"].mean().reset_index()
        replicate_std = df.groupby(["Time_min", "Assay"])["CMC"].std().reset_index()
        replicate_std.rename(columns={"CMC": "replicate_std"}, inplace=True)

        assay_std = rep_avg.groupby("Time_min")["CMC"].std().reset_index()
        assay_std.rename(columns={"CMC": "assay_std"}, inplace=True)

        time_std = rep_avg.groupby("Assay")["CMC"].std().reset_index()
        time_std.rename(columns={"CMC": "time_std"}, inplace=True)

        replicate_std.to_csv(os.path.join(output_folder, f"{surf}_replicate_std.csv"), index=False)
        assay_std.to_csv(os.path.join(output_folder, f"{surf}_assay_std.csv"), index=False)
        time_std.to_csv(os.path.join(output_folder, f"{surf}_time_std.csv"), index=False)

        # Plot 1: Std Dev of Replicates vs Time (per Trial)
        plt.figure()
        for assay in replicate_std["Assay"].unique():
            sub = replicate_std[replicate_std["Assay"] == assay]
            plt.plot(sub["Time_min"], sub["replicate_std"], marker='o', label=f"Trial {assay}")
        plt.title(f"Std Dev of Replicates vs Time ({surf})")
        plt.xlabel("Time (min)")
        plt.ylabel("Std Dev (CMC)")
        plt.legend(title="Trial")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{surf}_plot_replicate_std.png"))
        plt.close()

        # Plot 2: Std Dev Across Trials vs Time
        plt.figure()
        plt.plot(assay_std["Time_min"], assay_std["assay_std"], marker='o')
        plt.title(f"Std Dev Across Trials vs Time ({surf})")
        plt.xlabel("Time (min)")
        plt.ylabel("Std Dev (CMC across trial averages)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{surf}_plot_assay_std.png"))
        plt.close()

        # Plot 3: Std Dev Across Timepoints per Trial
        plt.figure()
        plt.bar(time_std["Assay"], time_std["time_std"])
        plt.title(f"Std Dev Across Timepoints per Trial ({surf})")
        plt.xlabel("Trial")
        plt.ylabel("Std Dev (CMC over time)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{surf}_plot_time_std.png"))
        plt.close()
