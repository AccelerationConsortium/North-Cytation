import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def load_avg_spectra(folder_name):
    base_path = os.path.join(".", folder_name)
    if not os.path.exists(base_path):
        print(f"Folder not found: {folder_name}")
        return None

    all_files = [f for f in os.listdir(base_path) if f.startswith("output_")]
    file_time_pairs = []
    for f in all_files:
        match = re.search(r"output_(\d+)", f)
        if match:
            file_time_pairs.append((f, int(match.group(1))))

    file_time_pairs.sort(key=lambda x: x[1])
    result_df = None

    for filename, time_min in file_time_pairs:
        path = os.path.join(base_path, filename)
        df = pd.read_csv(path)
        wavelengths = df['Wavelengths']
        replicate_cols = df.columns[2:]
        avg_spectrum = df[replicate_cols].mean(axis=1)

        if result_df is None:
            result_df = pd.DataFrame({'Wavelength': wavelengths})

        result_df[f"{time_min}"] = avg_spectrum

    return result_df.set_index('Wavelength')

def normalize_then_average(folder_name, ref_wavelength=700):
    base_path = os.path.join(".", folder_name)
    if not os.path.exists(base_path):
        print(f"Folder not found: {folder_name}")
        return None

    all_files = [f for f in os.listdir(base_path) if f.startswith("output_")]
    file_time_pairs = []
    for f in all_files:
        match = re.search(r"output_(\d+)", f)
        if match:
            file_time_pairs.append((f, int(match.group(1))))

    file_time_pairs.sort(key=lambda x: x[1])
    result_df = None

    for filename, time_min in file_time_pairs:
        path = os.path.join(base_path, filename)
        df = pd.read_csv(path)
        wavelengths = df['Wavelengths']
        replicate_cols = df.columns[2:]

        normalized_reps = []
        for col in replicate_cols:
            series = df[col]
            ref_val = series[df['Wavelengths'] == ref_wavelength].values
            if len(ref_val) == 0 or ref_val[0] == 0:
                print(f"Warning: bad ref value in {filename}, replicate {col}")
                continue
            norm_series = series / ref_val[0]
            normalized_reps.append(norm_series)

        if normalized_reps:
            avg_norm = pd.concat(normalized_reps, axis=1).mean(axis=1)
            if result_df is None:
                result_df = pd.DataFrame({'Wavelength': wavelengths})
            result_df[f"{time_min}"] = avg_norm

    return result_df.set_index('Wavelength')

# Updated save functions to direct to correct subfolders
def save_plot(df, title, ylabel, filename, graph_type="Spectra"):
    if df is None:
        print(f"Skipping plot for: {title} (no data)")
        return
    plt.figure(figsize=(10, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=f"{col} min")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    full_path = os.path.join("Graphs", graph_type, filename)
    plt.savefig(full_path)
    plt.close()

def extract_absorbance_at_wavelength(df, target_wavelength=595):
    """Return a Series of absorbance at target wavelength over time"""
    if df is None or target_wavelength not in df.index:
        return None
    return df.loc[target_wavelength]


def save_absorbance_vs_time_plot(abs_series, folder_name, ylabel, filename):
    if abs_series is None:
        print(f"Skipping absorbance plot for {folder_name} (no data at target wavelength)")
        return
    times = [int(col) for col in abs_series.index]
    values = abs_series.values

    plt.figure(figsize=(6, 4))
    plt.plot(times, values, marker='o')
    plt.xlabel("Time (min)")
    plt.ylabel(ylabel)
    plt.title(f"{folder_name} - {ylabel} at 595 nm")
    plt.grid(True)
    plt.tight_layout()
    full_path = os.path.join("Graphs", "Absorbance", filename)
    plt.savefig(full_path)
    plt.close()


def absorbance_to_concentration(absorbance_series):
    """Apply standard curve: C(t) = 0.0036*t + 0.0256 => invert: t = (C - 0.0256)/0.0036"""
    return (absorbance_series - 0.0256) / 0.0036  # returns concentration in µM

def plot_concentration_vs_time(conc_series, folder_name, label, filename):
    if conc_series is None:
        print(f"Skipping concentration plot for {folder_name} ({label})")
        return
    times = [int(col) for col in conc_series.index]
    values = conc_series.values

    plt.figure(figsize=(6, 4))
    plt.plot(times, values, marker='o')
    plt.xlabel("Time (min)")
    plt.ylabel("Concentration (µM)")
    plt.title(f"{folder_name} - {label} at 595 nm")
    plt.grid(True)
    plt.tight_layout()
    full_path = os.path.join("Graphs", "Concentration", filename)
    plt.savefig(full_path)
    plt.close()

def compute_dcdt(conc_series):
    """Compute dC/dt using central differences"""
    times = [int(col) for col in conc_series.index]
    values = conc_series.values
    df = pd.DataFrame({'time': times, 'conc': values}).sort_values('time')
    dcdt = df['conc'].diff().dropna() / df['time'].diff().dropna()
    return dcdt.mean()  # Average rate of change

def compute_dcdt_limited(conc_series, time_limit=15):
    """Compute dC/dt using only values up to a certain time limit"""
    if conc_series is None:
        return None
    df = pd.DataFrame({
        'time': [int(col) for col in conc_series.index],
        'conc': conc_series.values
    }).sort_values('time')
    df = df[df['time'] <= time_limit]
    if len(df) < 2:
        return None  # Not enough points
    dcdt = df['conc'].diff().dropna() / df['time'].diff().dropna()
    return dcdt.mean()

def get_reference_700nm_values(folder_name):
    """Return a dict of mean absorbance at 700nm per timepoint from raw data"""
    base_path = os.path.join(".", folder_name)
    all_files = [f for f in os.listdir(base_path) if f.startswith("output_")]

    file_time_pairs = []
    for f in all_files:
        match = re.search(r"output_(\d+)", f)
        if match:
            file_time_pairs.append((f, int(match.group(1))))

    file_time_pairs.sort(key=lambda x: x[1])
    ref_dict = {}

    for filename, time_min in file_time_pairs:
        path = os.path.join(base_path, filename)
        df = pd.read_csv(path)
        if 700 in df['Wavelengths'].values:
            idx = df['Wavelengths'] == 700
            replicate_cols = df.columns[2:]
            values = df.loc[idx, replicate_cols].values.flatten()
            ref_dict[str(time_min)] = values.mean()
        else:
            print(f"700 nm not found in {filename}")
            ref_dict[str(time_min)] = None

    return ref_dict

# Prepare text output
results = []

# Folders you specified
folders = [
    "June_2_2025_COF_PH_200uL",
    "June_3_2025_COF_PH_r2_200uL",
    "May_26_2025_COF-AO_200uL",
    "June_2_2025_COF_AO_200uL_run2",
    "May_26_2025_COF-BS_200uL",
    "June_2_2025_COF_BS_200uL_run2"
]

os.makedirs("Graphs", exist_ok=True)
# Ensure subfolders exist for organizing output
os.makedirs("Graphs/Spectra", exist_ok=True)
os.makedirs("Graphs/Absorbance", exist_ok=True)
os.makedirs("Graphs/Concentration", exist_ok=True)

for folder in folders:
    print(f"Processing corrected normalized concentration for: {folder}")
    try:
        avg_df = load_avg_spectra(folder)
        norm_df = normalize_then_average(folder, ref_wavelength=700)
        ref_700 = get_reference_700nm_values(folder)

        # Save spectra plots
        save_plot(avg_df, f"{folder} - Averaged Spectra", "Absorbance", f"{folder}_avg_spectra.png", graph_type="Spectra")
        save_plot(norm_df, f"{folder} - Normalized Spectra (Ref: 700 nm)", "Normalized Absorbance", f"{folder}_norm_spectra.png", graph_type="Spectra")

        # Absorbance at 595 nm
        abs_595 = extract_absorbance_at_wavelength(avg_df, 595)
        norm_595 = extract_absorbance_at_wavelength(norm_df, 595)

        save_absorbance_vs_time_plot(abs_595, folder, "Absorbance", f"{folder}_avg_595nm.png")
        save_absorbance_vs_time_plot(norm_595, folder, "Normalized Absorbance", f"{folder}_norm_595nm.png")

        # Concentration (avg)
        conc_595 = absorbance_to_concentration(abs_595)
        plot_concentration_vs_time(conc_595, folder, "Concentration", f"{folder}_conc_595nm.png")

        # Corrected normalized concentration
        corrected_norm_595 = pd.Series({
            col: norm_595[col] * ref_700.get(col, 1)
            for col in norm_595.index
        })
        corrected_conc_595 = absorbance_to_concentration(corrected_norm_595)
        plot_concentration_vs_time(corrected_conc_595, folder, "Corrected Normalized Concentration", f"{folder}_norm_corrected_conc_595nm.png")

        # Compute rates
        dcdt_avg = compute_dcdt(conc_595)
        dcdt_avg_limited = compute_dcdt_limited(conc_595, time_limit=15)
        dcdt_norm = compute_dcdt(corrected_conc_595)

        results.append(f"{folder}\n"
                       f"  dC/dt (avg): {dcdt_avg:.4f} µM/min\n"
                       f"  dC/dt (avg, up to 15min): {dcdt_avg_limited:.4f} µM/min\n"
                       f"  dC/dt (norm, corrected): {dcdt_norm:.4f} µM/min\n")

    except FileNotFoundError:
        print(f"⚠️ Skipped missing folder: {folder}")

# Save summary
summary_path = os.path.join("Graphs", "dC_dt_summary.txt")
with open(os.path.join("Graphs", "dC_dt_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print("✅ All figures and dC/dt summary saved in organized subfolders inside 'Graphs'.")