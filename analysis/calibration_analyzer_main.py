import sys
import os
import pandas as pd

# Ensure calibration_analyzer is importable
sys.path.append("../utoronto_demo")
import calibration_analyzer as analyzer

import ast

def clean_tuple_strings(df):
    """Convert stringified tuples like '(3.4, None)' into 3.4."""
    def convert(val):
        if isinstance(val, str) and val.startswith("(") and val.endswith(")"):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, tuple) and parsed[1] is None:
                    return parsed[0]
                return val
            except:
                return val
        return val

    return df.map(convert)

def run_analysis_on_existing_output(folder_path):
    folder_path = os.path.abspath(folder_path)
    summary_file = os.path.join(folder_path, "experiment_summary.csv")

    print("Looking for file at:", summary_file)
    
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"experiment_summary.csv not found in {summary_file}")
    
    df = pd.read_csv(summary_file)
    df = clean_tuple_strings(df)

    print(df)
    for metric in ['deviation', 'time', 'variability']:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')

        print(f"Loaded {len(df)} rows from {summary_file}")
    
    if "liquid" not in df.columns:
        df["liquid"] = "Water"
    
    # Skip SHAP analysis since XGBoost is broken
    print("Skipping SHAP analysis due to XGBoost issues")
    
    # Generate all the plots that should work
    try:
        analyzer.plot_time_vs_deviation(df, folder_path)
        print("✅ Time vs deviation scatter plot created")
    except Exception as e:
        print(f"❌ Time vs deviation plot failed: {e}")
    
    try:
        analyzer.plot_boxplots(df, folder_path)
        print("✅ Boxplots created")
    except Exception as e:
        print(f"❌ Boxplots failed: {e}")
    
    try:
        best_trials = analyzer.get_top_trials(df, folder_path, weight_time=1.0, weight_deviation=0.5, weight_variability=2.0, top_n=3)
        analyzer.plot_top_trial_histograms(best_trials, folder_path)
        print("✅ Top trials histograms created")
    except Exception as e:
        print(f"❌ Top trials histograms failed: {e}")
    
    try:
        analyzer.plot_learning_curves(df, folder_path)
        print("✅ Learning curves created")
    except Exception as e:
        print(f"❌ Learning curves failed: {e}")
    
    try:
        analyzer.plot_improvement_summary(df, folder_path)
        print("✅ Improvement summary created")
    except Exception as e:
        print(f"❌ Improvement summary failed: {e}")
    
    # Look for raw measurements file for volume plot
    import glob
    raw_files = glob.glob(os.path.join(folder_path, "*raw_measurements*.csv"))
    if raw_files:
        try:
            raw_df = pd.read_csv(raw_files[0])
            analyzer.plot_measured_volume_over_time(raw_df, folder_path)
            print("✅ Measured volume over time plot created")
        except Exception as e:
            print(f"❌ Measured volume plot failed: {e}")
    else:
        print("ℹ️ No raw measurements file found - skipping volume over time plot")
    
    print("✅ Analysis complete. Results saved to:", folder_path)

if __name__ == "__main__":    
    run_analysis_on_existing_output(r"C:\Users\Imaging Controller\Desktop\Calibration_SDL_Output\New_Method\20251007_134359_water")
