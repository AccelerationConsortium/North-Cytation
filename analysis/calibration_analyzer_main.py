import sys
import os
import pandas as pd

# Ensure calibration_analyzer is importable
sys.path.append("../utoronto_demo")
import calibration_analyzer as analyzer

def run_analysis_on_existing_output(folder_path):
    folder_path = os.path.abspath(folder_path)
    summary_file = os.path.join(folder_path, "experiment_summary.csv")

    print("Looking for file at:", summary_file)
    
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"experiment_summary.csv not found in {summary_file}")
    
    df = pd.read_csv(summary_file)
    print(f"Loaded {len(df)} rows from {summary_file}")
    
    if "liquid" not in df.columns:
        df["liquid"] = "Water"
    
    analyzer.run_shap_analysis(df, folder_path)
    analyzer.plot_boxplots(df, folder_path)
    analyzer.plot_pairplot(df, folder_path)
    analyzer.plot_learning_curves(df, folder_path)
    analyzer.plot_improvement_summary(df, folder_path)
    analyzer.plot_time_vs_deviation(df, folder_path)
    
    print("✅ Analysis complete. Results saved to:", folder_path)

if __name__ == "__main__":
    run_analysis_on_existing_output("output/experiment_calibration_20250619_163030")
