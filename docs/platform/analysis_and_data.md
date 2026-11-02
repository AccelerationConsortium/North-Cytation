# Analysis and Data Handling

Reference for processing, saving, and plotting experimental data produced by SDL workflows.
See also: `workflow_construction.md`, `recommender_guide.md`

---

## Central Data Structure: The Well Recipes DataFrame

Every workflow revolves around a single `well_recipes_df` that accumulates data over time:

```
wellplate_index | param_1 | param_2 | ... | volume_ul | well_type | output_1 | output_2 | ...
```

- **Recipe columns** (set before dispensing): input parameters, volumes, vial assignments
- **Measurement columns** (merged in after measurement): raw sensor readings, derived values
- **Metadata columns**: `well_type` (`experiment` | `control`), `replicate`, `control_type`

**Critical rule**: measurement results are always merged back into `well_recipes_df` by `wellplate_index` — they are never kept in a separate parallel structure.

```python
# Correct pattern: merge measurements into the recipe DataFrame
for _, row in measurement_data.iterrows():
    idx = int(row["wellplate_index"])
    mask = well_recipes_df["wellplate_index"] == idx
    well_recipes_df.loc[mask, "my_output"] = row["my_output"]
```

---

## Output Folder Setup

Create a timestamped experiment folder at the start of each run. All files for the run go inside it.

```python
import os
from datetime import datetime

def setup_experiment_folder(lash_e, experiment_tag=""):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_folder = "simulated" if lash_e.simulate else "experimental"
    name = f"{experiment_tag}_{timestamp}"
    base = os.path.join("output", sim_folder, name)
    os.makedirs(base, exist_ok=True)
    lash_e.logger.info(f"Output folder: {base}")
    return base
```

Subfolders for long runs:
```python
os.makedirs(os.path.join(base, "measurement_backups"), exist_ok=True)
os.makedirs(os.path.join(base, "plots"), exist_ok=True)
```

---

## Save at Every Step

**Always save `well_recipes_df` to CSV after each measurement or analysis step** — not just at the end. This allows recovery if the run is interrupted and provides a complete audit trail.

```python
def save_results(df, output_folder, label="results"):
    path = os.path.join(output_folder, f"{label}.csv")
    df.to_csv(path, index=False)
    return path
```

Canonical save points in a workflow:
1. After building well recipes (before dispensing) — saves the plan
2. After each measurement step — saves raw data immediately
3. After each analysis step — saves derived values
4. After each recommender call — saves updated results for that iteration
5. At the end of the run — final save with a clear `_final` label

```python
# Example: save after each plate measurement
well_recipes_df = measure_and_process(lash_e, well_recipes_df)
save_results(well_recipes_df, output_folder, label=f"results_after_plate_{plate_num}")

# Iteration saves
save_results(well_recipes_df, output_folder, label=f"results_iter_{iteration}")

# Final save
save_results(well_recipes_df, output_folder, label="results_final")
```

For timestamped backups (use when overwriting the same label each iteration):
```python
from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
df.to_csv(os.path.join(output_folder, f"results_iter_{iteration}_{ts}.csv"), index=False)
```

---

## Handling Raw Measurement Data from Cytation

`lash_e.measure_wellplate()` returns a `pd.DataFrame` with MultiIndex columns `(rep_protocol, wavelength)` — or `None` in simulate mode.

```python
data = lash_e.measure_wellplate(
    protocol_file_path=["protocols/my_protocol.prt"],
    wells_to_measure=None,
    plate_type="96 WELL PLATE",
)
# data.columns = MultiIndex: (replicate_protocol_name, wavelength_nm)
```

Extracting a single wavelength (e.g. absorbance at 600 nm):
```python
# Get first replicate, 600 nm column
turb_col = ("rep1_my_protocol", 600)  # actual key depends on protocol name
abs_600 = data[turb_col]
```

In practice, use helper functions that know the column structure for each protocol type. Always check for `None` before indexing:
```python
if data is not None:
    # process data
    pass
```

---

## Analysis Module Pattern

Analysis scripts live in `analysis/`. The standard signature:

```python
def analyze(measurement_df, well_recipes_df, output_folder, logger=None):
    """
    Args:
        measurement_df: Raw data from lash_e.measure_wellplate()
        well_recipes_df: Recipe DataFrame — receives derived columns in-place
        output_folder: Where to save plots and result CSVs
        logger: Optional logger (use lash_e.logger in workflows)

    Returns:
        well_recipes_df: Updated with new derived output columns
    """
    import logging
    _logger = logger or logging.getLogger(__name__)

    # 1. Extract values from raw measurement data
    # 2. Merge into well_recipes_df by wellplate_index
    # 3. Compute derived quantities (ratios, baselines, etc.)
    # 4. Save intermediate CSVs
    # 5. Generate plots
    # 6. Return updated DataFrame

    return well_recipes_df
```

Keep control wells and experiment wells separate during analysis:
```python
exp_df = well_recipes_df[well_recipes_df["well_type"] == "experiment"].copy()
ctrl_df = well_recipes_df[well_recipes_df["well_type"] == "control"].copy()
```

---

## Saving CSVs

```python
# Standard saves — no index (cleaner for downstream loading)
df.to_csv(os.path.join(output_folder, "results.csv"), index=False)

# When the index is meaningful (e.g. MultiIndex measurement data)
data.to_csv(os.path.join(output_folder, "raw_measurement.csv"), index=True)
```

For resilient saves with error logging:
```python
def safe_save_csv(df, path, logger=None):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        if logger:
            logger.info(f"Saved: {os.path.basename(path)}")
        return path
    except Exception as e:
        if logger:
            logger.error(f"Failed to save {path}: {e}")
        return None
```

---

## Plotting Conventions

**Always use `matplotlib.use('Agg')`** at the top of any analysis module — this prevents GUI windows from appearing during hardware runs and avoids threading errors.

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

Standard plot save pattern:
```python
fig, ax = plt.subplots(figsize=(8, 6))
# ... plot code ...
fig.tight_layout()
fig.savefig(os.path.join(output_folder, "my_plot.png"), dpi=150, bbox_inches='tight')
plt.close(fig)  # ALWAYS close — prevents memory leaks in long runs
```

**Never call `plt.show()`** in workflow or analysis code — it blocks execution. Save to file only.

Per-iteration plots: include iteration number and timestamp in the filename to avoid overwrites:
```python
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
fig.savefig(os.path.join(output_folder, f"iter{iteration}_{ts}.png"), dpi=150)
plt.close(fig)
```

For large datasets, use `rasterized=True` in scatter plots to reduce file size:
```python
ax.scatter(x, y, s=6, alpha=0.5, rasterized=True)
```

---

## Simulate Mode Handling

When `lash_e.simulate = True`, `measure_wellplate()` returns `None`. Analysis code must handle this and substitute synthetic data rather than propagating `None` into downstream steps:

```python
if lash_e.simulate:
    well_recipes_df = inject_synthetic_data(well_recipes_df)
else:
    data = lash_e.measure_wellplate(...)
    # merge data into well_recipes_df
```

---

## Synthetic Data for Simulation

When running in simulate mode, replace the `None` returned by the instrument with a function that computes plausible outputs from the input parameters already in `well_recipes_df`. This gives the recommender realistic signal to train on during development.

The synthetic function should capture the qualitative shape of your expected response — it does not need to be physically exact, just smooth and non-trivial so Bayesian optimization has something to optimize against.

```python
import numpy as np

def synthetic_output(row):
    """
    Compute a fake measurement result from input parameters.
    Replace this with a function that approximates your real system.
    """
    x = row["param_1"]
    y = row["param_2"]

    # Example: a noisy hill function in x, linear in y
    signal = (x ** 2) / (1.0 + x ** 2) + 0.1 * y
    noise = np.random.normal(0, 0.02)
    return signal + noise


def inject_synthetic_data(well_recipes_df):
    """Fill output columns with synthetic values when running in simulate mode."""
    well_recipes_df = well_recipes_df.copy()
    well_recipes_df["my_output"] = well_recipes_df.apply(synthetic_output, axis=1)
    return well_recipes_df
```

Keep the synthetic function in the same file as your analysis module. When the real experiment runs, the `if lash_e.simulate` branch is never entered and the function is never called.

**Simulation structure in a workflow:**
```python
if lash_e.simulate:
    well_recipes_df = inject_synthetic_data(well_recipes_df)
    lash_e.logger.info("Simulate mode: synthetic data injected")
else:
    data = lash_e.measure_wellplate(protocol_file_path=[...], ...)
    if data is None:
        raise RuntimeError("measure_wellplate() returned None in non-simulate mode")
    # merge data into well_recipes_df
    for _, row in data.iterrows():
        idx = int(row["wellplate_index"])
        mask = well_recipes_df["wellplate_index"] == idx
        well_recipes_df.loc[mask, "my_output"] = row["my_output"]
```

---

## Typical Analysis File Structure

```
analysis/
    my_experiment_analysis.py   # Main analysis entry point
    my_experiment_plots.py      # Plot helpers (import into main)
```

For simple experiments, everything can live in a single file. For complex multi-step analysis, split plotting into a separate module.

The workflow calls analysis functions inline:
```python
# In the workflow:
from analysis.my_experiment_analysis import analyze, plot_summary
well_recipes_df = analyze(raw_data, well_recipes_df, output_folder, lash_e.logger)
plot_summary(well_recipes_df, output_folder)
save_results(well_recipes_df, output_folder, label=f"results_iter_{iteration}")
```
