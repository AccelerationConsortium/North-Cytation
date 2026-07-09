# SDL Pipette Calibration

Automated pipetting-parameter calibration using multi-objective Bayesian
optimization. Given a target volume and a way to measure delivered volume, the
system searches over a hardware parameter space (speeds, wait times, air gaps,
blowout, overaspirate) and returns an optimized recipe balancing accuracy,
precision, and time.

The decision layer is **purely software** — no specific robot or balance is
required. You plug in a protocol module that implements four functions
(`initialize`, `measure`, `wrapup`, `get_parameter_constraints`) and the
optimizer takes it from there. A simulated protocol is bundled so you can run
the entire pipeline end-to-end with no hardware at all.

## Features

- **Hardware Abstraction** — swap protocols by editing one line of YAML
- **Bayesian Optimization** — multi-objective search via the Ax platform (qNEHVI, GPEI)
- **Adaptive Measurements** — optional extra replicates on noisy trials
- **Volume-Dependent Parameters** — re-optimize per target volume with transfer learning
- **Type-Safe Configuration** — YAML with strict schema validation
- **External Data Bootstrapping** — seed the optimizer from prior experiments (optional)
- **LLM-Guided Screening** — experimental AI-assisted parameter suggestions

## Quick Start

### 1. Install

```bash
cd sdl_pipette_calibration
pip install -r requirements.txt
```

### 2. Run a simulated calibration

Edit `experiment_config.yaml` and set:

```yaml
experiment:
  simulate: true
```

Then:

```bash
python run_calibration.py
```

Results land in `output/<run_name>/` — CSV summaries, optimized-parameter
files, and plots. No hardware needed.

### 3. Validate the optimized parameters

Point `validation.optimal_conditions_file` at the CSV the calibration produced,
then:

```bash
python run_validation.py
```

## Customizing Your Setup

All behavior is controlled by [`experiment_config.yaml`](experiment_config.yaml).
The file is organized into commented sections: `experiment`,
`calibration_parameters`, `hardware_parameters`, `optimization`, `output`,
`validation`, `screening`, `tolerances`, `adaptive_measurement`. Skim it once
— the inline comments explain each block.

### Basic experiment settings

```yaml
experiment:
  liquid: water                                    # label, propagated to outputs
  volume_targets_ml: [0.005, 0.01, 0.025, 0.05]    # volumes to calibrate
  simulate: true                                    # false = run on real hardware
  hardware_protocol: calibration_protocol_myrobot   # your protocol (used when simulate: false)
  simulation_protocol: calibration_protocol_simulated
  name: my_calibration_run
  description: Testing accuracy across four volumes
  random_seed: 30
  max_total_measurements: 96
  num_screening_trials: 8
```

### Hardware parameter search space

Each entry in `hardware_parameters` defines one dimension of the search:

```yaml
hardware_parameters:
  aspirate_speed:
    bounds: [2, 30]
    default: 10
    type: integer
    round_to_nearest: 1
    time_affecting: true
    description: Aspiration speed (relative units).

  aspirate_wait_time:
    bounds: [0.0, 30.0]
    default: 10.0
    type: float
    round_to_nearest: 0.1
    time_affecting: true
    description: Wait time after aspiration (seconds).
```

### Pinning parameters (skip optimization without deleting them)

Any parameter listed under `experiment.fixed_parameters` is held at the given
value and excluded from the search space, while its `hardware_parameters`
block (bounds/default/description) stays intact. Toggle a parameter in or out
of the optimizer by adding or removing its entry in `fixed_parameters` —
no need to rewrite the parameter definition.

```yaml
experiment:
  fixed_parameters:
    post_asp_air_vol: 0        # held constant; not tuned
    retract_speed: 5.0         # held constant; not tuned
```

### Optimization settings

```yaml
optimization:
  objectives:
    # weights must sum to 1.0
    accuracy_weight: 0.4
    precision_weight: 0.5
    time_weight: 0.1

  optimizer:
    type: multi_objective       # or single_objective
    backend: qNEHVI             # first-stage screening backend
    backend_subsequent: GPEI    # backend for subsequent volumes
```

### Writing your own protocol

Copy the template and fill in the TODO sections:

```bash
cp protocols/calibration_protocol_template.py protocols/calibration_protocol_myrobot.py
```

See `protocols/calibration_protocol_template.py` for a minimal, annotated
example showing exactly what to implement.

Then point the config at it:

```yaml
experiment:
  hardware_protocol: calibration_protocol_myrobot   # filename without .py
  simulation_protocol: calibration_protocol_simulated
  simulate: false
```

## Protocol Interface Requirements

Your protocol **must** implement these four methods:

### `initialize(cfg) -> Dict[str, Any]`
- Initialize hardware
- Return state dictionary with hardware objects/settings
- State will be passed to all subsequent calls

### `measure(state, volume_mL, params, replicates) -> List[Dict[str, Any]]`
- Perform pipetting measurements
- `params` contains optimization parameters (speeds, volumes, etc.)
- Return list of measurement dictionaries, one per replicate
- Each result must have: `replicate`, `volume` (measured in mL), `elapsed_s`

### `wrapup(state) -> None`
- Clean up hardware resources
- Move to safe positions, close connections, etc.

### `get_parameter_constraints(target_volume_ml) -> List[str]`
- Return hardware-specific optimization constraints
- Called for each target volume during optimization
- Return empty list `[]` if no constraints apply

## Measurement Result Format

Each measurement result must include:

```python
{
    'replicate': 1,                    # Replicate number (1-based)
    'volume': 0.0089,                  # Measured volume in mL
    'elapsed_s': 3.2,                  # Time taken in seconds
    'target_volume_mL': 0.01,          # Target volume
    # Plus any parameters you want to echo back
    'my_hardware_param': 15.5,         # Your hardware-specific parameter
    'my_timing_param': 2.0,            # Your hardware-specific timing
    # etc.
}
```

## Advanced Features

### Hardware Constraints
Define hardware-specific parameter relationships in your protocol file:

```python
def get_parameter_constraints(self, target_volume_ml: float) -> List[str]:
    """Return constraint strings for the optimizer."""
    constraints = []
    
    # Example: Tip volume constraint
    tip_volume_ml = 1.0  # Your tip capacity
    available_volume = tip_volume_ml - target_volume_ml
    constraints.append(f"my_air_param + overaspirate_vol <= {available_volume}")
    
    # Example: Hardware limits
    constraints.append("my_speed1 * my_speed2 <= 1000")
    
    return constraints
```

### External Data Integration
Load existing calibration data to bootstrap optimization:

```yaml
screening:
  external_data:
    enabled: true
    data_path: "my_previous_data.csv" 
    volume_filter_ml: 0.01  # Only use data for specific volume
```

### LLM-Powered Parameter Suggestions
Enable AI-powered parameter suggestions (experimental):

```yaml
optimization:
  llm_optimization:
    enabled: true
    config_path: "llm_recommender/calibration_screening_llm_template.json"

screening:
  use_llm_suggestions: true
  llm_config_path: "llm_recommender/calibration_screening_llm_template.json"
```

### Adaptive Measurements
Control when additional replicates are performed:

```yaml
adaptive_measurement:
  enabled: true
  base_replicates: 1                  # baseline replicate count per trial
  deviation_threshold_pct: 100.0      # trigger extra replicates if deviation > this
  penalty_variability_pct: 100.0      # trigger extra replicates if variability > this
```

Extra replicates are drawn adaptively (capped by
`experiment.max_replicates_per_trial`) when a trial's deviation or variability
exceeds either threshold.

## Directory Structure

```
sdl_pipette_calibration/
├── run_calibration.py                   # Main entry point
├── run_validation.py                    # Validation entry point
├── experiment_config.yaml               # Configuration file
├── experiment.py                        # Experiment orchestration
├── config_manager.py                    # Configuration loading & validation
├── data_structures.py                   # Type-safe data classes
├── optimization_structures.py           # Optimization objective definitions
├── bayesian_recommender.py              # Ax/BoTorch optimization engine
├── analysis.py                          # Per-trial statistical analysis
├── experiment_analysis.py               # Post-hoc analysis (feature importance, etc.)
├── visualization.py                     # Plot generation
├── csv_export.py                        # Results export
├── external_data.py                     # External data loader
├── protocol_loader.py                   # Protocol discovery
├── constraint_calibration.py            # Two-point overaspirate calibration
├── pipetting_wizard.py                  # Load & interpolate calibrated parameters
├── yaml_io.py                           # Round-trip YAML writes (preserves comments)
├── input_data/                          # Sample / external datasets
├── protocols/                           # Hardware protocol modules
│   ├── calibration_protocol_base.py     # Abstract base class
│   ├── calibration_protocol_template.py # Start here for new hardware
│   ├── calibration_protocol_simulated.py# Simulation (no hardware needed)
│   └── calibration_protocol_northrobot.py, ...  # Reference implementations
├── llm_recommender/                     # Optional LLM-guided screening
├── tools/                               # Demo GUI and dashboards (not required)
└── output/                              # Run outputs — results and plots (gitignored)
```

## Troubleshooting

### Common Issues

**Import Errors**: Make sure `requirements.txt` is installed and you're in the right directory.

**Protocol Not Found**: Check that your protocol file is in `sdl_pipette_calibration/protocols/` and the name in config matches the filename.

**Configuration Errors**: The config is strictly validated. Check YAML syntax and required fields.

**Hardware Simulation**: If you want to test without hardware, set `experiment.simulate: true` in the config.

### Getting Help

1. Start from `protocols/calibration_protocol_template.py` — it has TODO comments for every required method
2. Look at `data_structures.py` for the expected types and fields
3. `experiment_config.yaml` is fully annotated — it documents every option inline

## Example Workflow

1. **Start with simulation** — set `experiment.simulate: true`, run `python run_calibration.py`
2. **Inspect outputs** — look at `output/<run_name>/` for plots and CSVs
3. **Adjust parameter bounds** — tune `hardware_parameters` for your setup
4. **Write your protocol** — copy `protocols/calibration_protocol_template.py`
5. **Run on hardware** — set `experiment.simulate: false`, run `python run_calibration.py`
6. **Validate** — point `validation.optimal_conditions_file` at the run's CSV, run `python run_validation.py`

The optimizer handles replication, transfer learning between volumes, and
produces analysis outputs (plots, feature importance, statistical summaries)
automatically.
