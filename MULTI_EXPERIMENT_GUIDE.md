# Multi-Experiment Calibration System

## Overview

The calibration system now supports running multiple experiments with different configurations automatically. This allows you to:

- Compare different liquids, volumes, optimization methods
- Run parameter sensitivity studies
- Batch experiments with automatic error handling and reporting

## Quick Start

### Single Experiment with Custom Parameters

```python
from workflows.calibration_sdl_modular import run_single_experiment

# Run one experiment with custom settings
result = run_single_experiment({
    'liquid': 'water',
    'volumes': [0.1, 0.05, 0.01], 
    'seed': 42,
    'precision_replicates': 6
})

print(f"Success: {result['success']}")
print(f"Completed: {result['completed_volumes']}/{result['total_volumes']} volumes")
```

### Multiple Experiments

```python
from workflows.calibration_sdl_modular import run_multiple_experiments

# Compare different liquids
experiments = [
    {'liquid': 'water', 'seed': 1},
    {'liquid': 'glycerol', 'seed': 2},
    {'liquid': 'ethanol', 'seed': 3}
]

results = run_multiple_experiments(experiments)
```

## Configuration Parameters

You can override any of these parameters:

### Basic Settings
- `liquid`: 'water', 'glycerol', 'ethanol', etc.
- `simulate`: True/False 
- `seed`: Random seed for reproducibility
- `volumes`: List of volumes in mL, e.g., [0.1, 0.05, 0.01]
- `max_wells`: Maximum wells to use (e.g., 96, 48, 144)

### Experiment Control
- `precision_replicates`: Number of precision test replicates (3-6)
- `initial_suggestions`: Number of SOBOL trials (default 5)
- `base_time_seconds`: Time threshold for optimization acceptance

### Optimization Settings
- `use_llm_for_screening`: True/False - Use LLM vs SOBOL for initial exploration
- `use_llm_for_optimization`: True/False - Use LLM vs Bayesian for optimization
- `bayesian_model_type`: 'qEI', 'qLogEI', 'qNEHVI'

### Overaspirate Settings
- `overaspirate_base_ul`: Base overaspirate volume (μL)
- `overaspirate_scaling_percent`: Additional percentage of volume
- `auto_calibrate_overvolume`: True/False - Auto-calibrate overaspirate parameters

## Common Use Cases

### 1. Liquid Comparison Study
```python
liquid_study = [
    {'liquid': 'water', 'seed': 1},
    {'liquid': 'glycerol', 'seed': 2, 'base_time_seconds': 30},  # Longer time for viscous
    {'liquid': 'ethanol', 'seed': 3}
]
run_multiple_experiments(liquid_study)
```

### 2. Volume Range Optimization
```python
volume_study = [
    {'volumes': [0.2, 0.1, 0.05], 'seed': 10},      # Large volumes
    {'volumes': [0.1, 0.05, 0.025], 'seed': 20},    # Medium volumes  
    {'volumes': [0.05, 0.025, 0.01], 'seed': 30}    # Small volumes
]
run_multiple_experiments(volume_study)
```

### 3. Precision Sensitivity Study
```python
precision_study = [
    {'precision_replicates': 3, 'seed': 100},
    {'precision_replicates': 4, 'seed': 200}, 
    {'precision_replicates': 6, 'seed': 300}
]
run_multiple_experiments(precision_study)
```

### 4. Optimization Method Comparison
```python
optimization_study = [
    {'use_llm_for_optimization': False, 'bayesian_model_type': 'qEI', 'seed': 1},
    {'use_llm_for_optimization': False, 'bayesian_model_type': 'qLogEI', 'seed': 2}, 
    {'use_llm_for_optimization': True, 'seed': 3}
]
run_multiple_experiments(optimization_study)
```

## Output and Results

Each experiment returns a result dictionary with:
- `success`: Boolean indicating if all volumes completed
- `completed_volumes`: Number of volumes successfully calibrated
- `total_volumes`: Total number of volumes attempted
- `autosave_dir`: Directory containing experiment outputs
- `report_path`: Path to the calibration report
- `volume_report_data`: Detailed per-volume statistics

The multi-experiment runner provides:
- Real-time progress updates
- Error handling with continue/stop options
- Summary statistics across all experiments
- Individual experiment results for analysis

## Example Files

- `example_multi_experiments.py`: Complete examples of different experiment types
- Run the original calibration: `python calibration_sdl_modular.py` (single experiment)
- Run multi-experiments: Modify the `if __name__ == "__main__":` section or create your own script

## Tips

1. **Seeds**: Use different seeds for each experiment to get independent results
2. **Error Handling**: Failed experiments won't stop the batch - you can choose to continue
3. **Results Analysis**: Save the results list and analyze completion rates, optimal parameters, etc.
4. **Resource Management**: Consider `max_wells` limits when planning large batches
5. **Simulation Mode**: Test your experiment designs in simulation first (`simulate: True`)