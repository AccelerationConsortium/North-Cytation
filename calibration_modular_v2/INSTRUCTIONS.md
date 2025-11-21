# Calibration Modular V2 - User Instructions

## Overview

This is a next-generation calibration system that uses Bayesian optimization to find optimal pipetting parameters for different volumes. The system is hardware-agnostic with clean protocol abstraction, making it easy to adapt to different liquid handling systems.

## Key Features

- **Hardware Abstraction**: Easy to switch between simulation and real hardware
- **Bayesian Optimization**: Multi-objective optimization using Ax platform
- **Adaptive Measurements**: Conditional replicates based on measurement quality
- **Volume-Dependent Parameters**: Automatic re-optimization for different volumes
- **Type-Safe Configuration**: YAML-based config with comprehensive validation
- **Transfer Learning**: Parameter inheritance between volumes
- **Optional LLM Integration**: AI-powered parameter suggestions (experimental)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Basic Calibration

```bash
cd calibration_modular_v2
python run_calibration.py
```

This will use the default configuration with simulated measurements.

## Customizing Your Setup

### Step 1: Update Your Experiment Config

Edit `experiment_config.yaml` to customize your calibration:

#### Basic Experiment Settings
```yaml
experiment:
  name: "my_calibration_test"
  liquid: "water"  # or "glycerol", "ethanol", etc.
  description: "Your experiment description"

volumes:
  targets_ml: [0.005, 0.01, 0.025, 0.05]  # Volumes to calibrate

execution:
  simulate: false  # Set to true for simulation, false for hardware
```

#### Parameter Space (Hardware-Specific)
```yaml
hardware_parameters:
  aspirate_speed:
    bounds: [5, 35]
    default: 17
    description: "Aspiration speed (hardware units)"
    
  dispense_speed:
    bounds: [5, 35] 
    default: 17
    description: "Dispense speed (hardware units)"
    
  # Add your hardware-specific parameters here
```

#### Optimization Settings
```yaml
optimization:
  objectives:
    accuracy_weight: 0.5    # Weight for volume accuracy
    precision_weight: 0.4   # Weight for measurement precision  
    time_weight: 0.1        # Weight for operation speed
    
  optimizer:
    type: "multi_objective"  # or "single_objective"
    backend: "qNEHVI"       # Optimizer algorithm
```

### Step 2: Write Your Hardware Protocol

Create a new protocol file (e.g., `calibration_protocol_myrobot.py`) implementing the abstract interface:

```python
from calibration_protocol_base import CalibrationProtocolBase
from typing import Dict, Any, List, Optional

class MyRobotCalibrationProtocol(CalibrationProtocolBase):
    """Custom calibration protocol for your liquid handling system."""
    
    def initialize(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize your hardware."""
        liquid = cfg['experiment']['liquid']
        
        # Initialize your hardware here
        # my_robot = MyRobot()
        # my_robot.initialize()
        
        return {
            'initialized_at': datetime.now(),
            'hardware_type': 'my_robot_system',
            'liquid': liquid,
            # Add your hardware state here
        }
    
    def measure(self, state: Dict[str, Any], volume_mL: float, 
                params: Dict[str, Any], replicates: int = 1) -> List[Dict[str, Any]]:
        """Perform measurement with given parameters."""
        results = []
        
        for rep in range(replicates):
            # Extract parameters
            aspirate_speed = params.get('aspirate_speed', 20)
            dispense_speed = params.get('dispense_speed', 15)
            
            # Perform your pipetting operation
            # measured_volume_mL = my_robot.pipette_and_measure(
            #     volume_mL, aspirate_speed, dispense_speed
            # )
            
            # For now, simulate
            import random
            measured_volume_mL = volume_mL + random.uniform(-0.01, 0.01) * volume_mL
            
            results.append({
                'replicate': rep + 1,
                'volume': measured_volume_mL,
                'elapsed_s': 2.0,  # Your actual timing
                'target_volume_mL': volume_mL,
                **params  # Echo back parameters
            })
            
        return results
    
    def wrapup(self, state: Dict[str, Any]) -> None:
        """Clean up hardware resources."""
        # my_robot.shutdown()
        print("Hardware cleanup completed")

# Backward compatibility functions
_protocol_instance = MyRobotCalibrationProtocol()

def initialize(cfg): return _protocol_instance.initialize(cfg)
def measure(state, volume_mL, params, replicates=1): 
    return _protocol_instance.measure(state, volume_mL, params, replicates)
def wrapup(state): return _protocol_instance.wrapup(state)
```

### Step 3: Update Protocol Configuration

In `experiment_config.yaml`, point to your protocol:

```yaml
protocol:
  hardware_module: "calibration_protocol_myrobot"  # Your protocol file (no .py)
  simulation_module: "calibration_protocol_simulated"  # Keep this for simulation
```

## Protocol Interface Requirements

Your protocol **must** implement these three methods:

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

## Measurement Result Format

Each measurement result must include:

```python
{
    'replicate': 1,                    # Replicate number (1-based)
    'volume': 0.0089,                  # Measured volume in mL
    'elapsed_s': 3.2,                  # Time taken in seconds
    'target_volume_mL': 0.01,          # Target volume
    # Plus any parameters you want to echo back
    'aspirate_speed': 15,
    'dispense_speed': 12,
    # etc.
}
```

## Advanced Features

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
    config_path: "llm_config.json"

screening:
  use_llm_suggestions: true
  llm_config_path: "calibration_screening_llm_template.json"
```

### Adaptive Measurements
Control when additional replicates are performed:

```yaml
adaptive_measurement:
  enabled: true
  deviation_threshold_pct: 10.0    # Run more replicates if >10% deviation
  base_replicates: 1               # Start with 1 replicate
  additional_replicates: 2         # Add 2 more if needed
```

## Directory Structure

```
calibration_modular_v2/
├── run_calibration.py              # Main entry point
├── experiment_config.yaml          # Configuration file
├── calibration_protocol_hardware.py    # North Robot protocol (example)
├── calibration_protocol_simulated.py   # Simulation protocol
├── calibration_protocol_base.py        # Abstract base class
├── experiment.py                   # Main experiment orchestration
├── config_manager.py              # Configuration loading
├── data_structures.py             # Type-safe data classes
├── bayesian_recommender.py        # Optimization engine
├── analysis.py                    # Statistical analysis
├── output/                        # Results and plots
└── INSTRUCTIONS.md                # This file
```

## Troubleshooting

### Common Issues

**Import Errors**: Make sure `requirements.txt` is installed and you're in the right directory.

**Protocol Not Found**: Check that your protocol file is in `calibration_modular_v2/` and the name in config matches the filename.

**Configuration Errors**: The config is strictly validated. Check YAML syntax and required fields.

**Hardware Simulation**: If you want to test without hardware, set `execution.simulate: true` in config.

### Getting Help

1. Check the example protocols (`calibration_protocol_hardware.py`, `calibration_protocol_simulated.py`)
2. Look at the type definitions in `data_structures.py` for required data formats
3. Examine `experiment_config.yaml` for all configuration options

## Example Workflow

1. **Start with simulation**: Set `simulate: true`, run with default config
2. **Customize parameters**: Edit parameter bounds for your hardware
3. **Implement your protocol**: Create new protocol file with your hardware interface
4. **Test with real hardware**: Set `simulate: false`, run calibration
5. **Analyze results**: Check `output/` directory for results and plots

The system will automatically optimize parameters, handle measurement replicates, and provide comprehensive analysis of calibration quality.