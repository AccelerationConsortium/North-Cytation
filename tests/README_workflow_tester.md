# Surfactant Workflow Component Tester

## Overview
This GUI application allows you to test individual components of the surfactant workflow repeatedly to validate consistency and reliability before running critical experiments.

## Usage

### 1. Launch the Application
```bash
cd tests/
python surfactant_workflow_tester.py
```

### 2. Initialize System
- **Simulation Mode**: Click "Initialize System (Simulate)" for safe testing without hardware
- **Real Hardware**: Click "Initialize System (Real Hardware)" to test with actual equipment

### 3. Test Categories

#### Wellplate Movement Tests
- **Cytation Movement Cycle**: Move wellplate to Cytation → back to pipetting area
- **Pipetting Area Cycle**: Origin → pipetting area positioning  
- **Full Position Cycle**: Complete cycle through all positions
- **Position Status Check**: Query current wellplate position

#### Robot Operation Tests  
- **Vial Movement Cycle**: Move test vial to clamp → return home
- **Pipet Usage Cycle**: Test pipet attachment/removal operations
- **Liquid Handling Test**: Safe aspirate/dispense cycle with water
- **Vortex Operation Test**: Test vial vortexing with safe parameters

#### Cytation Operation Tests
- **Carrier In/Out Cycle**: Test Cytation carrier movement
- **Quick Measurement**: Fast measurement protocol test
- **Full Cytation Workflow**: Complete move → measure → return cycle  
- **Shake Protocol Test**: Test plate shaking protocols

### 4. Test Configuration
- **Repeat Count**: Number of times to repeat each test (default: 3)
- **Delay**: Seconds between test iterations (default: 2.0)
- **STOP Button**: Emergency stop for any running test

### 5. Results Monitoring
- **Progress Bar**: Real-time test progress
- **Statistics**: Success/error counters 
- **Test Log**: Detailed timestamped log of all operations

## Safety Features
- **Emergency Stop**: Stop button halts current test cycle immediately
- **Error Isolation**: Individual test failures don't stop the entire cycle
- **Simulation Mode**: Test logic without moving hardware
- **Safe Test Values**: All liquid volumes and movements use conservative parameters

## Typical Use Cases

### Pre-Experiment Validation
1. Run "Full Position Cycle" 5 times to ensure wellplate movement reliability
2. Run "Cytation Movement Cycle" 3 times to verify Cytation positioning 
3. Run "Vial Movement Cycle" 3 times to check robot precision

### Troubleshooting Issues  
1. Use single iterations with detailed logging to isolate problems
2. Compare simulation vs. hardware results
3. Test specific operations that failed in full workflow

### Equipment Maintenance
1. Run extended cycles (10+ repetitions) to check for wear/drift
2. Monitor timing consistency between iterations
3. Validate after hardware adjustments or calibration

## Tips
- Always test in simulation mode first to validate logic
- Use shorter delays (0.5s) for rapid testing, longer delays (5s+) for thorough validation
- Check the test log for any warnings or unexpected behaviors
- Monitor success rates - consistent 100% success indicates reliable operation
- Stop and investigate if error rates exceed 5%

## File Dependencies
- Requires `master_usdl_coordinator.py` and related system files
- Uses vial configuration from `../status/surfactant_grid_vials_expanded.csv`
- Compatible with existing workflow structure and patterns