#!/usr/bin/env python3
"""
Example script showing how to run multiple calibration experiments with different configurations.

This demonstrates the new multi-experiment capability that allows you to:
1. Run the same experiment with different parameters
2. Compare different liquids, volumes, optimization methods, etc.
3. Automatically batch multiple experiments with error handling
"""

from workflows.calibration_sdl_modular import run_multiple_experiments, run_single_experiment

def main():
    print("=== CALIBRATION MULTI-EXPERIMENT EXAMPLES ===\n")
    
    # Example 1: Single experiment with custom parameters
    print("📋 Example 1: Single experiment with custom parameters")
    result = run_single_experiment({
        'liquid': 'water',
        'volumes': [0.1, 0.05],
        'seed': 42,
        'precision_replicates': 3
    })
    print(f"Result: {result['success']} - {result['completed_volumes']}/{result['total_volumes']} volumes\n")
    
    # Example 2: Compare different liquids
    print("📋 Example 2: Compare different liquids")
    liquid_experiments = [
        {'liquid': 'water', 'seed': 1},
        {'liquid': 'glycerol', 'seed': 2},
        {'liquid': 'ethanol', 'seed': 3}
    ]
    results = run_multiple_experiments(liquid_experiments)
    print(f"Completed {len(results)} liquid comparison experiments\n")
    
    # Example 3: Test different volume sets
    print("📋 Example 3: Test different volume sets")
    volume_experiments = [
        {'volumes': [0.1, 0.05, 0.01], 'seed': 10},
        {'volumes': [0.2, 0.1, 0.05], 'seed': 20},
        {'volumes': [0.05, 0.025, 0.01], 'seed': 30}
    ]
    results = run_multiple_experiments(volume_experiments)
    print(f"Completed {len(results)} volume comparison experiments\n")
    
    # Example 4: Optimization method comparison
    print("📋 Example 4: Compare optimization methods")
    optimization_experiments = [
        {'use_llm_for_optimization': False, 'bayesian_model_type': 'qEI', 'seed': 100},
        {'use_llm_for_optimization': False, 'bayesian_model_type': 'qLogEI', 'seed': 200},
        {'use_llm_for_optimization': True, 'seed': 300}
    ]
    results = run_multiple_experiments(optimization_experiments)
    print(f"Completed {len(results)} optimization method experiments\n")
    
    # Example 5: Parameter sensitivity study
    print("📋 Example 5: Parameter sensitivity study")
    sensitivity_experiments = [
        {'precision_replicates': 3, 'max_wells': 48, 'seed': 1000},
        {'precision_replicates': 4, 'max_wells': 96, 'seed': 2000},
        {'precision_replicates': 6, 'max_wells': 144, 'seed': 3000}
    ]
    results = run_multiple_experiments(sensitivity_experiments)
    print(f"Completed {len(results)} sensitivity study experiments\n")

def custom_experiment_example():
    """Example of creating a completely custom experiment batch"""
    
    # Define your own experiment configurations
    my_experiments = [
        {
            'liquid': 'water',
            'volumes': [0.1, 0.05, 0.01],
            'precision_replicates': 4,
            'seed': 42,
            'max_wells': 96,
            'use_llm_for_optimization': False
        },
        {
            'liquid': 'glycerol', 
            'volumes': [0.1, 0.05, 0.01],
            'precision_replicates': 6,  # More replicates for viscous liquid
            'seed': 43,
            'max_wells': 120,
            'base_time_seconds': 30,  # Longer time for viscous liquid
            'use_llm_for_optimization': True
        },
        {
            'liquid': 'water',
            'volumes': [0.2, 0.1, 0.05],  # Different volume range
            'precision_replicates': 3,
            'seed': 44,
            'overaspirate_base_ul': 10.0,  # Higher overaspirate
            'overaspirate_scaling_percent': 8.0
        }
    ]
    
    print("🔬 Running custom experiment batch...")
    results = run_multiple_experiments(my_experiments)
    
    # Analyze results
    print(f"\n📊 Custom Experiment Analysis:")
    for result in results:
        if result.get('success', False):
            print(f"✅ Experiment {result['experiment_number']}: "
                  f"{result['liquid']} - {result['completed_volumes']}/{result['total_volumes']} volumes completed")
        else:
            print(f"❌ Experiment {result['experiment_number']}: Failed")
    
    return results

if __name__ == "__main__":
    # Run the examples
    main()
    
    # Uncomment to run custom experiments
    # custom_experiment_example()