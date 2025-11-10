"""
Test script for both Bayesian and LLM recommenders working together.
"""

from config_manager import ExperimentConfig
from bayesian_recommender import BayesianRecommender
from llm_recommender import create_llm_recommender

def test_recommenders():
    # Load configuration
    config = ExperimentConfig.from_yaml('experiment_config.yaml')
    
    print("=== Recommender Integration Test ===")
    print(f"Experiment: {config.get_experiment_name()}")
    print(f"Liquid: {config.get_liquid_name()}")
    print(f"Simulation mode: {config.is_simulation()}")
    print()
    
    # Test Bayesian recommender
    print("--- Bayesian Recommender ---")
    bayesian_rec = BayesianRecommender(config, 0.05)
    print(f"Optimizing parameters: {bayesian_rec.optimize_params}")
    print(f"Fixed parameters: {bayesian_rec.fixed_params}")
    
    # Generate suggestions
    bayesian_suggestions = bayesian_rec.suggest_parameters(3)
    print(f"Generated {len(bayesian_suggestions)} Bayesian suggestions")
    for i, suggestion in enumerate(bayesian_suggestions, 1):
        print(f"  Suggestion {i}: aspirate_speed={suggestion.aspirate_speed:.2f}, "
              f"dispense_speed={suggestion.dispense_speed:.2f}, "
              f"aspirate_wait_time={suggestion.aspirate_wait_time:.2f}s")
    print()
    
    # Test LLM recommender (screening phase)
    print("--- LLM Recommender (Screening) ---")
    llm_rec_screening = create_llm_recommender(config, 'screening')
    if llm_rec_screening:
        llm_suggestions = llm_rec_screening.suggest_parameters(2)
        print(f"Generated {len(llm_suggestions)} LLM screening suggestions")
        for i, suggestion in enumerate(llm_suggestions, 1):
            print(f"  Suggestion {i}: aspirate_speed={suggestion.aspirate_speed:.2f}, "
                  f"dispense_speed={suggestion.dispense_speed:.2f}")
    else:
        print("LLM screening disabled in configuration")
    print()
    
    # Test LLM recommender (optimization phase)  
    print("--- LLM Recommender (Optimization) ---")
    llm_rec_optimization = create_llm_recommender(config, 'optimization')
    if llm_rec_optimization:
        llm_suggestions = llm_rec_optimization.suggest_parameters(2)
        print(f"Generated {len(llm_suggestions)} LLM optimization suggestions")
    else:
        print("LLM optimization disabled in configuration")
    print()
    
    # Test parameter unification
    print("--- Parameter Unification Check ---")
    sample_bayesian = bayesian_suggestions[0]
    print("Bayesian parameter fields:")
    for field in sample_bayesian.__dataclass_fields__:
        value = getattr(sample_bayesian, field)
        print(f"  {field}: {value}")
    
    print("\nConfiguration parameters:")
    for param_name in config._config['parameters'].keys():
        param_config = config._config['parameters'][param_name]
        print(f"  {param_name}: bounds={param_config['bounds']}, default={param_config['default']}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_recommenders()