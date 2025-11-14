#!/usr/bin/env python3
"""
Universal Calibration System Runner
==================================

Universal entry point for running calibration experiments.
Works with any hardware configuration - no hardware-specific assumptions.

Usage:
    python run_calibration.py

Configuration:
    Edit experiment_config.yaml to configure your specific:
    - Hardware (simulation or real instruments)
    - Target volumes
    - Parameter bounds
    - Liquid properties
    - Measurement budget

Output:
    Results automatically saved to output/ directory with:
    - Raw measurement data (CSV)
    - Optimal parameter sets (CSV) 
    - Experiment summary (JSON)
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration_modular_v2 import ExperimentConfig, CalibrationExperiment


def setup_logging():
    """Configure logging for the calibration run."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('calibration_experiment.log')
        ]
    )


def main():
    """Run calibration experiment based on configuration."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Universal Calibration System")
    
    try:
        # Load configuration
        config_path = Path(__file__).parent / "experiment_config.yaml"
        logger.info(f"Loading configuration from {config_path}")
        
        config = ExperimentConfig.from_yaml(str(config_path))
        logger.info(f"Configuration loaded: {config.get_experiment_name()}")
        logger.info(f"Target volumes: {config.get_target_volumes_ml()} mL")
        logger.info(f"Execution mode: {'Simulation' if config.is_simulation() else 'Hardware'}")
        
        # Create and run experiment
        experiment = CalibrationExperiment(config)
        logger.info("Starting experiment execution...")
        
        results = experiment.run()
        
        # Display results summary (hardware-agnostic)
        logger.info("=" * 60)
        logger.info("CALIBRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        logger.info(f"Experiment: {results.experiment_name}")
        logger.info(f"Total measurements: {results.total_measurements}")
        logger.info(f"Total duration: {results.total_duration_s:.1f} seconds")
        logger.info(f"Volumes calibrated: {len(results.volume_results)}")
        
        # Overall statistics
        stats = results.overall_statistics
        logger.info(f"Success rate: {stats['success_rate']:.1%}")
        logger.info(f"Mean score: {stats['mean_score']:.3f}")
        logger.info(f"High quality trials: {stats['within_tolerance_count']}")
        
        # Results location
        logger.info(f"\nResults saved to: {experiment.output_dir}")
        logger.info("Files generated:")
        logger.info("  - optimal_conditions.csv (best parameters for each volume)")
        logger.info("  - raw_measurements.csv (all measurement data)")
        logger.info("  - experiment_summary.json (complete statistics)")
        
        logger.info("\n[SUCCESS] Calibration experiment completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.error("Check configuration and try again")
        raise


if __name__ == "__main__":
    main()