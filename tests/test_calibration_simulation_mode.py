"""Simulation policy and the calibration-to-validation in-memory contract."""
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from sdl_pipette_calibration.config_manager import ExperimentConfig
from sdl_pipette_calibration.csv_export import CleanCSVExporter
from sdl_pipette_calibration.experiment import CalibrationExperiment
from sdl_pipette_calibration.run_validation import ValidationRunner
from sdl_pipette_calibration import bayesian_recommender as recommender
from sdl_pipette_calibration.optimization_structures import OptimizerType

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw():
    data = yaml.safe_load((ROOT / "sdl_pipette_calibration/experiment_config.yaml").read_text())
    data["experiment"]["simulate"] = True
    data["output"]["save_files"] = False
    data["optimization"]["use_bayesian"] = False
    return data


@pytest.mark.parametrize("simulate", [True, False])
@pytest.mark.parametrize("save_files", [True, False])
@pytest.mark.parametrize("bayesian", [True, False])
def test_independent_switches_and_hardware_defaults(raw, simulate, save_files, bayesian):
    raw["experiment"]["simulate"] = simulate
    raw["output"]["save_files"] = save_files
    raw["optimization"]["use_bayesian"] = bayesian
    config = ExperimentConfig(raw)
    assert config.should_save_files() == (not simulate or save_files)
    assert config.use_bayesian_optimization() == (not simulate or bayesian)


def test_standalone_defaults_remain_enabled(raw):
    del raw["output"]["save_files"]
    del raw["optimization"]["use_bayesian"]
    config = ExperimentConfig(raw)
    assert config.should_save_files()
    assert config.use_bayesian_optimization()


def test_all_calibration_output_sinks_are_disabled(raw, tmp_path):
    raw["output"]["base_directory"] = str(tmp_path)
    experiment = CalibrationExperiment(ExperimentConfig(raw))
    assert experiment.output_dir is None
    # None inputs intentionally prove we skip serialization as well as writes.
    experiment._save_incremental_data()
    experiment._save_incremental_optimal_conditions()
    experiment._save_measurement_immediately(None, 0.02, "screening")
    experiment._save_constraint_calibration_results(0.02, None, [None])
    experiment._export_results(None)
    experiment._generate_enhanced_outputs(None)
    assert list(tmp_path.iterdir()) == []


def test_shared_table_matches_csv_and_validation_does_not_read_file(raw, tmp_path):
    records = [{"volume_ul": 20, "measured_volume_ul": 19.5,
                "parameters": {"calibration": {"overaspirate_vol": 0.002},
                               "hardware_parameters": {"aspirate_speed": 10}},
                "status": "success"}]
    table = CleanCSVExporter.optimal_conditions_table(records)
    exporter = CleanCSVExporter(tmp_path, liquid="water")
    exporter._export_optimal_conditions(records)
    pd.testing.assert_frame_equal(table, pd.read_csv(tmp_path / "optimal_conditions_water.csv"),
                                  check_dtype=False)
    validator = ValidationRunner(ExperimentConfig(raw), optimal_conditions=table)
    table.loc[0, "volume_target_ul"] = 999
    with patch("pandas.read_csv", side_effect=AssertionError("Unexpected disk read")):
        conditions = validator._load_calibration_results()
    assert conditions.loc[0, "volume_target_ul"] == 20
    conditions.loc[0, "volume_target_ul"] = 888
    assert validator._load_calibration_results().loc[0, "volume_target_ul"] == 20
    validator._generate_outputs(None, None)
    assert validator.output_dir is None


@pytest.mark.skipif(not recommender.AX_AVAILABLE, reason="Requires calibration Ax 0.4.3 dependencies")
@pytest.mark.parametrize("first_volume", [True, False])
def test_real_ax_sobol_never_builds_bayesian_model_or_predicts(raw, first_volume):
    raw["experiment"]["random_seed"] = 0
    config = ExperimentConfig(raw)
    with patch.object(recommender.AxBayesianOptimizer, "_get_model_and_kwargs",
                      side_effect=AssertionError("Bayesian model requested")):
        optimizer = recommender.create_optimizer(config, 0.02, OptimizerType.MULTI_OBJECTIVE,
                                                num_sobol_trials=0, is_first_volume=first_volume)
    assert optimizer.sobol_only
    strategy = optimizer.state.ax_client.generation_strategy
    assert len(strategy._steps) == 1
    assert strategy._steps[0].num_trials == -1
    with patch.object(optimizer.state.ax_client, "get_model_predictions_for_parameterizations") as predictions:
        first = optimizer.suggest_parameters()
        second = optimizer.suggest_parameters()
        predictions.assert_not_called()
    assert first.to_protocol_dict() != second.to_protocol_dict()
    assert optimizer.config.random_seed == 0
    for params in [first, second]:
        assert params.calibration.overaspirate_vol >= 0


@pytest.mark.skipif(not recommender.AX_AVAILABLE, reason="Requires calibration Ax 0.4.3 dependencies")
@pytest.mark.parametrize("simulate", [True, False])
def test_bayesian_strategy_is_retained_when_enabled(raw, simulate):
    raw["experiment"]["simulate"] = simulate
    raw["optimization"]["use_bayesian"] = simulate  # Hardware ignores False.
    config = ExperimentConfig(raw)
    optimizer = recommender.create_optimizer(config, 0.02, num_sobol_trials=0)
    assert not optimizer.sobol_only
    assert optimizer.state.ax_client.generation_strategy._steps[0].model != recommender.Models.SOBOL
