"""Each run saves its config and records its output folder in experiment_runs.csv."""
import csv
import logging
import os
from unittest.mock import Mock

import yaml

from experiment_tracking import experiment_run_logger
from master_usdl_coordinator import Lash_E


def start(tmp_path, config):
    logger = logging.getLogger("test_run_config_saving")
    tracker = experiment_run_logger.start_run("demo", True, "demo.log", logger, str(tmp_path / "logs"), config)
    tracker.csv_path = str(tmp_path / "experiment_runs.csv")  # Keep the real CSV untouched
    return tracker


def test_start_run_saves_config_and_finalize_records_it_with_output_folder(tmp_path):
    tracker = start(tmp_path, {"SIMULATE": True, "VOLUME_ML": 0.5})
    assert yaml.safe_load(open(tracker.config_file)) == {"SIMULATE": True, "VOLUME_ML": 0.5}
    assert tracker.config_file == os.path.join(str(tmp_path / "logs"), "experiment_run_configs", f"{tracker.run_id}.yaml")
    experiment_run_logger.record_output_folder(tracker, str(tmp_path / "out"))
    experiment_run_logger.finalize_run(tracker)
    row = list(csv.DictReader(open(tmp_path / "experiment_runs.csv")))[-1]
    assert row["config_file"] == tracker.config_file
    assert row["output_folder"] == str(tmp_path / "out")


def test_run_without_config_leaves_config_file_blank(tmp_path):
    tracker = start(tmp_path, None)
    experiment_run_logger.finalize_run(tracker)
    row = list(csv.DictReader(open(tmp_path / "experiment_runs.csv")))[-1]
    assert row["config_file"] == "" and row["output_folder"] == ""


def test_save_run_info_writes_config_and_run_id_into_output_folder(tmp_path):
    lash = object.__new__(Lash_E)
    lash.workflow_config = {"SIMULATE": True, "N_SAMPLES": 4}
    lash.workflow_name = "demo"
    lash.simulate = True
    lash._run_tracker = Mock(run_id="abc12345", datetime_started="2026-11-13 10:00:00")
    recorded = Mock()
    original, experiment_run_logger.record_output_folder = experiment_run_logger.record_output_folder, recorded
    try:
        lash.save_run_info(str(tmp_path / "run"))
    finally:
        experiment_run_logger.record_output_folder = original
    assert yaml.safe_load(open(tmp_path / "run" / "config.yaml")) == {"SIMULATE": True, "N_SAMPLES": 4}
    info = yaml.safe_load(open(tmp_path / "run" / "run_info.yaml"))
    assert info["run_id"] == "abc12345" and info["workflow_name"] == "demo"
    recorded.assert_called_once_with(lash._run_tracker, str(tmp_path / "run"))
