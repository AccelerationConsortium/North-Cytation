"""Exercise output policy without loading hardware drivers."""
import ast
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from workflow_config_manager import ConfigManager


@pytest.fixture
def workflow():
    path = Path(__file__).resolve().parents[1] / "workflows/surfactant_multidimensional_workflow.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"setup_experiment_folder", "save_results"}]
    namespace = dict(os=os, datetime=datetime, SURFACTANTS=["SDS", "TTAB"],
                     n_surfactants=2, RECOMMENDER_TYPE="bayesian", INIT_STRATEGY="grid",
                     EXPERIMENT_TAG="test", _SAVE_FILES_ON_SIMULATE=False)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("simulate,save_simulation,expected", [
    (True, False, False), (True, True, True),
    (False, False, True), (False, True, True),
])
def test_output_policy(workflow, tmp_path, monkeypatch, simulate, save_simulation, expected):
    monkeypatch.chdir(tmp_path)
    workflow["_SAVE_FILES_ON_SIMULATE"] = save_simulation
    coordinator = SimpleNamespace(simulate=simulate, logger=Mock())
    folder, name = workflow["setup_experiment_folder"](coordinator)
    assert coordinator.current_experiment_name == name
    frame = Mock()
    result = workflow["save_results"](frame, folder)
    if expected:
        assert Path(folder, "measurement_backups").is_dir()
        assert Path(folder, "calibration_validation").is_dir()
        frame.to_csv.assert_called_once_with(result, index=False)
    else:
        assert folder is None
        assert result is None
        frame.to_csv.assert_not_called()
        assert list(tmp_path.iterdir()) == []


def test_internal_switch_does_not_change_gui_config(workflow):
    before = ConfigManager.workflow_constants(workflow)
    workflow["_SAVE_FILES_ON_SIMULATE"] = True
    assert ConfigManager.workflow_constants(workflow) == before
    assert "_SAVE_FILES_ON_SIMULATE" not in before
