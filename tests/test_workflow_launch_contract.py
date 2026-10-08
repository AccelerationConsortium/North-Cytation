import ast
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from workflow_config_manager import ConfigManager


WORKFLOWS = (
    "Degradation_serena", "fluorescence_calibration_workflow",
    "surfactant_grid_adaptive_concentrations", "surfactant_grid_ailsa",
    "surfactant_multidimensional_workflow",
)


class ExperimentBodyReached(Exception):
    pass


class WorkflowLaunchContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.vials = Path(self.temporary.name) / "vials.csv"
        self.vials.touch()
        plotting = types.ModuleType("matplotlib.pyplot")
        plotting.ioff = Mock()
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.use = Mock()
        matplotlib.pyplot = plotting
        self.plot_modules = {"matplotlib": matplotlib, "matplotlib.pyplot": plotting}

    def entrypoint(self, name):
        """Load the actual launch function without importing hardware/dependencies."""
        path = REPO_ROOT / "workflows" / f"{name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        namespace = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and (target.id.isupper() or target.id == "_CONFIG_KEYS"):
                    namespace[target.id] = value
        config_path = REPO_ROOT / "workflow_configs" / f"{name}.yaml"
        namespace.update(yaml.safe_load(config_path.read_text(encoding="utf-8")))
        namespace.update(SIMULATE=True, INPUT_VIAL_STATUS_FILE=str(self.vials), VALIDATE_LIQUIDS=False)
        keys = namespace["_CONFIG_KEYS"] if "_CONFIG_KEYS" in namespace else [
            key for key, value in namespace.items()
            if key.isupper() and not key.startswith("_")
            and isinstance(value, (str, int, float, bool, list, dict))
        ]
        config = deepcopy({key: namespace[key] for key in keys})
        body = Mock(side_effect=ExperimentBodyReached())
        robot = types.SimpleNamespace(VIAL_FILE=str(self.vials), home_robot_components=body)
        lash = types.SimpleNamespace(_workflow_should_continue=True, simulate=True, nr_robot=robot)
        def initialize(*args, **kwargs):
            selected = ConfigManager.resolve_workflow_config(
                kwargs['workflow_name'], kwargs['workflow_globals'],
                config=kwargs['config'], show_gui=kwargs['show_gui'],
            )
            lash.workflow_config = selected
            lash.simulate = selected['SIMULATE']
            return lash

        constructor = Mock(side_effect=initialize)
        namespace.update(Lash_E=constructor, fill_water_vial=body, execute_study_workflow=body,
                         build_plan=body, run_multidim_workflow=body)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        return namespace, config, lash, constructor, body

    def test_every_workflow_accepts_complete_config_without_gui_or_yaml(self):
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                namespace, config, lash, constructor, body = self.entrypoint(name)
                original = deepcopy(config)
                modules = {**self.plot_modules, "workflow_config_manager": None}
                if name == "fluorescence_calibration_workflow":
                    modules["master_usdl_coordinator"] = types.SimpleNamespace(
                        Lash_E=constructor, flatten_cytation_data=Mock()
                    )
                with patch.dict(sys.modules, modules):
                    with self.assertRaises(ExperimentBodyReached):
                        namespace["execute"](config=config, show_gui=False)
                self.assertFalse(constructor.call_args.kwargs["show_gui"])
                self.assertIs(constructor.call_args.kwargs["workflow_globals"], namespace)
                self.assertNotIn("simulate", constructor.call_args.kwargs)
                self.assertEqual(constructor.call_args.kwargs["config"], config)
                self.assertEqual(config, original)
                self.assertTrue(body.called)

    def test_every_normal_launch_uses_reviewed_values(self):
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                namespace, config, lash, constructor, body = self.entrypoint(name)
                def review(*args, **kwargs):
                    self.assertTrue(kwargs["show_gui"])
                    self.assertEqual(kwargs["workflow_name"], name)
                    self.assertNotIn("simulate", kwargs)
                    kwargs["workflow_globals"]["SIMULATE"] = True
                    lash.workflow_config = deepcopy(config)
                    return lash

                constructor.side_effect = review
                modules = self.plot_modules.copy()
                if name == "fluorescence_calibration_workflow":
                    modules["master_usdl_coordinator"] = types.SimpleNamespace(
                        Lash_E=constructor, flatten_cytation_data=Mock()
                    )
                with patch.dict(sys.modules, modules):
                    with self.assertRaises(ExperimentBodyReached):
                        namespace["execute"]()
                self.assertTrue(namespace["SIMULATE"])
                self.assertTrue(body.called)
                if name == "surfactant_grid_ailsa":
                    self.assertTrue(body.call_args.kwargs["simulate"])

    def test_every_cancel_returns_before_automation(self):
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                namespace, config, lash, constructor, body = self.entrypoint(name)
                lash._workflow_should_continue = False
                modules = self.plot_modules.copy()
                if name == "fluorescence_calibration_workflow":
                    modules["master_usdl_coordinator"] = types.SimpleNamespace(
                        Lash_E=constructor, flatten_cytation_data=Mock()
                    )
                with patch.dict(sys.modules, modules):
                    self.assertIsNone(namespace["execute"](config, show_gui=False))
                body.assert_not_called()

    def test_partial_or_ambiguous_supplied_config_is_rejected(self):
        namespace = {"SIMULATE": True, "INPUT_VIAL_STATUS_FILE": str(self.vials), "REPLICATES": 3}
        with self.assertRaisesRegex(ValueError, "show_gui=False"):
            ConfigManager.resolve_workflow_config("test", namespace, namespace.copy(), True)
        with self.assertRaises(KeyError):
            ConfigManager.resolve_workflow_config("test", namespace, {"SIMULATE": True}, False)

    def test_config_manager_preserves_explicit_null_constants(self):
        from workflow_config_manager import ConfigManager
        namespace = {"RANDOMIZATION_SEED": None, "SIMULATE": True, "_PRIVATE": None}
        with patch.object(ConfigManager, "CONFIG_DIR", self.temporary.name):
            ConfigManager.setup_config_if_missing("created", namespace)
            ConfigManager.setup_workflow_config("detected", namespace)
            for name in ("created", "detected"):
                config = yaml.safe_load((Path(self.temporary.name) / f"{name}.yaml").read_text())
                self.assertIn("RANDOMIZATION_SEED", config)
                self.assertIsNone(config["RANDOMIZATION_SEED"])
                self.assertNotIn("_PRIVATE", config)

    def test_script_entrypoints_preserve_normal_run_and_no_top_level_hardware(self):
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                tree = ast.parse((REPO_ROOT / "workflows" / f"{name}.py").read_text(encoding="utf-8-sig"))
                execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
                coordinator_calls = [node for node in ast.walk(execute) if isinstance(node, ast.Call)
                                     and isinstance(node.func, ast.Name) and node.func.id == "Lash_E"]
                self.assertEqual(len(coordinator_calls), 1, f"{name} must construct exactly one coordinator")
                main = next(node for node in tree.body if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test))
                self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                                    and node.func.id == "execute" for node in ast.walk(main)))
                for node in tree.body:
                    if isinstance(node, (ast.Assign, ast.Expr)):
                        self.assertFalse(any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                                             and call.func.id == "Lash_E" for call in ast.walk(node)))


if __name__ == "__main__":
    unittest.main()