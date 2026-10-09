import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO_ROOT = Path(__file__).resolve().parents[1]


class VortexSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.vial = Path(self.temp.name) / "vials.csv"
        self.vial.touch()
        self.robot = Mock(spec_set=["VIAL_FILE", "vortex_vial", "return_vial_home", "move_home"])
        self.robot.VIAL_FILE = str(self.vial)
        self.lash = types.SimpleNamespace(simulate=True, vial_file=str(self.vial),
                                          nr_robot=self.robot, logger=Mock(), _workflow_should_continue=True)
        def initialize(*args, **kwargs):
            self.lash.workflow_config = kwargs['config'].copy()
            return self.lash

        self.constructor = Mock(side_effect=initialize)
        with patch.dict(sys.modules, {"master_usdl_coordinator": types.SimpleNamespace(Lash_E=self.constructor)}):
            self.namespace = runpy.run_path(str(REPO_ROOT / "workflows" / "test_vortex_scheduler.py"))
        self.config = {"SIMULATE": True, "INPUT_VIAL_STATUS_FILE": str(self.vial),
                       "TARGET_VIAL": "test_vial", "VORTEX_TIME": 4}

    def test_import_does_not_start_task(self):
        self.constructor.assert_not_called()

    def test_supplied_config_runs_original_actions_in_order(self):
        self.namespace["execute"](self.config, show_gui=False)
        self.assertEqual(self.robot.method_calls, [call.vortex_vial("test_vial", vortex_time=4),
                                                  call.return_vial_home("test_vial"), call.move_home()])
        kwargs = self.constructor.call_args.kwargs
        self.assertFalse(kwargs["show_gui"])
        self.assertIsNotNone(kwargs["workflow_globals"])
        self.assertEqual(kwargs["config"], self.config)
        self.assertFalse(kwargs["initialize_track"])
        self.assertFalse(kwargs["initialize_biotek"])

    def test_normal_gui_uses_edited_target_and_time(self):
        def review(*args, **kwargs):
            self.assertTrue(kwargs["show_gui"])
            self.assertEqual(kwargs["workflow_name"], "test_vortex_scheduler")
            kwargs["workflow_globals"].update(TARGET_VIAL="edited_vial", VORTEX_TIME=8)
            self.lash.workflow_config = {**self.config, "TARGET_VIAL": "edited_vial", "VORTEX_TIME": 8}
            return self.lash

        self.constructor.side_effect = review
        self.namespace["execute"]()
        self.robot.vortex_vial.assert_called_once_with("edited_vial", vortex_time=8)

    def test_no_vial_input_cannot_vortex(self):
        self.lash.vial_file = None
        self.robot.VIAL_FILE = None
        with self.assertRaisesRegex(ValueError, "requires a vial CSV"):
            self.namespace["execute"]({**self.config, "INPUT_VIAL_STATUS_FILE": None}, show_gui=False)
        self.robot.vortex_vial.assert_not_called()

    def test_cancel_does_not_move_robot(self):
        self.lash._workflow_should_continue = False
        self.assertIsNone(self.namespace["execute"](self.config, show_gui=False))
        self.assertEqual(self.robot.method_calls, [])


if __name__ == "__main__":
    unittest.main()