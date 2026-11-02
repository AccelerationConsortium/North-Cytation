import unittest
from importlib import import_module
from unittest.mock import MagicMock, patch
from pathlib import Path
import os
import shutil
import tempfile
import types
import yaml

import numpy as np
import pandas as pd

from workflows import fluorescence_calibration_workflow as workflow
from workflows import interrupt_cleanup
from workflows.fluorescence_calibration_workflow import build_plan, execute, fit_calibration, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = {key: getattr(workflow, key) for key in workflow._CONFIG_KEYS}


class FluorescenceCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.vial_file = self.root / "vials.csv"
        shutil.copyfile(REPO_ROOT / "status" / "fluorescence_calibration_vials.csv", self.vial_file)
        inventory = pd.read_csv(self.vial_file)
        # Keep the legacy test inventory stable when the local rack is relabeled.
        inventory["vial_name"] = inventory.vial_name.replace({
            "DMSO_rf_1": "DMSO_working", "DMSO_rf_2": "DMSO_2", "DMSO_rf_3": "DMSO_3",
        })
        inventory.loc[inventory.vial_name.isin(["DMSO_2", "DMSO_3"]), "vial_volume"] = 20.0
        if "DMSO" not in set(inventory.vial_name):
            bulk = inventory.loc[inventory.vial_name == "DMSO_refill"].copy()
            bulk.loc[:, "vial_index"] = int(inventory.vial_index.max()) + 1
            bulk.loc[:, "vial_name"] = "DMSO"
            bulk.loc[:, "location_index"] = 4
            bulk.loc[:, "home_location_index"] = 4
            inventory = pd.concat([inventory, bulk], ignore_index=True)
        inventory.loc[inventory.vial_name == "DMSO", "vial_volume"] = 20.0
        inventory.loc[inventory.vial_name == "DMSO_working", "vial_volume"] = 0.0
        inventory.loc[inventory.vial_name == "surfactant_refill", "vial_volume"] = 19.9
        for name, template, location_index, volume in (
            ("acetone", "DMSO", 5, 20.0),
            ("acetone_working", "DMSO_working", 41, 6.0),
            ("coumarin6_acetone_stock", "pyrene_DMSO_stock", 42, 7.8),
            ("coumarin6_acetone_s1", "pyrene_DMSO_s1", 43, 0.0),
            ("coumarin6_acetone_s2", "pyrene_DMSO_s2", 44, 0.0),
            ("coumarin6_acetone_s3", "pyrene_DMSO_s3", 45, 0.0),
        ):
            if name not in set(inventory.vial_name):
                row = inventory.loc[inventory.vial_name == template].copy()
                row.loc[:, "vial_index"] = int(inventory.vial_index.max()) + 1
                row.loc[:, "vial_name"] = name
                row.loc[:, "location_index"] = location_index
                row.loc[:, "home_location_index"] = location_index
                row.loc[:, "vial_volume"] = volume
                inventory = pd.concat([inventory, row], ignore_index=True)
        inventory.loc[inventory.vial_name.str.contains(r'_s\d+$'), 'vial_volume'] = 0.0
        inventory.to_csv(self.vial_file, index=False)
        self.original_directory = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.original_directory)
        self.config = {**BASE_CONFIG, "INPUT_VIAL_STATUS_FILE": str(self.vial_file),
                       "MEASUREMENT_SCHEDULE_MIN": [0],
                       "SOLVENT_SOURCE_VIALS": ["DMSO", "DMSO_2"]}
        slack = patch.object(workflow, "slack_agent", MagicMock())
        slack.start()
        self.addCleanup(slack.stop)

    def fake_coordinator(self, simulate):
        lash = MagicMock()
        lash._workflow_should_continue = True
        lash.simulate = simulate
        lash.nr_robot.VIAL_FILE = str(self.vial_file)
        from robot_state.plate_inventory import PlateInventory
        lash.nr_track.inventory = PlateInventory({"wellplate_type": "96 WELL PLATE", "num_in_source": 1},
            yaml.safe_load((REPO_ROOT / "robot_state/wellplates.yaml").read_text()),
            yaml.safe_load((REPO_ROOT / "robot_state/track_positions.yaml").read_text()))
        lash.nr_track.NUM_SOURCE = 1
        lash.nr_robot.WELLPLATES = {"96 WELL PLATE": {"max_volume_per_well": 0.3}}
        lash.nr_robot.get_vial_in_location.return_value = None
        lash.nr_robot.VIAL_DF = pd.read_csv(self.vial_file).set_index("vial_index", drop=False)
        def vial_info(vial, column):
            rows = lash.nr_robot.VIAL_DF
            return rows.loc[rows.vial_name == vial, column].iloc[0]
        lash.nr_robot.get_vial_info.side_effect = vial_info
        lash.nr_robot.is_vial_movable.side_effect = lambda vial: (
            vial_info(vial, "vial_type") == "8_mL" and bool(vial_info(vial, "capped")))
        return lash

    def test_solvent_pool_exhausts_first_source_before_using_later_sources(self):
        lash = self.fake_coordinator(simulate=True)
        volumes = {"DMSO": 2.25, "DMSO_2": 7.0, "DMSO_3": 20.0}
        lash.nr_robot.get_vial_info.side_effect = lambda vial, column: volumes[vial]
        with patch.object(workflow, "dispense_vial_to_vial") as transfer:
            workflow.dispense_solvent(lash, list(volumes), "destination", 4.5, "DMSO", 2.0)
        self.assertEqual([call.args[1] for call in transfer.call_args_list], ["DMSO", "DMSO_2"])
        self.assertAlmostEqual(transfer.call_args_list[0].args[3], 0.25)
        self.assertAlmostEqual(transfer.call_args_list[1].args[3], 4.25)

    def test_solvent_allocation_skips_tiny_leftovers_and_avoids_tiny_final_portion(self):
        self.assertEqual(workflow.plan_solvent_sources(
            {"first": 1.001, "second": 5.0}, 3.0, "DMSO", 1.0),
            [("second", 3.0)])
        transfers = workflow.plan_solvent_sources(
            {"first": 3.9, "second": 2.0}, 3.0, "DMSO", 1.0)
        self.assertEqual([source for source, _ in transfers], ["first", "second"])
        self.assertAlmostEqual(sum(amount for _, amount in transfers), 3.0)
        self.assertTrue(all(amount >= 0.2 - 1e-9 for _, amount in transfers))

    def test_solvent_pool_combines_containers_to_supply_whole_addition(self):
        lash = self.fake_coordinator(simulate=True)
        lash.nr_robot.get_vial_info.side_effect = lambda vial, column: 4.0
        with patch.object(workflow, "dispense_vial_to_vial") as transfer:
            workflow.dispense_solvent(lash, ["DMSO", "DMSO_2"], "destination", 3.0, "DMSO", 2.0)
        self.assertEqual([call.args[1:] for call in transfer.call_args_list], [
            ("DMSO", "destination", 2.0, "DMSO"),
            ("DMSO_2", "destination", 1.0, "DMSO"),
        ])

    def test_low_working_vial_is_refilled_before_validation_and_rechecked_after(self):
        inventory = pd.read_csv(self.vial_file)
        inventory.loc[inventory.vial_name.str.contains(r"_s\d+$"), "vial_volume"] = 4.0
        inventory.loc[inventory.vial_name == "DMSO_working", "vial_volume"] = 0.354
        inventory.to_csv(self.vial_file, index=False)
        lash = self.fake_coordinator(simulate=True)
        lash.measure_wellplate.return_value = None
        lash.workflow_config = {
            **self.config, "SIMULATE": True, "VALIDATE_LIQUIDS": True,
            "TEST_SOLVENT": True, "TOTAL_VOLUME_UL": 250, "DYE_VOLUME_UL": 5,
            "SUBSTOCK_VOLUME_ML": 4.0,
        }
        rows = lash.nr_robot.VIAL_DF
        def transfer(controller, source, destination, amount, liquid):
            rows.loc[rows.vial_name == source, "vial_volume"] -= amount
            rows.loc[rows.vial_name == destination, "vial_volume"] += amount

        def validate(controller, sources, output, *, replicates, vial_transfers):
            self.assertAlmostEqual(controller.nr_robot.get_vial_info(
                "DMSO_working", "vial_volume"), 6.0)
            controller.nr_robot.dispense_from_vials_into_wellplate.assert_not_called()
            rows.loc[rows.vial_name == "DMSO_working", "vial_volume"] = 0.3
            return {}

        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                            flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), \
                patch.object(workflow, "dispense_vial_to_vial", side_effect=transfer), \
                patch("pipetting_data.embedded_liquid_validations.run_liquid_validations",
                      side_effect=validate):
            execute(lash.workflow_config, show_gui=False)
        self.assertAlmostEqual(lash.nr_robot.get_vial_info(
            "DMSO_working", "vial_volume"), 5.72)
        lash.nr_robot.dispense_from_vials_into_wellplate.assert_called()

    def test_working_refill_accepts_less_than_six_ml_and_skips_when_sufficient(self):
        # Queue job 007 previously skipped filling 4.070 mL, then drained
        # the working vial to 0.350 mL despite having another bulk source.
        for current, bulk, expected in (
                (4.069666666666679, 7.599, 6.0 - 4.069666666666679),
                (4.28, 2.5, 1.5), (5.72, 1.0, 0.0)):
            with self.subTest(current=current):
                inventory = pd.read_csv(self.vial_file)
                inventory.loc[inventory.vial_name.str.contains(r"_s\d+$"), "vial_volume"] = 4.0
                inventory.loc[inventory.vial_name == "DMSO_working", "vial_volume"] = current
                inventory.loc[inventory.vial_name == "DMSO", "vial_volume"] = bulk
                inventory.to_csv(self.vial_file, index=False)
                lash = self.fake_coordinator(simulate=True)
                lash.measure_wellplate.return_value = None
                lash.workflow_config = {
                    **self.config, "SIMULATE": True, "TEST_SOLVENT": True,
                    "TOTAL_VOLUME_UL": 250, "DYE_VOLUME_UL": 5,
                    "SOLVENT_SOURCE_VIALS": ["DMSO"], "SOLVENT_SOURCE_RESERVE_ML": 1.0,
                }
                coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                                    flatten_cytation_data=MagicMock())
                with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), \
                        patch.object(workflow, "dispense_vial_to_vial") as transfer:
                    execute(lash.workflow_config, show_gui=False)
                fills = [call.args[3] for call in transfer.call_args_list
                         if call.args[2] == "DMSO_working"]
                self.assertAlmostEqual(sum(fills), expected)

    def test_working_refill_rejects_supply_below_small_tip_reserve(self):
        inventory = pd.read_csv(self.vial_file)
        inventory.loc[inventory.vial_name.str.contains(r"_s\d+$"), "vial_volume"] = 4.0
        inventory.loc[inventory.vial_name == "DMSO_working", "vial_volume"] = 4.07
        inventory.loc[inventory.vial_name == "DMSO", "vial_volume"] = 1.0
        inventory.to_csv(self.vial_file, index=False)
        lash = self.fake_coordinator(simulate=True)
        config = {**self.config, "SIMULATE": True, "TEST_SOLVENT": True,
                  "TOTAL_VOLUME_UL": 250, "DYE_VOLUME_UL": 5,
                  "SOLVENT_SOURCE_VIALS": ["DMSO"], "SOLVENT_SOURCE_RESERVE_ML": 1.0}
        lash.workflow_config = config
        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                            flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), \
                patch.object(workflow, "dispense_vial_to_vial") as transfer:
            with self.assertRaisesRegex(ValueError, "including 2 mL reserve"):
                execute(config, show_gui=False)
        transfer.assert_not_called()
        lash.nr_robot.dispense_from_vials_into_wellplate.assert_not_called()

    def test_solvent_pool_rejects_insufficient_volume_before_transfers(self):
        lash = self.fake_coordinator(simulate=True)
        lash.nr_robot.get_vial_info.side_effect = None
        lash.nr_robot.get_vial_info.return_value = 0.15
        with patch.object(workflow, "dispense_vial_to_vial") as transfer:
            with self.assertRaisesRegex(ValueError, "Insufficient bulk DMSO"):
                workflow.dispense_solvent(lash, ["DMSO", "DMSO_2"], "destination", 0.2, "DMSO", 0.1)
        transfer.assert_not_called()

    def test_working_solvent_refill_uses_configured_pool_without_identity_named_source(self):
        inventory = pd.read_csv(self.vial_file)
        inventory = inventory[inventory.vial_name != "DMSO"]
        inventory.loc[inventory.vial_name == "DMSO_working", "vial_volume"] = 0.0
        inventory.to_csv(self.vial_file, index=False)
        lash = self.fake_coordinator(simulate=True)
        lash.workflow_config = {
            **self.config, "SIMULATE": True, "PREPARE_ONLY": True,
            "SOLVENT_SOURCE_VIALS": ["DMSO_2", "DMSO_3"],
        }
        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                            flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), \
                patch.object(workflow, "dispense_vial_to_vial") as transfer:
            execute(lash.workflow_config, show_gui=False)
        working = [call for call in transfer.call_args_list if call.args[2] == "DMSO_working"]
        self.assertEqual(len(working), 1)
        self.assertEqual(working[0].args[1:], ("DMSO_2", "DMSO_working", 6.0, "DMSO"))

    def test_six_preparations_share_solvent_pool_and_reuse_each_dyes_substocks(self):
        inventory = pd.read_csv(self.vial_file)
        inventory.loc[inventory.vial_name == "coumarin6_DMSO_stock", "vial_volume"] = 7.8
        inventory.to_csv(self.vial_file, index=False)
        transfers = []
        for run, dye in enumerate(("nile-red", "nile-red", "coumarin-6", "coumarin-6", "pyrene", "pyrene")):
            lash = self.fake_coordinator(simulate=True)
            robot = lash.nr_robot
            lash.workflow_config = {
                **self.config, "SIMULATE": True, "DYE": dye, "DYE_SOLVENT": "DMSO",
                "SOLVENT_SOURCE_VIALS": ["DMSO", "DMSO_2", "DMSO_3"],
                "PREPARE_ONLY": True, "TEST_WATER": False, "TEST_SURFACTANT": False,
            }

            def transfer(controller, source, destination, volume, liquid):
                rows = robot.VIAL_DF
                source_index = rows.index[rows.vial_name == source][0]
                destination_index = rows.index[rows.vial_name == destination][0]
                self.assertGreaterEqual(rows.at[source_index, "vial_volume"] + 1e-9, volume)
                rows.at[source_index, "vial_volume"] -= volume
                rows.at[destination_index, "vial_volume"] += volume
                transfers.append((source, destination, volume))

            coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                                flatten_cytation_data=MagicMock())
            before = len(transfers)
            with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), \
                    patch.object(workflow, "dispense_vial_to_vial", side_effect=transfer):
                execute(lash.workflow_config, show_gui=False)
            if run % 2:
                self.assertEqual(len(transfers), before)
            robot.VIAL_DF.to_csv(self.vial_file, index=False)
        bulk_transfers = [entry for entry in transfers if entry[0] in ("DMSO", "DMSO_2", "DMSO_3")]
        # Three sets of substocks plus the initially empty working-vial refill.
        self.assertAlmostEqual(sum(entry[2] for entry in bulk_transfers), 44.7)
        stock_transfers = [entry for entry in transfers if entry[0].endswith("_stock")]
        self.assertEqual(len(stock_transfers), 9)
        final = pd.read_csv(self.vial_file).set_index("vial_name")
        self.assertTrue((final.loc[["DMSO", "DMSO_2", "DMSO_3"], "vial_volume"] >= 2.0 - 1e-9).all())

    def test_simulation_keeps_automation_and_analysis_without_saving(self):
        for reader_has_data in (False, True):
            with self.subTest(reader_has_data=reader_has_data):
                lash = self.fake_coordinator(simulate=True)
                lash.workflow_config = self.config
                plan, _ = build_plan(self.config)
                raw = pd.DataFrame({"well_position": plan.well_position,
                                    "334_373": 10 + plan.concentration_relative * 100,
                                    "334_384": 20 + plan.concentration_relative * 200})
                lash.measure_wellplate.return_value = raw if reader_has_data else None
                coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                                    flatten_cytation_data=lambda data, _: data)
                before = set(self.root.rglob("*"))
                with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), \
                        patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("Unexpected CSV write")), \
                        patch.object(Path, "write_text", side_effect=AssertionError("Unexpected text write")), \
                        patch.object(workflow, "fit_calibration", wraps=fit_calibration) as fit, \
                        patch.object(workflow, "plot_calibration") as calibration_plot, \
                        patch.object(workflow, "plot_kinetics") as kinetics_plot:
                    self.assertIsNone(execute(self.config, show_gui=False))
                self.assertEqual(set(self.root.rglob("*")), before)
                fit.assert_called_once()
                calibration_plot.assert_not_called()
                kinetics_plot.assert_not_called()
                lash.nr_robot.dispense_from_vial_into_vial.assert_called()
                lash.nr_robot.dispense_from_vials_into_wellplate.assert_called()
                lash.measure_wellplate.assert_called_once()
                lash.grab_new_wellplate.assert_called_once()
                lash.discard_used_wellplate.assert_called_once()

    def test_simulation_saving_can_be_reenabled(self):
        lash = self.fake_coordinator(simulate=True)
        lash.workflow_config = self.config
        lash.measure_wellplate.return_value = None
        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                            flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), \
                patch.object(workflow, "_SAVE_FILES_ON_SIMULATE", True), \
                patch.object(workflow, "plot_calibration") as calibration_plot, \
                patch.object(workflow, "plot_kinetics") as kinetics_plot:
            output = execute(self.config, show_gui=False)
        self.assertEqual([path.resolve() for path in
                          (self.root / "output" / "fluorescence_calibration").iterdir()],
                         [output.resolve()])
        self.assertIn("_simulate_", output.name)
        for name in ("well_plan.csv", "substock_recipes.csv", "config.yaml",
                     "fluorescence_results.csv", "calibration_summary.csv", "calibration_fits.csv"):
            self.assertTrue((output / name).is_file(), name)
        calibration_plot.assert_called_once()
        kinetics_plot.assert_called_once()

    def test_simulation_save_switch_is_private_and_disabled(self):
        from workflow_config_manager import ConfigManager
        self.assertIs(workflow._SAVE_FILES_ON_SIMULATE, False)
        self.assertNotIn("_SAVE_FILES_ON_SIMULATE", workflow._CONFIG_KEYS)
        self.assertNotIn("_SAVE_FILES_ON_SIMULATE", ConfigManager.workflow_constants(vars(workflow)))
    def test_transfer_staging_depends_on_either_closed_cap(self):
        for source_cap, destination_cap in (("open", "open"), ("closed", "open"),
                                            ("open", "closed"), ("closed", "closed")):
            with self.subTest(source=source_cap, destination=destination_cap):
                lash = self.fake_coordinator(simulate=True)
                robot = lash.nr_robot
                robot.VIAL_DF.at[0, "cap_type"] = source_cap
                robot.VIAL_DF.at[4, "cap_type"] = destination_cap
                original = robot.VIAL_DF[["home_location", "home_location_index"]].copy()

                def transfer(*args, **kwargs):
                    if "closed" in (source_cap, destination_cap):
                        self.assertEqual(robot.get_vial_info("pyrene_DMSO_stock", "home_location_index"), 0)
                        self.assertEqual(robot.get_vial_info("pyrene_DMSO_s1", "home_location_index"), 1)
                    return "transferred"

                robot.dispense_from_vial_into_vial.side_effect = transfer
                result = workflow.dispense_vial_to_vial(
                    lash, "pyrene_DMSO_stock", "pyrene_DMSO_s1", 3.0, "DMSO")
                self.assertEqual(result, "transferred")
                pd.testing.assert_frame_equal(original, robot.VIAL_DF[original.columns])
                if source_cap == destination_cap == "open":
                    robot.move_vial_to_location.assert_not_called()
                    robot.remove_pipet.assert_not_called()
                    robot.return_vial_home.assert_not_called()
                else:
                    self.assertEqual([call.args for call in robot.move_vial_to_location.call_args_list],
                                     [("pyrene_DMSO_stock", "main_8mL_rack", 0),
                                      ("pyrene_DMSO_s1", "main_8mL_rack", 1)])
                    calls = robot.method_calls
                    move_index = next(i for i, call in enumerate(calls) if call[0] == "move_vial_to_location")
                    self.assertTrue(any(call[0] == "remove_pipet" for call in calls[:move_index]))
                    return_index = next(i for i, call in enumerate(calls) if call[0] == "return_vial_home")
                    self.assertTrue(any(call[0] == "remove_pipet" for call in calls[move_index:return_index]))

    def test_large_source_stays_in_place_with_closed_destination(self):
        lash = self.fake_coordinator(simulate=True)
        lash.nr_robot.VIAL_DF.at[4, "cap_type"] = "closed"
        workflow.dispense_vial_to_vial(lash, "DMSO", "pyrene_DMSO_s1", 3.0, "DMSO")
        lash.nr_robot.move_vial_to_location.assert_called_once_with("pyrene_DMSO_s1", "main_8mL_rack", 1)
        lash.nr_robot.return_vial_home.assert_called_once_with("pyrene_DMSO_s1")

    def test_split_closed_cap_transfer_uses_safe_homes_for_clamp_exchanges(self):
        from North_Safe import North_Robot
        from pipetting_data.pipetting_parameters import PipettingParameters
        from pipetting_data.pipetting_wizard import PipettingWizard

        lash = self.fake_coordinator(simulate=True)
        robot = lash.nr_robot
        rows = robot.VIAL_DF
        rows.loc[[0, 4], "cap_type"] = "closed"
        original_homes = [int(rows.at[index, "home_location_index"]) for index in (0, 4)]
        robot.normalize_vial_index.side_effect = lambda vial: (
            int(rows.loc[rows.vial_name == vial, "vial_index"].iloc[0])
            if isinstance(vial, str) else vial)
        robot.get_vial_index_from_name.side_effect = robot.normalize_vial_index.side_effect
        robot.get_vial_info.side_effect = lambda vial, column: rows.at[robot.normalize_vial_index(vial), column]
        robot.get_vial_in_location.side_effect = lambda location, slot: next(
            iter(rows.loc[(rows.location == location) & (rows.location_index == slot), "vial_index"]), None)
        robot.is_vial_movable.side_effect = lambda vial: North_Robot.is_vial_movable(robot, vial)
        robot.is_vial_pipetable.side_effect = lambda vial: North_Robot.is_vial_pipetable(robot, vial)
        robot.GRIPPER_STATUS = None

        def move(vial, location, location_index, **kwargs):
            index = robot.normalize_vial_index(vial)
            rows.at[index, "location"] = location
            rows.at[index, "location_index"] = location_index

        def cap(capped, **kwargs):
            rows.at[robot.get_vial_in_location("clamp", 0), "capped"] = capped
            robot.GRIPPER_STATUS = None if capped else "Cap"

        returns = []

        def return_home(vial, **kwargs):
            returns.append((robot.get_vial_info(vial, "home_location_index"), robot.GRIPPER_STATUS))
            North_Robot.return_vial_home(robot, vial, **kwargs)

        robot.move_vial_to_location.side_effect = move
        robot.uncap_clamp_vial.side_effect = lambda **kwargs: cap(False)
        robot.recap_clamp_vial.side_effect = lambda **kwargs: cap(True)
        robot.return_vial_home.side_effect = return_home
        robot.PIPET_TIPS = {"large_tip": {"volume": 1.0}}
        robot._get_optimized_parameters.return_value = PipettingParameters()
        robot.resolve_pipetting_setup.side_effect = lambda *args, **kwargs: (
            North_Robot.resolve_pipetting_setup(robot, *args, **kwargs))
        robot.plan_vial_transfer.side_effect = lambda *args, **kwargs: (
            North_Robot.plan_vial_transfer(robot, *args, **kwargs))
        robot.aspirate_from_vial.side_effect = lambda vial, *args, **kwargs: (
            North_Robot._ensure_vial_accessible_for_pipetting(robot, vial))
        robot.dispense_into_vial.side_effect = lambda vial, *args, **kwargs: (
            North_Robot._ensure_vial_accessible_for_pipetting(robot, vial), None)[1]
        robot.dispense_from_vial_into_vial.side_effect = lambda *args, **kwargs: (
            North_Robot.dispense_from_vial_into_vial(robot, *args, **kwargs))
        with patch.object(PipettingWizard, "get_max_calibrated_volume", return_value=None):
            workflow.dispense_vial_to_vial(
                lash, "pyrene_DMSO_stock", "pyrene_DMSO_s1", 2.5, "DMSO")
        self.assertGreater(robot.aspirate_from_vial.call_count, 1)
        self.assertTrue(all(slot in (0, 1) for slot, _ in returns[:-2]))
        self.assertEqual([slot for slot, _ in returns[-2:]], original_homes)
        self.assertEqual(robot.get_vial_info("pyrene_DMSO_stock", "location_index"), original_homes[0])
        self.assertEqual(robot.get_vial_info("pyrene_DMSO_s1", "location_index"), original_homes[1])

    def test_occupied_staging_slot_fails_before_movements(self):
        lash = self.fake_coordinator(simulate=True)
        lash.nr_robot.VIAL_DF.at[4, "cap_type"] = "closed"
        lash.nr_robot.get_vial_in_location.side_effect = [None, 2]
        with self.assertRaisesRegex(RuntimeError, "staging.*occupied"):
            workflow.dispense_vial_to_vial(
                lash, "pyrene_DMSO_stock", "pyrene_DMSO_s1", 3.0, "DMSO")
        lash.nr_robot.remove_pipet.assert_not_called()
        lash.nr_robot.move_vial_to_location.assert_not_called()
        lash.nr_robot.dispense_from_vial_into_vial.assert_not_called()

    def test_transfer_failure_restores_homes_without_recovery_movements(self):
        lash = self.fake_coordinator(simulate=True)
        lash.nr_robot.VIAL_DF.at[4, "cap_type"] = "closed"
        robot = lash.nr_robot
        original = robot.VIAL_DF[["home_location", "home_location_index"]].copy()
        robot.dispense_from_vial_into_vial.side_effect = RuntimeError("transfer interrupted")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            workflow.dispense_vial_to_vial(
                lash, "pyrene_DMSO_stock", "pyrene_DMSO_s1", 3.0, "DMSO")
        pd.testing.assert_frame_equal(original, robot.VIAL_DF[original.columns])
        robot.save_robot_status.assert_called_once()
        robot.return_vial_home.assert_not_called()

    def test_optional_shake_runs_before_first_read_at_each_timepoint(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                lash = self.fake_coordinator(simulate=True)
                config = {**self.config, "SHAKE_BEFORE_MEASUREMENT": enabled,
                          "MEASUREMENT_SCHEDULE_MIN": [0, 10], "MEASUREMENT_REPLICATES": 2}
                lash.workflow_config = config
                lash.measure_wellplate.return_value = None
                coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                                    flatten_cytation_data=MagicMock())
                with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), \
                        patch.object(workflow, "plot_calibration"):
                    execute(config, show_gui=False)
                protocol = workflow._PROTOCOLS["pyrene"][0]
                self.assertEqual([call.args[0] for call in lash.measure_wellplate.call_args_list],
                                 [protocol] * 4)
                first = config["SHAKE_PROTOCOL_FILE"] if enabled else None
                self.assertEqual(
                    [call.kwargs.get("preparation_protocol") for call in lash.measure_wellplate.call_args_list],
                    [first, None, first, None])

    def test_enabled_shake_requires_protocol(self):
        with self.assertRaisesRegex(ValueError, "Set SHAKE_PROTOCOL_FILE"):
            build_plan({**self.config, "SHAKE_BEFORE_MEASUREMENT": True, "SHAKE_PROTOCOL_FILE": ""})

    def test_two_runs_share_plate_across_reloaded_state_then_third_loads_new(self):
        state_file = self.root / "track_status.yaml"
        state_file.write_text(yaml.safe_dump({"wellplate_type": "96 WELL PLATE", "num_in_source": 2, "num_in_waste": 0}))
        loaded, discarded, measured = [], [], []
        for run in range(3):
            lash = self.fake_coordinator(simulate=True)
            track = lash.nr_track
            saved = yaml.safe_load(state_file.read_text())
            from robot_state.plate_inventory import PlateInventory
            track.inventory = PlateInventory(saved, track.inventory.definitions, track.inventory.position_config)
            track.NUM_SOURCE = len(track.inventory.contents("source_stack"))
            track.TRACK_STATUS_FILE = str(state_file)
            track._scheduler_state_root = str(self.root)

            def save():
                state_file.write_text(yaml.safe_dump(track.inventory.data))

            def grab():
                loaded.append(run)
                track.inventory.relocate(track.inventory.top("source_stack"), "pipetting_area")
                track.NUM_SOURCE = len(track.inventory.contents("source_stack"))

            def discard():
                discarded.append(run)
                track.inventory.relocate(track.inventory.top("pipetting_area"), "waste_stack")
                save()

            track.save_track_status.side_effect = save
            lash.grab_new_wellplate.side_effect = grab
            lash.discard_used_wellplate.side_effect = discard
            lash.measure_wellplate.return_value = None
            config = {**self.config, "TWO_RUNS_PER_PLATE": True}
            lash.workflow_config = config
            coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                                flatten_cytation_data=MagicMock())
            with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), \
                    patch.object(workflow, "plot_calibration"), patch.object(workflow, "plot_kinetics"), \
                    patch.object(workflow, "_SAVE_FILES_ON_SIMULATE", True), \
                    patch.object(import_module("scheduler.simulation_state"), "can_save", return_value=True):
                output = execute(config, show_gui=False)
            save()
            indices = lash.measure_wellplate.call_args.args[1]
            measured.append(indices)
            self.assertEqual(pd.read_csv(output / "well_plan.csv").well_index.tolist(), indices)
            dispensed = {index for call in lash.nr_robot.dispense_from_vials_into_wellplate.call_args_list
                         for index in call.args[0].index}
            self.assertEqual(dispensed, set(indices))
        self.assertEqual(loaded, [0, 2])
        self.assertEqual(discarded, [1])
        self.assertEqual(measured, [list(range(45)), list(range(48, 93)), list(range(45))])

    def test_two_run_plate_blocks_unfinished_run_before_dispensing(self):
        lash = self.fake_coordinator(simulate=True)
        lash.nr_track.inventory.relocate(lash.nr_track.inventory.top("source_stack"), "pipetting_area")
        lash.nr_track.TRACK_STATUS_FILE = str(self.root / "track_status.yaml")
        (self.root / "fluorescence_plate.yaml").write_text(yaml.safe_dump(
            {"status": "in_progress", "plate_type": "96 WELL PLATE"}))
        lash.workflow_config = {**self.config, "TWO_RUNS_PER_PLATE": True}
        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                            flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}):
            with self.assertRaisesRegex(ValueError, "unfinished run"):
                execute(lash.workflow_config, show_gui=False)
        lash.grab_new_wellplate.assert_not_called()
        lash.nr_robot.dispense_from_vial_into_vial.assert_not_called()

    def test_constant_solvent_and_concentration(self):
        plan, recipes = build_plan({**self.config, "STOCK_CONCENTRATION_UM": 100})
        self.assertEqual(len(plan), 45)
        self.assertEqual(len(recipes), 3)
        self.assertEqual((plan.dilution_factor == 0).sum(), 9)
        np.testing.assert_allclose(plan.dye_volume_ul + plan.medium_volume_ul, 200)
        np.testing.assert_allclose(plan.concentration_um, plan.dilution_factor * 2.5)
        np.testing.assert_allclose(plan[plan.medium != "solvent"].solvent_fraction, 0.025)
        np.testing.assert_allclose(recipes.stock_ml + recipes.solvent_ml, 6)
        np.testing.assert_allclose(recipes.stock_ml, [0.6, 1.5, 3.0])
        np.testing.assert_allclose(recipes.solvent_ml, [5.4, 4.5, 3.0])

    def test_volatile_solvent_keeps_target_medium_volume_and_skips_solvent_wells(self):
        plan, recipes = build_plan({**self.config, "VOLATILE_SOLVENT": True,
                                    "TOTAL_VOLUME_UL": 250, "DYE_VOLUME_UL": 20,
                                    "STOCK_CONCENTRATION_UM": 100})
        self.assertEqual(set(plan.medium), {"water", "surfactant"})
        self.assertEqual(len(plan), 30)
        np.testing.assert_allclose(plan.medium_volume_ul, 250)
        np.testing.assert_allclose(plan.dye_volume_ul, 20)
        np.testing.assert_allclose(plan.concentration_um, plan.dilution_factor * 8)
        np.testing.assert_allclose(plan.solvent_fraction, 0)
        self.assertEqual(len(recipes), 3)

    def test_four_volatile_substocks_preserve_five_microliter_concentrations(self):
        original = {**self.config, "TOTAL_VOLUME_UL": 250}
        before, _ = build_plan(original)
        after, recipes = build_plan({**original, "VOLATILE_SOLVENT": True,
                                    "DYE_VOLUME_UL": 20,
                                    "DILUTION_FACTORS": [f / 4 for f in original["DILUTION_FACTORS"]]})
        np.testing.assert_allclose(sorted(before.concentration_um.unique()),
                                   sorted(after.concentration_um.unique()))
        self.assertEqual(len(recipes), 4)
        self.assertAlmostEqual(recipes.stock_ml.sum(), 2.775)
        self.assertAlmostEqual(recipes.solvent_ml.sum(), 21.225)

    def test_inventory_name_overrides_drive_plan_recipes_and_execution(self):
        config = {**self.config, "SIMULATE": True, "VIAL_NAME_OVERRIDES": {
            "DMSO_working": "DMSO_rf_1", "pyrene_DMSO_stock": "parent",
            "pyrene_DMSO_s1": "first_substock",
        }}
        plan, recipes = build_plan(config)
        self.assertEqual(set(plan.loc[plan.dilution_factor == 0, "dye_source"]), {"DMSO_rf_1"})
        self.assertEqual(set(plan.loc[plan.dilution_factor == 1, "dye_source"]), {"parent"})
        self.assertIn("first_substock", set(recipes.vial))
        inventory = pd.read_csv(self.vial_file)
        inventory["vial_name"] = inventory.vial_name.replace(config["VIAL_NAME_OVERRIDES"])
        inventory.to_csv(self.vial_file, index=False)
        lash = self.fake_coordinator(simulate=True)
        lash.workflow_config = config
        lash.measure_wellplate.return_value = None
        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                            flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}):
            execute(config, show_gui=False)
        transfers = lash.nr_robot.dispense_from_vial_into_vial.call_args_list
        self.assertTrue(any(call.args[0] == "parent" and call.args[1] == "first_substock"
                            for call in transfers))

    def test_measurement_validation_uses_plate_volumes_when_substocks_are_prepared(self):
        inventory = pd.read_csv(self.vial_file)
        inventory.loc[inventory.vial_name.str.contains(r"_s\d+$"), "vial_volume"] = 4.0
        inventory.to_csv(self.vial_file, index=False)
        for volatile, dye_ul, medium_ul in ((True, 20, 250), (False, 5, 245)):
            with self.subTest(volatile=volatile):
                lash = self.fake_coordinator(simulate=True)
                lash.workflow_config = {**self.config, "SIMULATE": True, "VALIDATE_LIQUIDS": True,
                                        "TEST_SOLVENT": False, "TOTAL_VOLUME_UL": 250,
                                        "VOLATILE_SOLVENT": volatile, "DYE_VOLUME_UL": dye_ul,
                                        "SUBSTOCK_VOLUME_ML": 4.0}
                lash.measure_wellplate.return_value = None
                coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                                    flatten_cytation_data=MagicMock())

                def validate(controller, sources, output, *, vial_transfers):
                    controller.nr_robot.dispense_from_vials_into_wellplate.assert_not_called()
                    self.assertEqual({vial: set(volumes) for vial, _, volumes in sources}, {
                        "DMSO_working": {dye_ul}, "water": {medium_ul}, "surfactant": {medium_ul},
                    })
                    return {}

                with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), \
                        patch.object(import_module("pipetting_data.embedded_liquid_validations"),
                                     "run_liquid_validations", side_effect=validate) as validation:
                    execute(lash.workflow_config, show_gui=False)
                validation.assert_called_once()

    def test_liquid_shortages_stop_before_startup_transfers(self):
        for source, overrides in (
            ("pyrene_DMSO_stock", {}),
            ("water", {"REPLICATES": 20, "TEST_SURFACTANT": False, "TEST_SOLVENT": False}),
            ("surfactant", {"REPLICATES": 20, "TEST_WATER": False, "TEST_SOLVENT": False}),
        ):
            with self.subTest(source=source):
                inventory = pd.read_csv(self.vial_file)
                if source == "pyrene_DMSO_stock":
                    inventory.loc[inventory.vial_name == source, "vial_volume"] = 0.2
                inventory.to_csv(self.vial_file, index=False)
                lash = self.fake_coordinator(simulate=True)
                lash.workflow_config = {**self.config, "SIMULATE": True, **overrides}
                lash.nr_track.NUM_SOURCE = 10
                coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                                    flatten_cytation_data=MagicMock())
                with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), \
                        self.assertRaisesRegex(ValueError, f"Insufficient liquid in {source}"):
                    execute(lash.workflow_config, show_gui=False)
                lash.nr_robot.dispense_from_vial_into_vial.assert_not_called()
                lash.nr_robot.dispense_into_vial_from_reservoir.assert_not_called()
                # Restore stock for the following subtests.
                inventory.loc[inventory.vial_name == "pyrene_DMSO_stock", "vial_volume"] = 7.3
                inventory.to_csv(self.vial_file, index=False)

    def test_fragmented_solvent_pool_supplies_preparation(self):
        inventory = pd.read_csv(self.vial_file)
        inventory.loc[inventory.vial_name.isin(["DMSO", "DMSO_2"]), "vial_volume"] = 3.4
        inventory.loc[inventory.vial_name == "DMSO_working", "vial_volume"] = 6.0
        inventory.to_csv(self.vial_file, index=False)
        lash = self.fake_coordinator(simulate=True)
        lash.measure_wellplate.return_value = None
        lash.workflow_config = {**self.config, "SIMULATE": True, "TEST_SOLVENT": False,
                                "SUBSTOCK_VOLUME_ML": 2.0,
                                "DILUTION_FACTORS": [0, 0.25, 0.5]}
        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                            flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}):
            execute(lash.workflow_config, show_gui=False)
        lash.nr_robot.dispense_from_vial_into_vial.assert_called()

    def test_selected_pair_generates_pair_specific_vials(self):
        config = {**self.config, "DYE": "coumarin-6", "DYE_SOLVENT": "acetone"}
        plan, recipes = build_plan(config)
        self.assertEqual(set(plan.dye), {"coumarin-6"})
        self.assertEqual(set(plan.dye_solvent), {"acetone"})
        self.assertEqual(set(recipes.vial), {
            "coumarin6_acetone_s1", "coumarin6_acetone_s2",
            "coumarin6_acetone_s3",
        })
        self.assertEqual(set(plan.loc[plan.dilution_factor == 0, "dye_source"]), {"acetone_working"})
        self.assertEqual(set(plan.loc[plan.dilution_factor == 1, "dye_source"]),
                         {"coumarin6_acetone_stock"})

    def test_selected_pair_uses_actual_solvent_vial_in_execution(self):
        lash = self.fake_coordinator(simulate=True)
        config = {**self.config, "DYE": "coumarin-6", "DYE_SOLVENT": "acetone",
                  "SOLVENT_SOURCE_VIALS": ["acetone"]}
        lash.workflow_config = config
        lash.measure_wellplate.return_value = None
        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                            flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}):
            execute(config, show_gui=False)
        preparation_sources = [call.args[0] for call in lash.nr_robot.dispense_from_vial_into_vial.call_args_list]
        self.assertIn("acetone", preparation_sources)
        self.assertIn("coumarin6_acetone_stock", preparation_sources)
        self.assertNotIn("DMSO", preparation_sources)
        plate_sources = {column for call in lash.nr_robot.dispense_from_vials_into_wellplate.call_args_list
                         for column in call.args[0].columns}
        self.assertIn("acetone_working", plate_sources)
        self.assertNotIn("acetone", plate_sources)
        self.assertNotIn("DMSO_working", plate_sources)
        self.assertNotIn("DMSO", plate_sources)

    def test_prefilled_working_solvent_is_selected_for_each_pair(self):
        inventory = pd.read_csv(self.vial_file)
        inventory.loc[inventory.vial_name.str.endswith("_working"), "vial_volume"] = 6.0
        inventory.to_csv(self.vial_file, index=False)
        for dye, solvent in (("pyrene", "DMSO"), ("coumarin-6", "acetone")):
            with self.subTest(dye=dye, solvent=solvent):
                lash = self.fake_coordinator(simulate=True)
                lash.workflow_config = {**self.config, "DYE": dye, "DYE_SOLVENT": solvent,
                                        "SIMULATE": True, "SOLVENT_SOURCE_VIALS": [solvent]}
                lash.measure_wellplate.return_value = None
                coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                                    flatten_cytation_data=MagicMock())
                with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), \
                        patch.object(workflow, "_SAVE_FILES_ON_SIMULATE", True):
                    output = execute(lash.workflow_config, show_gui=False)
                transfers = lash.nr_robot.dispense_from_vial_into_vial.call_args_list
                self.assertTrue(all(not call.args[1].endswith("_working") for call in transfers))
                plate_sources = {column for call in lash.nr_robot.dispense_from_vials_into_wellplate.call_args_list
                                 for column in call.args[0].columns}
                self.assertIn(f"{solvent}_working", plate_sources)
                self.assertNotIn(solvent, plate_sources)
                plan = pd.read_csv(output / "well_plan.csv")
                self.assertEqual(set(plan.loc[plan.dilution_factor == 0, "dye_source"]),
                                 {f"{solvent}_working"})

    def test_interrupt_during_staged_transfer_restores_homes_and_runs_cleanup(self):
        lash = self.fake_coordinator(simulate=True)
        lash.workflow_config = {**self.config, "SIMULATE": True}
        original = lash.nr_robot.VIAL_DF[["home_location", "home_location_index"]].copy()
        lash.nr_robot.dispense_from_vial_into_vial.side_effect = KeyboardInterrupt
        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                            flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), patch.object(
                interrupt_cleanup, "cleanup_after_keyboard_interrupt") as cleanup:
            with self.assertRaises(KeyboardInterrupt):
                execute(lash.workflow_config, show_gui=False)
        cleanup.assert_called_once_with(lash)
        pd.testing.assert_frame_equal(original, lash.nr_robot.VIAL_DF[original.columns])

    def test_queued_run_refills_depleted_water_before_plate_dispensing(self):
        lash = self.fake_coordinator(simulate=True)
        robot = lash.nr_robot
        robot.VIAL_DF.loc[robot.VIAL_DF.vial_name == "water", "vial_volume"] = 3.325
        lash.workflow_config = {**self.config, "SIMULATE": True, "TOTAL_VOLUME_UL": 250.0,
                                "TEST_SOLVENT": False}
        lash.measure_wellplate.return_value = None
        def refill(reservoir, vial, volume, **kwargs):
            index = robot.VIAL_DF.index[robot.VIAL_DF.vial_name == vial][0]
            robot.VIAL_DF.at[index, "vial_volume"] += volume
        robot.dispense_into_vial_from_reservoir.side_effect = refill
        def dispense(volumes, **kwargs):
            if "water" in volumes.columns:
                required = volumes.water.sum()
                available = robot.get_vial_info("water", "vial_volume")
                self.assertGreaterEqual(available, required,
                                        f"Not enough solution in vial water: need {required:.3f} mL, have {available:.3f} mL")
        robot.dispense_from_vials_into_wellplate.side_effect = dispense
        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                            flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}):
            execute(lash.workflow_config, show_gui=False)
        robot.dispense_into_vial_from_reservoir.assert_called_once_with(1, "water", 4.675)
        calls = robot.method_calls
        refill_index = next(i for i, call in enumerate(calls) if call[0] == "dispense_into_vial_from_reservoir")
        self.assertEqual(calls[refill_index - 1][0], "remove_pipet")
        first_plate_index = next(i for i, call in enumerate(calls) if call[0] == "dispense_from_vials_into_wellplate")
        self.assertLess(refill_index, first_plate_index)

    def test_water_refill_skips_full_vial_and_disabled_water_medium(self):
        for enabled, volume in ((True, 8.0), (True, 7.95), (False, 3.325)):
            with self.subTest(enabled=enabled, volume=volume):
                lash = self.fake_coordinator(simulate=True)
                robot = lash.nr_robot
                robot.VIAL_DF.loc[robot.VIAL_DF.vial_name == "water", "vial_volume"] = volume
                lash.workflow_config = {**self.config, "SIMULATE": True, "TEST_WATER": enabled}
                lash.measure_wellplate.return_value = None
                coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                                    flatten_cytation_data=MagicMock())
                with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}):
                    execute(lash.workflow_config, show_gui=False)
                robot.dispense_into_vial_from_reservoir.assert_not_called()

    def test_startup_refills_depleted_surfactant_from_large_vial(self):
        lash = self.fake_coordinator(simulate=True)
        robot = lash.nr_robot
        rows = robot.VIAL_DF
        rows.loc[rows.vial_name == "surfactant", "vial_volume"] = 0.55
        lash.workflow_config = {**self.config, "SIMULATE": True, "TOTAL_VOLUME_UL": 250.0,
                                "TEST_SOLVENT": False, "SURFACTANT_LIQUID": "water"}
        lash.measure_wellplate.return_value = None
        def transfer(source, destination, volume, **kwargs):
            if destination == "surfactant":
                rows.loc[rows.vial_name == source, "vial_volume"] -= volume
                rows.loc[rows.vial_name == destination, "vial_volume"] += volume
        robot.dispense_from_vial_into_vial.side_effect = transfer
        def dispense(volumes, **kwargs):
            if "surfactant" in volumes.columns:
                self.assertGreaterEqual(robot.get_vial_info("surfactant", "vial_volume"),
                                        volumes.surfactant.sum())
        robot.dispense_from_vials_into_wellplate.side_effect = dispense
        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                            flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}):
            execute(lash.workflow_config, show_gui=False)
        refill = [call for call in robot.dispense_from_vial_into_vial.call_args_list
                  if call.args[1] == "surfactant"]
        self.assertEqual(len(refill), 1)
        self.assertEqual(refill[0].args[:2], ("surfactant_refill", "surfactant"))
        self.assertAlmostEqual(refill[0].args[2], 6.95)
        self.assertEqual(refill[0].kwargs["liquid"], "water")
        self.assertAlmostEqual(robot.get_vial_info("surfactant", "vial_volume"), 7.5)
        self.assertAlmostEqual(robot.get_vial_info("surfactant_refill", "vial_volume"), 12.95)
        calls = robot.method_calls
        refill_index = next(i for i, call in enumerate(calls) if call[0] == "dispense_from_vial_into_vial"
                            and call.args[1] == "surfactant")
        plate_index = next(i for i, call in enumerate(calls) if call[0] == "dispense_from_vials_into_wellplate")
        self.assertLess(refill_index, plate_index)

    def test_surfactant_refill_skips_full_vial_and_disabled_medium(self):
        for enabled, volume in ((True, 7.5), (True, 7.45), (False, 0.55)):
            with self.subTest(enabled=enabled, volume=volume):
                lash = self.fake_coordinator(simulate=True)
                robot = lash.nr_robot
                robot.VIAL_DF.loc[robot.VIAL_DF.vial_name == "surfactant", "vial_volume"] = volume
                lash.workflow_config = {**self.config, "SIMULATE": True, "TEST_SURFACTANT": enabled}
                lash.measure_wellplate.return_value = None
                coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                                    flatten_cytation_data=MagicMock())
                with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}):
                    execute(lash.workflow_config, show_gui=False)
                self.assertTrue(all(call.args[1] != "surfactant" for call in
                                    robot.dispense_from_vial_into_vial.call_args_list))

    def test_insufficient_surfactant_refill_stops_before_transfers(self):
        lash = self.fake_coordinator(simulate=True)
        robot = lash.nr_robot
        robot.VIAL_DF.loc[robot.VIAL_DF.vial_name == "surfactant", "vial_volume"] = 0.55
        robot.VIAL_DF.loc[robot.VIAL_DF.vial_name == "surfactant_refill", "vial_volume"] = 1.0
        lash.workflow_config = {**self.config, "SIMULATE": True}
        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                            flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), self.assertRaisesRegex(
                ValueError, "Insufficient surfactant_refill"):
            execute(lash.workflow_config, show_gui=False)
        robot.dispense_from_vial_into_vial.assert_not_called()
        robot.dispense_into_vial_from_reservoir.assert_not_called()

    def test_missing_generated_vials_are_reported_before_dispensing(self):
        inventory = pd.read_csv(self.vial_file)
        inventory = inventory[inventory.vial_name != "coumarin6_acetone_s2"]
        inventory.to_csv(self.vial_file, index=False)
        lash = self.fake_coordinator(simulate=True)
        config = {**self.config, "DYE": "coumarin-6", "DYE_SOLVENT": "acetone"}
        lash.workflow_config = config
        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                            flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}):
            with self.assertRaisesRegex(ValueError, "coumarin6_acetone_s2"):
                execute(config, show_gui=False)
        lash.nr_robot.dispense_from_vial_into_vial.assert_not_called()

    def test_repeated_randomized_multiplate_plan(self):
        config = {**self.config, "REPETITIONS": 2, "REPLICATES": 5,
                  "WELLPLATE_TYPE": "48 WELL PLATE", "RANDOMIZED_ORDER": True,
                  "FRESH_SUBSTOCKS": True}
        plan, recipes = build_plan(config)
        pd.testing.assert_frame_equal(plan, build_plan(config)[0])
        self.assertEqual(plan.plate.nunique(), 4)
        self.assertEqual(len(recipes), 6)
        self.assertEqual(set(recipes.vial), {
            "pyrene_DMSO_r1_s1", "pyrene_DMSO_r1_s2", "pyrene_DMSO_r1_s3",
            "pyrene_DMSO_r2_s1", "pyrene_DMSO_r2_s2", "pyrene_DMSO_r2_s3",
        })
        self.assertFalse(plan.duplicated(["plate", "well_position"]).any())
        self.assertLess(plan.well_index.max(), 48)
        self.assertEqual(set(plan.substock_batch), {1, 2})

    def test_invalid_plans_fail(self):
        for override in ({"DYE_VOLUME_UL": 200}, {"REPLICATES": 0},
                         {"DILUTION_FACTORS": [0.5, 0.5]}, {"WELLPLATE_TYPE": "12 WELL PLATE"},
                         {"SURFACTANT_CONCENTRATION_MM": 10, "SURFACTANT_CMC_MM": 9.9},
                         {"REPETITIONS": 21, "SUBSTOCK_VOLUME_ML": 1.0},
                         {"TOTAL_VOLUME_UL": float("nan")},
                         {"SOLVENT_SOURCE_VIALS": []},
                         {"SOLVENT_SOURCE_VIALS": ["DMSO", "DMSO"]},
                         {"SOLVENT_SOURCE_VIALS": "DMSO"},
                         {"SOLVENT_SOURCE_RESERVE_ML": -0.1},
                         {"SOLVENT_SOURCE_RESERVE_ML": float("nan")},
                         {"DYE": "unknown"}, {"DYE_SOLVENT": "---"}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                build_plan({**self.config, **override})

    def test_plate_scale_preserves_dye_fraction_and_splits_wells(self):
        base = self.config.copy()
        base["DYE_VOLUME_UL"] = 5
        base["TOTAL_VOLUME_UL"] = 200
        for plate_type, scale, plate_count in (
            ("96 WELL PLATE", 1, 1), ("48 WELL PLATE", 2, 1),
            ("24 WELL PLATE", 4, 2),
        ):
            with self.subTest(plate_type=plate_type):
                plan, recipes = build_plan({**base, "WELLPLATE_TYPE": plate_type})
                self.assertEqual(plan.plate.nunique(), plate_count)
                self.assertTrue((plan.dye_volume_ul == 5 * scale).all())
                self.assertTrue((plan.medium_volume_ul == 195 * scale).all())
                self.assertTrue((plan.volume_scale == scale).all())
                self.assertTrue((plan.concentration_relative == plan.dilution_factor * 0.025).all())
                self.assertEqual(recipes.stock_ml.sum(), build_plan(base)[1].stock_ml.sum())

    def test_blank_correction_and_repeat_reads_are_not_independent_wells(self):
        test_config = {**self.config, "DILUTION_FACTORS": [0.0, 0.5, 1.0], "SUBSTOCK_VOLUME_ML": 1.0}
        plan, _ = build_plan(test_config)
        plan["signal"] = 50 + plan.concentration_relative * 1000
        reads = pd.concat([plan.assign(measurement_replicate=i, timepoint_min=0) for i in (1, 2)])
        summary = summarize(reads, ["signal"])
        self.assertTrue((summary.signal_count == 3).all())
        np.testing.assert_allclose(summary.signal_blank_corrected, summary.concentration_relative * 1000)
        fits = fit_calibration(summary, ["signal"], 100)
        np.testing.assert_allclose(fits.slope_per_um, 10)
        np.testing.assert_allclose(fits.r_squared, 1)

    def test_execution_with_fake_hardware(self):
        lash = self.fake_coordinator(simulate=False)
        test_config = {**self.config, "DILUTION_FACTORS": [0.0, 0.25, 0.5, 1.0], "SUBSTOCK_VOLUME_ML": 2.0}
        plan, _ = build_plan(test_config)
        data = pd.DataFrame({"well_position": plan.well_position,
                             "334_373": 10 + plan.concentration_relative * 100,
                             "334_384": 20 + plan.concentration_relative * 200})
        lash.measure_wellplate.side_effect = [data]
        module = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                       flatten_cytation_data=lambda raw, _: raw)
        with tempfile.TemporaryDirectory() as temporary:
            protocol = Path(temporary) / "test.prt"
            protocol.touch()
            lash.workflow_config = {**test_config, "SIMULATE": False, "PROTOCOL_FILE": str(protocol)}
            with patch.dict("sys.modules", {"master_usdl_coordinator": module}):
                output = execute({**test_config, "SIMULATE": False,
                                  "PROTOCOL_FILE": str(protocol)}, show_gui=False)
        self.assertEqual(lash.nr_robot.dispense_from_vial_into_vial.call_count, 5)
        self.assertEqual(lash.nr_robot.dispense_from_vial_into_vial.call_args_list[0].args,
                         ("DMSO", "DMSO_working", 6.0))
        lash.discard_used_wellplate.assert_called_once()
        measured = pd.read_csv(output / "fluorescence_results.csv")
        self.assertEqual(len(measured), 36)
        self.assertEqual(len(pd.read_csv(output / "calibration_fits.csv")), 6)
        total_dispensed = sum(call.args[0].sum().sum() for call in
                              lash.nr_robot.dispense_from_vials_into_wellplate.call_args_list)
        self.assertAlmostEqual(total_dispensed, 7.2)
        plate_calls = lash.nr_robot.dispense_from_vials_into_wellplate.call_args_list
        self.assertTrue(all("DMSO" not in call.args[0].columns for call in plate_calls))
        working_dispensed = sum(call.args[0]["DMSO_working"].sum() for call in plate_calls
                                if "DMSO_working" in call.args[0].columns)
        self.assertAlmostEqual(working_dispensed, 2.385)
        moves = [call for call in lash.nr_robot.move_vial_to_location.call_args_list
                 if call.args[1] == "clamp"]
        self.assertEqual({call.args[0] for call in moves},
                         {"water", "surfactant", "pyrene_DMSO_stock", "pyrene_DMSO_s1", "pyrene_DMSO_s2", "DMSO_working"})
        self.assertTrue(all(call.args[1:] == ("clamp", 0) for call in moves))
        calls = lash.nr_robot.method_calls
        for i, call in enumerate(calls):
            if call[0] == "move_vial_to_location" and call.args[1] == "clamp":
                self.assertEqual(calls[i - 2][0], "remove_pipet")
                self.assertEqual(calls[i + 1][0], "dispense_from_vials_into_wellplate")
                self.assertEqual(list(calls[i + 1].args[0].columns), [call.args[0]])
                self.assertEqual(calls[i + 1].kwargs["strategy"], "serial")

    def test_resuming_nearly_full_working_solvent_does_not_refill(self):
        inventory = pd.read_csv(self.vial_file)
        inventory.loc[inventory.vial_name == "DMSO_working", "vial_volume"] = 5.97
        inventory.to_csv(self.vial_file, index=False)
        lash = self.fake_coordinator(simulate=True)
        lash.workflow_config = {**self.config, "SIMULATE": True}
        lash.measure_wellplate.return_value = None
        module = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                       flatten_cytation_data=lambda raw, _: raw)
        with patch.dict("sys.modules", {"master_usdl_coordinator": module}):
            execute(lash.workflow_config, show_gui=False)
        transfers = lash.nr_robot.dispense_from_vial_into_vial.call_args_list
        self.assertTrue(all(call.args[1] != "DMSO_working" for call in transfers))

    def test_insufficient_bulk_or_working_solvent_stops_before_transfers(self):
        for changes in ({"bulk": 16.0}, {"repetitions": 3}):
            with self.subTest(changes=changes):
                inventory = pd.read_csv(self.vial_file)
                inventory.loc[inventory.vial_name == "DMSO", "vial_volume"] = changes.get("bulk", 19.9)
                inventory.to_csv(self.vial_file, index=False)
                lash = self.fake_coordinator(simulate=True)
                lash.workflow_config = {**self.config, "SIMULATE": True,
                                        "SOLVENT_SOURCE_VIALS": ["DMSO"],
                                        "REPETITIONS": changes.get("repetitions", 1)}
                lash.nr_track.NUM_SOURCE = 10
                module = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                               flatten_cytation_data=lambda raw, _: raw)
                with patch.dict("sys.modules", {"master_usdl_coordinator": module}), self.assertRaisesRegex(
                        ValueError, "[Ii]nsufficient.*solvent"):
                    execute(lash.workflow_config, show_gui=False)
                lash.nr_robot.dispense_from_vial_into_vial.assert_not_called()

    def test_gui_changes_drive_plan_recipes_protocol_and_saved_config(self):
        initial = {**self.config, "SIMULATE": False}
        confirmed = {**initial, "SIMULATE": True, "REPLICATES": 1, "DILUTION_FACTORS": [0.0, 0.5, 1.0],
                     "DYE_VOLUME_UL": 10.0, "SUBSTOCK_VOLUME_ML": 2.0,
                     "PROTOCOL_FILE": "confirmed.prt", "RAW_CHANNELS": ["confirmed_signal"],
                     "MEASUREMENT_REPLICATES": 2}
        lash = self.fake_coordinator(simulate=True)
        lash.measure_wellplate.return_value = None
        events = []

        def load_config(name, namespace):
            namespace.update(initial)

        def review(*args, **kwargs):
            events.append("review")
            self.assertTrue(kwargs["show_gui"])
            self.assertFalse((self.root / "output").exists())
            kwargs["workflow_globals"].update(confirmed)
            lash.workflow_config = confirmed.copy()
            return lash

        def confirmed_plan(config):
            events.append("plan")
            self.assertEqual(config, confirmed)
            return build_plan(config)

        manager = types.SimpleNamespace(setup_and_reload_config=load_config)
        coordinator = types.SimpleNamespace(Lash_E=review, flatten_cytation_data=MagicMock())
        with patch.dict(workflow.__dict__), patch.dict("sys.modules", {
            "workflow_config_manager": types.SimpleNamespace(ConfigManager=manager),
            "master_usdl_coordinator": coordinator,
        }), patch.object(workflow, "build_plan", side_effect=confirmed_plan), patch.object(workflow, "plot_calibration"), \
            patch.object(workflow, "_SAVE_FILES_ON_SIMULATE", True):
            output = execute()
        self.assertEqual(events, ["review", "plan"])
        self.assertEqual(yaml.safe_load((output / "config.yaml").read_text()), confirmed)
        expected_plan, expected_recipes = build_plan(confirmed)
        saved_plan = pd.read_csv(output / "well_plan.csv")
        self.assertEqual(len(saved_plan), 9)
        np.testing.assert_allclose(saved_plan.dye_volume_ul, expected_plan.dye_volume_ul)
        pd.testing.assert_frame_equal(pd.read_csv(output / "substock_recipes.csv"), expected_recipes)
        reads = pd.read_csv(output / "fluorescence_results.csv")
        self.assertEqual(len(reads), 18)
        self.assertIn("confirmed_signal", reads.columns)
        self.assertEqual([call.args[0] for call in lash.measure_wellplate.call_args_list],
                         ["confirmed.prt", "confirmed.prt"])

    def test_cancel_does_not_build_plan_or_dispense(self):
        lash = self.fake_coordinator(simulate=True)
        lash._workflow_should_continue = False
        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash), flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), patch.object(workflow, "build_plan") as planner:
            self.assertIsNone(execute(self.config, show_gui=False))
        planner.assert_not_called()
        lash.nr_robot.dispense_from_vial_into_vial.assert_not_called()
        self.assertFalse((self.root / "output").exists())

    def test_supplied_config_is_delegated_before_planning(self):
        lash = self.fake_coordinator(simulate=True)
        lash.workflow_config = {**self.config, "SIMULATE": True}
        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash), flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), patch.object(workflow, "build_plan", side_effect=RuntimeError("test stop before automation")) as planner:
            with self.assertRaisesRegex(RuntimeError, "test stop"):
                execute({**self.config, "SIMULATE": True}, show_gui=False)
        coordinator.Lash_E.assert_called_once()
        self.assertNotIn("simulate", coordinator.Lash_E.call_args.kwargs)
        self.assertFalse(coordinator.Lash_E.call_args.kwargs['show_gui'])
        self.assertEqual(planner.call_args.args[0], lash.workflow_config)
        lash.nr_robot.dispense_from_vial_into_vial.assert_not_called()
        self.assertFalse((self.root / "output").exists())

    def test_selected_vial_path_comes_from_confirmed_coordinator_config(self):
        (self.root / "other.csv").touch()
        lash = self.fake_coordinator(simulate=True)
        lash.workflow_config = {**self.config, "SIMULATE": True, "INPUT_VIAL_STATUS_FILE": str(self.root / "other.csv")}
        coordinator = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash), flatten_cytation_data=MagicMock())
        with patch.dict("sys.modules", {"master_usdl_coordinator": coordinator}), patch.object(workflow, "build_plan", side_effect=RuntimeError("test stop before automation")) as planner:
            with self.assertRaisesRegex(RuntimeError, "test stop"):
                execute({**self.config, "SIMULATE": True, "INPUT_VIAL_STATUS_FILE": str(self.root / "other.csv")}, show_gui=False)
        self.assertEqual(planner.call_args.args[0]['INPUT_VIAL_STATUS_FILE'], str(self.root / "other.csv"))
        lash.nr_robot.dispense_from_vial_into_vial.assert_not_called()
        self.assertFalse((self.root / "output").exists())

    def test_invalid_confirmed_config_does_not_save_plan_or_dispense(self):
        initial = {**self.config, "SIMULATE": True}
        lash = self.fake_coordinator(simulate=True)

        def review(*args, **kwargs):
            kwargs["workflow_globals"].update(initial, REPLICATES=0)
            lash.workflow_config = {**initial, "REPLICATES": 0}
            return lash

        manager = types.SimpleNamespace(setup_and_reload_config=lambda name, namespace: namespace.update(initial))
        with patch.dict(workflow.__dict__), patch.dict("sys.modules", {
            "workflow_config_manager": types.SimpleNamespace(ConfigManager=manager),
            "master_usdl_coordinator": types.SimpleNamespace(Lash_E=review, flatten_cytation_data=MagicMock()),
        }):
            with self.assertRaisesRegex(ValueError, "REPLICATES"):
                execute()
        lash.nr_robot.dispense_from_vial_into_vial.assert_not_called()
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
