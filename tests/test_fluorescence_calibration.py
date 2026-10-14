import unittest
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
from workflows.fluorescence_calibration_workflow import build_plan, execute, fit_calibration, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = {key: getattr(workflow, key) for key in workflow._CONFIG_KEYS}
BASE_CONFIG["DYE"] = "pyrene"


class FluorescenceCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.vial_file = self.root / "vials.csv"
        shutil.copyfile(REPO_ROOT / "status" / "fluorescence_calibration_vials.csv", self.vial_file)
        inventory = pd.read_csv(self.vial_file)
        inventory.loc[inventory.vial_name.str.startswith('dye_b'), 'vial_volume'] = 0.0
        inventory.to_csv(self.vial_file, index=False)
        self.original_directory = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.original_directory)
        self.config = {**BASE_CONFIG, "INPUT_VIAL_STATUS_FILE": str(self.vial_file), "MEASUREMENT_SCHEDULE_MIN": [0]}
        slack = patch.object(workflow, "slack_agent", MagicMock())
        slack.start()
        self.addCleanup(slack.stop)

    def fake_coordinator(self, simulate):
        lash = MagicMock()
        lash._workflow_should_continue = True
        lash.simulate = simulate
        lash.nr_robot.VIAL_FILE = str(self.vial_file)
        lash.nr_track.CURRENT_WP_TYPE = "96 WELL PLATE"
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
                        self.assertEqual(robot.get_vial_info("dye_stock", "home_location_index"), 0)
                        self.assertEqual(robot.get_vial_info("dye_b1_s1", "home_location_index"), 1)
                    return "transferred"
                robot.dispense_from_vial_into_vial.side_effect = transfer
                result = workflow.dispense_vial_to_vial(lash, "dye_stock", "dye_b1_s1", 3.0, "DMSO")
                self.assertEqual(result, "transferred")
                pd.testing.assert_frame_equal(original, robot.VIAL_DF[original.columns])
                if source_cap == destination_cap == "open":
                    robot.move_vial_to_location.assert_not_called()
                    robot.remove_pipet.assert_not_called()
                    robot.return_vial_home.assert_not_called()
                else:
                    self.assertEqual([call.args for call in robot.move_vial_to_location.call_args_list],
                                     [("dye_stock", "main_8mL_rack", 0), ("dye_b1_s1", "main_8mL_rack", 1)])
                    calls = robot.method_calls
                    move_index = next(i for i, call in enumerate(calls) if call[0] == "move_vial_to_location")
                    self.assertTrue(any(call[0] == "remove_pipet" for call in calls[:move_index]))
                    return_index = next(i for i, call in enumerate(calls) if call[0] == "return_vial_home")
                    self.assertTrue(any(call[0] == "remove_pipet" for call in calls[move_index:return_index]))

    def test_large_source_stays_in_place_with_closed_destination(self):
        lash = self.fake_coordinator(simulate=True)
        workflow.dispense_vial_to_vial(lash, "solvent", "dye_b1_s1", 3.0, "DMSO")
        lash.nr_robot.move_vial_to_location.assert_called_once_with("dye_b1_s1", "main_8mL_rack", 1)
        lash.nr_robot.return_vial_home.assert_called_once_with("dye_b1_s1")

    def test_split_closed_cap_transfer_uses_safe_homes_for_clamp_exchanges(self):
        from North_Safe import North_Robot
        from pipetting_data.pipetting_parameters import PipettingParameters
        from pipetting_data.pipetting_wizard import PipettingWizard

        lash = self.fake_coordinator(simulate=True)
        robot = lash.nr_robot
        rows = robot.VIAL_DF
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
        robot.move_vial_to_location.side_effect = move
        def cap(capped, **kwargs):
            rows.at[robot.get_vial_in_location("clamp", 0), "capped"] = capped
            robot.GRIPPER_STATUS = None if capped else "Cap"
        robot.uncap_clamp_vial.side_effect = lambda **kwargs: cap(False)
        robot.recap_clamp_vial.side_effect = lambda **kwargs: cap(True)
        returns = []
        def return_home(vial, **kwargs):
            returns.append((robot.get_vial_info(vial, "home_location_index"), robot.GRIPPER_STATUS))
            North_Robot.return_vial_home(robot, vial, **kwargs)
        robot.return_vial_home.side_effect = return_home
        robot.PIPET_TIPS = {"large_tip": {"volume": 1.0}}
        robot._get_optimized_parameters.return_value = PipettingParameters()
        robot.aspirate_from_vial.side_effect = lambda vial, *args, **kwargs: (
            North_Robot._ensure_vial_accessible_for_pipetting(robot, vial))
        robot.dispense_into_vial.side_effect = lambda vial, *args, **kwargs: (
            North_Robot._ensure_vial_accessible_for_pipetting(robot, vial), None)[1]
        robot.dispense_from_vial_into_vial.side_effect = lambda *args, **kwargs: (
            North_Robot.dispense_from_vial_into_vial(robot, *args, **kwargs))
        with patch.object(PipettingWizard, "get_max_calibrated_volume", return_value=None):
            workflow.dispense_vial_to_vial(lash, "dye_stock", "dye_b1_s1", 2.5, "DMSO")
        self.assertGreater(robot.aspirate_from_vial.call_count, 1)
        self.assertTrue(all(slot in (0, 1) for slot, _ in returns[:-2]))
        self.assertEqual([slot for slot, _ in returns[-2:]], [18, 24])
        self.assertEqual(robot.get_vial_info("dye_stock", "location_index"), 18)
        self.assertEqual(robot.get_vial_info("dye_b1_s1", "location_index"), 24)

    def test_occupied_staging_slot_fails_before_movements(self):
        lash = self.fake_coordinator(simulate=True)
        lash.nr_robot.get_vial_in_location.side_effect = [None, 2]
        with self.assertRaisesRegex(RuntimeError, "staging.*occupied"):
            workflow.dispense_vial_to_vial(lash, "dye_stock", "dye_b1_s1", 3.0, "DMSO")
        lash.nr_robot.remove_pipet.assert_not_called()
        lash.nr_robot.move_vial_to_location.assert_not_called()
        lash.nr_robot.dispense_from_vial_into_vial.assert_not_called()

    def test_transfer_failure_restores_homes_without_recovery_movements(self):
        lash = self.fake_coordinator(simulate=True)
        robot = lash.nr_robot
        original = robot.VIAL_DF[["home_location", "home_location_index"]].copy()
        robot.dispense_from_vial_into_vial.side_effect = RuntimeError("transfer interrupted")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            workflow.dispense_vial_to_vial(lash, "dye_stock", "dye_b1_s1", 3.0, "DMSO")
        pd.testing.assert_frame_equal(original, robot.VIAL_DF[original.columns])
        robot.save_robot_status.assert_called_once()
        robot.return_vial_home.assert_not_called()

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

    def test_repeated_randomized_multiplate_plan(self):
        config = {**self.config, "REPETITIONS": 2, "REPLICATES": 5,
                  "WELLPLATE_TYPE": "48 WELL PLATE", "RANDOMIZED_ORDER": True,
                  "FRESH_SUBSTOCKS": True}
        plan, recipes = build_plan(config)
        pd.testing.assert_frame_equal(plan, build_plan(config)[0])
        self.assertEqual(plan.plate.nunique(), 4)
        self.assertEqual(len(recipes), 6)
        self.assertFalse(plan.duplicated(["plate", "well_position"]).any())
        self.assertLess(plan.well_index.max(), 48)
        self.assertEqual(set(plan.substock_batch), {1, 2})

    def test_invalid_plans_fail(self):
        for override in ({"DYE_VOLUME_UL": 200}, {"REPLICATES": 0},
                         {"DILUTION_FACTORS": [0.5, 0.5]}, {"WELLPLATE_TYPE": "24 WELL PLATE"},
                         {"SURFACTANT_CONCENTRATION_MM": 10, "SURFACTANT_CMC_MM": 9.9},
                         {"REPETITIONS": 21, "SUBSTOCK_VOLUME_ML": 1.0},
                         {"TOTAL_VOLUME_UL": float("nan")}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                build_plan({**self.config, **override})

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
                         ("solvent", "solvent_working", 6.0))
        lash.discard_used_wellplate.assert_called_once()
        measured = pd.read_csv(output / "fluorescence_results.csv")
        self.assertEqual(len(measured), 36)
        self.assertEqual(len(pd.read_csv(output / "calibration_fits.csv")), 6)
        total_dispensed = sum(call.args[0].sum().sum() for call in
                              lash.nr_robot.dispense_from_vials_into_wellplate.call_args_list)
        self.assertAlmostEqual(total_dispensed, 7.2)
        plate_calls = lash.nr_robot.dispense_from_vials_into_wellplate.call_args_list
        self.assertTrue(all("solvent" not in call.args[0].columns for call in plate_calls))
        working_dispensed = sum(call.args[0]["solvent_working"].sum() for call in plate_calls
                                if "solvent_working" in call.args[0].columns)
        self.assertAlmostEqual(working_dispensed, 2.385)
        moves = [call for call in lash.nr_robot.move_vial_to_location.call_args_list
                 if call.args[1] == "clamp"]
        self.assertEqual({call.args[0] for call in moves},
                         {"water", "surfactant", "dye_stock", "dye_b1_s1", "dye_b1_s2", "solvent_working"})
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
        inventory.loc[inventory.vial_name == "solvent_working", "vial_volume"] = 5.97
        inventory.to_csv(self.vial_file, index=False)
        lash = self.fake_coordinator(simulate=True)
        lash.workflow_config = {**self.config, "SIMULATE": True}
        lash.measure_wellplate.return_value = None
        module = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                       flatten_cytation_data=lambda raw, _: raw)
        with patch.dict("sys.modules", {"master_usdl_coordinator": module}):
            execute(lash.workflow_config, show_gui=False)
        transfers = lash.nr_robot.dispense_from_vial_into_vial.call_args_list
        self.assertTrue(all(call.args[1] != "solvent_working" for call in transfers))

    def test_insufficient_bulk_or_working_solvent_stops_before_transfers(self):
        for changes in ({"bulk": 18.0}, {"repetitions": 3}):
            with self.subTest(changes=changes):
                inventory = pd.read_csv(self.vial_file)
                inventory.loc[inventory.vial_name == "solvent", "vial_volume"] = changes.get("bulk", 19.9)
                inventory.to_csv(self.vial_file, index=False)
                lash = self.fake_coordinator(simulate=True)
                lash.workflow_config = {**self.config, "SIMULATE": True,
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
        }), patch.object(workflow, "build_plan", side_effect=confirmed_plan), patch.object(workflow, "plot_calibration"):
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
