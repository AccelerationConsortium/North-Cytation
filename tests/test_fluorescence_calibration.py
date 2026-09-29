import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import types

import numpy as np
import pandas as pd

from workflows.fluorescence_calibration_workflow import (
    DEFAULTS, build_plan, execute, fit_calibration, summarize,
)


class FluorescenceCalibrationTests(unittest.TestCase):
    def test_constant_solvent_and_concentration(self):
        plan, recipes = build_plan({**DEFAULTS, "STOCK_CONCENTRATION_UM": 100})
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
        config = {**DEFAULTS, "REPETITIONS": 2, "REPLICATES": 5,
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
                build_plan({**DEFAULTS, **override})

    def test_blank_correction_and_repeat_reads_are_not_independent_wells(self):
        test_config = {**DEFAULTS, "DILUTION_FACTORS": [0.0, 0.5, 1.0], "SUBSTOCK_VOLUME_ML": 1.0}
        plan, _ = build_plan(test_config)
        plan["signal"] = 50 + plan.concentration_relative * 1000
        reads = pd.concat([plan.assign(measurement_replicate=i) for i in (1, 2)])
        summary = summarize(reads, ["signal"])
        self.assertTrue((summary.signal_count == 3).all())
        np.testing.assert_allclose(summary.signal_blank_corrected, summary.concentration_relative * 1000)
        fits = fit_calibration(summary, ["signal"], 100)
        np.testing.assert_allclose(fits.slope_per_um, 10)
        np.testing.assert_allclose(fits.r_squared, 1)

    def test_execution_with_fake_hardware(self):
        lash = MagicMock()
        lash.nr_track.CURRENT_WP_TYPE = "96 WELL PLATE"
        lash.nr_track.NUM_SOURCE = 1
        lash.nr_robot.WELLPLATES = {"96 WELL PLATE": {"max_volume_per_well": 0.3}}
        lash.nr_robot.get_vial_in_location.return_value = None
        test_config = {**DEFAULTS, "DILUTION_FACTORS": [0.0, 0.25, 0.5, 1.0], "SUBSTOCK_VOLUME_ML": 1.0}
        plan, _ = build_plan(test_config)
        data = pd.DataFrame({"well_position": plan.well_position,
                             "334_373": 10 + plan.concentration_relative * 100,
                             "334_384": 20 + plan.concentration_relative * 200})
        lash.measure_wellplate.side_effect = [None, data]
        module = types.SimpleNamespace(Lash_E=MagicMock(return_value=lash),
                                       flatten_cytation_data=lambda raw, _: raw)
        with tempfile.TemporaryDirectory() as temporary:
            protocol = Path(temporary) / "test.prt"
            protocol.touch()
            with patch.dict("sys.modules", {"master_usdl_coordinator": module}):
                output = execute({**test_config, "SIMULATE": False,
                                  "PROTOCOL_FILE": str(protocol), "SHAKE_PROTOCOL_FILE": str(protocol)})
        self.assertEqual(lash.nr_robot.dispense_from_vial_into_vial.call_count, 4)
        lash.discard_used_wellplate.assert_called_once()
        measured = pd.read_csv(output / "fluorescence_results.csv")
        self.assertEqual(len(measured), 36)
        self.assertEqual(len(pd.read_csv(output / "calibration_fits.csv")), 6)
        total_dispensed = sum(call.args[0].sum().sum() for call in
                              lash.nr_robot.dispense_from_vials_into_wellplate.call_args_list)
        self.assertAlmostEqual(total_dispensed, 7.2)
        moves = lash.nr_robot.move_vial_to_location.call_args_list
        self.assertEqual({call.args[0] for call in moves},
                         {"water", "surfactant", "dye_stock", "dye_b1_s1", "dye_b1_s2"})
        self.assertTrue(all(call.args[1:] == ("clamp", 0) for call in moves))
        calls = lash.nr_robot.method_calls
        for i, call in enumerate(calls):
            if call[0] == "move_vial_to_location":
                self.assertEqual(calls[i - 2][0], "remove_pipet")
                self.assertEqual(calls[i + 1][0], "dispense_from_vials_into_wellplate")
                self.assertEqual(list(calls[i + 1].args[0].columns), [call.args[0]])
                self.assertEqual(calls[i + 1].kwargs["strategy"], "serial")


if __name__ == "__main__":
    unittest.main()
