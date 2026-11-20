import unittest

import pandas as pd

from workflows import surfactant_dye_cmc as workflow
from workflows.surfactant_dye_cmc import build_plan, fill_chunks

BASE = {key: getattr(workflow, key) for key in workflow._CONFIG_KEYS}


def config(**changes):
    c = dict(BASE)
    c.update(changes)
    return c


class SurfactantDyeCmcPlanTests(unittest.TestCase):
    def test_default_plan_matches_config_blocks(self):
        plan = build_plan(config())
        wells = plan["wells"]
        cmc_wells = sum(n for _, n in BASE["CMC_LEVELS_X_CMC"])
        check_wells = len(BASE["STOCK_CHECK_FRACTIONS"]) * BASE["STOCK_CHECK_REPLICATES"] + BASE["STOCK_CHECK_BLANKS"]
        self.assertEqual((wells.block == "cmc").sum(), cmc_wells)
        self.assertEqual((wells.block == "stock_check").sum(), check_wells)
        self.assertTrue(wells.well_index.is_unique)
        self.assertTrue(((wells.loc[wells.matrix == "water", "dye_ul"]) == BASE["DYE_VOLUME_UL"]).all())
        cmc = wells[wells.block == "cmc"]
        pd.testing.assert_series_equal(cmc.surfactant_mM, cmc.level * BASE["SURFACTANT_CMC_MM"],
                                       check_names=False)

    def test_plate_contents_follow_config(self):
        plan = build_plan(config(CMC_LEVELS_X_CMC=[[0.0, 2], [1.0, 4]], CMC_DYE_UM=0.5,
                                 STOCK_CHECK_FRACTIONS=[0.5], STOCK_CHECK_BLANKS=1))
        wells = plan["wells"]
        self.assertEqual(len(wells), 8)
        self.assertEqual(set(wells.loc[wells.block == "cmc", "dye_uM"]), {0.5})
        self.assertEqual(sorted(wells.loc[wells.block == "cmc", "replicate"]), [1, 1, 2, 2, 3, 4])

    def test_batches_share_one_plate_and_reuse_substocks(self):
        plan = build_plan(config(CMC_BATCH=1, STOCK_CHECK_BATCH=2, BATCH_SCHEDULES_MIN=[[0, 30], [0]]))
        wells, recipes = plan["wells"], plan["recipes"]
        self.assertEqual(set(wells.loc[wells.block == "cmc", "batch"]), {1})
        self.assertEqual(set(wells.loc[wells.block == "stock_check", "batch"]), {2})
        self.assertTrue(wells.well_index.is_unique)
        # The 40 uM dye substock made for the CMC block is reused by the stock check.
        self.assertEqual(recipes.name.is_unique, True)
        made_in_1 = set(recipes.loc[recipes.batch == 1, "name"])
        used_in_2 = set(wells.loc[wells.batch == 2, "dye_source"])
        self.assertTrue(used_in_2 & made_in_1)

    def test_sequential_wells_without_randomization(self):
        wells = build_plan(config(RANDOMIZED_ORDER=False))["wells"]
        self.assertEqual(list(wells.well_index), list(range(len(wells))))
        self.assertEqual(wells.well_position.iloc[12], "B1")

    def test_default_surfactant_wells_meet_preferred_minimum(self):
        wells = build_plan(config())["wells"]
        used = wells.loc[wells.surfactant_ul > 0, "surfactant_ul"]
        self.assertGreaterEqual(used.min(), BASE["SURFACTANT_PREFERRED_MIN_UL"] - 1e-9)

    def test_invalid_config(self):
        with self.assertRaisesRegex(ValueError, "DYE_SUBSTOCK_REFILL_ML < DYE_SUBSTOCK_ML"):
            build_plan(config(DYE_SUBSTOCK_REFILL_ML=4.0))
        with self.assertRaisesRegex(ValueError, "SURFACTANT_PREFERRED_MIN_UL"):
            build_plan(config(SURFACTANT_PREFERRED_MIN_UL=10.0))
        with self.assertRaisesRegex(ValueError, "batch with a schedule"):
            build_plan(config(CMC_BATCH=2))
        with self.assertRaisesRegex(ValueError, "start at 0"):
            build_plan(config(BATCH_SCHEDULES_MIN=[[10, 30]]))
        with self.assertRaisesRegex(ValueError, "DISPENSE_ORDER"):
            build_plan(config(DISPENSE_ORDER=["water", "dye"]))
        with self.assertRaisesRegex(ValueError, "free on the plate"):
            build_plan(config(CMC_LEVELS_X_CMC=[[1.0, 100]]))

    def test_fill_chunks_fit_one_vial_fill(self):
        volumes = pd.Series([0.19] * 60)
        chunks = fill_chunks(volumes, 5.5)
        self.assertTrue(all(chunk.sum() <= 5.5 for chunk in chunks))
        self.assertEqual(sum(len(chunk) for chunk in chunks), 60)


if __name__ == "__main__":
    unittest.main()
