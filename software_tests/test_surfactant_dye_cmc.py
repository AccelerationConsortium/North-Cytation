import unittest
from datetime import datetime, timedelta

import pandas as pd

from workflows import surfactant_dye_cmc as workflow
from workflows.surfactant_dye_cmc import (build_plan, fill_chunks, plate_layout, record_rows,
                                          reusable_substocks, total_demand, update_substock_record)

BASE = {key: getattr(workflow, key) for key in workflow._CONFIG_KEYS}


# For tests whose own blocks have no pause after dye.
NO_PAUSE = dict(DYE_SOLVENT_EVAPORATES=False, DISPENSE_PAUSES=[])


def config(**changes):
    c = dict(BASE)
    c.update(changes)
    return c


def block(name="b", batch=1, matrix="water", replicates=1, **levels):
    b = dict(name=name, batch=batch, matrix=matrix, replicates=replicates)
    b.update(levels)
    return b


class SurfactantDyeCmcPlanTests(unittest.TestCase):
    def test_default_plan_is_the_coumarin_sds_plate(self):
        wells = build_plan(config())["wells"]
        counts = wells.groupby(["batch", "block"]).size().to_dict()
        self.assertEqual(counts, {(1, "stock_check"): 6, (2, "cmc"): 42, (2, "dye_matrix"): 46})
        self.assertTrue(wells.well_index.is_unique)
        self.assertTrue((wells.loc[wells.matrix == "water", "dye_ul"] == BASE["DYE_VOLUME_UL"]).all())
        matrix = wells[wells.block == "dye_matrix"]
        pairs = set(zip(matrix.surfactant_mM, matrix.dye_uM))
        self.assertNotIn((0.0, 1.0), pairs)
        self.assertNotIn((16.0, 1.0), pairs)
        self.assertEqual(len(pairs), 23)

    def test_stock_check_reuses_later_dye_substocks(self):
        plan = build_plan(config())
        wells, recipes = plan["wells"], plan["recipes"]
        later = set(wells.loc[wells.batch == 2, "dye_source"].dropna())
        check = set(wells.loc[wells.block == "stock_check", "dye_source"].dropna())
        self.assertTrue(check <= later | {BASE["DYE_STOCK_VIAL"]})
        self.assertEqual((recipes.component == "dye").sum(), 4)

    def test_substock_made_in_one_batch_is_not_charged_again_later(self):
        plan = build_plan(config())
        made = plan["recipes"]
        stock_ml = made.loc[made.parent == BASE["DYE_STOCK_VIAL"], "parent_ml"].sum()
        self.assertAlmostEqual(total_demand(plan)[BASE["DYE_STOCK_VIAL"]], stock_ml)

    def test_grid_replicates_and_multiples_of_cmc(self):
        b = block(replicates=2, surfactant_x_cmc=[0.0, 1.0], dye_uM=[0.5])
        wells = build_plan(config(BLOCKS=[b], BATCH_SCHEDULES_MIN=[[0]], **NO_PAUSE))["wells"]
        self.assertEqual(len(wells), 4)
        self.assertEqual(sorted(wells.surfactant_mM.unique()), [0.0, BASE["SURFACTANT_CMC_MM"]])
        self.assertEqual(sorted(wells.replicate), [1, 1, 2, 2])

    def test_other_plate_sizes_come_from_wellplates_yaml(self):
        layout = plate_layout("48 WELL PLATE")
        self.assertEqual((layout["wells"], layout["columns"]), (48, 8))
        b = block(surfactant_mM=[1.0], dye_uM=[1.0])
        wells = build_plan(config(BLOCKS=[b], BATCH_SCHEDULES_MIN=[[0]], WELLPLATE_TYPE="48 WELL PLATE",
                                  RANDOMIZED_ORDER=False, **NO_PAUSE))["wells"]
        self.assertEqual(wells.well_position.iloc[0], "A1")
        with self.assertRaisesRegex(ValueError, "free on the plate"):
            build_plan(config(WELLPLATE_TYPE="48 WELL PLATE"))

    def test_sequential_wells_without_randomization(self):
        wells = build_plan(config(RANDOMIZED_ORDER=False))["wells"]
        self.assertEqual(list(wells.well_index), list(range(len(wells))))
        self.assertEqual(wells.well_position.iloc[12], "B1")

    def test_default_surfactant_wells_meet_preferred_minimum(self):
        wells = build_plan(config())["wells"]
        used = wells.loc[wells.surfactant_ul > 0, "surfactant_ul"]
        self.assertGreaterEqual(used.min(), BASE["SURFACTANT_PREFERRED_MIN_UL"] - 1e-9)

    def test_extra_block_targets_feed_a_later_batch(self):
        original = workflow.extra_block_targets
        try:
            workflow.extra_block_targets = lambda c, batch, results: pd.DataFrame(
                [dict(batch=2, block="adaptive", replicate=1, matrix="water", surfactant_mM=9.0, dye_uM=1.0)])
            b = block(surfactant_mM=[0.0, 5.0], dye_uM=[1.0])
            wells = build_plan(config(BLOCKS=[b], BATCH_SCHEDULES_MIN=[[0], [0]], **NO_PAUSE))["wells"]
            self.assertEqual(wells.loc[wells.block == "adaptive", "batch"].tolist(), [2])
        finally:
            workflow.extra_block_targets = original

    def test_invalid_config(self):
        with self.assertRaisesRegex(ValueError, "DYE_SUBSTOCK_REFILL_ML < DYE_SUBSTOCK_ML"):
            build_plan(config(DYE_SUBSTOCK_REFILL_ML=4.0))
        with self.assertRaisesRegex(ValueError, "start at 0"):
            build_plan(config(BATCH_SCHEDULES_MIN=[[10, 30], [0]]))
        with self.assertRaisesRegex(ValueError, "DISPENSE_ORDER"):
            build_plan(config(DISPENSE_ORDER=["water", "dye"]))
        with self.assertRaisesRegex(ValueError, "batch with a schedule"):
            build_plan(config(BATCH_SCHEDULES_MIN=[[0]]))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_plan(config(BLOCKS=[block(dye_uM=[1.0])], BATCH_SCHEDULES_MIN=[[0]]))
        with self.assertRaisesRegex(ValueError, "skip entry"):
            build_plan(config(BLOCKS=[block(surfactant_mM=[1.0], dye_uM=[1.0], skip=[[2.0, 1.0]])],
                              BATCH_SCHEDULES_MIN=[[0]]))
        with self.assertRaisesRegex(ValueError, "cannot contain surfactant"):
            build_plan(config(BLOCKS=[block(matrix="solvent", surfactant_mM=[1.0], dye_uM=[1.0])],
                              BATCH_SCHEDULES_MIN=[[0]]))
        with self.assertRaisesRegex(ValueError, "unique name"):
            b = block(surfactant_mM=[1.0], dye_uM=[1.0])
            build_plan(config(BLOCKS=[b, b], BATCH_SCHEDULES_MIN=[[0]]))
        with self.assertRaisesRegex(ValueError, "TOTAL_VOLUME_UL"):
            build_plan(config(TOTAL_VOLUME_UL=400.0))

    # ---- reuse of substocks left from earlier runs ----

    def leftovers(self, volumes=None, made_at=None, **changes):
        """Plan a run, then pretend it left its substocks (volume mL each) in their vials."""
        c = config(**changes)
        first = build_plan(c)["recipes"]
        now = datetime(2026, 11, 21, 12, 0)
        record = record_rows(c, first, made_at or now - timedelta(hours=10))
        # Roughly what the simulated run leaves: SDS substocks ~5.4 mL, dye substocks ~3.5 mL.
        left = {n: (5.4 if comp == "surfactant" else 3.5) for n, comp in zip(first.name, first.component)}
        inventory = pd.DataFrame({"vial_volume": volumes or left})
        return c, first, record, inventory, now

    def test_second_run_reuses_every_leftover_and_makes_nothing_new(self):
        c, first, record, inventory, now = self.leftovers()
        reused, problems = reusable_substocks(c, inventory, record, now)
        self.assertEqual(problems, [])
        self.assertEqual(sorted(s["name"] for s in reused), sorted(first.name))
        plan = build_plan(c, reused)
        self.assertTrue(plan["recipes"].empty)
        self.assertEqual(total_demand(plan).get(BASE["DYE_STOCK_VIAL"], 0.0), 0.0)

    def test_partial_leftovers_new_substocks_take_free_names(self):
        c, first, record, inventory, now = self.leftovers()
        keep = first.name[first.component == "dye"].iloc[[0, 2]].tolist()  # e.g. dye_sub_1, dye_sub_3
        inventory = inventory.assign(vial_volume=[3.5 if n in keep else 0.0 for n in inventory.index])
        reused, problems = reusable_substocks(c, inventory, record, now)
        self.assertEqual(sorted(s["name"] for s in reused), sorted(keep))
        recipes = build_plan(c, reused)["recipes"]
        self.assertFalse(set(recipes.name) & set(keep))
        self.assertEqual((recipes.component == "dye").sum(), 2)

    def test_old_unrecorded_or_other_leftovers_are_problems(self):
        c, first, record, inventory, now = self.leftovers(made_at=datetime(2026, 11, 1))
        _, problems = reusable_substocks(c, inventory, record, now)
        self.assertTrue(problems and all("SUBSTOCK_MAX_AGE_HOURS" in p for p in problems))
        c, first, record, inventory, now = self.leftovers()
        _, problems = reusable_substocks(config(DYE="nile-red"), inventory, record, now)
        self.assertTrue(any("nile-red" in p for p in problems))
        _, problems = reusable_substocks(c, inventory, record.iloc[:0], now)
        self.assertTrue(problems and all("no record" in p for p in problems))

    def test_reused_concentration_follows_a_remeasured_stock(self):
        c, first, record, inventory, now = self.leftovers()
        reused, _ = reusable_substocks(config(DYE_STOCK_UM=2500.0), inventory, record, now)
        top = max(s["conc"] for s in reused if s["component"] == "dye")
        made = first.loc[first.component == "dye", "conc"].max()
        self.assertAlmostEqual(top, made * 2500.0 / BASE["DYE_STOCK_UM"])

    def test_record_keeps_other_vials_and_replaces_remade_ones(self):
        c, first, record, inventory, now = self.leftovers()
        again = record_rows(c, first.iloc[:1], now)
        merged = update_substock_record(record, again)
        self.assertEqual(len(merged), len(record))
        self.assertEqual(merged.set_index("vial").loc[first.name.iloc[0], "made_at"], again.made_at.iloc[0])

    def test_record_round_trips_through_its_file(self):
        import tempfile
        from pathlib import Path
        c, first, record, inventory, now = self.leftovers()
        with tempfile.TemporaryDirectory() as folder:
            path = workflow.substock_record_path(Path(folder) / "vials.csv")
            self.assertEqual(path.suffix, ".yaml")
            workflow.save_substock_record(record, path)
            loaded = workflow.load_substock_record(path)
        self.assertEqual(reusable_substocks(c, inventory, loaded, now)[1], [])
        self.assertEqual(len(loaded), len(record))

    # ---- pauses between components ----

    def dispense_log(self, batch, **changes):
        """Run dispense_batch on a fake robot; return the sequence of dispenses and pauses."""
        from types import SimpleNamespace
        from unittest import mock
        c = config(**changes)
        plan = build_plan(c)
        wells = plan["wells"][plan["wells"].batch == batch]
        events = []
        robot = SimpleNamespace(dispense_from_vials_into_wellplate=lambda frame, **kw: events.append(
            ("dispense", frame.columns[0])))
        lash = SimpleNamespace(nr_robot=robot, simulate=True, logger=mock.Mock())
        with mock.patch.object(workflow, "ensure_volume"), \
                mock.patch.object(workflow, "pause", lambda l, b, comp, m: events.append(("pause", comp, m))):
            workflow.dispense_batch(lash, c, wells, workflow.refill_table(c, plan["recipes"]), batch)
        return c, events

    def test_pause_after_dye_in_its_batch_only(self):
        settings = dict(DISPENSE_ORDER=["dye", "water", "surfactant", "solvent"],
                        DISPENSE_PAUSES=[{"batch": 2, "after": "dye", "minutes": 30}])
        c, events = self.dispense_log(2, **settings)
        pauses = [i for i, e in enumerate(events) if e[0] == "pause"]
        self.assertEqual(len(pauses), 1)
        self.assertEqual(events[pauses[0]], ("pause", "dye", 30.0))
        before = {e[1] for e in events[:pauses[0]]}
        after = {e[1] for e in events[pauses[0] + 1:]}
        plan = build_plan(c)["wells"]
        dye_sources = set(plan.loc[plan.batch == 2, "dye_source"].dropna())
        self.assertEqual(before, dye_sources)  # only dye (and the ethanol blank) before the pause
        self.assertFalse(after & dye_sources)
        self.assertIn(c["WATER_VIAL"], after)
        _, events = self.dispense_log(1, **settings)
        self.assertFalse([e for e in events if e[0] == "pause"])

    def test_evaporating_dye_solvent_goes_on_top(self):
        wells = build_plan(config())["wells"]
        water = wells[wells.matrix == "water"]
        self.assertTrue(((water.water_ul + water.surfactant_ul) - BASE["TOTAL_VOLUME_UL"]).abs().max() < 1e-9)
        sds = water[water.surfactant_source == BASE["SURFACTANT_STOCK_VIAL"]]
        conc = sds.surfactant_ul * BASE["SURFACTANT_STOCK_MM"] / BASE["TOTAL_VOLUME_UL"]
        self.assertTrue(((conc - sds.surfactant_mM).abs() < 1e-9).all())
        kept = build_plan(config(DYE_SOLVENT_EVAPORATES=False))["wells"]
        kept = kept[kept.matrix == "water"]
        self.assertTrue(((kept.water_ul + kept.surfactant_ul + kept.dye_ul) - BASE["TOTAL_VOLUME_UL"]).abs().max() < 1e-9)

    def test_evaporation_needs_dye_first_and_a_pause(self):
        with self.assertRaisesRegex(ValueError, "dye\" first"):
            build_plan(config(DISPENSE_ORDER=["water", "solvent", "surfactant", "dye"]))
        with self.assertRaisesRegex(ValueError, r"after dye in batch\(es\) \[2\]"):
            build_plan(config(DISPENSE_PAUSES=[]))
        with self.assertRaisesRegex(ValueError, "TOTAL_VOLUME_UL \\+ DYE_VOLUME_UL"):
            build_plan(config(TOTAL_VOLUME_UL=290.0))

    def test_invalid_pauses(self):
        for bad, message in (([{"batch": 3, "after": "dye", "minutes": 30}], "batch with a schedule"),
                             ([{"batch": 2, "after": "ethanol", "minutes": 30}], "after must be"),
                             ([{"batch": 2, "after": "dye", "minutes": 0}], "positive"),
                             ([{"batch": 2, "after": "dye"}], "entries are"),
                             ([{"batch": 2, "after": "dye", "minutes": 5}] * 2, "Two DISPENSE_PAUSES")):
            with self.assertRaisesRegex(ValueError, message):
                build_plan(config(DISPENSE_PAUSES=bad))

    def test_fill_chunks_fit_one_vial_fill(self):
        volumes = pd.Series([0.19] * 60)
        chunks = fill_chunks(volumes, 5.5)
        self.assertTrue(all(chunk.sum() <= 5.5 for chunk in chunks))
        self.assertEqual(sum(len(chunk) for chunk in chunks), 60)


if __name__ == "__main__":
    unittest.main()
