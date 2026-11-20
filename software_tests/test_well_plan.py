import unittest

import pandas as pd

from workflows.helpers.well_plan import after_batch, plan_wells, top_up

RULES = dict(
    total_ul=200.0, dye_ul=5.0, well_min_ul=20.0, well_max_ul=150.0,
    substock_min_transfer_ul=200.0,
    stocks={"surfactant": {"vial": "SDS_stock", "conc": 100.0},
            "dye": {"vial": "dye_stock", "conc": 50.0}},
    water_vial="water", solvent_vial="ethanol",
    substock_ml={"surfactant": 6.0, "dye": 4.0},
    substock_refill_ml={"surfactant": 4.0, "dye": 2.0},
    vial_capacity_ml=8.0,
    substock_prefix={"surfactant": "SDS_sub", "dye": "dye_sub"},
)


def targets(surfactant=(), dye_in_water=1.0, solvent_dye=()):
    rows = [dict(matrix="water", surfactant_mM=c, dye_uM=dye_in_water) for c in surfactant]
    rows += [dict(matrix="solvent", surfactant_mM=0.0, dye_uM=d) for d in solvent_dye]
    return pd.DataFrame(rows)


def achieved(wells, sources, component):
    conc = {s["name"]: s["conc"] for s in sources}
    source, volume = f"{component}_source", f"{component}_ul"
    return wells.apply(lambda w: 0.0 if w[source] is None else
                       conc[w[source]] * w[volume] / RULES["total_ul"], axis=1)


class WellPlanTests(unittest.TestCase):
    def test_volumes_and_concentrations_match_targets(self):
        t = targets([0, 2.58, 8.6, 12.9, 43.0], solvent_dye=[0, 2.5, 25, 37.5])
        wells, sources, _ = plan_wells(t, RULES)
        total = wells[["surfactant_ul", "dye_ul", "water_ul", "solvent_ul"]].sum(axis=1)
        self.assertTrue(((total - 200).abs() < 1e-9).all())
        pd.testing.assert_series_equal(achieved(wells, sources, "surfactant"), t.surfactant_mM,
                                       check_names=False)
        pd.testing.assert_series_equal(achieved(wells, sources, "dye"), t.dye_uM, check_names=False)

    def test_water_wells_get_the_same_dye_volume_including_dye_free_wells(self):
        wells, _, _ = plan_wells(targets([0, 8.6, 43.0], dye_in_water=0.0).assign(
            dye_uM=[0.0, 1.0, 1.0]), RULES)
        self.assertEqual(set(wells.dye_ul), {5.0})
        self.assertEqual(wells.dye_source.iloc[0], "ethanol")

    def test_variable_volumes_stay_in_range_and_prefer_large_volumes(self):
        wells, _, _ = plan_wells(targets([0.5, 2.58, 8.6, 12.9, 30.1, 43.0]), RULES)
        used = wells.surfactant_ul
        self.assertTrue(((used >= 20 - 1e-9) & (used <= 150 + 1e-9)).all())
        # 12.9 mM fits stock (25.8 uL, in range), so no extra substock is made.
        self.assertEqual(wells.surfactant_source.iloc[3], "SDS_stock")

    def test_largest_volume_wins_when_several_sources_fit(self):
        # Stock check 25 uM: stock gives 100 uL, the 40 uM dye substock 125 uL.
        wells, _, _ = plan_wells(targets([8.6], solvent_dye=[25.0]), RULES)
        self.assertEqual(wells.dye_source.iloc[1], "dye_sub_1")
        self.assertAlmostEqual(wells.dye_ul.iloc[1], 125.0)

    def test_fewest_surfactant_substocks_for_proposed_cmc_block(self):
        cmc = 8.6
        levels = [0, 0.3, 0.6] + [0.75 + i * 0.6 / 11 for i in range(12)] + [1.5, 2, 2.5, 3.5, 5]
        _, _, recipes = plan_wells(targets([x * cmc for x in levels for _ in range(3)]), RULES)
        self.assertEqual((recipes.component == "surfactant").sum(), 1)
        self.assertEqual((recipes.component == "dye").sum(), 1)

    def test_substock_transfers_respect_minimum_and_parents_come_first(self):
        _, _, recipes = plan_wells(targets([0.002, 0.5, 8.6]), RULES)
        self.assertTrue((recipes.parent_ml >= 0.2 - 1e-12).all())
        made = {"SDS_stock", "dye_stock"}
        for recipe in recipes.itertuples():
            self.assertIn(recipe.parent, made)
            made.add(recipe.name)

    def test_most_concentrated_parent_is_used_when_transfer_is_large_enough(self):
        _, _, recipes = plan_wells(targets([0.5, 8.6]), RULES)
        surf = recipes[recipes.component == "surfactant"].set_index("name")
        # 8.6 mM substock: 0.69 mL of stock. 0.5 mM substock: stock would need
        # only 40 uL, so it is made from the first substock instead.
        self.assertEqual(surf.loc["SDS_sub_1", "parent"], "SDS_stock")
        self.assertEqual(surf.loc["SDS_sub_2", "parent"], "SDS_sub_1")

    def test_later_batch_reuses_substocks(self):
        _, sources, recipes = plan_wells(targets([2.58, 8.6]), RULES)
        sub = recipes[recipes.component == "surfactant"].name.iloc[0]
        second, _, new = plan_wells(targets([3.0, 7.0]), RULES, existing=after_batch(sources))
        self.assertTrue(new.empty)
        self.assertEqual(set(second.surfactant_source), {sub})

    def test_top_up_keeps_recipe_ratio(self):
        parent, diluent = top_up(6.0, 0.35, 4.0, 0.2, 8.0)
        self.assertAlmostEqual(parent + diluent, 2.0)
        self.assertAlmostEqual(parent / (parent + diluent), 0.35)
        # A parent part under the minimum transfer is raised, diluent scaled with it.
        parent, diluent = top_up(4.0, 0.08, 2.0, 0.2, 8.0)
        self.assertAlmostEqual(parent, 0.2)
        self.assertAlmostEqual(parent / (parent + diluent), 0.08)
        with self.assertRaisesRegex(ValueError, "overfills"):
            top_up(6.0, 0.02, 4.0, 0.2, 8.0)

    def test_heavy_use_plans_top_ups_and_charges_parent(self):
        # 60 wells x ~150 uL from one substock: 9 mL, so several 2 mL top-ups.
        wells, sources, recipes = plan_wells(targets([1.0] * 60), RULES)
        used = wells.surfactant_source.iloc[0]
        sub = next(s for s in sources if s["name"] == used)
        draw = wells.surfactant_ul.sum() / 1000
        self.assertEqual(sub["topups"], -(-(draw - 2.0) // 2.0))
        stock = next(s for s in sources if s["name"] == sub["parent"])
        per_topup = 2.0 * sub["parent_ml"] / 6.0
        self.assertAlmostEqual(stock["demand_ml"], sub["parent_ml"] + sub["topups"] * max(per_topup, 0.2))

    def test_demand_includes_make_up_and_substock_preparation(self):
        wells, sources, recipes = plan_wells(targets([2.58], solvent_dye=[25]), RULES)
        demand = {s["name"]: s["demand_ml"] for s in sources}
        water = wells.water_ul.sum() / 1000 + recipes.loc[recipes.diluent == "water", "diluent_ml"].sum()
        self.assertAlmostEqual(demand["water"], water)
        ethanol = wells.solvent_ul.sum() / 1000 + recipes.loc[recipes.diluent == "ethanol", "diluent_ml"].sum()
        self.assertAlmostEqual(demand["ethanol"], ethanol)

    def test_preferred_minimum_adds_substocks_and_raises_volumes(self):
        levels = [0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 2, 5]
        t = targets([x * 8.6 for x in levels])
        rules = dict(RULES, stocks=dict(RULES["stocks"], surfactant={"vial": "SDS_stock", "conc": 50.0}),
                     well_max_ul=195.0)
        loose, _, loose_recipes = plan_wells(t, rules)
        tight, _, tight_recipes = plan_wells(t, dict(rules, preferred_min_ul={"surfactant": 75.0}))
        self.assertLess(loose.surfactant_ul.min(), 75)
        self.assertGreaterEqual(tight.surfactant_ul.min(), 75 - 1e-9)
        self.assertGreater((tight_recipes.component == "surfactant").sum(),
                           (loose_recipes.component == "surfactant").sum())
        # The preferred minimum is for the surfactant only; dye keeps well_min_ul.
        check = targets([], solvent_dye=[5.0])
        dye_only, _, _ = plan_wells(check, dict(rules, preferred_min_ul={"surfactant": 75.0}))
        self.assertAlmostEqual(dye_only.dye_ul.iloc[0], 20.0)

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "stronger stock"):
            plan_wells(targets([80.0]), RULES)
        with self.assertRaisesRegex(ValueError, "Solvent wells cannot contain surfactant"):
            plan_wells(pd.DataFrame([dict(matrix="solvent", surfactant_mM=1.0, dye_uM=1.0)]), RULES)
        with self.assertRaisesRegex(ValueError, "refill level"):
            plan_wells(targets([1.0]), dict(RULES, substock_refill_ml={"surfactant": 6.0, "dye": 2.0}))


if __name__ == "__main__":
    unittest.main()
