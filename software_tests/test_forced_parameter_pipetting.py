"""Hardware-free checks for forced-parameter pipetting: rows, forcing, tip groups, resume, faults."""
import csv
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows import forced_parameter_pipetting as fp
from pipetting_data.pipetting_parameters import PipettingParameters

LIQUIDS = {"water": {"density": 1.0, "refill_pipets": False, "condition_tip": True},
           "glycerol": {"density": 1.26, "refill_pipets": True, "condition_tip": False}}
TIPS = {"small_tip": {"volume": 0.2, "capacity_margin": 0.005, "min_suggested_volume": 0.001},
        "large_tip": {"volume": 1.0, "capacity_margin": 0.01, "min_suggested_volume": 0.15}}


def row(set_id, liquid="water", volume=50, **changes):
    base = dict(param_set_id=set_id, liquid=liquid, volume_ul=str(volume), aspirate_speed="10",
                dispense_speed="10", retract_speed="10", blowout_speed="12", aspirate_wait_time="1",
                dispense_wait_time="0", post_retract_wait_time="0", overaspirate_vol_ul="2",
                pre_asp_air_vol_ul="0", post_asp_air_vol_ul="0", blowout_vol_ul="10", asp_disp_cycles="0")
    base.update({k: str(v) for k, v in changes.items()})
    return base


class Robot:
    """Enough of North_Robot: tip choice by volume (liquid=None), tracked vials, recorded calls."""
    def __init__(self, volumes=None, masses=None, change=None):
        self.PIPET_TIPS = TIPS
        self.HELD_PIPET_TYPE = None
        self.volumes = dict(volumes or {"water": 7.0, "water_2": 7.0, "glycerol": 7.0})
        self.locations = {}
        self.masses = list(masses or [])
        self.change = change or {}
        self.calls = []
        self.logger = logging.getLogger("forced_test")
        self.logger.addHandler(logging.NullHandler())
        self.logger.propagate = False

    def get_config_parameter(self, config, key, parameter):
        assert (config, key, parameter) == ("pumps", 0, "volume")
        return 1.0

    def resolve_pipetting_setup(self, volume, liquid, params, specified_tip=None):
        assert liquid is None
        out = PipettingParameters(**{k: getattr(params, k) for k in vars(params)})
        for k, v in self.change.items():
            setattr(out, k, v)
        return out, ("small_tip" if volume < 0.15 else "large_tip")

    def get_vial_info(self, vial, field):
        if vial not in self.volumes:
            raise KeyError(vial)
        return self.volumes[vial] if field == "vial_volume" else self.locations.get(vial, "main_8mL_rack")

    def aspirate_from_vial(self, vial, volume, parameters=None, liquid=None, specified_tip=None):
        assert liquid is None and parameters is not None
        if self.HELD_PIPET_TYPE is None:
            self.calls.append(("get_tip", specified_tip))
        assert self.locations.get(vial) == "clamp", "pipetted off the scale"
        self.HELD_PIPET_TYPE = specified_tip
        self.calls.append(("aspirate", vial, round(volume * 1000, 3), parameters.blowout_speed))

    def dispense_into_vial(self, vial, volume, parameters=None, liquid=None, measure_weight=False,
                           continuous_mass_monitoring=False, save_mass_data=False, record_measurement=True):
        assert liquid is None and measure_weight and not record_measurement
        self.calls.append(("dispense", vial, round(volume * 1000, 3)))
        mass = self.masses.pop(0) if self.masses else volume
        return mass, {"pre_baseline_std": 0.0001, "post_baseline_std": 0.0001, "pre_stable_count": 10,
                      "pre_total_count": 10, "post_stable_count": 10, "post_total_count": 10}

    def condition_tip(self, vial, volume_per_cycle_mL=0.1, cycles=3, specified_tip=None):
        self.calls.append(("condition", vial, specified_tip))
        self.HELD_PIPET_TYPE = specified_tip

    def remove_pipet(self):
        self.calls.append(("remove_tip",))
        self.HELD_PIPET_TYPE = None

    def move_vial_to_location(self, vial, location, index):
        assert self.HELD_PIPET_TYPE is None, "a vial moved under a held tip"
        assert location == "clamp" and not [v for v, l in self.locations.items() if l == "clamp"]
        self.locations[vial] = location
        self.calls.append(("clamp", vial))

    def return_vial_home(self, vial):
        self.locations[vial] = "main_8mL_rack"
        self.calls.append(("home", vial))


def run(robot, rows, folder, replicates=1, minutes=0.0, source_vials=None, simulate=False):
    plan, rejected = fp.check_rows(rows, robot, LIQUIDS, source_vials or {}, 1.0)
    lash_e = SimpleNamespace(nr_robot=robot, logger=robot.logger, simulate=simulate)
    config = {"CAMPAIGN_ID": "t", "MAX_RUN_MINUTES": minutes, "VIAL_LOW_THRESHOLD_ML": 1.0}
    campaign = fp._Campaign(lash_e, config, folder)
    done = fp.load_done(folder)
    stopped = None
    with patch.object(campaign.validation, "_scale_ready", return_value=True), \
         patch.object(campaign.validation, "_get_latest_environmental_data", return_value=None), \
         patch.object(campaign.validation, "_notify"):
        try:
            for key, items in fp.pending_groups(plan, set(done), replicates):
                campaign.run_group(key, items)
        except fp._Stop as why:
            stopped = str(why)
    return campaign, rejected, stopped


def saved(folder):
    with open(Path(folder) / "measurements.csv", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class RowTests(unittest.TestCase):
    def test_every_value_is_required_and_checked_against_robot_limits(self):
        rows = [row("ok"), row("blank", blowout_speed=""), row("fast", aspirate_speed=99),
                row("frac", asp_disp_cycles=1.5), row("oil", liquid="oil")]
        plan, rejected = fp.check_rows(rows, Robot(), LIQUIDS, {}, 1.0)
        self.assertEqual([r["param_set_id"] for r in plan], ["ok"])
        reasons = dict(rejected)
        self.assertIn("blowout_speed is blank", reasons["blank"])
        self.assertIn("aspirate_speed", reasons["fast"])  # robot_hardware.yaml parameter_limits
        self.assertIn("whole number", reasons["frac"])
        self.assertIn("liquids.yaml", reasons["oil"])

    def test_uL_columns_become_mL_and_every_field_is_set(self):
        volume, params = fp.parse_row(row("a", volume=500, overaspirate_vol_ul=5, blowout_vol_ul=50))
        self.assertAlmostEqual(volume, 0.5)
        self.assertAlmostEqual(params.overaspirate_vol, 0.005)
        self.assertAlmostEqual(params.blowout_vol, 0.05)
        self.assertEqual(params.blowout_speed, 12)  # North_Safe's default is None; forced here
        self.assertTrue(all(getattr(params, f) is not None for f, _ in fp.PARAMETER_COLUMNS.values()))

    def test_more_air_and_liquid_than_the_pump_holds_is_rejected(self):
        plan, rejected = fp.check_rows([row("big", volume=800, pre_asp_air_vol_ul=300)], Robot(), LIQUIDS, {}, 1.0)
        self.assertEqual(plan, [])
        self.assertIn("pipetting pump", rejected[0][1])

    def test_a_row_north_safe_would_change_is_rejected(self):
        plan, rejected = fp.check_rows([row("a")], Robot(change={"blowout_speed": 20}), LIQUIDS, {}, 1.0)
        self.assertEqual(plan, [])
        self.assertIn("rejected_override_mismatch", rejected[0][1])

    def test_a_liquid_with_no_usable_vial_is_rejected(self):
        robot = Robot(volumes={"water": 0.5})
        _, rejected = fp.check_rows([row("a")], robot, LIQUIDS, {}, 1.0)
        self.assertIn("no source vial", rejected[0][1])

    def test_duplicate_ids_and_missing_columns_refuse_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.csv"
            path.write_text("liquid,volume_ul\nwater,50\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                fp.read_rows(path)
            fields = list(row("a"))
            with open(path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows([row("a"), row("a")])
            with self.assertRaisesRegex(ValueError, "unique"):
                fp.read_rows(path)


class RunTests(unittest.TestCase):
    def test_rows_share_a_conditioned_tip_per_liquid_and_tip_type(self):
        robot = Robot()
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _, stopped = run(robot, [row("s1"), row("l1", volume=500), row("s2", volume=20)], tmp,
                                       replicates=2)
            self.assertIsNone(stopped)
            self.assertEqual(campaign.tips_used, 2)
            self.assertEqual([c for c in robot.calls if c[0] == "condition"],
                             [("condition", "water", "small_tip"), ("condition", "water", "large_tip")])
            rows = saved(tmp)
            self.assertEqual(len(rows), 6)
            self.assertEqual([r["new_tip"] for r in rows], ["True", "False", "False", "False", "True", "False"])
            self.assertEqual({r["validation_type"] for r in rows}, {"forced_parameters"})

    def test_a_rows_replicates_column_overrides_replicates(self):
        robot = Robot()
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _, _ = run(robot, [{**row("control"), "replicates": "3"}, row("b")], tmp, replicates=1)
            self.assertEqual([r["param_set_id"] for r in saved(tmp)], ["control"] * 3 + ["b"])
        _, rejected = fp.check_rows([{**row("x"), "replicates": "2.5"}], Robot(), LIQUIDS, {}, 1.0)
        self.assertIn("replicates", rejected[0][1])

    def test_tip_group_column_gives_a_fresh_tip(self):
        robot = Robot()
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _, _ = run(robot, [row("a"), {**row("b"), "tip_group": "second"}], tmp)
            self.assertEqual(campaign.tips_used, 2)

    def test_a_refill_liquid_gets_a_new_unconditioned_tip_each_time(self):
        robot = Robot()
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _, _ = run(robot, [row("g", liquid="glycerol")], tmp, replicates=3)
            self.assertEqual(campaign.tips_used, 3)
            self.assertFalse([c for c in robot.calls if c[0] == "condition"])
            self.assertEqual(sum(c[0] == "remove_tip" for c in robot.calls), 3)

    def test_mass_becomes_volume_with_the_liquids_density(self):
        robot = Robot(masses=[0.063])
        with tempfile.TemporaryDirectory() as tmp:
            run(robot, [row("g", liquid="glycerol", volume=50)], tmp)
            (rec,) = saved(tmp)
            self.assertAlmostEqual(float(rec["measured_ul"]), 50.0, places=3)
            self.assertAlmostEqual(float(rec["accuracy_pct"]), 0.0, places=2)

    def test_resume_skips_what_is_saved_and_a_changed_row_is_not_mixed_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            run(Robot(), [row("a"), row("b")], tmp, replicates=2)
            self.assertEqual(len(saved(tmp)), 4)
            robot = Robot()
            campaign, _, _ = run(robot, [row("a"), row("b"), row("c")], tmp, replicates=2)
            self.assertEqual(campaign.saved, 2)  # only c's two replicates
            plan, _ = fp.check_rows([row("a", blowout_vol_ul=20), row("b")], Robot(), LIQUIDS, {}, 1.0)
            self.assertEqual(fp.changed_rows(plan, fp.load_done(tmp)), ["a"])

    def test_an_empty_source_moves_to_the_next_vial_then_stops(self):
        robot = Robot(volumes={"water": 0.5, "water_2": 7.0})
        with tempfile.TemporaryDirectory() as tmp:
            run(robot, [row("a")], tmp, source_vials={"water": ["water", "water_2"]})
            self.assertEqual(saved(tmp)[0]["source_vial"], "water_2")
            self.assertIn(("clamp", "water_2"), robot.calls)
            robot = Robot(volumes={"water": 0.5, "water_2": 0.5})
            _, rejected, _ = run(robot, [row("a")], tmp, source_vials={"water": ["water", "water_2"]})
            self.assertIn("no source vial", rejected[0][1])

    def test_three_tiny_dispenses_in_a_row_stop_the_run_with_data_kept(self):
        robot = Robot(masses=[0.05, 0.001, 0.001, 0.001, 0.05])
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _, stopped = run(robot, [row("a")], tmp, replicates=5)
            self.assertIn("under 10%", stopped)
            self.assertEqual(len(saved(tmp)), 4)

    def test_no_time_left_stops_starting_measurements(self):
        robot = Robot()
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _, stopped = run(robot, [row("a")], tmp, replicates=3, minutes=1.0)  # < END_RESERVE_S
            self.assertIn("out of time", stopped)
            self.assertEqual(campaign.saved, 0)

    def test_an_unstable_reading_is_recorded_not_retried(self):
        robot = Robot()
        bad = {"pre_baseline_std": 0.5, "post_baseline_std": 0.5, "pre_stable_count": 0, "pre_total_count": 10,
               "post_stable_count": 0, "post_total_count": 10}
        original = robot.dispense_into_vial
        robot.dispense_into_vial = lambda *a, **k: (original(*a, **k)[0], bad)
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _, stopped = run(robot, [row("a")], tmp, replicates=2)
            self.assertIsNone(stopped)  # the scale settled again (patched check)
            self.assertEqual([r["measurement_acceptable"] for r in saved(tmp)], ["False", "False"])
            self.assertEqual(sum(c[0] == "dispense" for c in robot.calls), 2)


class RefillTests(unittest.TestCase):
    def test_only_liquids_with_a_reservoir_are_topped_up_once_per_vial(self):
        robot = Robot()
        calls = []
        robot.refill_vial = lambda vial, target, reservoir=None, liquid=None: calls.append(
            (vial, target, reservoir, liquid)) or 1.0
        plan, _ = fp.check_rows([row("a"), row("b"), row("g", liquid="glycerol")], robot, LIQUIDS, {}, 1.0)
        groups = fp.pending_groups(plan, set(), 1)
        added = fp.refill_sources(robot, plan, groups, {"water": 1}, 8.0, robot.logger)
        self.assertEqual(calls, [("water", 8.0, 1, "water")])
        self.assertEqual(added, {"water": 1.0})


if __name__ == "__main__":
    unittest.main()
