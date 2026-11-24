"""Planning in workflows/surfactant_target_samples.py (no hardware, fake error model)."""
import math

import pandas as pd
import pytest

from workflows import surfactant_target_samples as w

STOCKS = [dict(stock_id="SDS-001", material="SDS", concentration="50", units="mM"),
          dict(stock_id="CTAB-001", material="CTAB", concentration="20", units="mM"),
          dict(stock_id="WATER-001", material="water", concentration="", units="")]


def fake_error(volume_ul):
    return math.hypot(0.5 / volume_ul, 0.005)


def config(**over):
    c = {k: getattr(w, k) for k in w._CONFIG_KEYS}
    c.update(SURFACTANTS={"SDS": {"liquid": "SDS"}, "CTAB": {"liquid": "water"}},
             ADDITIVES={"pyrene": {"vial": "pyrene", "volume_ul": 5.0, "liquid": "ethanol"}})
    c.update(over)
    return c


def vials(*rows):
    base = [("SDS_stock", 8.0, "8_mL", "SDS-001"), ("CTAB_stock", 8.0, "8_mL", "CTAB-001"),
            ("water", 7.5, "8_mL", "WATER-001"), ("pyrene", 5.0, "8_mL", "")]
    rows = base + [(f"SDS_sub_{k}", 0.0, "8_mL", "") for k in range(1, 9)] + list(rows)
    return pd.DataFrame([dict(vial_name=n, vial_volume=v, vial_type=t, container_id="", stock_id=s)
                         for n, v, t, s in rows])


def targets(rows, c):
    frame = pd.DataFrame(rows)
    path = pytest.TMP / "targets.csv"
    frame.to_csv(path, index=False)
    return w.load_targets(c, path)


@pytest.fixture(autouse=True)
def _tmp(tmp_path):
    pytest.TMP = tmp_path


def plan(rows, c=None, vial_df=None):
    c = c or config()
    vial_df = vial_df if vial_df is not None else vials()
    sources = w.inventory(vial_df, lambda cid: None, STOCKS, list(c["SURFACTANTS"]))
    errors = {s: fake_error for s in c["SURFACTANTS"]}
    return w.build_plan(c, targets(rows, c), sources, errors, vial_df)


def test_replicates_blank_and_volumes_add_up():
    p = plan([dict(sample_id="blank", SDS_mM=0, CTAB_mM=0, pyrene=False, replicates=2),
              dict(sample_id="a", SDS_mM=1.0, CTAB_mM=0, pyrene=True, replicates=3)])
    wells = p["wells"]
    assert len(wells) == 5
    total = wells.SDS_ul + wells.CTAB_ul + wells.pyrene_ul + wells.water_ul
    assert total.round(9).eq(250.0).all()
    a = wells[wells.sample_id == "a"]
    assert a.SDS_planned_mM.round(9).eq(1.0).all() and a.pyrene_ul.eq(5.0).all()


def test_two_surfactants_share_the_well():
    p = plan([dict(sample_id="mix", SDS_mM=30.0, CTAB_mM=6.0, pyrene=True)])
    (row,) = p["wells"].to_dict("records")
    assert row["SDS_ul"] + row["CTAB_ul"] + 5.0 <= 250 + 1e-9
    assert row["SDS_planned_mM"] == pytest.approx(30.0) and row["CTAB_planned_mM"] == pytest.approx(6.0)


def test_physically_impossible_wells_are_skipped_with_reason():
    p = plan([dict(sample_id="too_much", SDS_mM=60.0, CTAB_mM=0, pyrene=False),
              dict(sample_id="overflow", SDS_mM=40.0, CTAB_mM=15.0, pyrene=False),
              dict(sample_id="ok", SDS_mM=1.0, CTAB_mM=0, pyrene=False)])
    reasons = dict(zip(p["skipped"].well_id, p["skipped"].reason))
    assert "above WELL_MAX_UL" in reasons["too_much_r1"]
    assert "together" in reasons["overflow_r1"]
    assert list(p["wells"].sample_id) == ["ok"]
    assert any("skipped" in x for x in p["warnings"])


def test_over_limit_samples_are_made_and_warned():
    p = plan([dict(sample_id="a", SDS_mM=1.0, CTAB_mM=0, pyrene=False)], c=config(MAX_ERROR_PCT=0.1))
    assert p["wells"].SDS_over_limit.all()
    assert any("above MAX_ERROR_PCT" in x for x in p["warnings"])


def test_unknown_stock_id_stops_the_run():
    with pytest.raises(ValueError, match="Unknown stock_id"):
        plan([dict(sample_id="a", SDS_mM=1.0, CTAB_mM=0, pyrene=False)],
             vial_df=vials(("mystery", 5.0, "8_mL", "NOPE-001")))


def test_surfactant_without_a_source_stops_the_run():
    c = config(SURFACTANTS={"SDS": {"liquid": "SDS"}, "Brij": {"liquid": "water"}}, ADDITIVES={})
    with pytest.raises(ValueError, match="No vial holds a known stock of Brij"):
        plan([dict(sample_id="a", SDS_mM=0, Brij_mM=1.0)], c=c)


def test_unknown_surfactant_column_is_an_error():
    with pytest.raises(ValueError, match="not in SURFACTANTS"):
        plan([dict(sample_id="a", SDS_mM=1.0, CTAB_mM=0, Tween_mM=1.0, pyrene=False)])


def test_new_substocks_skip_vials_that_hold_something():
    full = vials()
    full.loc[full.vial_name == "SDS_sub_1", "vial_volume"] = 3.0  # unknown leftover: not a source
    p = plan([dict(sample_id="low", SDS_mM=0.05, CTAB_mM=0, pyrene=False)], vial_df=full)
    assert not p["substocks"].empty and "SDS_sub_1" not in set(p["substocks"].vial)
    assert set(p["substocks"].vial) <= {f"SDS_sub_{k}" for k in range(2, 9)}


def test_identical_stock_vials_form_one_group():
    v = vials(("SDS_refill", 15.0, "20_mL", "SDS-001"))
    sources = w.inventory(v, lambda cid: None, STOCKS, ["SDS"])
    (group,) = sources["SDS"]
    assert group["name"] == "SDS_stock" and group["vials"] == ["SDS_stock", "SDS_refill"]
    assert group["available_ml"] == 23.0


def test_layout_splits_into_batches_and_plates():
    rows = [dict(sample_id=f"s{i}", SDS_mM=1.0 + i * 0.1, CTAB_mM=0, pyrene=False) for i in range(100)]
    p = plan(rows, c=config(BATCH_SIZE=24))
    wells = p["wells"]
    assert wells.plate.nunique() == 2 and wells.groupby("batch").plate.nunique().max() == 1
    assert not wells.duplicated(["plate", "well_index"]).any()


def test_check_vials_finds_missing_and_short_vials():
    c = config()
    p = plan([dict(sample_id="a", SDS_mM=1.0, CTAB_mM=0, pyrene=True, replicates=4)], c=c)
    short = vials()
    short = short[short.vial_name != "pyrene"]
    with pytest.raises(ValueError, match="add vial pyrene"):
        w.check_vials(c, p, short)
    full = vials()
    full.loc[full.vial_name == p["substocks"].vial.iloc[0], "vial_volume"] = 2.0 if not p["substocks"].empty else 0
    if not p["substocks"].empty:
        with pytest.raises(ValueError, match="must start empty"):
            w.check_vials(c, p, full)
    w.check_vials(c, p, vials())


class _FakeRobot:
    def __init__(self, volumes, types):
        self.volumes, self.types, self.calls = dict(volumes), types, []

    def get_vial_info(self, vial, key):
        return self.volumes[vial] if key == "vial_volume" else self.types[vial]

    def refill_vial(self, vial, target, reservoir=None, source_vial=None, liquid=None):
        add = target - self.volumes[vial]
        self.volumes[vial] = target
        if source_vial:
            self.volumes[source_vial] -= add
        self.calls.append(("refill", vial, round(add, 6), source_vial or f"reservoir {reservoir}"))

    def create_substock(self, vial, components, vortex_time=None):
        for comp in components:
            self.volumes[vial] += comp["volume_mL"]
            if comp.get("source_vial"):
                self.volumes[comp["source_vial"]] -= comp["volume_mL"]
        self.calls.append(("make", vial))


class _FakeLashE:
    def __init__(self, robot):
        import logging
        self.nr_robot, self.logger, self.simulate = robot, logging.getLogger("test"), True


def test_ensure_volume_refills_stock_from_its_group_and_tops_up_substocks():
    c = config()
    robot = _FakeRobot({"SDS_stock": 2.5, "SDS_refill": 15.0, "SDS_sub_1": 2.2, "water": 7.5, "pyrene": 3.0},
                       {"SDS_stock": "8_mL", "SDS_refill": "20_mL", "SDS_sub_1": "8_mL", "water": "8_mL"})
    recipe = pd.DataFrame([dict(surfactant="SDS", vial="SDS_sub_1", conc_mM=5.0, error=0.01, parent="SDS_stock",
                                parent_ul=400.0, water_ul=3600.0, fills=2, used_ml=3.0)]).iloc[0]
    sources = {"SDS": [dict(name="SDS_stock", vials=["SDS_stock", "SDS_refill"], conc=50.0, error=0.0,
                            available_ml=17.5)], "CTAB": []}
    plan = dict(substocks=pd.DataFrame([recipe]))
    table = w.refill_table(c, plan, sources)
    lash_e = _FakeLashE(robot)
    w.ensure_volume(lash_e, c, "SDS_sub_1", 1.0, table)       # 2.2 - 1.0 < 2.0 dead -> top up by recipe
    assert ("make", "SDS_sub_1") in robot.calls and robot.volumes["SDS_sub_1"] == pytest.approx(6.2)
    w.ensure_volume(lash_e, c, "SDS_stock", 1.5, table)       # 2.1 - 1.5 < 2.0 -> refill from SDS_refill
    assert robot.calls[-1][0] == "refill" and robot.calls[-1][3] == "SDS_refill"
    assert robot.volumes["SDS_stock"] == pytest.approx(7.5)
    with pytest.raises(RuntimeError, match="never refilled"):
        w.ensure_volume(lash_e, c, "pyrene", 4.0, {})
