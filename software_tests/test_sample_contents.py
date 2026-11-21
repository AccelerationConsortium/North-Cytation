"""Stored contents travel with the sample: the worked example, uncertainty by run, the robot.

The worked example is the one in Integration_Plan/SAMPLE_CONTENTS_DESIGN.md: 50 µL of 10 mM SDS
(small tip) and 200 µL of water (large tip) into one well gives 2.00 ± 0.045 mM, ± 0.011 mM within
the run, the same as the log-based calculation already checked against real validation data.
"""
import logging
import math
from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiment_tracking import experiment_run_logger, sample_contents as sc, sample_tracking as st
import master_usdl_coordinator
from scheduler import simulation_state as state

ROOT = Path(__file__).resolve().parents[1]
STOCKS = [{"stock_id": "SDS-001", "material": "SDS", "concentration": "10", "units": "mM"},
          {"stock_id": "WATER-001", "material": "water", "concentration": "", "units": ""}]
SDS = "SDS|mM"
# (random, shared) errors in mL, from the screened validation values in the design notes
SMALL_50 = (0.26e-3, math.sqrt(0.35 ** 2 + 0.42 ** 2 + 0.45 ** 2) * 1e-3)
LARGE_200 = (0.86e-3, math.sqrt(2.11 ** 2 + 3.46 ** 2 + 2.46 ** 2) * 1e-3)


def aspirate(tip, source, volume, run="r1"):
    portion, left = sc.take(source, volume)
    return sc.add(tip, portion, volume, (0.0, 0.0), run), left


def dispense(tip, destination, volume, error, run="r1"):
    portion, tip = sc.take(tip, volume)
    return tip, sc.add(destination, portion, volume, error, run)


def worked_example():
    sds_vial = sc.stock_record("SDS-001", STOCKS, 5.0)
    water_vial = sc.stock_record("WATER-001", STOCKS, 6.0)
    tip, sds_vial = aspirate(sc.empty_record(), sds_vial, 0.30)              # step 1
    tip, a1 = dispense(tip, sc.empty_record(), 0.05, SMALL_50)                # step 2
    tip, a2 = dispense(tip, sc.empty_record(), 0.05, SMALL_50)                # step 3
    tip = sc.empty_record()                                                   # step 4: tip dropped
    tip, water_vial = aspirate(tip, water_vial, 0.40)                         # step 5
    tip, a1 = dispense(tip, a1, 0.20, LARGE_200)                              # step 6
    return sds_vial, a1, a2


def test_worked_example_step_by_step():
    sds_vial, a1, a2 = worked_example()
    assert sds_vial["volume_mL"] == pytest.approx(4.70) and sds_vial["materials"][SDS] == pytest.approx(10.0)
    assert a2["volume_mL"] == pytest.approx(0.05) and a2["materials"][SDS] == pytest.approx(10.0)
    assert a1["volume_mL"] == pytest.approx(0.25)
    assert a1["materials"][SDS] == pytest.approx(2.0)
    assert sc.sd_absolute(a1, SDS) == pytest.approx(0.0455, abs=5e-4)
    assert sc.sd_within_run(a1, SDS, "r1") == pytest.approx(0.0108, abs=2e-4)
    assert a1["stocks"] == ["SDS-001", "WATER-001"]


def test_stored_numbers_equal_the_log_based_numbers():
    class Errors:
        def lookup(self, liquid, tip, volume_ul):
            return {("SDS", "small_tip", 50.0): tuple(e * 1000 for e in SMALL_50),
                    ("water", "large_tip", 200.0): tuple(e * 1000 for e in LARGE_200)}.get((liquid, tip, round(volume_ul, 3)))
    log = [{"source": "vial-sds", "destination": "w", "amount": "0.05", "units": "mL", "method": "small_tip", "liquid": "SDS"},
           {"source": "vial-water", "destination": "w", "amount": "0.2", "units": "mL", "method": "large_tip", "liquid": "water"}]
    within, absolute = st.concentration_sd("w", log, STOCKS, {"vial-sds": "SDS-001", "vial-water": "WATER-001"},
                                           Errors())[("SDS", "mM")]
    _, a1, _ = worked_example()
    assert sc.sd_absolute(a1, SDS) == pytest.approx(absolute, rel=1e-9)
    assert sc.sd_within_run(a1, SDS, "r1") == pytest.approx(within, rel=1e-9)


def test_a_dilution_carries_its_own_error_into_the_wells():
    err = (1e-3, 5e-3)
    tip, _ = aspirate(sc.empty_record(), sc.stock_record("SDS-001", STOCKS, 5.0), 1.0, "yesterday")
    tip, mix = dispense(tip, sc.empty_record(), 1.0, err, "yesterday")
    tip, _ = aspirate(sc.empty_record(), sc.stock_record("WATER-001", STOCKS, 5.0), 1.0, "yesterday")
    tip, mix = dispense(tip, mix, 1.0, err, "yesterday")
    assert mix["materials"][SDS] == pytest.approx(5.0)
    tip, _ = aspirate(sc.empty_record(), mix, 0.1, "today")
    tip, well = dispense(tip, sc.empty_record(), 0.1, (0.3e-3, 1e-3), "today")
    tip, _ = aspirate(sc.empty_record(), sc.stock_record("WATER-001", STOCKS, 5.0), 0.1, "today")
    tip, well = dispense(tip, well, 0.1, (0.3e-3, 1e-3), "today")
    parts = sc.variance_by_run(well, SDS)
    assert set(parts) == {"yesterday", "today"}
    assert math.sqrt(parts["yesterday"]) == pytest.approx(0.5 * sc.sd_absolute(mix, SDS))  # carried, halved
    # Within today's run only today's random error counts; yesterday's dilution is common to all.
    direct = math.sqrt(2 * ((5.0 - 2.5) / 0.2 * 0.3e-3) ** 2)
    assert sc.sd_within_run(well, SDS, "today") == pytest.approx(direct)
    assert sc.sd_absolute(well, SDS) > sc.sd_within_run(well, SDS, "today")


def test_a_volume_correction_changes_only_the_amount():
    _, a1, _ = worked_example()
    before = (a1["materials"][SDS], sc.sd_absolute(a1, SDS))
    sc.rescale(a1, 0.20)
    assert a1["volume_mL"] == pytest.approx(0.20)
    assert (a1["materials"][SDS], sc.sd_absolute(a1, SDS)) == pytest.approx(before)


def test_the_same_liquid_changes_only_the_volume():
    # Conditioning: 5 cycles of 0.2 mL out of a mixed vial and back in leave its record as it was
    _, a1, _ = worked_example()
    vial = sc.rescale(a1, 2.0)
    before = (vial["materials"], sc.variance_by_run(vial, SDS), vial["stocks"])
    for _ in range(5):
        tip, vial = aspirate(sc.empty_record(), vial, 0.2, "today")
        tip, vial = dispense(tip, vial, 0.2, LARGE_200, "today")
    assert vial["volume_mL"] == pytest.approx(2.0)
    assert (vial["materials"], vial["stocks"]) == (before[0], before[2])
    assert sc.variance_by_run(vial, SDS) == pytest.approx(before[1])
    # A refill from the same stock changes only the volume too; a different liquid still mixes
    stock = sc.stock_record("SDS-001", STOCKS, 2.0)
    tip, _ = aspirate(sc.empty_record(), sc.stock_record("SDS-001", STOCKS, 8.0), 1.0)
    _, stock = dispense(tip, stock, 1.0, LARGE_200)
    assert stock["volume_mL"] == pytest.approx(3.0) and sc.sd_absolute(stock, SDS) == pytest.approx(0.0)
    tip, _ = aspirate(sc.empty_record(), sc.stock_record("WATER-001", STOCKS, 8.0), 1.0)
    _, stock = dispense(tip, stock, 1.0, LARGE_200)
    assert stock["materials"][SDS] == pytest.approx(7.5) and sc.sd_absolute(stock, SDS) > 0


def test_no_estimate_means_no_value():
    tip, _ = aspirate(sc.empty_record(), sc.stock_record("SDS-001", STOCKS, 5.0), 0.1)
    tip, well = dispense(tip, sc.empty_record(), 0.1, None)  # e.g. an unvalidated volume
    assert well["materials"][SDS] == pytest.approx(10.0)
    assert sc.sd_absolute(well, SDS) is None and sc.sd_within_run(well, SDS, "r1") is None


def test_unknown_liquid_makes_the_record_partly_unknown():
    tip, _ = aspirate(sc.empty_record(), sc.unknown_record(3.0), 0.1)
    tip, well = dispense(tip, sc.empty_record(), 0.1, (1e-4, 1e-4))
    assert not well["complete"] and "partly unknown" in sc.describe(well)


def test_powder_waits_for_liquid():
    vial = sc.add_solid(sc.empty_record(), ["TTAB"], 12.5, "TTAB-004")
    tip, _ = aspirate(sc.empty_record(), sc.stock_record("WATER-001", STOCKS, 6.0), 5.0)
    tip, vial = dispense(tip, vial, 5.0, (1e-3, 1e-3))
    assert vial["materials"]["TTAB|mg/mL"] == pytest.approx(2.5) and "TTAB-004" in vial["stocks"]


def test_the_store_round_trips_and_is_readable(tmp_path):
    _, a1, _ = worked_example()
    path = tmp_path / "container_contents.csv"
    store = sc.ContentsStore(path)
    store.put("plate-p:0", a1)
    again = sc.ContentsStore(path).get("plate-p:0")
    assert again["materials"] == pytest.approx(a1["materials"])
    assert sc.sd_absolute(again, SDS) == pytest.approx(sc.sd_absolute(a1, SDS))
    rows = pd.read_csv(path)
    assert rows.loc[0, ["material", "units"]].tolist() == ["SDS", "mM"] and rows.loc[0, "concentration"] == pytest.approx(2.0)


def test_plain_simulation_writes_no_contents(tmp_path):
    class Robot:
        simulate = True
    assert sc.contents_path(Robot()) is None


# --- the robot, in a scheduler simulation sandbox -------------------------------------------------

@pytest.fixture
def session(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path / "private")
    stocks = tmp_path / "stocks.csv"
    sds = st.add_stock([{"material": "SDS", "concentration": 10, "units": "mM", "solvent": "water"}], stocks)
    monkeypatch.setattr(st, "STOCKS_FILE", stocks)
    real_start = experiment_run_logger.start_run

    def private_start(*args, **kwargs):
        tracker = real_start(*args, **kwargs)
        tracker.csv_path = str(tmp_path / "experiment_runs.csv")
        return tracker
    monkeypatch.setattr(experiment_run_logger, "start_run", private_start)
    vials = tmp_path / "vials.csv"
    base = dict(location="main_8mL_rack", capped=True, cap_type="open", vial_type="8_mL", home_location="main_8mL_rack")
    pd.DataFrame([
        {**base, "vial_index": 0, "vial_name": "stock", "location_index": 1, "home_location_index": 1,
         "vial_volume": 5.0, "container_id": "vial-aaaaaaaaaaaa", "stock_id": sds},
        {**base, "vial_index": 1, "vial_name": "mix", "location_index": 2, "home_location_index": 2,
         "vial_volume": 0.0, "container_id": "vial-bbbbbbbbbbbb", "stock_id": ""},
    ]).to_csv(vials, index=False)
    root = state.create_session([str(vials)])
    return root, next((root / "vials").glob("*.csv")), tmp_path


def test_robot_transfer_stores_the_mix_and_keeps_the_stock_vial_a_stock(session):
    root, vials, tmp_path = session
    real = ROOT / "status" / "container_contents.csv"
    real_before = real.read_bytes() if real.exists() else None
    with state.workflow_state(root, "contents"):
        lash = master_usdl_coordinator.Lash_E(vial_file=str(vials), simulate=True, initialize_biotek=False,
                                              initialize_track=False, show_gui=False,
                                              logging_folder=str(tmp_path / "logs"))
        lash.nr_robot.dispense_from_vial_into_vial("stock", "mix", 0.5, liquid="water")
    store = sc.ContentsStore(root / "container_contents.csv")
    mix = store.get("vial-bbbbbbbbbbbb")
    assert mix is not None and mix["volume_mL"] == pytest.approx(0.5)
    assert mix["materials"][SDS] == pytest.approx(10.0)
    assert mix["stocks"] == ["SDS-001"]
    assert store.get("vial-aaaaaaaaaaaa") is None  # still just its stock: no record needed
    assert (real.read_bytes() if real.exists() else None) == real_before  # real file untouched


def test_simulated_mix_vial_keeps_one_tip_and_the_volume(session, caplog):
    root, vials, tmp_path = session
    with state.workflow_state(root, "mix"):
        lash = master_usdl_coordinator.Lash_E(vial_file=str(vials), simulate=True, initialize_biotek=False,
                                              initialize_track=False, show_gui=False,
                                              logging_folder=str(tmp_path / "logs"))
        robot = lash.nr_robot
        with caplog.at_level(logging.WARNING):
            robot.mix_vial("stock", 0.3, repeats=3, liquid="water")
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert robot.HELD_PIPET_TYPE is not None  # same tip, still attached
        assert robot.get_vial_info("stock", "vial_volume") == pytest.approx(5.0)
        assert robot.PIPET_FLUID_VOLUME == pytest.approx(0.0)
        robot.remove_pipet()


def test_the_tip_carries_its_contents_through_robot_status(session):
    root, vials, tmp_path = session
    with state.workflow_state(root, "tip"):
        lash = master_usdl_coordinator.Lash_E(vial_file=str(vials), simulate=True, initialize_biotek=False,
                                              initialize_track=False, show_gui=False,
                                              logging_folder=str(tmp_path / "logs"))
        robot = lash.nr_robot
        robot.aspirate_from_vial("stock", 0.3, liquid="water")
        saved = yaml.safe_load(Path(robot.ROBOT_STATUS_FILE).read_text())["pipet_fluid_contents"]
        assert saved["parts"] == [["vial-aaaaaaaaaaaa", pytest.approx(0.3)]]
        record = sc.from_json(saved["record"])
        assert record["materials"][SDS] == pytest.approx(10.0) and record["volume_mL"] == pytest.approx(0.3)
        # A restart reloads it: a fresh tip state restored from the saved one is the same.
        robot._clear_tip_tracking()
        robot._restore_tip_state(saved)
        assert robot._tip_contents().parts == [["vial-aaaaaaaaaaaa", pytest.approx(0.3)]]
        assert robot._tip_record()["materials"][SDS] == pytest.approx(10.0)
        robot.remove_pipet()
        assert yaml.safe_load(Path(robot.ROBOT_STATUS_FILE).read_text())["pipet_fluid_contents"] is None


def test_snapshots_and_well_contents_come_from_the_stored_records(monkeypatch, tmp_path):
    monkeypatch.setattr(st, "WELL_CONTENTS_DIR", tmp_path / "sample_tracking")
    _, a1, _ = worked_example()
    contents = tmp_path / "container_contents.csv"
    sc.ContentsStore(contents).put("plate-p:0", a1)
    log = tmp_path / "transfers.csv"

    class Robot:
        simulate = False
        run_id = "r1"
    st.record_transfer(Robot(), "vial-sds", "plate-p:0", 0.05, "mL", "small_tip", "SDS", path=log)
    vials = tmp_path / "vials.csv"
    vials.write_text("vial_index,vial_name,location,location_index,vial_volume\n0,a,rack,0,1\n")
    st.snapshot("r1", "start", vials, contents_file=contents)
    st.snapshot("r1", "end", vials, contents_file=contents)
    path = st.write_well_contents("r1", transfers_file=log, contents_file=contents)
    folder = tmp_path / "sample_tracking" / "r1"
    assert sorted(p.name for p in folder.iterdir()) == ["end_contents.csv", "end_vials.csv", "start_contents.csv",
                                                       "start_vials.csv", "well_contents.csv"]
    row = pd.read_csv(path).iloc[0]
    assert row.concentration == pytest.approx(2.0)
    assert row.sd_absolute == pytest.approx(0.0455, abs=5e-4) and row.sd_within_run == pytest.approx(0.0108, abs=2e-4)
