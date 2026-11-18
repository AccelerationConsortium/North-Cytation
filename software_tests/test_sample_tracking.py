"""Sample tracking: stocks, the tip's contents, the transfer log and worked-out contents."""
from pathlib import Path

import pandas as pd
import pytest

from experiment_tracking import experiment_run_logger, sample_tracking as st
import master_usdl_coordinator
from scheduler import simulation_state as state

ROOT = Path(__file__).resolve().parents[1]


class Controller:
    def __init__(self, simulate=False, root=None, run_id="run1"):
        self.simulate = simulate
        self.run_id = run_id
        if root is not None:
            self._scheduler_state_root = str(root)


# --- stocks ---------------------------------------------------------------------------

def test_stock_ids_count_up_per_material_and_are_never_reused(tmp_path):
    path = tmp_path / "stocks.csv"
    assert st.add_stock([{"material": "TTAB", "concentration": 50, "units": "mM", "solvent": "water"}], path) == "TTAB-001"
    assert st.add_stock([{"material": "TTAB", "concentration": 25, "units": "mM"}], path) == "TTAB-002"
    assert st.add_stock([{"material": "HCl", "concentration": 6, "units": "M"}], path) == "HCL-001"
    buffer = st.add_stock([{"material": "Tris", "concentration": 10, "units": "mM"},
                           {"material": "NaCl", "concentration": 150, "units": "mM"}], path)
    rows = st.load_stocks(path)
    assert [r["stock_id"] for r in rows].count(buffer) == 2  # one row per component


def test_stock_labels_and_changing_a_vials_stock_makes_a_new_vial(tmp_path):
    path = tmp_path / "stocks.csv"
    ttab = st.add_stock([{"material": "TTAB", "concentration": 50, "units": "mM", "solvent": "water", "lot": "S123"}], path)
    labels = dict(st.stock_choices(st.load_stocks(path)))
    assert labels[ttab] == "TTAB-001: TTAB 50 mM in water (lot S123)"
    vial = {"vial_name": "a", "container_id": "vial-old000000000", "stock_id": ""}
    changed = st.vial_after_stock_choice(vial, ttab)
    assert changed["stock_id"] == ttab and changed["container_id"] != "vial-old000000000"
    same = st.vial_after_stock_choice(changed, ttab)
    assert same["container_id"] == changed["container_id"]


def test_only_notes_change_and_usage_is_summarised(tmp_path):
    stocks, sources = tmp_path / "stocks.csv", tmp_path / "sources.csv"
    sds = st.add_stock([{"material": "SDS", "concentration": 10, "units": "mM", "notes": "old"}], stocks)
    st.update_stock_notes(sds, "opened 2026-11-18", stocks)
    row = st.load_stocks(stocks)[0]
    assert row["notes"] == "opened 2026-11-18" and row["concentration"] == "10"
    with pytest.raises(ValueError):
        st.update_stock_notes("NOPE-001", "x", stocks)
    st.record_load("reservoir-1", sds, 500, "mL", sources)
    write_vials(tmp_path / "vials_a.csv", [dict(vial_name="a", vial_volume=1, stock_id=sds),
                                           dict(vial_name="b", vial_volume=1, stock_id=sds)])
    usage = st.stock_usage(tmp_path, sources)
    assert usage[sds][0] == "Reservoir 1" and len(usage[sds]) == 3
    assert st.usage_summary(usage[sds]) == "Reservoir 1; 2 vials"
    assert st.usage_summary([]) == "—"


def test_each_reservoir_load_is_its_own_container(tmp_path, monkeypatch):
    sources, log = tmp_path / "sources.csv", tmp_path / "transfers.csv"
    monkeypatch.setattr(st, "SOURCES_FILE", sources)
    assert st.reservoir_id(1) == "reservoir-1"  # nothing loaded yet
    first = st.record_load("reservoir-1", "WATER-001", 500, "mL")
    assert st.reservoir_id(1) == first
    st.record_transfer(Controller(), first, "vial-x", 120, "mL", "reservoir", path=log)
    assert st.amount_left("reservoir-1", transfers_file=log) == (pytest.approx(380), "mL")
    second = st.record_load("reservoir-1", "WATER-002", 500, "mL")
    assert second != first and st.reservoir_id(1) == second
    assert st.amount_left("reservoir-1", transfers_file=log) == (pytest.approx(500), "mL")
    assert st.source_stocks() == {first: "WATER-001", second: "WATER-002"}


def test_reservoir_status_and_low_warning(tmp_path):
    pumps = tmp_path / "pumps.yaml"
    pumps.write_text("0: {description: Pipetting pump, carousel_angle: null}\n"
                     "1: {description: Water reservoir pump, liquid: water, carousel_angle: 45,"
                     " capacity_mL: 1500, low_warning_mL: 500}\n")
    sources, log = tmp_path / "sources.csv", tmp_path / "transfers.csv"
    rows = st.reservoir_status(pumps, sources, log)
    assert [r["index"] for r in rows] == [1] and rows[0]["load"] is None and not rows[0]["low"]
    load = st.record_load("reservoir-1", "WATER-001", 1500, "mL", sources)
    st.record_transfer(Controller(), load, "vial-x", 900, "mL", "reservoir", path=log)
    assert st.reservoir_warnings(pumps, sources, log) == []  # 600 mL left
    st.record_transfer(Controller(), load, "vial-y", 200, "mL", "reservoir", path=log)
    rows = st.reservoir_status(pumps, sources, log)
    assert rows[0]["left_mL"] == pytest.approx(400) and rows[0]["low"]
    assert "about 400 mL left" in st.reservoir_warnings(pumps, sources, log)[0]


def test_real_pump_config_lists_the_water_reservoir():
    rows = st.reservoir_status(sources_path=Path("does-not-exist.csv"), transfers_file=Path("none.csv"))
    assert [(r["index"], r["liquid"], r["capacity_mL"], r["low_warning_mL"]) for r in rows] == [(1, "water", 1500, 500)]


def test_vial_dialog_text(tmp_path):
    stocks, log = tmp_path / "stocks.csv", tmp_path / "transfers.csv"
    sds = st.add_stock([{"material": "SDS", "concentration": 10, "units": "mM"}], stocks)
    st.record_transfer(Controller(), "vial-sds000000000", "vial-mix000000000", 1.0, "mL", "large_tip", path=log)
    st.record_transfer(Controller(), "vial-unknown0000", "vial-mix000000000", 1.0, "mL", "large_tip", path=log)
    assert st.describe_contents("vial-x", sds, stocks_file=stocks).startswith("SDS-001: SDS 10 mM")
    assert st.describe_contents("", None, stocks_file=stocks).startswith("No ID yet")
    assert st.describe_contents("vial-none", None, log, stocks, tmp_path / "s.csv") == "Nothing recorded"


# --- the tip ---------------------------------------------------------------------------

def test_tip_splits_a_dispense_between_its_sources():
    tip = st.TipContents()
    tip.aspirate("vial-a", 0.3)
    tip.aspirate("vial-b", 0.1)
    out = dict(tip.dispense(0.2))
    assert out["vial-a"] == pytest.approx(0.15)
    assert out["vial-b"] == pytest.approx(0.05)
    assert tip.total() == pytest.approx(0.2)


def test_tip_never_gives_out_more_than_it_holds():
    tip = st.TipContents()
    tip.aspirate("vial-a", 0.1)
    assert dict(tip.dispense(0.5))["vial-a"] == pytest.approx(0.1)
    assert tip.dispense(0.1) == []


# --- the transfer log -------------------------------------------------------------------

def test_only_live_runs_and_scheduler_sandboxes_write(tmp_path):
    assert st.transfers_path(Controller(simulate=True)) is None
    assert st.transfers_path(Controller(simulate=False)) == st.TRANSFERS_FILE
    assert st.transfers_path(Controller(simulate=True, root=tmp_path)) == tmp_path / "transfers.csv"


def test_rows_are_appended_and_self_transfers_skipped(tmp_path):
    path = tmp_path / "transfers.csv"
    robot = Controller()
    assert st.record_transfer(robot, "vial-a", "vial-b", 0.25, "mL", "large_tip", "water", path=path)
    assert not st.record_transfer(robot, "vial-a", "vial-a", 0.25, "mL", "large_tip", path=path)  # mixing
    assert not st.record_transfer(robot, None, "vial-b", 0.25, "mL", "large_tip", path=path)      # unknown source
    rows = st.load_transfers(path)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run1" and rows[0]["source"] == "vial-a" and rows[0]["method"] == "large_tip"


def test_a_tracking_failure_never_raises(tmp_path):
    blocked = tmp_path / "a_file"
    blocked.write_text("x")
    assert st.record_transfer(Controller(), "vial-a", "vial-b", 0.1, "mL", "small_tip",
                              path=blocked / "transfers.csv") is False


# --- worked-out contents -------------------------------------------------------------------

def test_contents_follow_dilutions_through_the_log():
    stocks = [{"stock_id": "SDS-001", "material": "SDS", "concentration": "10", "units": "mM"},
              {"stock_id": "WATER-001", "material": "water", "concentration": "", "units": ""}]
    holds = {"vial-sds": "SDS-001", "reservoir-1": "WATER-001"}
    log = [
        {"source": "vial-sds", "destination": "vial-mix", "amount": "1.0", "units": "mL"},
        {"source": "reservoir-1", "destination": "vial-mix", "amount": "3.0", "units": "mL"},
        {"source": "vial-mix", "destination": "plate-x:0", "amount": "0.05", "units": "mL"},
        {"source": "reservoir-1", "destination": "plate-x:0", "amount": "0.05", "units": "mL"},
    ]
    mix = st.contents("vial-mix", log, stocks, holds)
    assert mix["materials"][("SDS", "mM")] == pytest.approx(2.5)
    well = st.contents("plate-x:0", log, stocks, holds)
    assert well["materials"][("SDS", "mM")] == pytest.approx(1.25)
    assert st.contents("vial-sds", log, stocks, holds)["stock_id"] == "SDS-001"
    assert st.contents("vial-unknown", log, stocks, holds) is None


def test_powder_is_reported_per_mL_of_liquid():
    stocks = [{"stock_id": "TTAB-004", "material": "TTAB", "concentration": "", "units": ""}]
    holds = {"powder-0": "TTAB-004", "reservoir-1": None}
    log = [{"source": "powder-0", "destination": "vial-m", "amount": "12.5", "units": "mg"},
           {"source": "reservoir-1", "destination": "vial-m", "amount": "5.0", "units": "mL"}]
    assert st.contents("vial-m", log, stocks, holds)["materials"][("TTAB", "mg/mL")] == pytest.approx(2.5)


# --- end-of-run well contents -------------------------------------------------------------------

def test_well_contents_lists_what_this_run_filled(monkeypatch, tmp_path):
    monkeypatch.setattr(st, "WELL_CONTENTS_DIR", tmp_path / "well_contents")
    stocks = tmp_path / "stocks.csv"
    sds = st.add_stock([{"material": "SDS", "concentration": 10, "units": "mM"}], stocks)
    vials = tmp_path / "vials.csv"
    write_vials(vials, [dict(vial_name="sds", vial_volume=5, container_id="vial-sds000000000", stock_id=sds),
                        dict(vial_name="old", vial_volume=1, container_id="vial-old000000000", stock_id="")])
    log = tmp_path / "transfers.csv"
    st.record_transfer(Controller(run_id="earlier"), "vial-sds000000000", "vial-old000000000", 1.0, "mL", "large_tip", path=log)
    st.record_transfer(Controller(run_id="r2"), "vial-sds000000000", "plate-p:0", 0.05, "mL", "small_tip", path=log)
    st.record_transfer(Controller(run_id="r2"), "reservoir-1", "plate-p:0", 0.2, "mL", "reservoir", path=log)
    out = tmp_path / "run_output"
    path = st.write_well_contents("r2", vials, out, transfers_file=log, stocks_file=stocks)
    rows = pd.read_csv(path)
    assert rows.container.tolist() == ["plate-p:0"]  # only what run r2 filled
    assert rows.concentration.iloc[0] == pytest.approx(2.0)
    assert rows.units.iloc[0] == "mM" and rows.from_stocks.iloc[0] == sds
    assert rows.total_mL.iloc[0] == pytest.approx(0.25)
    assert (out / "sample_tracking" / "well_contents.csv").exists()  # copied next to the run's own data


class FixedErrors:
    """Stand-in estimator: (random_ul, shared_ul) per (liquid, tip, volume_ul)."""
    def __init__(self, table):
        self.table = table

    def lookup(self, liquid, tip, volume_ul):
        return self.table.get((liquid, tip, round(volume_ul, 3)))


EXAMPLE_STOCKS = [{"stock_id": "SDS-001", "material": "SDS", "concentration": "10", "units": "mM"}]
EXAMPLE_LOG = [{"source": "vial-sds", "destination": "plate-p:0", "amount": "0.05", "units": "mL",
                "method": "small_tip", "liquid": "SDS"},
               {"source": "vial-water", "destination": "plate-p:0", "amount": "0.2", "units": "mL",
                "method": "large_tip", "liquid": "water"}]
EXAMPLE_HOLDS = {"vial-sds": "SDS-001", "vial-water": "WATER-001"}


def test_uncertainty_matches_the_worked_example():
    # 50 uL of 10 mM SDS (small tip) + 200 uL water (large tip), with the screened values
    errors = FixedErrors({("SDS", "small_tip", 50.0): (0.26, (0.35**2 + 0.42**2 + 0.45**2) ** 0.5),
                          ("water", "large_tip", 200.0): (0.86, (2.11**2 + 3.46**2 + 2.46**2) ** 0.5)})
    within, absolute = st.concentration_sd("plate-p:0", EXAMPLE_LOG, EXAMPLE_STOCKS, EXAMPLE_HOLDS, errors)[("SDS", "mM")]
    assert within == pytest.approx(0.0108, abs=2e-4)
    assert absolute == pytest.approx(0.0455, abs=5e-4)


def test_no_uncertainty_when_any_dispense_cannot_be_estimated():
    errors = FixedErrors({("SDS", "small_tip", 50.0): (0.26, 0.7)})  # water dispense unknown
    assert st.concentration_sd("plate-p:0", EXAMPLE_LOG, EXAMPLE_STOCKS, EXAMPLE_HOLDS, errors)[("SDS", "mM")] == (None, None)


def test_a_robot_made_source_passes_on_its_own_error():
    log = [{"source": "vial-sds", "destination": "vial-mix", "amount": "1.0", "units": "mL", "method": "large_tip", "liquid": "SDS"},
           {"source": "vial-water", "destination": "vial-mix", "amount": "1.0", "units": "mL", "method": "large_tip", "liquid": "water"},
           {"source": "vial-mix", "destination": "plate-p:1", "amount": "0.1", "units": "mL", "method": "small_tip", "liquid": "SDS"},
           {"source": "vial-water", "destination": "plate-p:1", "amount": "0.1", "units": "mL", "method": "small_tip", "liquid": "water"}]
    errors = FixedErrors({("SDS", "large_tip", 1000.0): (1.0, 5.0), ("water", "large_tip", 1000.0): (1.0, 5.0),
                          ("SDS", "small_tip", 100.0): (0.3, 1.0), ("water", "small_tip", 100.0): (0.3, 1.0)})
    mix = st.concentration_sd("vial-mix", log, EXAMPLE_STOCKS, EXAMPLE_HOLDS, errors)[("SDS", "mM")]
    well = st.concentration_sd("plate-p:1", log, EXAMPLE_STOCKS, EXAMPLE_HOLDS, errors)[("SDS", "mM")]
    direct_only = (2 * ((5.0 - 2.5) / 0.2 * 0.001044) ** 2) ** 0.5  # this well's own two dispenses
    assert well[1] > direct_only            # the mix vial's error is carried into the well
    assert well[1] == pytest.approx((direct_only**2 + (0.5 * mix[1]) ** 2) ** 0.5, rel=1e-3)


def test_real_validation_data_gives_estimates_only_where_validated():
    from pipetting_data.dispense_uncertainty import DispenseUncertainty
    errors = DispenseUncertainty()
    small20 = errors.lookup("water", "small_tip", 20.0)
    small50 = errors.lookup("water", "small_tip", 50.0)
    assert small20 is not None and small50 is not None
    between = errors.lookup("water", "small_tip", 35.0)
    assert min(small20[1], small50[1]) <= between[1] <= max(small20[1], small50[1])
    assert errors.lookup("ethanol", "small_tip", 100.0) is None  # no sessions with its current settings
    assert errors.lookup("water", "reservoir", 500.0) is None
    assert errors.lookup("water", "small_tip", 1.0) is None      # below the calibrated range


def test_no_well_contents_for_a_run_that_moved_nothing(tmp_path):
    assert st.write_well_contents("nothing", transfers_file=tmp_path / "none.csv") is None


# --- container IDs in vial files ----------------------------------------------------------------

def write_vials(path, rows):
    base = dict(location="main_8mL_rack", capped=True, cap_type="open", vial_type="8_mL",
                home_location="main_8mL_rack")
    pd.DataFrame([{**base, "vial_index": i, "location_index": i, "home_location_index": i, **row}
                  for i, row in enumerate(rows)]).to_csv(path, index=False)


def test_old_vial_files_gain_ids_and_keep_their_data(tmp_path):
    vials = tmp_path / "vials.csv"
    write_vials(vials, [dict(vial_name="a", vial_volume=2.0), dict(vial_name="b", vial_volume=0.0)])
    before = pd.read_csv(vials)
    changes = st.prepare_vial_file(vials)
    after = pd.read_csv(vials)
    assert len(changes) == 4  # two columns added, two IDs filled
    assert after.container_id.str.match(r"^vial-[0-9a-f]{12}$").all()
    assert after.container_id.nunique() == 2
    pd.testing.assert_frame_equal(after[before.columns], before)
    assert st.prepare_vial_file(vials) == []  # nothing left to do; file not rewritten


def test_copied_rows_get_their_own_id(tmp_path):
    vials = tmp_path / "vials.csv"
    write_vials(vials, [dict(vial_name="a", vial_volume=1, container_id="vial-111111111111"),
                        dict(vial_name="b", vial_volume=1, container_id="vial-111111111111")])
    st.prepare_vial_file(vials)
    ids = pd.read_csv(vials).container_id.tolist()
    assert ids[0] == "vial-111111111111" and ids[1] != ids[0]


def test_live_start_retires_only_empty_vials_with_history(tmp_path):
    log = tmp_path / "transfers.csv"
    for destination in ("vial-aaaaaaaaaaaa", "vial-bbbbbbbbbbbb"):
        st.record_transfer(Controller(), "vial-stock00000", destination, 1.0, "mL", "large_tip", path=log)
    vials = tmp_path / "vials.csv"
    write_vials(vials, [
        dict(vial_name="replaced", vial_volume=0.0, container_id="vial-aaaaaaaaaaaa"),  # emptied by a person
        dict(vial_name="kept", vial_volume=1.0, container_id="vial-bbbbbbbbbbbb"),      # left in place
        dict(vial_name="new_empty", vial_volume=0.0, container_id="vial-cccccccccccc"),  # no history
    ])
    st.prepare_vial_file(vials, retire_emptied=True, transfers_file=log)
    ids = dict(zip(*[pd.read_csv(vials)[c] for c in ("vial_name", "container_id")]))
    assert ids["replaced"] != "vial-aaaaaaaaaaaa"
    assert ids["kept"] == "vial-bbbbbbbbbbbb"
    assert ids["new_empty"] == "vial-cccccccccccc"
    assert len(st.load_transfers(log)) == 2  # history is kept


def test_a_bad_vial_file_is_left_alone(tmp_path):
    vials = tmp_path / "vials.csv"
    vials.write_text("not,a,vial,file\n1,2,3,4\n")
    assert st.prepare_vial_file(vials) == []
    assert vials.read_text() == "not,a,vial,file\n1,2,3,4\n"


def test_simulation_sandbox_copy_gets_ids_but_real_file_does_not(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path / "private")
    vials = tmp_path / "real.csv"
    write_vials(vials, [dict(vial_name="a", vial_volume=1.0)])
    original = vials.read_bytes()
    root = state.create_session([str(vials)])
    assert vials.read_bytes() == original
    copy = pd.read_csv(next((root / "vials").glob("*.csv")))
    assert copy.container_id.str.startswith("vial-").all()


# --- a real simulated transfer, in a scheduler simulation sandbox -------------------------------

@pytest.fixture
def session(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path / "private")
    real_start = experiment_run_logger.start_run

    def private_start(*args, **kwargs):
        tracker = real_start(*args, **kwargs)
        tracker.csv_path = str(tmp_path / "experiment_runs.csv")
        return tracker
    monkeypatch.setattr(experiment_run_logger, "start_run", private_start)
    vials = tmp_path / "vials.csv"
    pd.DataFrame([
        dict(vial_index=0, vial_name="stock", location="main_8mL_rack", location_index=1, vial_volume=5.0,
             capped=True, cap_type="open", vial_type="8_mL", home_location="main_8mL_rack", home_location_index=1,
             container_id="vial-aaaaaaaaaaaa"),
        dict(vial_index=1, vial_name="mix", location="main_8mL_rack", location_index=2, vial_volume=0.0,
             capped=True, cap_type="open", vial_type="8_mL", home_location="main_8mL_rack", home_location_index=2,
             container_id="vial-bbbbbbbbbbbb"),
    ]).to_csv(vials, index=False)
    root = state.create_session([str(vials)])
    return root, next((root / "vials").glob("*.csv")), tmp_path


def test_simulated_transfer_writes_one_row_to_the_sandbox_log(session):
    root, vials, tmp_path = session
    with state.workflow_state(root, "tracking"):
        lash = master_usdl_coordinator.Lash_E(vial_file=str(vials), simulate=True, initialize_biotek=False,
                                              initialize_track=False, show_gui=False,
                                              logging_folder=str(tmp_path / "logs"))
        lash.nr_robot.dispense_from_vial_into_vial("stock", "mix", 0.5, liquid="water")
    rows = st.load_transfers(root / "transfers.csv")
    assert [(r["source"], r["destination"]) for r in rows] == [("vial-aaaaaaaaaaaa", "vial-bbbbbbbbbbbb")]
    assert float(rows[0]["amount"]) == pytest.approx(0.5)
    assert rows[0]["method"] in ("small_tip", "large_tip")
    assert rows[0]["run_id"] == (lash.run_id or "")
    assert not st.TRANSFERS_FILE.exists() or st.TRANSFERS_FILE.stat().st_mtime < (root / "transfers.csv").stat().st_mtime
