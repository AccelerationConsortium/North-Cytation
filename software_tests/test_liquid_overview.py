"""Liquids tab: overview, adding and editing a liquid, and dated calibration merges."""
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from pipetting_data import liquid_overview as lo
from pipetting_data.liquid_registry import LIQUID_REGISTRY_PATH, load_liquids

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipetting_data"))
from import_calibration_run import convert  # noqa: E402


@pytest.fixture
def registry(tmp_path):
    copy = tmp_path / "liquids.yaml"
    shutil.copy(LIQUID_REGISTRY_PATH, copy)
    return copy


def test_add_and_edit_keep_the_rest_of_the_file(registry):
    before = registry.read_text(encoding="utf-8")
    lo.save_liquid("PEG400", 1.13, reuse_tip=False, condition_tip=True, registry=registry, new=True)
    liquids = load_liquids(registry)
    assert liquids["PEG400"] == {"density": 1.13, "refill_pipets": True, "condition_tip": True}
    lo.save_liquid("water", 0.998, reuse_tip=True, condition_tip=False, registry=registry)
    assert load_liquids(registry)["water"]["density"] == pytest.approx(0.998)
    after = registry.read_text(encoding="utf-8")
    # Comments and every other liquid's line are untouched
    assert [l for l in before.splitlines() if not l.strip().startswith("water:")] == \
           [l for l in after.splitlines() if not l.strip().startswith(("water:", "PEG400:"))]


@pytest.mark.parametrize("name, density, new, message", [
    ("bad name", 1.0, True, "only use"),
    ("water", 1.0, True, "already"),
    ("nothing_here", 1.0, False, "isn't in the list"),
    ("ok_name", 0.0, True, "positive"),
])
def test_bad_input_is_refused_and_the_file_is_unchanged(registry, name, density, new, message):
    before = registry.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        lo.save_liquid(name, density, True, False, registry=registry, new=new)
    assert registry.read_text(encoding="utf-8") == before


def test_merge_dates_new_rows_and_the_tab_shows_the_latest(tmp_path):
    legacy = tmp_path / "optimal_conditions_testliquid.csv"
    pd.DataFrame({"volume_target": [50.0], "overaspirate_vol": [0.001], "tip_type": ["small_tip"]}).to_csv(legacy, index=False)
    run = tmp_path / "new_run.csv"
    pd.DataFrame({"volume_target_ul": [100.0], "calibration_overaspirate_vol": [0.002], "tip_type": ["small_tip"]}).to_csv(run, index=False)
    convert(run, legacy)
    merged = pd.read_csv(legacy)
    assert merged.loc[merged.volume_target == 50.0, "calibrated_on"].isna().all()  # older row: unknown
    assert merged.loc[merged.volume_target == 100.0, "calibrated_on"].notna().all()
    summary = lo.calibration_summary("testliquid", tmp_path)
    assert summary["small_tip"] == "50–100 µL (2)" and summary["large_tip"] == "—"
    assert summary["last_calibrated"] == merged.calibrated_on.dropna().iloc[0]


def test_never_calibrated_and_never_validated(tmp_path):
    assert lo.calibration_summary("nothing", tmp_path)["small_tip"] == "not calibrated"
    sessions = pd.DataFrame({"timestamp": ["20261101_120000", "20261110_090000"], "liquid_type": ["x", "x"],
                             "simulate": [False, True]})
    assert lo.last_validated("x", sessions=sessions) == "2026-11-01"  # the simulated one doesn't count
    assert lo.last_validated("y", sessions=sessions) is None


def test_delete_removes_only_that_line(registry):
    lo.save_liquid("to_delete_4%", 1.01, True, False, registry=registry, new=True)  # independent of the real list
    before = registry.read_text(encoding="utf-8").splitlines()
    lo.delete_liquid("to_delete_4%", registry=registry)
    after = registry.read_text(encoding="utf-8").splitlines()
    assert "to_delete_4%" not in load_liquids(registry)
    assert [l for l in before if not l.strip().startswith("to_delete_4%:")] == after
    with pytest.raises(ValueError, match="isn't in the list"):
        lo.delete_liquid("to_delete_4%", registry=registry)


def test_usage_matches_the_whole_name_only(tmp_path):
    (tmp_path / "workflows").mkdir()
    (tmp_path / "workflows" / "a.py").write_text("LIQUID = 'agar_water'\n")
    (tmp_path / "workflows" / "b.py").write_text("LIQUID = 'agar_water_4%'\n")  # a different liquid
    (tmp_path / "workflow_configs").mkdir()
    (tmp_path / "workflow_configs" / "c.yaml").write_text("liquid: agar_water\n")
    usage = lo.liquid_usage("agar_water", repo_root=tmp_path, sessions_file=tmp_path / "none.csv")
    assert usage["files"] == ["workflow_configs/c.yaml", "workflows/a.py"]
    assert usage["calibration_files"] == [] and usage["validation_sessions"] == 0


def test_recent_use_counts_dispenses_and_validations(tmp_path):
    sessions = pd.DataFrame({"timestamp": ["20261110_120000", "20261101_120000", "20260101_120000", "20261112_120000"],
                             "liquid_type": ["SDS", "SDS", "SDS", "water"], "simulate": [False, False, False, True]})
    log = tmp_path / "transfers.csv"
    log.write_text("time,run_id,source,destination,amount,units,method,liquid\n"
                   "2026-11-15 10:00:00,r,a,b,0.1,mL,small_tip,water\n"
                   "2026-11-15 10:01:00,r,a,c,0.1,mL,small_tip,water\n")
    use = lo.recent_use(sessions, log, now="2026-11-18")
    assert use == {"sds": 2, "water": 2}  # old and simulated sessions don't count


def test_the_real_files_give_a_row_per_liquid():
    rows = lo.liquids_table()
    assert len(rows) == len(load_liquids())
    water = next(r for r in rows if r["liquid"] == "water")
    assert water["reuse_tip"] is True and water["small_tip"] != "not calibrated"
