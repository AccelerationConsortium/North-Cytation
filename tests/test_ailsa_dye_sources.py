"""Dye backup switching with recorded volumes and no hardware motion."""
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from workflows import surfactant_grid_ailsa as ailsa


@pytest.fixture
def dye_rig(monkeypatch):
    monkeypatch.setattr(ailsa, "DYE", "coumarin-6")
    monkeypatch.setattr(ailsa, "DYE_VIAL", "primary")
    monkeypatch.setattr(ailsa, "DYE_BACKUP_VIALS", ["backup"])
    monkeypatch.setattr(ailsa, "DYE_SOURCE_RESERVE_ML", 2.0)
    monkeypatch.setattr(ailsa, "DYE_VOLUME_UL", 5.0)
    monkeypatch.setattr(ailsa, "WELLPLATE_TYPE", "24 WELL PLATE")
    monkeypatch.setattr(ailsa, "VOLUME_SCALE_BY_PLATE", {"24 WELL PLATE": 4})
    rows = {
        "primary": {"vial_volume": 2.05, "vial_type": "8_mL", "home_location": "main_8mL_rack", "home_location_index": 5},
        "backup": {"vial_volume": 8.0, "vial_type": "8_mL", "home_location": "main_8mL_rack", "home_location_index": pd.Series([18]).iloc[0]},
    }
    robot = MagicMock()
    robot.get_vial_info.side_effect = lambda vial, column: rows[vial][column]
    lash = SimpleNamespace(nr_robot=robot, logger=logging.getLogger(__name__))
    deliveries = []
    def dispense(controller, wells, source, liquid, column):
        volume = float(wells[column].sum()) / 1000
        rows[source]["vial_volume"] -= volume
        assert rows[source]["vial_volume"] >= 2.0 - 1e-9
        deliveries.append((source, wells.wellplate_index.tolist(), volume))
    monkeypatch.setattr(ailsa, "dispense_study_component_to_wellplate", dispense)
    return lash, rows, deliveries


def test_switches_between_whole_wells_and_records_actual_sources(dye_rig):
    lash, rows, deliveries = dye_rig
    wells = pd.DataFrame({"wellplate_index": [7, 2, 9, 4, 0]}, index=[30, 11, 22, 8, 17])
    result = ailsa.dispense_dye(lash, wells)
    assert result is wells
    assert deliveries == [("primary", [7, 2], 0.04), ("backup", [9, 4, 0], 0.06)]
    assert wells.dye_source_vial.tolist() == ["primary", "primary", "backup", "backup", "backup"]
    assert rows["primary"]["vial_volume"] == pytest.approx(2.01)
    assert lash.nr_robot.move_vial_to_location.call_args_list[0].args == ("primary", "clamp", 0)
    assert lash.nr_robot.move_vial_to_location.call_args_list[1].args == ("backup", "clamp", 0)
    calls = lash.nr_robot.method_calls
    primary_home = next(i for i, call in enumerate(calls) if call[0] == "return_vial_home" and call.args == ("primary",))
    backup_move = next(i for i, call in enumerate(calls) if call[0] == "move_vial_to_location" and call.args[0] == "backup")
    assert primary_home < backup_move


def test_next_plate_uses_backup_from_live_volume(dye_rig):
    lash, _, deliveries = dye_rig
    ailsa.dispense_dye(lash, pd.DataFrame({"wellplate_index": [0, 1, 2]}))
    deliveries.clear()
    ailsa.dispense_dye(lash, pd.DataFrame({"wellplate_index": [3, 4]}))
    assert deliveries == [("backup", [3, 4], 0.04)]


def test_insufficient_pool_fails_before_staging_or_dispensing(dye_rig):
    lash, rows, deliveries = dye_rig
    rows["backup"]["vial_volume"] = 2.01
    with pytest.raises(ValueError, match="Insufficient dye supply.*short by 1 doses"):
        ailsa.dispense_dye(lash, pd.DataFrame({"wellplate_index": [0, 1, 2]}))
    assert deliveries == []
    lash.nr_robot.move_vial_to_location.assert_not_called()


def test_empty_primary_is_skipped(dye_rig):
    lash, rows, deliveries = dye_rig
    rows["primary"]["vial_volume"] = 0.0
    ailsa.dispense_dye(lash, pd.DataFrame({"wellplate_index": [0]}))
    assert deliveries == [("backup", [0], 0.02)]


def test_legacy_single_vial_config_still_dispenses(dye_rig, monkeypatch):
    lash, _, deliveries = dye_rig
    monkeypatch.setattr(ailsa, "DYE_BACKUP_VIALS", [])
    ailsa.dispense_dye(lash, pd.DataFrame({"wellplate_index": [0, 1]}))
    assert deliveries == [("primary", [0, 1], 0.04)]


def test_plan_for_both_runs_retains_two_ml_in_each_source(dye_rig):
    lash, rows, _ = dye_rig
    rows["primary"]["vial_volume"] = 7.32
    # Both dye volumes expressed at the active 24-well dose: 8.64 mL overall.
    allocations = ailsa.plan_dye_sources(lash, 432)
    assert allocations == [("primary", 266), ("backup", 166)]


@pytest.mark.parametrize("backups", ["backup", ["primary"], ["backup", "backup"], [""]])
def test_bad_backup_names_are_rejected(dye_rig, monkeypatch, backups):
    monkeypatch.setattr(ailsa, "DYE_BACKUP_VIALS", backups)
    with pytest.raises(ValueError, match="DYE_BACKUP_VIALS"):
        ailsa.get_dye_protocol()


@pytest.mark.parametrize("reserve", [-1, float("nan"), True])
def test_invalid_reserve_is_rejected(dye_rig, monkeypatch, reserve):
    monkeypatch.setattr(ailsa, "DYE_SOURCE_RESERVE_ML", reserve)
    with pytest.raises(ValueError, match="DYE_SOURCE_RESERVE_ML"):
        ailsa.get_dye_protocol()


def test_dye_inventory_validation_accepts_backup_main_rack_vial(dye_rig):
    lash, _, _ = dye_rig
    ailsa.validate_dye_sources(lash, ["SDS_stock", "BDDAC_stock"], [])


@pytest.mark.parametrize("column,value", [
    ("vial_type", "20_mL"), ("home_location", "large_vial_rack"),
    ("home_location_index", 5), ("home_location_index", 44),
    ("home_location_index", 48), ("vial_volume", 9.0), ("vial_volume", float("nan")),
])
def test_invalid_backup_inventory_is_rejected(dye_rig, column, value):
    lash, rows, _ = dye_rig
    rows["backup"][column] = value
    with pytest.raises(ValueError):
        ailsa.validate_dye_sources(lash, ["SDS_stock", "BDDAC_stock"], [])


def test_missing_backup_is_rejected(dye_rig):
    lash, rows, _ = dye_rig
    del rows["backup"]
    with pytest.raises(ValueError, match="Missing vial"):
        ailsa.validate_dye_sources(lash, ["SDS_stock", "BDDAC_stock"], [])


def test_dye_cannot_share_a_surfactant_source(dye_rig, monkeypatch):
    lash, _, _ = dye_rig
    monkeypatch.setattr(ailsa, "STOCK_REFILL_SOURCES", {"SDS_stock": ["backup"]})
    with pytest.raises(ValueError, match="Dye sources must be separate"):
        ailsa.validate_dye_sources(lash, ["SDS_stock", "BDDAC_stock"], [])


def test_metadata_preserves_actual_dye_source(dye_rig):
    rows = pd.DataFrame({"dye_source_vial": ["primary", "backup"]})
    result = ailsa.add_study_metadata(rows, 1)
    assert result.dye_vial.tolist() == ["primary", "backup"]
