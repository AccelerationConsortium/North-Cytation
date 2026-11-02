"""Exercise backup switching and scaled dispensing without hardware calls."""
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from workflows import surfactant_grid_ailsa as ailsa


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setattr(ailsa, "STOCK_REFILL_SOURCES", {
        "SDS_stock": ["SDS_refill", "SDS_refill_2"],
        "BDDAC_stock": ["BDDAC_refill"],
    })
    monkeypatch.setattr(ailsa, "STOCK_REFILL_RESERVE_ML", 0.1)
    volumes = {"SDS_stock": 0.5, "SDS_refill": 2.1, "SDS_refill_2": 10.1,
               "BDDAC_stock": 7.5, "BDDAC_refill": 20.0}
    robot = MagicMock()
    robot.get_vial_info.side_effect = lambda vial, column: volumes[vial]
    def transfer(source_vial_name, dest_vial_name, volume, **kwargs):
        volumes[source_vial_name] -= volume
        volumes[dest_vial_name] += volume
    robot.dispense_from_vial_into_vial.side_effect = transfer
    return SimpleNamespace(nr_robot=robot, logger=logging.getLogger(__name__)), volumes


def test_refill_splits_topup_and_next_refill_skips_depleted_source(rig):
    lash, volumes = rig
    ailsa.validate_stock_refill_sources(lash, ["SDS_stock", "BDDAC_stock"], [])
    ailsa.refill_surfactant_vial(lash, "SDS_stock")
    calls = lash.nr_robot.dispense_from_vial_into_vial.call_args_list
    assert [c.kwargs["source_vial_name"] for c in calls] == ["SDS_refill", "SDS_refill_2"]
    assert [c.kwargs["volume"] for c in calls] == pytest.approx([2.0, 5.0])
    assert volumes["SDS_stock"] == pytest.approx(7.5)
    assert volumes["SDS_refill"] == pytest.approx(0.1)
    volumes["SDS_stock"] = 3.0
    lash.nr_robot.dispense_from_vial_into_vial.reset_mock()
    ailsa.refill_surfactant_vial(lash, "SDS_stock")
    assert lash.nr_robot.dispense_from_vial_into_vial.call_args.kwargs["source_vial_name"] == "SDS_refill_2"


def test_insufficient_pool_fails_before_any_transfer(rig):
    lash, volumes = rig
    volumes["SDS_refill_2"] = 1.0
    with pytest.raises(ValueError, match="Insufficient usable stock refill supply.*SDS_refill_2"):
        ailsa.refill_surfactant_vial(lash, "SDS_stock")
    lash.nr_robot.dispense_from_vial_into_vial.assert_not_called()
    assert volumes["SDS_stock"] == 0.5


def test_small_remainder_is_left_and_transfers_preserve_total():
    transfers = ailsa.plan_stock_refill({"a": 1.1, "b": 1.1}, 1.1, 0.1)
    assert sum(v for _, v in transfers) == pytest.approx(1.1)
    assert all(v >= 0.2 - 1e-9 for _, v in transfers)
    assert ailsa.plan_stock_refill({"a": 0.25, "b": 2.0}, 1.0, 0.1) == [("b", 1.0)]


@pytest.mark.parametrize("sources", [[], "SDS_refill", ["SDS_refill", "SDS_refill"],
                                         ["SDS_stock"], ["BDDAC_refill"], ["missing"]])
def test_invalid_pool_rejected_before_transfers(rig, monkeypatch, sources):
    lash, _ = rig
    monkeypatch.setattr(ailsa, "STOCK_REFILL_SOURCES", {
        "SDS_stock": sources, "BDDAC_stock": ["BDDAC_refill"]})
    with pytest.raises(ValueError):
        ailsa.validate_stock_refill_sources(lash, ["SDS_stock", "BDDAC_stock"], [])
    lash.nr_robot.dispense_from_vial_into_vial.assert_not_called()


@pytest.mark.parametrize("volume", [-1, float("nan"), float("inf")])
def test_invalid_recorded_volume_rejected(rig, volume):
    lash, volumes = rig
    volumes["SDS_refill_2"] = volume
    with pytest.raises(ValueError, match="Invalid recorded volume"):
        ailsa.validate_stock_refill_sources(lash, ["SDS_stock", "BDDAC_stock"], [])


def test_legacy_source_name_still_works(rig, monkeypatch):
    lash, volumes = rig
    monkeypatch.setattr(ailsa, "STOCK_REFILL_SOURCES", {})
    volumes["SDS_stock"] = 6.5
    ailsa.refill_surfactant_vial(lash, "SDS_stock")
    assert volumes["SDS_stock"] == pytest.approx(7.5)
    assert lash.nr_robot.dispense_from_vial_into_vial.call_count == 1


def test_large_well_chunks_refill_before_exceeding_capacity(rig, monkeypatch):
    lash, volumes = rig
    volumes["SDS_stock"] = 4.1  # Above threshold but insufficient for the next chunk.
    volumes["SDS_refill_2"] = 30.0
    monkeypatch.setattr(ailsa, "REFILL_CHECK_CHUNK_SIZE", 24)
    wells = pd.DataFrame({"substock_A_name": ["SDS_stock"] * 24,
                          "surf_A_volume_ul": [600.0] * 24,
                          "wellplate_index": range(24)})
    dispensed = []
    def dispense(controller, chunk, vial, *args, **kwargs):
        amount = chunk.surf_A_volume_ul.sum() / 1000
        assert amount + 0.1 <= volumes[vial] + 1e-9
        dispensed.extend(chunk.wellplate_index)
        volumes[vial] -= amount
    monkeypatch.setattr(ailsa, "dispense_component_to_wellplate", dispense)
    ailsa._dispense_vial_in_chunks(lash, wells, "SDS_stock", "SDS", "surf_A_volume_ul", [], True)
    assert dispensed == list(range(24))
    assert volumes["SDS_stock"] >= 0.1 - 1e-9


def test_substock_preparation_refills_its_parent_stock(rig, monkeypatch):
    lash, volumes = rig
    volumes["SDS_dilution_0"] = 0.0
    recipe = {"Vial_Name": "SDS_dilution_0", "Target_Conc_mM": 10.0,
              "Source_Vial": "SDS_stock", "Source_Volume_mL": 1.2,
              "Water_Volume_mL": 4.8, "Final_Volume_mL": 6.0}
    ailsa.create_substocks_from_recipes(lash, [recipe])
    calls = lash.nr_robot.dispense_from_vial_into_vial.call_args_list
    assert calls[-1].kwargs["dest_vial_name"] == "SDS_dilution_0"
    assert calls[0].kwargs["source_vial_name"] == "SDS_refill"
    assert volumes["SDS_stock"] == pytest.approx(6.3)


def test_substock_shortage_propagates_before_preparation(rig):
    lash, volumes = rig
    volumes.update(SDS_dilution_0=0.0, SDS_refill=0.0, SDS_refill_2=0.0)
    recipe = {"Vial_Name": "SDS_dilution_0", "Target_Conc_mM": 10.0,
              "Source_Vial": "SDS_stock", "Source_Volume_mL": 1.2,
              "Water_Volume_mL": 4.8, "Final_Volume_mL": 6.0}
    with pytest.raises(ValueError, match="Insufficient usable stock refill supply"):
        ailsa.create_substocks_from_recipes(lash, [recipe])
    lash.nr_robot.dispense_from_vial_into_vial.assert_not_called()


def test_circular_substock_sources_are_rejected(rig):
    lash, _ = rig
    with pytest.raises(ValueError, match="Circular substock"):
        ailsa.validate_stock_refill_sources(lash, ["SDS_stock", "BDDAC_stock"], [
            {"Vial_Name": "a", "Source_Vial": "b"},
            {"Vial_Name": "b", "Source_Vial": "a"},
        ])


def test_near_full_stock_can_cover_chunk_with_reserve(rig):
    lash, volumes = rig
    volumes["SDS_stock"] = 7.4
    assert ailsa.ensure_vial_above_threshold(lash, "SDS_stock", [], required_ml=7.35)
    assert 7.45 <= volumes["SDS_stock"] <= 8.0


def test_upstream_dilution_is_replenished_before_downstream_preparation(rig):
    lash, volumes = rig
    volumes.update(parent=0.0, child=0.0)
    parent = {"Vial_Name": "parent", "Target_Conc_mM": 10.0,
              "Source_Vial": "SDS_stock", "Source_Volume_mL": 1.2,
              "Water_Volume_mL": 4.8, "Final_Volume_mL": 6.0}
    child = {**parent, "Vial_Name": "child", "Target_Conc_mM": 2.0, "Source_Vial": "parent"}
    # Use a minimal volume-tracking reservoir alongside the tracked vial transfers.
    def water(reservoir_index, vial_index, volume, **kwargs):
        volumes[vial_index] += volume
    lash.nr_robot.dispense_into_vial_from_reservoir.side_effect = water
    ailsa.create_substocks_from_recipes(lash, [child], all_recipes=[parent, child])
    assert volumes["parent"] == pytest.approx(4.8)
    assert volumes["child"] == pytest.approx(6.0)


def test_substock_above_normal_threshold_is_topped_up_for_large_chunk(rig):
    lash, volumes = rig
    volumes["SDS_dilution_0"] = 4.5
    recipe = {"Vial_Name": "SDS_dilution_0", "Target_Conc_mM": 10.0,
              "Source_Vial": "SDS_stock", "Source_Volume_mL": 1.2,
              "Water_Volume_mL": 4.8, "Final_Volume_mL": 6.0}
    def water(reservoir_index, vial_index, volume, **kwargs):
        volumes[vial_index] += volume
    lash.nr_robot.dispense_into_vial_from_reservoir.side_effect = water
    ailsa.ensure_vial_above_threshold(lash, "SDS_dilution_0", [recipe], required_ml=5.4)
    assert volumes["SDS_dilution_0"] == pytest.approx(6.0)
