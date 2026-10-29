import logging
import math
import random
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from workflows import surfactant_grid_ailsa as ailsa


def test_configured_dye_vial_is_used_for_dispensing(monkeypatch):
    monkeypatch.setattr(ailsa, "DYE", "pyrene")
    monkeypatch.setattr(ailsa, "DYE_VIAL", "dye_nile_red")
    assert "DYE_VIAL" in ailsa._CONFIG_KEYS
    assert ailsa.get_dye_protocol()["dye_vial"] == "dye_nile_red"


def test_measurement_schedule_requires_zero_and_increasing_times(monkeypatch):
    for schedule in ([0], [0, 10, 30]):
        monkeypatch.setattr(ailsa, "MEASUREMENT_SCHEDULE_MIN", schedule)
        assert ailsa.validate_measurement_schedule() == schedule

    for schedule in ([], [5], [0, 10, 10], [0, -1], [0, float("nan")]):
        monkeypatch.setattr(ailsa, "MEASUREMENT_SCHEDULE_MIN", schedule)
        with pytest.raises(ValueError, match="MEASUREMENT_SCHEDULE_MIN"):
            ailsa.validate_measurement_schedule()


def test_measurement_means_keep_timepoints_separate(tmp_path):
    rows = []
    for timepoint, signal in ((0, 10.0), (30, 30.0)):
        for read, offset in ((1, 0.0), (2, 2.0)):
            rows.append({
                "wellplate_index": 0,
                "timepoint_min": timepoint,
                "elapsed_actual_min": timepoint + read / 10,
                "measurement_replicate": read,
                "turbidity_600": signal + offset,
                "fluorescence_334_373": signal + offset,
                "fluorescence_334_384": signal + offset,
                "ratio": signal + offset,
                "fluorescence_metric": signal + offset,
            })
    tables = ailsa.save_measurement_analysis_tables(pd.DataFrame(rows), tmp_path, logging.getLogger(__name__))
    mean = pd.read_csv(tables["mean_csv"]).sort_values("timepoint_min")
    assert mean["timepoint_min"].tolist() == [0, 30]
    assert mean["turbidity_600"].tolist() == [11.0, 31.0]
    assert mean["elapsed_actual_min"].tolist() == pytest.approx([0.15, 30.15])


def test_random_measurement_order_is_reproducible_and_independent_of_recipe_order(monkeypatch):
    monkeypatch.setattr(ailsa, "RANDOMIZED_MEASUREMENT_ORDER", True)
    recipes = pd.DataFrame({"wellplate_index": [3, 1, 2, 0]})
    first = ailsa.get_measurement_well_order(recipes, random.Random(42))
    second = ailsa.get_measurement_well_order(recipes.iloc[::-1], random.Random(42))
    assert first == second
    assert first != [0, 1, 2, 3]


def test_ordered_reads_use_singleton_requests_for_both_assays():
    lash = SimpleNamespace(logger=logging.getLogger(__name__))
    wells = [3, 1, 2]
    recipes = pd.DataFrame({"wellplate_index": wells})
    with patch.object(ailsa, "position_study_wellplate_at_cytation") as enter, \
         patch.object(ailsa, "position_study_wellplate_at_track") as leave, \
         patch.object(ailsa, "shake_study_wellplate") as shake, \
         patch.object(ailsa, "measure_study_turbidity_protocol_only",
                      side_effect=lambda _, indices, __: pd.DataFrame({
                          "wellplate_index": indices, "turbidity_600": [1.0]})) as turbidity, \
         patch.object(ailsa, "measure_dye_fluorescence_protocol_only",
                      side_effect=lambda _, indices, __, ___: pd.DataFrame({
                          "wellplate_index": indices, "ratio": [1.0]})) as fluorescence:
        ailsa.measure_study_turbidity(
            lash, wells, recipes, return_wellplate=False, one_well_at_a_time=True,
        )
        ailsa.measure_dye_fluorescence(
            lash, wells, recipes, ailsa.get_dye_protocol(),
            shake_and_wait=False, one_well_at_a_time=True,
        )
    assert [call.args[1] for call in turbidity.call_args_list] == [[3], [1], [2]]
    assert [call.args[1] for call in fluorescence.call_args_list] == [[3], [1], [2]]
    assert enter.call_count == 2
    assert leave.call_count == 1  # Turbidity leaves the plate at Cytation for fluorescence.
    assert shake.call_count == 1


@pytest.mark.parametrize(
    ("dye", "measurement_column"),
    (
        ("nile-red", "fluorescence_550_648"),
        ("coumarin-6", "fluorescence_485_530"),
    ),
)
def test_intensity_dye_simulation_returns_synthetic_protocol_data(
    monkeypatch, dye, measurement_column
):
    monkeypatch.setattr(ailsa, "DYE", dye)
    dye_protocol = ailsa.get_dye_protocol()
    recipes = pd.DataFrame({
        "wellplate_index": [0, 1, 2, 3],
        "well_type": ["experiment", "experiment", "control", "control"],
        "control_type": ["experiment", "experiment", "water_blank", "cmc_SDS_1"],
        "surf_A_conc_mm": [2.0, 8.0, None, 4.0],
        "surf_B_conc_mm": [1.0, 5.0, None, None],
    })

    simulated = ailsa.simulate_dye_fluorescence([0, 1, 2, 3], recipes, dye_protocol)

    assert simulated["wellplate_index"].tolist() == [0, 1, 2, 3]
    assert measurement_column in simulated.columns
    values = pd.to_numeric(simulated[measurement_column], errors="raise")
    assert values.map(math.isfinite).all()
    assert values.nunique() > 1
