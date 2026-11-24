"""Substock planner (workflows/helpers/substock_planning.py), with a fake error model."""
import math

import pytest

from workflows.helpers.substock_planning import plan_substocks, make_transfer_error

SETTINGS = dict(well_ul=240, well_min_ul=20, well_max_ul=200, substock_ml=4.0,
                substock_min_ul=100, dead_ml=0.5, max_error=0.03, name_prefix="sub_")
STOCK = [dict(name="stock", conc=50.0, error=0.0, available_ml=20.0)]


def fake_error(volume_ul):
    """0.5 uL absolute + 0.5% relative: small volumes are worse, like the real data."""
    return math.hypot(0.5 / volume_ul, 0.005)


def plan(targets, sources=STOCK, max_ul=None, **over):
    return plan_substocks(targets, sources, {**SETTINGS, **over}, fake_error, max_ul=max_ul)


def test_targets_reachable_from_stock_need_no_substock():
    p = plan({"a": 10.0, "b": 30.0})
    assert p["substocks"] == [] and {w["source"] for w in p["wells"]} == {"stock"}


def test_every_well_within_limits_and_concentration_exact():
    targets = {f"s{i}": 0.01 * 1.15 ** i for i in range(60)}
    p = plan(targets)
    src = p["sources"]
    assert not p["infeasible"]
    for w in p["wells"]:
        assert 20 - 1e-9 <= w["volume_ul"] <= 200 + 1e-9
        assert w["error"] <= 0.03 + 1e-9
        assert src[w["source"]]["conc"] * w["volume_ul"] / 240 == pytest.approx(w["target"])
    for x in p["substocks"]:
        assert 100 - 1e-9 <= x["transfer_ul"] <= 4000


def test_nearby_targets_share_one_substock():
    p = plan({f"s{i}": 1.0 + 0.02 * i for i in range(10)})
    assert len(p["substocks"]) == 1


def test_big_dilution_inserts_intermediate():
    p = plan({"a": 0.001})
    assert len(p["substocks"]) >= 2 and not p["infeasible"]


def test_above_stock_is_infeasible_with_reason():
    p = plan({"a": 60.0, "b": 0.0})
    assert [x["sample_id"] for x in p["infeasible"]] == ["a"]
    assert "above max" in p["infeasible"][0]["reason"]


def test_over_limit_target_is_still_made_by_lowest_error_route():
    p = plan({"a": 1.0}, max_error=0.001)
    assert not p["infeasible"]
    (well,) = p["wells"]
    assert well["over_limit"] and well["error"] > 0.001
    best = min(math.hypot(p["sources"][n]["error"], fake_error(1.0 * 240 / p["sources"][n]["conc"]))
               for n in p["sources"] if 20 <= 1.0 * 240 / p["sources"][n]["conc"] <= 200)
    assert well["error"] <= best * 1.1 + 1e-12


def test_per_sample_cap_limits_the_volume():
    p = plan({"a": 5.0, "b": 5.0}, max_ul={"a": 60}, max_error=0.012)
    vols = {w["sample_id"]: w["volume_ul"] for w in p["wells"]}
    assert vols["a"] <= 60 + 1e-9 and not p["infeasible"]
    p = plan({"a": 30.0}, max_ul={"a": 50})
    assert "above max" in p["infeasible"][0]["reason"]


def test_existing_substock_reused_and_bad_one_ignored():
    good = STOCK + [dict(name="old", conc=2.0, error=0.01, available_ml=3.0)]
    assert {w["source"] for w in plan({"a": 1.0}, sources=good)["wells"]} == {"old"}
    bad = STOCK + [dict(name="old", conc=2.0, error=0.5, available_ml=3.0)]
    assert "old" not in {w["source"] for w in plan({"a": 1.0}, sources=bad)["wells"]}


def test_shortfall_flagged():
    p = plan({f"s{i}": 40.0 for i in range(10)},
             sources=[dict(name="stock", conc=50.0, error=0.0, available_ml=1.0)])
    assert p["usage"]["stock"]["short"]


def test_deterministic():
    targets = {f"s{i}": 0.02 * 1.3 ** i for i in range(30)}
    assert plan(targets) == plan(targets)


class _FakeUncertainty:
    def table(self, liquid, tip):
        return [(200.0, 1.0, 2.0), (800.0, 1.0, 4.0)] if tip == "large_tip" else [(20.0, 0.3, 0.6)]

    def lookup(self, liquid, tip, volume_ul):
        return (1.0, 2.0) if tip == "large_tip" and 200 <= volume_ul <= 800 else None


def test_transfer_error_splits_large_volumes_and_adds_shared_linearly():
    err = make_transfer_error(_FakeUncertainty(), "SDS", [("small_tip", 195), ("large_tip", 990)])
    assert err(500) == pytest.approx(math.hypot(1, 2) / 500)
    assert err(1600) == pytest.approx(math.sqrt(2 * 1 + (2 * 2) ** 2) / 1600)
    assert err(100) is None
