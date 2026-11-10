"""Deck check comparison and records: mocked camera and server, no hardware."""
import copy
import csv
import json
from datetime import datetime

import pytest

from scheduler import deck_check
from tools.labaware_client import DeckReading, LabawareError

EMPTY_GRID = [["empty"] * 6 for _ in range(8)]


def response(grid=None, clamp_1="empty", clamp_2="empty", warnings=()):
    return {"app_id": "north_deck", "states": {"vial_rack": copy.deepcopy(EMPTY_GRID if grid is None else grid),
                                               "clamp_1": clamp_1, "clamp_2": clamp_2},
            "timing_ms": 1.0, "warnings": list(warnings), "history_id": 7}


def rack(*cells):
    """cells: (location_index, state) pairs."""
    grid = copy.deepcopy(EMPTY_GRID)
    for index, state in cells:
        grid[index // 6][index % 6] = state
    return grid


def vial(name, location="main_8mL_rack", index=0, capped="True", cap_type="closed"):
    return {(location, index): [{"vial_name": name, "location": location, "location_index": str(index),
                                 "capped": capped, "cap_type": cap_type, "source_file": "x.csv"}]}


def occupancy(*vials):
    merged = {}
    for item in vials:
        merged.update(item)
    return merged


def compare(occ, data):
    return deck_check.compare(occ, DeckReading.from_response(data))


def kinds(issues):
    return sorted(issue.kind for issue in issues)


def test_match_open_and_closed_caps_and_clamp():
    occ = occupancy(vial("a", index=0, cap_type="closed"), vial("b", index=9, cap_type="open"),
                    vial("c", location="clamp", index=0, cap_type="open"))
    issues, notes, checked, unchecked = compare(occ, response(rack((0, "closed-cap"), (9, "open-cap")),
                                                              clamp_1="open-cap"))
    assert issues == [] and checked == 3 and unchecked == [] and notes == []


def test_missing_unexpected_and_wrong_cap():
    occ = occupancy(vial("a", index=0, cap_type="closed"), vial("b", index=1, cap_type="closed"))
    issues, *_ = compare(occ, response(rack((1, "open-cap"), (5, "open-cap"))))
    assert kinds(issues) == ["missing", "unexpected", "wrong_cap"]
    by_kind = {issue.kind: issue for issue in issues}
    assert by_kind["missing"].position == "main_8mL_rack [0]"
    assert "should have a closed cap but the camera sees an open cap" in by_kind["wrong_cap"].message
    assert by_kind["wrong_cap"].expected == "b" and by_kind["wrong_cap"].observed == "open-cap"
    assert by_kind["unexpected"].location_index == 5


def test_missing_vial_suggests_likely_new_position():
    occ = occupancy(vial("a", index=0, cap_type="closed"))
    issues, *_ = compare(occ, response(rack((10, "closed-cap"), (11, "open-cap"))))
    missing = next(issue for issue in issues if issue.kind == "missing")
    assert "Possibly moved to main_8mL_rack [10]" in missing.message
    assert "[11]" not in missing.message   # wrong cap type, not a candidate


def test_uncapped_vial_checks_presence_only():
    occ = occupancy(vial("u", index=3, capped="False", cap_type=""))
    issues, notes, checked, _ = compare(occ, response(rack((3, "open-cap"))))
    assert issues == [] and checked == 1 and "cap not checked" in notes[0]
    issues, *_ = compare(occ, response())
    assert kinds(issues) == ["missing"]


def test_locations_camera_cannot_see_are_listed_not_flagged():
    occ = occupancy(vial("big", location="large_vial_rack", index=0))
    issues, _, checked, unchecked = compare(occ, response())
    assert issues == [] and checked == 0 and unchecked == ["big at large_vial_rack [0]"]


def test_labware_not_detected_and_server_warnings():
    data = response(warnings=["rack vial_rack (vial_rack) not detected"])
    data["states"]["vial_rack"] = None
    issues, _, checked, _ = compare(occupancy(vial("a", index=4)), data)
    assert kinds(issues) == ["not_seen", "server_warning"] and checked == 0
    assert issues[0].position == "main_8mL_rack"


def test_rack_not_detected_with_nothing_expected_there_is_only_a_warning():
    data = response(warnings=["not detected"])
    data["states"]["vial_rack"] = None
    issues, *_ = compare({}, data)
    assert kinds(issues) == ["server_warning"]


class FakeClient:
    def __init__(self, data=None, error=None):
        self.data, self.error = data, error

    def infer_raw(self, _frame, detail="compact"):
        self.detail = detail
        if self.error:
            raise self.error
        return self.data


def fake_capture(path, _folder, _timeout):
    path.write_bytes(b"\xff\xd8jpeg")


def full_response(compact):
    """Convert a compact response to the detail=full shape, with 10 px cells."""
    states = compact["states"]
    labware = [{"id": "vial_rack", "missing": states["vial_rack"] is None, "slots": [
        {"row": r, "col": c, "state": s, "cell": [[c * 10, r * 10], [c * 10 + 9, r * 10],
                                                  [c * 10 + 9, r * 10 + 9], [c * 10, r * 10 + 9]]}
        for r, row in enumerate(states["vial_rack"] or []) for c, s in enumerate(row)]}]
    for i, key in enumerate(("clamp_1", "clamp_2")):
        labware.append({"id": key, "missing": states[key] is None, "slots": [] if states[key] is None else [
            {"row": 0, "col": 0, "state": states[key], "cell": [[70 + i * 10, 0], [79 + i * 10, 0],
                                                               [79 + i * 10, 9], [70 + i * 10, 9]]}]})
    return {"app_id": "north_deck", "labware": labware, "timing_ms": 1.0,
            "warnings": compact["warnings"], "history_id": compact["history_id"]}


def real_jpeg(path, _folder, _timeout):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    cv2.imwrite(str(path), np.full((90, 100, 3), 255, dtype=np.uint8))


def test_full_response_parses_like_compact_and_keeps_boxes():
    compact = response(rack((0, "closed-cap"), (47, "open-cap")), clamp_1="open-cap")
    full = DeckReading.from_response(full_response(compact))
    assert full.rack == DeckReading.from_response(compact).rack
    assert full.clamps == {"clamp": "open-cap", "clamp_2": "empty"}
    assert len(full.boxes) == 50
    box = next(b for b in full.boxes if b["location_index"] == 47)
    assert box["state"] == "open-cap" and box["cell"][0] == [50, 70]


def test_run_check_requests_full_and_saves_rotated_overlay(tmp_path):
    cv2 = pytest.importorskip("cv2")
    client = FakeClient(full_response(response(rack((1, "open-cap")))))
    check = deck_check.run_check(occupancy(vial("a", index=0)), tmp_path, SETTINGS, real_jpeg, client, NOW)
    assert client.detail == "full"
    assert check.outcome == "mismatch" and check.annotated.endswith("annotated.jpg")
    raw, shown = cv2.imread(check.frame), cv2.imread(check.annotated)
    assert raw.shape[:2] == (90, 100) and shown.shape[:2] == (180, 200)   # raw frame untouched; display 2x
    # Index 0 (cell 0-9 px, top-left in the raw frame) is flagged; rotated 180 it is drawn bottom-right, orange.
    bottom_right = shown[160:180, 180:200].reshape(-1, 3)
    assert any(tuple(int(v) for v in px) == deck_check.ISSUE_BGR for px in bottom_right)
    assert all(tuple(int(v) for v in px) != deck_check.ISSUE_BGR for px in shown[0:20, 0:20].reshape(-1, 3))


def test_low_confidence_vials_count_as_empty(tmp_path):
    # Expected: vial at [0]. Camera: real vial at [0] (0.99) and a phantom closed cap at [47] (0.56).
    data = full_response(response(rack((0, "closed-cap"), (47, "closed-cap"))))
    for slot in data["labware"][0]["slots"]:
        slot["conf"] = 0.56 if slot["row"] * 6 + slot["col"] == 47 else 0.99
    check = deck_check.run_check(occupancy(vial("a", index=0)), tmp_path, SETTINGS, fake_capture,
                                 FakeClient(data), NOW)
    assert check.outcome == "match"
    assert check.notes == ["main_8mL_rack [47]: camera unsure (closed-cap, confidence 0.56); treated as empty."]
    strict = dict(SETTINGS, min_vial_confidence=0.5)
    check = deck_check.run_check(occupancy(vial("a", index=0)), tmp_path, strict, fake_capture,
                                 FakeClient(data), datetime(2026, 11, 10, 16))
    assert [issue.kind for issue in check.issues] == ["unexpected"]


def test_overlay_failure_does_not_fail_the_check(tmp_path):
    # fake_capture writes bytes cv2 can't decode: the check still completes, without an overlay.
    check = deck_check.run_check(occupancy(vial("a", index=0)), tmp_path, SETTINGS, fake_capture,
                                 FakeClient(full_response(response())), NOW)
    assert check.outcome == "mismatch" and check.annotated is None


SETTINGS = dict(deck_check.DEFAULT_SETTINGS)
NOW = datetime(2026, 11, 10, 15, 0, 0)


def test_run_check_mismatch_saves_records(tmp_path):
    occ = occupancy(vial("a", index=0))
    check = deck_check.run_check(occ, tmp_path, SETTINGS, fake_capture, FakeClient(response()), NOW)
    assert check.outcome == "mismatch" and check.history_id == 7
    folder = tmp_path / "20261110_150000_000000"
    assert {p.name for p in folder.iterdir()} == {"frame.jpg", "response.json", "expected.csv", "report.json"}
    deck_check.finalize(check, "continue", " looked fine ")
    report = json.loads((folder / "report.json").read_text())
    assert report["decision"] == "continue" and report["note"] == "looked fine"
    assert report["issues"][0]["kind"] == "missing"
    rows = list(csv.DictReader((tmp_path / "index.csv").open()))
    assert rows[0]["outcome"] == "mismatch" and rows[0]["issues"] == "1" and rows[0]["history_id"] == "7"
    deck_check.mark_used(folder, "live_123")
    assert json.loads((folder / "report.json").read_text())["live_run"] == "live_123"


def test_run_check_match(tmp_path):
    check = deck_check.run_check(occupancy(vial("a", index=0)), tmp_path, SETTINGS, fake_capture,
                                 FakeClient(response(rack((0, "closed-cap")))), NOW)
    assert check.outcome == "match" and "deck matches" in check.summary()


@pytest.mark.parametrize("capture,client,text", [
    (lambda *_: (_ for _ in ()).throw(RuntimeError("no frame")), FakeClient(response()), "camera capture failed"),
    (fake_capture, FakeClient(error=LabawareError("unreachable")), "Labaware server"),
    (fake_capture, FakeClient({"bad": 1}), "Labaware server"),
])
def test_run_check_comms_failures_never_raise(tmp_path, capture, client, text):
    check = deck_check.run_check({}, tmp_path, SETTINGS, capture, client, NOW)
    assert check.outcome == "comms_failure" and text in check.error
    assert json.loads((tmp_path / check.timestamp / "report.json").read_text())["outcome"] == "comms_failure"


def test_camera_errors_copy_photo_and_upgrade_old_index(tmp_path):
    old = tmp_path / "index.csv"
    old.write_text("timestamp,outcome,decision,issues,history_id,error,note,folder\n"
                   "old,mismatch,cancel,1,1,,,x\n")
    occ = occupancy(vial("a", index=0))
    check = deck_check.run_check(occ, tmp_path, SETTINGS, fake_capture,
                                 FakeClient(response(rack((0, "closed-cap"), (47, "closed-cap")))), NOW)
    deck_check.finalize(check, "continue", "47 is empty", camera_wrong=[0])
    assert deck_check.all_camera_wrong(check)
    errors = tmp_path / "camera_errors"
    assert (errors / f"{check.timestamp}_7.jpg").read_bytes() == b"\xff\xd8jpeg"
    assert "main_8mL_rack [47]: camera said closed-cap" in (errors / f"{check.timestamp}_7.txt").read_text()
    rows = list(csv.DictReader(old.open()))
    assert [r["timestamp"] for r in rows] == ["old", check.timestamp]
    assert rows[0]["camera_wrong"] == "" and rows[1]["camera_wrong"] == "1"
    assert json.loads((tmp_path / check.timestamp / "report.json").read_text())["issues"][0]["camera_wrong"]


def test_unflagged_check_saves_no_camera_error(tmp_path):
    check = deck_check.run_check({}, tmp_path, SETTINGS, fake_capture,
                                 FakeClient(response(rack((47, "closed-cap")))), NOW)
    deck_check.finalize(check, "cancel")
    assert not deck_check.all_camera_wrong(check) and not (tmp_path / "camera_errors").exists()


def test_load_settings_falls_back_to_defaults(tmp_path):
    assert deck_check.load_settings(tmp_path / "missing.yaml") == deck_check.DEFAULT_SETTINGS
    path = tmp_path / "s.yaml"
    path.write_text("labaware_url: http://x:1\n")
    assert deck_check.load_settings(path)["labaware_url"] == "http://x:1"
    assert deck_check.load_settings()["labaware_app_id"] == "north_deck"   # real repo file parses
