"""Labaware client: offline tests with a fake HTTP session (no network, no hardware)."""
import copy

import pytest

from tools.labaware_client import (DeckReading, LabawareClient, LabawareError, slot_name)
from vial_csv import read_vial_csv

# Example response from docs/platform/labaware_api.md
RESPONSE = {
    "app_id": "north_deck",
    "states": {
        "vial_rack": [
            ["open-cap", "open-cap", "open-cap", "empty", "open-cap", "empty"],
            ["empty", "open-cap", "open-cap", "empty", "empty", "open-cap"],
            ["open-cap", "empty", "open-cap", "open-cap", "open-cap", "open-cap"],
            ["closed-cap"] * 6,
            ["empty"] * 6, ["empty"] * 6, ["empty"] * 6, ["empty"] * 6,
        ],
        "clamp_1": "open-cap",
        "clamp_2": "empty",
    },
    "timing_ms": 170.4,
    "warnings": [],
    "history_id": 219,
}


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text
        self.ok = 200 <= status < 300

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return self.response

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return self.response


def client_for(response=None, error=None):
    session = FakeSession(response, error)
    return LabawareClient(session=session), session


def test_slot_name_matches_location_index():
    assert slot_name(0) == "A1" and slot_name(1) == "A2" and slot_name(6) == "B1"
    assert slot_name(47) == "H6"
    with pytest.raises(ValueError):
        slot_name(48)


def test_read_deck_posts_file_and_parses(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNGfake")
    client, session = client_for(FakeResponse(payload=RESPONSE))

    reading = client.read_deck(image)

    url, kwargs = session.calls[0]
    assert url == "http://sdl4-not-for-gaming:8010/api/apps/north_deck/infer"
    assert kwargs["files"]["image"] == ("frame.png", b"\x89PNGfake", "image/png")
    assert reading.trustworthy and reading.history_id == 219
    assert len(reading.rack) == 48
    assert reading.state_at("main_8mL_rack", 0) == "open-cap"
    assert reading.state_at("main_8mL_rack", 3) == "empty"
    assert reading.state_at("main_8mL_rack", 18) == "closed-cap"   # D1
    assert reading.state_at("clamp") == "open-cap"
    assert reading.state_at("clamp_2") == "empty"
    assert len(reading.occupied()) == 4 + 3 + 5 + 6 + 1  # rack rows A-D + clamp


def test_bytes_and_numpy_frames_are_accepted():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    client, session = client_for(FakeResponse(payload=RESPONSE))
    client.read_deck(b"\xff\xd8jpeg")
    client.read_deck(np.zeros((10, 10, 3), dtype=np.uint8))
    assert session.calls[0][1]["files"]["image"][2] == "image/jpeg"
    name, payload, ctype = session.calls[1][1]["files"]["image"]
    assert ctype == "image/jpeg" and payload[:2] == b"\xff\xd8"


def test_missing_labware_is_untrustworthy_not_an_error():
    data = copy.deepcopy(RESPONSE)
    data["states"]["vial_rack"] = None
    data["warnings"] = ["rack vial_rack (vial_rack) not detected"]
    reading = DeckReading.from_response(data)
    assert not reading.trustworthy
    assert reading.state_at("main_8mL_rack", 5) is None
    assert reading.occupied() == [("clamp", 0, "open-cap")]


def test_writes_valid_vial_csv(tmp_path):
    reading = DeckReading.from_response(RESPONSE)
    path = reading.write_vial_csv(tmp_path / "detected.csv")
    fields, records = read_vial_csv(path)
    assert fields[:5] == ["vial_index", "vial_name", "location", "location_index", "vial_volume"]
    by_name = {r["vial_name"]: r for r in records}
    assert by_name["main_8mL_rack_A1"]["cap_type"] == "open"
    assert by_name["main_8mL_rack_D1"]["cap_type"] == "closed"
    assert by_name["main_8mL_rack_D1"]["location_index"] == "18"
    assert by_name["clamp"]["location"] == "clamp"
    assert "main_8mL_rack_A4" not in by_name


def test_empty_deck_refuses_to_write(tmp_path):
    data = copy.deepcopy(RESPONSE)
    data["states"] = {"vial_rack": [["empty"] * 6] * 8, "clamp_1": "empty", "clamp_2": "empty"}
    with pytest.raises(ValueError):
        DeckReading.from_response(data).write_vial_csv(tmp_path / "x.csv")


@pytest.mark.parametrize("response,error", [
    (FakeResponse(status=400, text="bad image"), None),
    (FakeResponse(payload=None), None),
    (FakeResponse(payload={"oops": 1}), None),
    (None, ConnectionError("down")),
])
def test_failures_raise_labaware_error(response, error):
    client, _ = client_for(response, error)
    with pytest.raises(LabawareError):
        client.read_deck(b"\xff\xd8jpeg")


def test_bad_grid_or_state_raises():
    data = copy.deepcopy(RESPONSE)
    data["states"]["vial_rack"] = data["states"]["vial_rack"][:7]
    with pytest.raises(LabawareError):
        DeckReading.from_response(data)
    data = copy.deepcopy(RESPONSE)
    data["states"]["clamp_2"] = "half-cap"
    with pytest.raises(LabawareError):
        DeckReading.from_response(data)


def test_health_never_raises():
    assert client_for(FakeResponse(payload={"status": "ok"}))[0].health() is True
    assert client_for(error=ConnectionError("down"))[0].health() is False
