"""Client for the Labaware deck-state service (see docs/platform/labaware_api.md).

Send one north-deck camera frame and get back the state of every vial position,
either as a ``DeckReading`` or as vial-CSV rows in the standard repo schema.

Typical use from a workflow or GUI::

    from tools.labaware_client import LabawareClient

    client = LabawareClient()
    reading = client.read_deck("frame.jpg")      # path, bytes, or cv2/numpy frame
    if not reading.trustworthy:
        print(reading.warnings)
    reading.state_at("main_8mL_rack", 9)        # -> "open-cap" / "closed-cap" / "empty" / None
    reading.write_vial_csv("detected_vials.csv")

Importing this module never contacts the server. Every call to ``read_deck`` is
stored on the server (frame + result), so don't send frames that shouldn't be kept.
"""
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "http://sdl4-not-for-gaming:8010"
DEFAULT_APP_ID = "north_deck"

EMPTY, OPEN_CAP, CLOSED_CAP = "empty", "open-cap", "closed-cap"
STATES = (EMPTY, OPEN_CAP, CLOSED_CAP)

RACK_LOCATION = "main_8mL_rack"
RACK_ROWS, RACK_COLS = 8, 6
ROW_LETTERS = "ABCDEFGH"
# Server labware id -> robot location name in robot_state/vial_positions.yaml.
CLAMP_LOCATIONS = {"clamp_1": "clamp", "clamp_2": "clamp_2"}

VIAL_CSV_FIELDS = ["vial_index", "vial_name", "location", "location_index", "vial_volume",
                   "capped", "cap_type", "vial_type", "home_location", "home_location_index"]
_CAP_COLUMNS = {OPEN_CAP: ("True", "open"), CLOSED_CAP: ("True", "closed")}

_IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff",
                ".webp": "image/webp"}


class LabawareError(RuntimeError):
    """The service could not be reached or returned an unusable response."""


def slot_name(location_index):
    """``main_8mL_rack`` location_index -> slot name (0 -> 'A1', 47 -> 'H6')."""
    row, col = divmod(int(location_index), RACK_COLS)
    if not 0 <= row < RACK_ROWS:
        raise ValueError(f"location_index {location_index} outside 0..{RACK_ROWS * RACK_COLS - 1}")
    return f"{ROW_LETTERS[row]}{col + 1}"


@dataclass(frozen=True)
class DeckReading:
    """State of the north deck from one frame.

    ``rack`` is a flat list of 48 states indexed by ``main_8mL_rack`` location_index,
    or ``None`` if the rack was not found. ``clamps`` maps robot location name
    ('clamp', 'clamp_2') to a state, or ``None`` if that clamp was not found.
    """
    rack: list | None
    clamps: dict
    warnings: list = field(default_factory=list)
    history_id: int | None = None
    timing_ms: float | None = None
    raw: dict = field(default_factory=dict, repr=False)
    # Only with detail=full: [{"location", "location_index", "state", "conf", "cell": [[x, y] x4]}] in frame pixels.
    boxes: list = field(default_factory=list, repr=False)

    @classmethod
    def from_response(cls, data):
        """Parse either a compact (``states``) or full (``labware``) response."""
        if isinstance(data, dict) and isinstance(data.get("labware"), list) and "states" not in data:
            states, boxes = _states_from_full(data["labware"])
            return cls._from_states(states, data, boxes)
        return cls._from_states(data.get("states") if isinstance(data, dict) else None, data)

    @classmethod
    def _from_states(cls, states, data, boxes=()):
        if not isinstance(states, dict):
            raise LabawareError("Labaware response has no 'states' mapping")
        grid = states.get("vial_rack")
        rack = None
        if grid is not None:
            if len(grid) != RACK_ROWS or any(len(row) != RACK_COLS for row in grid):
                raise LabawareError(f"vial_rack is not {RACK_ROWS}x{RACK_COLS}")
            rack = [_check_state(s) for row in grid for s in row]
        clamps = {loc: (None if states.get(sid) is None else _check_state(states[sid]))
                  for sid, loc in CLAMP_LOCATIONS.items()}
        return cls(rack=rack, clamps=clamps, warnings=list(data.get("warnings") or []),
                   history_id=data.get("history_id"), timing_ms=data.get("timing_ms"), raw=data,
                   boxes=list(boxes))

    @property
    def trustworthy(self):
        """True only if there are no warnings and every labware was found."""
        return (not self.warnings and self.rack is not None
                and all(s is not None for s in self.clamps.values()))

    def state_at(self, location, location_index=0):
        """State at a robot location, or ``None`` if that labware wasn't detected."""
        if location == RACK_LOCATION:
            if self.rack is None:
                return None
            slot_name(location_index)  # range check
            return self.rack[int(location_index)]
        if location in self.clamps:
            if int(location_index) != 0:
                raise ValueError(f"{location} has a single position (index 0)")
            return self.clamps[location]
        raise KeyError(f"Labaware does not observe location {location!r}")

    def occupied(self):
        """``[(location, location_index, state), ...]`` for every non-empty position."""
        found = [(RACK_LOCATION, i, s) for i, s in enumerate(self.rack or []) if s != EMPTY]
        found += [(loc, 0, s) for loc, s in self.clamps.items() if s not in (None, EMPTY)]
        return found

    def to_vial_records(self, vial_type="8_mL"):
        """``(fields, records)`` in the standard vial-CSV schema, one row per occupied position.

        The camera can't see contents, so names are positional (e.g. 'main_8mL_rack_A1')
        and ``vial_volume`` is 0. 'open-cap' -> capped True / cap_type open (open-top cap);
        'closed-cap' -> capped True / cap_type closed.
        """
        records = []
        for location, index, state in self.occupied():
            capped, cap_type = _CAP_COLUMNS[state]
            name = f"{location}_{slot_name(index)}" if location == RACK_LOCATION else location
            records.append({"vial_index": str(len(records)), "vial_name": name,
                            "location": location, "location_index": str(index),
                            "vial_volume": "0", "capped": capped, "cap_type": cap_type,
                            "vial_type": vial_type, "home_location": location,
                            "home_location_index": str(index)})
        return list(VIAL_CSV_FIELDS), records

    def write_vial_csv(self, path, vial_type="8_mL"):
        """Write the detected vials to ``path`` (validated, atomic). Returns the path."""
        from vial_csv import save_vial_csv

        fields, records = self.to_vial_records(vial_type)
        if not records:
            raise ValueError("No vials detected; refusing to write an empty vial file")
        save_vial_csv(path, fields, records)
        return Path(path)


class LabawareClient:
    """Thin HTTP client for one Labaware app. Safe to share across threads for reads."""

    def __init__(self, base_url=DEFAULT_BASE_URL, app_id=DEFAULT_APP_ID, timeout=30, session=None):
        self.base_url = base_url.rstrip("/")
        self.app_id = app_id
        self.timeout = timeout
        self._session = session

    @property
    def session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    @property
    def infer_url(self):
        return f"{self.base_url}/api/apps/{self.app_id}/infer"

    def health(self):
        """True if the server answers its liveness check; never raises."""
        try:
            r = self.session.get(f"{self.base_url}/api/health", timeout=min(self.timeout, 5))
            return r.ok and r.json().get("status") == "ok"
        except Exception:
            return False

    def read_deck(self, image):
        """POST one frame and return a ``DeckReading``.

        ``image`` may be a file path, raw encoded bytes (JPEG/PNG/...), or a numpy/cv2
        BGR frame. Raises ``LabawareError`` on network/HTTP/response problems.
        Labware not found is *not* an error: check ``reading.trustworthy``/``warnings``.
        """
        return DeckReading.from_response(self.infer_raw(image))

    def infer_raw(self, image, detail="compact"):
        """POST one frame and return the server's JSON unchanged."""
        filename, payload, content_type = _encode_image(image)
        try:
            r = self.session.post(self.infer_url, params={"detail": detail},
                                  files={"image": (filename, payload, content_type)},
                                  timeout=self.timeout)
        except Exception as error:
            raise LabawareError(f"Could not reach Labaware at {self.infer_url}: {error}") from error
        if not r.ok:
            raise LabawareError(f"Labaware returned HTTP {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except ValueError as error:
            raise LabawareError("Labaware response was not JSON") from error


def _states_from_full(labware):
    """Turn a ``detail=full`` labware list into compact-style states plus slot outlines."""
    states = {"vial_rack": None, "clamp_1": None, "clamp_2": None}
    boxes = []
    for item in labware:
        if not isinstance(item, dict) or item.get("id") not in states or item.get("missing"):
            continue
        slots = item.get("slots") or []
        if item["id"] == "vial_rack":
            grid = [[None] * RACK_COLS for _ in range(RACK_ROWS)]
            for slot in slots:
                grid[slot["row"]][slot["col"]] = slot["state"]
            if any(state is None for row in grid for state in row):
                raise LabawareError("vial_rack full response is missing slots")
            states["vial_rack"] = grid
            for slot in slots:
                boxes.append({"location": RACK_LOCATION, "location_index": slot["row"] * RACK_COLS + slot["col"],
                              "state": slot["state"], "conf": slot.get("conf"), "cell": slot.get("cell")})
        elif slots:
            states[item["id"]] = slots[0]["state"]
            boxes.append({"location": CLAMP_LOCATIONS[item["id"]], "location_index": 0,
                          "state": slots[0]["state"], "conf": slots[0].get("conf"), "cell": slots[0].get("cell")})
    return states, [box for box in boxes if box["cell"]]


def _check_state(state):
    if state not in STATES:
        raise LabawareError(f"Unknown slot state {state!r}")
    return state


def _encode_image(image):
    if isinstance(image, (str, Path)):
        path = Path(image)
        content_type = _IMAGE_TYPES.get(path.suffix.lower())
        if content_type is None:
            raise ValueError(f"Unsupported image type: {path.suffix!r}")
        return path.name, path.read_bytes(), content_type
    if isinstance(image, (bytes, bytearray)):
        if not image:
            raise ValueError("Image bytes are empty")
        return "frame.jpg", bytes(image), _sniff_type(bytes(image))
    if hasattr(image, "shape"):  # numpy / cv2 frame
        import cv2
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Could not JPEG-encode frame")
        return "frame.jpg", buf.tobytes(), "image/jpeg"
    raise TypeError(f"Unsupported image input: {type(image).__name__}")


def _sniff_type(data):
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"
