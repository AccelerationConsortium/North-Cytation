"""Pre-run deck check: photograph the north deck, read it with Labaware, compare
with the queue's vial files. No Qt here, so it can be tested and reused.

Plan: docs/plans/scheduler_deck_check_plan.md. Never raises for camera or server
problems; those become a ``comms_failure`` outcome the operator can continue past.
"""
import csv
import json
import shutil
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import yaml

from tools.labaware_client import (CLOSED_CAP, EMPTY, OPEN_CAP, RACK_LOCATION, DeckReading,
                                   LabawareClient)

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = REPO_ROOT / "robot_state" / "deck_monitoring.yaml"
DEFAULT_SETTINGS = {
    "camera_folder": str(Path.home() / "Desktop" / "camera-control"),
    "capture_timeout_s": 20,
    "labaware_url": "http://sdl4-not-for-gaming:8010",
    "labaware_app_id": "north_deck",
    "request_timeout_s": 30,
    "display_rotate_180": True,
    "min_vial_confidence": 0.9,
}
OBSERVED_LOCATIONS = {RACK_LOCATION: 48, "clamp": 1, "clamp_2": 1}
INDEX_FIELDS = ["timestamp", "outcome", "decision", "issues", "camera_wrong", "history_id", "error", "note", "folder"]
_CAP_NAMES = {OPEN_CAP: "open cap", CLOSED_CAP: "closed cap", EMPTY: "empty", None: "not detected"}


def load_settings(path=SETTINGS_FILE):
    settings = dict(DEFAULT_SETTINGS)
    try:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            settings.update({k: v for k, v in loaded.items() if v is not None})
    except (OSError, yaml.YAMLError):
        pass  # defaults; a bad settings file shows up as a comms failure if it matters
    return settings


def position_label(location, index):
    """Same style as the vial files and Vial Layout tab: location [location_index]."""
    return f"{location} [{index}]"


def _a(state):
    name = _CAP_NAMES[state]
    return ("an " if name[0] in "aeiou" else "a ") + name


def expected_state(record):
    """Camera state a vial-file row should produce; ``None`` means presence-only (uncapped/unknown)."""
    capped = str(record.get("capped", "")).strip().casefold()
    cap_type = str(record.get("cap_type", "")).strip().casefold()
    if capped != "true":
        return None  # uncapped vials are not "open-cap"; the model has no uncapped state
    return {"open": OPEN_CAP, "closed": CLOSED_CAP}.get(cap_type)


@dataclass
class Issue:
    kind: str           # missing | unexpected | wrong_cap | not_seen | server_warning
    location: str | None
    location_index: int | None
    expected: str | None   # vial name
    expected_state: str | None
    observed: str | None
    message: str
    camera_wrong: bool = False   # operator says the camera misread this one (training example)

    @property
    def position(self):
        if self.location is None:
            return "-"
        if self.location_index is None:
            return self.location
        return position_label(self.location, self.location_index)


@dataclass
class DeckCheck:
    outcome: str                      # match | mismatch | comms_failure
    timestamp: str
    folder: str | None = None
    issues: list = field(default_factory=list)
    notes: list = field(default_factory=list)      # informational, never a mismatch
    checked: int = 0                                # expected vials the camera could verify
    unchecked: list = field(default_factory=list)  # expected vials at locations the camera can't see
    error: str | None = None
    history_id: int | None = None
    frame: str | None = None          # exactly as captured and sent to the server
    annotated: str | None = None      # display copy: slot boxes drawn, rotated to the operator's view
    rotate_display: bool = True       # rotate the raw frame if there is no annotated copy
    decision: str = "pending"
    note: str = ""

    def summary(self):
        if self.outcome == "comms_failure":
            return f"Deck check could not run: {self.error}"
        text = (f"{self.checked} expected vial(s) checked; "
                + ("deck matches the vial files." if self.outcome == "match"
                   else f"{len(self.issues)} issue(s) found."))
        if self.unchecked:
            text += f" {len(self.unchecked)} vial(s) at locations the camera can't see were not checked."
        return text


def compare(occupancy, reading):
    """Compare ``read_vial_occupancy`` output with a ``DeckReading``. Returns ``(issues, notes, checked, unchecked)``."""
    issues, notes, unchecked = [], [], []
    checked = 0
    expected = {key: claims[0] for key, claims in occupancy.items() if claims}
    for (location, index), record in sorted(expected.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        if location not in OBSERVED_LOCATIONS:
            unchecked.append(f"{record['vial_name']} at {location} [{index}]")
    not_seen = set()
    for location, size in OBSERVED_LOCATIONS.items():
        for index in range(size):
            record = expected.get((location, index))
            observed = reading.state_at(location, index)
            if observed is None:
                if record is not None:
                    not_seen.add(location)
                continue
            if record is None:
                if observed != EMPTY:
                    issues.append(Issue("unexpected", location, index, None, None, observed,
                                        f"Camera sees a vial ({_CAP_NAMES[observed]}) but no vial file has one here. "
                                        "Remove it or add it to the vial file."))
                continue
            checked += 1
            want = expected_state(record)
            name = record["vial_name"]
            if observed == EMPTY:
                issues.append(Issue("missing", location, index, name, want, observed,
                                    f"Expected '{name}'" + (f" ({_CAP_NAMES[want]})" if want else "")
                                    + " but the slot is empty. Place the vial or update the vial file."))
            elif want is None:
                notes.append(f"{position_label(location, index)}: '{name}' is uncapped or has no cap type; "
                             "presence checked, cap not checked.")
            elif observed != want:
                issues.append(Issue("wrong_cap", location, index, name, want, observed,
                                    f"'{name}' should have {_a(want)} but the camera sees {_a(observed)}. "
                                    "Swap the cap or update cap_type."))
    for location in sorted(not_seen):
        count = sum(1 for (loc, _), _r in expected.items() if loc == location)
        issues.append(Issue("not_seen", location, None, None, None, None,
                            f"{location} was not detected in the photo, so its {count} expected vial(s) "
                            "could not be checked. Check the camera view isn't blocked."))
    for warning in reading.warnings:
        issues.append(Issue("server_warning", None, None, None, None, None, f"Labaware warning: {warning}"))
    _suggest_moves(issues)
    return issues, notes, checked, unchecked


def _suggest_moves(issues):
    unexpected = [i for i in issues if i.kind == "unexpected"]
    for issue in issues:
        if issue.kind != "missing":
            continue
        candidates = [u for u in unexpected if issue.expected_state in (None, u.observed)]
        if candidates and len(candidates) <= 3:
            where = ", ".join(u.position for u in candidates)
            issue.message += f" Possibly moved to {where}."


def run_check(occupancy, records_root, settings=None, capture=None, client=None, now=None):
    """Capture, infer, compare and save a record. Never raises for camera/server problems."""
    settings = settings or load_settings()
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    folder = Path(records_root) / stamp
    folder.mkdir(parents=True, exist_ok=True)
    check = DeckCheck(outcome="comms_failure", timestamp=stamp, folder=str(folder),
                      rotate_display=bool(settings.get("display_rotate_180", True)))
    _write_expected(folder / "expected.csv", occupancy)
    frame = folder / "frame.jpg"
    try:
        if capture is None:
            from tools.c110_camera import capture_photo as capture
        capture(frame, settings["camera_folder"], settings["capture_timeout_s"])
        check.frame = str(frame)
    except Exception as error:
        check.error = f"camera capture failed: {error}"
        save_report(check)
        return check
    try:
        client = client or LabawareClient(settings["labaware_url"], settings["labaware_app_id"],
                                          timeout=settings["request_timeout_s"])
        response = client.infer_raw(frame, detail="full")  # full = per-slot outlines for the overlay
        (folder / "response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
        reading = DeckReading.from_response(response)
    except Exception as error:
        check.error = f"Labaware server: {error}"
        save_report(check)
        return check
    check.history_id = reading.history_id
    reading, unsure = treat_unsure_as_empty(reading, float(settings.get("min_vial_confidence", 0.9)))
    check.issues, check.notes, check.checked, check.unchecked = compare(occupancy, reading)
    check.notes += unsure
    check.outcome = "mismatch" if check.issues else "match"
    try:
        check.annotated = str(annotate(frame, reading.boxes, check.issues, folder / "annotated.jpg",
                                       rotate=check.rotate_display))
    except Exception:
        check.annotated = None  # overlay is cosmetic; the dialog falls back to the raw frame
    save_report(check)
    return check


def treat_unsure_as_empty(reading, threshold):
    """Vial readings below ``threshold`` confidence become empty (the model's phantom caps).

    Returns ``(adjusted_reading, notes)``. Only applies when the response had per-slot
    confidence (detail=full); empty readings are never changed.
    """
    rack = list(reading.rack) if reading.rack is not None else None
    clamps = dict(reading.clamps)
    boxes, notes = [], []
    for box in reading.boxes:
        conf = box.get("conf")
        if box["state"] != EMPTY and conf is not None and conf < threshold:
            if box["location"] == RACK_LOCATION and rack is not None:
                rack[box["location_index"]] = EMPTY
            elif box["location"] in clamps:
                clamps[box["location"]] = EMPTY
            notes.append(f"{position_label(box['location'], box['location_index'])}: camera unsure "
                         f"({box['state']}, confidence {conf:.2f}); treated as empty.")
            box = {**box, "state": EMPTY}
        boxes.append(box)
    return replace(reading, rack=rack, clamps=clamps, boxes=boxes), notes


ISSUE_BGR, OK_BGR, EMPTY_BGR =(0, 140, 255), (70, 190, 70), (170, 170, 170)


def annotate(frame_path, boxes, issues, out_path, rotate=True, scale=2):
    """Draw slot outlines (orange = issue, green = ok vial, grey = empty) and save a display copy.

    ``rotate`` turns the image 180 degrees so it matches the operator's view of the deck;
    labels are drawn after rotating so they stay upright.
    """
    import cv2
    import numpy as np

    image = cv2.imread(str(frame_path))
    if image is None:
        raise ValueError(f"Cannot read {frame_path}")
    height, width = image.shape[:2]
    image = cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    if rotate:
        image = cv2.rotate(image, cv2.ROTATE_180)

    def point(x, y):
        return (int(round((width - x if rotate else x) * scale)), int(round((height - y if rotate else y) * scale)))

    flagged = {(issue.location, issue.location_index) for issue in issues if issue.location_index is not None}
    labels = []
    for box in sorted(boxes, key=lambda b: (b["location"], b["location_index"]) in flagged):
        key = (box["location"], box["location_index"])
        corners = [point(x, y) for x, y in box["cell"]]
        if key in flagged:
            colour, thickness = ISSUE_BGR, 3
            text = str(box["location_index"]) if box["location"] == RACK_LOCATION else box["location"]
            labels.append((text, min(c[0] for c in corners), min(c[1] for c in corners)))
        elif box["state"] != EMPTY:
            colour, thickness = OK_BGR, 2
        else:
            colour, thickness = EMPTY_BGR, 1
        cv2.polylines(image, [np.array(corners, dtype=np.int32)], True, colour, thickness, cv2.LINE_AA)
    for text, x, y in labels:  # draw labels last so boxes don't cover them
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(y, th + 4)
        cv2.rectangle(image, (x, y - th - 4), (x + tw + 4, y), ISSUE_BGR, -1)
        cv2.putText(image, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(out_path), image):
        raise OSError(f"Cannot write {out_path}")
    return Path(out_path)


def save_report(check):
    data = asdict(check)
    (Path(check.folder) / "report.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def all_camera_wrong(check):
    """True when every listed issue was marked as a camera error (so the deck itself is fine)."""
    return bool(check.issues) and all(issue.camera_wrong for issue in check.issues)


def finalize(check, decision, note="", camera_wrong=()):
    """Record the operator's decision in report.json and the deck_checks index.

    ``camera_wrong``: indexes into ``check.issues`` the operator marked as camera errors.
    Those checks also get their photo copied to ``camera_errors/`` for later training.
    """
    check.decision, check.note = decision, note.strip()
    for number in camera_wrong:
        check.issues[number].camera_wrong = True
    save_report(check)
    root = Path(check.folder).parent
    flagged = [issue for issue in check.issues if issue.camera_wrong]
    if flagged:
        _save_camera_error(check, flagged, root / "camera_errors")
    _append_index(root / "index.csv", {
        "timestamp": check.timestamp, "outcome": check.outcome, "decision": decision,
        "issues": len(check.issues), "camera_wrong": len(flagged), "history_id": check.history_id,
        "error": check.error or "", "note": check.note, "folder": check.folder})


def _save_camera_error(check, flagged, folder):
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{check.timestamp}_{check.history_id}"
    if check.frame and Path(check.frame).is_file():
        shutil.copyfile(check.frame, folder / f"{name}.jpg")
    lines = [f"Labaware history id: {check.history_id}", f"Check folder: {check.folder}",
             "Camera misread:"] + [f"  {issue.position}: camera said {issue.observed}; {issue.kind}" for issue in flagged]
    if check.note:
        lines.append(f"Note: {check.note}")
    (folder / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_index(index, row):
    """Append one row; upgrade an index written with an older column list."""
    if index.exists():
        with index.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields, old_rows = reader.fieldnames, list(reader)
        if fields != INDEX_FIELDS:
            with index.open("w", newline="", encoding="utf-8") as out:
                writer = csv.DictWriter(out, fieldnames=INDEX_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(old_rows)
    new = not index.exists()
    with index.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def mark_used(folder, live_root):
    """Note in the report which live run started from this check."""
    path = Path(folder) / "report.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["live_run"] = str(live_root)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass


def _write_expected(path, occupancy):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["location", "location_index", "vial_name", "capped", "cap_type", "source_file"])
        for (location, index), claims in sorted(occupancy.items(), key=lambda item: (item[0][0], int(item[0][1]))):
            for record in claims:
                writer.writerow([location, index, record.get("vial_name"), record.get("capped", ""),
                                 record.get("cap_type", ""), record.get("source_file", "")])
