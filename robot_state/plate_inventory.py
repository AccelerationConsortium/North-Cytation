"""Physical plate/lid identity and placement, independent of hardware drivers.

Plate lists are bottom-to-top. A lid is either placed separately or attached,
never both. Legacy status is converted in memory; saving explicitly migrates it.
"""

from copy import deepcopy
from contextlib import contextmanager
import math
import os
from pathlib import Path
import tempfile
import threading
from uuid import uuid4

import yaml

LID_ALIASES = {"quartz_lid": "quartz", "96_wellplate_lid": "96 WELL PLATE"}
DEFAULT_HOLDERS = {
    "source_stack": "plate", "waste_stack": "plate",
    "pipetting_area": "plate", "cytation_tray": "plate",
    "lid_storage": "lid", "lid_storage_96": "lid",
}


def number(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing or invalid numeric {description}: {value!r}")
    return value


class PlateInventory:
    def __init__(self, data, definitions, positions):
        self.definitions = definitions
        self.position_config = positions
        self.lock = threading.RLock()
        self.legacy = data.get("schema_version") is None
        self.data = self.convert_legacy(data) if self.legacy else deepcopy(data)
        if self.data.get("schema_version") != 2:
            raise ValueError("Unsupported track status schema version")
        self.data.setdefault("revision", 0)
        self.data.setdefault("attachments", {})
        self.data.setdefault("pending_transfer", None)
        for location, kind in self.holders().items():
            self.data["positions"].setdefault(location, [] if kind == "plate" else None)
        self.data["positions"].setdefault("gripper", None)
        self.validate()

    @classmethod
    def load(cls, path, definitions, positions):
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}, definitions, positions)

    def holders(self):
        result = {}
        for name, definition in self.position_config.get("positions", {}).items():
            kind = definition.get("inventory_kind", DEFAULT_HOLDERS.get(name))
            if kind in ("plate", "lid"):
                result[name] = kind
        return result

    def convert_legacy(self, old):
        fields = {"wellplate_type", "num_in_source", "num_in_waste", "active_wellplate_position",
                  "current_gripper_location", "current_gripper_position"}
        data = {key: deepcopy(value) for key, value in old.items() if key not in fields}
        data.update(schema_version=2, revision=0, objects={}, positions={}, attachments={},
                    track_pose={"location": old.get("current_gripper_location", "origin"),
                                **(old.get("current_gripper_position") or {"x": None, "z": None})},
                    pending_transfer=None)
        plate_type = old.get("wellplate_type", "96 WELL PLATE")
        for location, field in (("source_stack", "num_in_source"), ("waste_stack", "num_in_waste")):
            count = old.get(field, 0)
            if type(count) is not int or count < 0:
                raise ValueError(f"Invalid legacy {field}")
            data["positions"][location] = []
            for index in range(count):
                object_id = f"legacy-{location}-{index + 1:03d}"
                data["objects"][object_id] = {"kind": "plate", "type": plate_type}
                data["positions"][location].append(object_id)
        active = old.get("active_wellplate_position")
        if active not in (None, "None", "null"):
            active = "cytation_tray" if active == "cytation" else active
            if active not in {*self.holders(), "gripper"}:
                raise ValueError(f"Review unknown legacy plate position: {active!r}")
            object_id = "legacy-active-001"
            data["objects"][object_id] = {"kind": "plate", "type": plate_type}
            if active == "gripper":
                data["positions"][active] = object_id
            elif self.holders()[active] == "plate":
                data["positions"].setdefault(active, []).append(object_id)
            else:
                raise ValueError(f"Review legacy plate recorded at lid position {active}")
        return data

    @property
    def objects(self):
        return self.data["objects"]

    @property
    def positions(self):
        return self.data["positions"]

    @property
    def attachments(self):
        return self.data["attachments"]

    def contents(self, location):
        value = self.positions[location]
        return list(value) if isinstance(value, list) else ([] if value is None else [value])

    def top(self, location):
        contents = self.contents(location)
        if not contents:
            raise ValueError(f"No object at {location}")
        return contents[-1]

    def plate_type(self, object_id):
        obj = self.objects[object_id]
        return obj["type"] if obj["kind"] == "plate" else obj["plate_type"]

    def locate(self, object_id):
        for location in self.positions:
            if object_id in self.contents(location):
                return location
        for plate, lid in self.attachments.items():
            if lid == object_id:
                return f"attached:{plate}"
        raise ValueError(f"Object {object_id} has no placement")

    def plate_at(self, location="pipetting_area", expected_type=None, plate_id=None, use_lid=None):
        found = self.top(location)
        if self.objects[found]["kind"] != "plate":
            raise ValueError(f"{location} contains a lid, not a plate")
        if plate_id is not None and plate_id != found:
            raise ValueError(f"Expected {plate_id} at {location}; found {found}")
        actual = self.plate_type(found)
        if expected_type is not None and expected_type != actual:
            raise ValueError(f"Plate {found} is {actual}, not expected {expected_type}")
        if use_lid is not None and use_lid != (found in self.attachments):
            raise ValueError(f"Lid expectation does not match plate {found}")
        return found

    def capacity(self, location):
        cfg = self.position_config["positions"][location]
        # null for source/waste preserves the existing height-based capacity;
        # an actual holder count can be supplied without guessing one here.
        return cfg.get("max_wellplates", None if location in ("source_stack", "waste_stack") else 1)

    def check_destination(self, object_id, location):
        if location == "gripper":
            if self.positions[location] is not None:
                raise ValueError("Track gripper is occupied")
            return
        if location not in self.holders():
            raise ValueError(f"{location} is not a configured holding position")
        obj = self.objects[object_id]
        contents = self.contents(location)
        if obj["kind"] == "plate":
            if self.holders()[location] != "plate":
                raise ValueError(f"{location} is a lid holder")
            maximum = self.capacity(location)
            if maximum is not None and len(contents) >= maximum:
                raise ValueError(f"{location} is at its plate capacity ({maximum})")
        elif self.holders()[location] == "plate":
            plate = self.plate_at(location)
            if plate in self.attachments:
                raise ValueError(f"Plate {plate} already has a lid")
        elif contents:
            raise ValueError(f"Lid holder {location} is occupied")

    def thickness_sum(self, object_ids):
        total = 0
        for object_id in object_ids:
            value = number(self.definitions[self.plate_type(object_id)].get("thickness"),
                           f"thickness for supporting plate {object_id}")
            if value < 0:
                raise ValueError("Plate thickness cannot be negative")
            total += value
        return total

    def stack_z(self, location, operation, kind="plate"):
        cfg = self.position_config["positions"][location]
        base = number(cfg.get("z_grab" if operation == "grab" else "z_release"), f"{location} base Z")
        contents = self.contents(location)
        if self.holders()[location] == "plate":
            supports = contents[:-1] if operation == "grab" or kind == "lid" else contents
            z = base - self.thickness_sum(supports)
        else:
            z = base
        # Preserve the existing stack limit, not for Cytation's distinct geometry.
        if location in ("source_stack", "waste_stack") or (self.holders()[location] == "plate" and self.capacity(location) != 1):
            minimum = self.position_config["limits"]["max_stack_height"]
            if not minimum <= z <= base:
                raise ValueError(f"{location} calculated Z {z} exceeds stack height limit {minimum}")
        return z

    @contextmanager
    def edit(self):
        with self.lock:
            before = deepcopy(self.data)
            try:
                yield
                self.validate()
            except BaseException:
                self.data = before
                raise

    def create(self, kind, plate_type):
        if plate_type not in self.definitions or "num_wells" not in self.definitions[plate_type]:
            raise ValueError(f"Unknown plate definition {plate_type}")
        object_id = f"{kind}-{uuid4().hex[:12]}"
        self.objects[object_id] = {"kind": kind, "type" if kind == "plate" else "plate_type": plate_type}
        return object_id

    def _remove_placement(self, object_id):
        location = self.locate(object_id)
        if location.startswith("attached:"):
            del self.attachments[location.split(":", 1)[1]]
        elif isinstance(self.positions[location], list):
            self.positions[location].remove(object_id)
        else:
            self.positions[location] = None

    def _place(self, object_id, destination):
        if destination != "gripper" and self.objects[object_id]["kind"] == "lid" and self.holders()[destination] == "plate":
            self.attachments[self.top(destination)] = object_id
        elif isinstance(self.positions[destination], list):
            self.positions[destination].append(object_id)
        else:
            self.positions[destination] = object_id

    def add(self, location, plate_type, kind="plate", with_lid=False):
        with self.edit():
            object_id = self.create(kind, plate_type)
            self.check_destination(object_id, location)
            self._place(object_id, location)
            if with_lid:
                if kind != "plate":
                    raise ValueError("Only plates have attached lids")
                self.attachments[object_id] = self.create("lid", plate_type)
            return object_id

    def relocate(self, object_id, destination, index=None):
        """Record an operator correction; not a hardware transfer."""
        with self.edit():
            source = self.locate(object_id)
            if source == destination and index is None:
                return
            if (source.startswith("attached:") and self.contents(destination)
                    and self.top(destination) == source.split(":", 1)[1]):
                return
            if index is not None:
                if (self.objects[object_id]["kind"] != "plate"
                        or not isinstance(self.positions.get(destination), list)):
                    raise ValueError("Only plate stacks have insertion layers")
                length = len(self.positions[destination]) - (source == destination)
                if type(index) is not int or not 0 <= index <= length:
                    raise ValueError("Invalid stack insertion layer")
            if source != destination:
                self.check_destination(object_id, destination)
            self._remove_placement(object_id)
            self._place(object_id, destination)
            if index is not None:
                contents = self.positions[destination]
                contents.remove(object_id)
                contents.insert(index, object_id)

    def check_relocation(self, object_id, destination, index=None):
        """Validate a GUI drop without editing the authoritative inventory."""
        draft = PlateInventory(self.data, self.definitions, self.position_config)
        draft.relocate(object_id, destination, index)

    def remove(self, object_id):
        with self.edit():
            self._remove_placement(object_id)
            lid = self.attachments.pop(object_id, None)
            if lid:
                del self.objects[lid]
            del self.objects[object_id]

    def begin(self, object_id, source, destination):
        pending = self.data.get("pending_transfer")
        if pending:
            involved = {source, destination, "gripper"}
            if involved.intersection({pending["source"], pending["destination"]}):
                raise ValueError(f"Reconcile pending transfer of {pending['object_id']} before this move")
        self.data["pending_transfer"] = {"object_id": object_id, "source": source,
                                          "destination": destination, "phase": "intent"}

    def commit_placement(self, object_id, destination):
        with self.edit():
            self._remove_placement(object_id)
            self._place(object_id, destination)
            self.data["pending_transfer"]["phase"] = "placed_before_retreat"

    def finish(self):
        self.data["pending_transfer"] = None

    def validate(self):
        seen = []
        for location, kind in self.holders().items():
            value = self.positions.get(location)
            if kind == "plate" and not isinstance(value, list):
                raise ValueError(f"{location} must contain a bottom-to-top list")
            contents = self.contents(location)
            for object_id in contents:
                if self.objects[object_id]["kind"] != kind:
                    raise ValueError(f"Wrong object kind at {location}")
            if kind == "plate":
                limit = self.capacity(location)
                if limit is not None and (type(limit) is not int or limit < 1):
                    raise ValueError(f"Invalid max_wellplates at {location}")
                # Loading recorded reality must allow the operator to correct
                # an overfull position. New placements enforce count locally.
        for location in self.positions:
            if location not in {*self.holders(), "gripper"}:
                raise ValueError(f"Unknown inventory holding position {location}")
            if location == "gripper" and isinstance(self.positions[location], list):
                raise ValueError("Track gripper holds one object, not a list")
            seen.extend(self.contents(location))
        for plate, lid in self.attachments.items():
            if self.objects[plate]["kind"] != "plate" or self.objects[lid]["kind"] != "lid":
                raise ValueError("Attachments must map a plate to a lid")
            seen.append(lid)
        if len(seen) != len(set(seen)) or set(seen) != set(self.objects):
            raise ValueError("Each object must have exactly one placement")
        for object_id, obj in self.objects.items():
            if obj["kind"] not in ("plate", "lid") or "num_wells" not in self.definitions.get(self.plate_type(object_id), {}):
                raise ValueError(f"Unknown object definition: {object_id}")

    def save(self, path):
        """Atomic compare-and-save, with a short cross-process writer lock."""
        path = Path(path)
        with self.lock:
            lock_path = path.with_suffix(path.suffix + ".lock")
            # Open the persistent lock file; OS locks are released on process exit.
            with lock_path.open("a+b") as lock_file:
                if os.name == "nt":
                    import msvcrt
                    lock_file.seek(0, 2)
                    if lock_file.tell() == 0:
                        lock_file.write(b"0")
                        lock_file.flush()
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    current = yaml.safe_load(path.read_text()) if path.exists() else {}
                    if current and current.get("revision", 0) != self.data["revision"]:
                        raise ValueError("Track state changed on disk; reload before saving")
                    self.validate()
                    if current and current.get("schema_version") is None:
                        backup = path.with_suffix(path.suffix + ".legacy.bak")
                        if not backup.exists():
                            backup.write_bytes(path.read_bytes())
                    saved = deepcopy(self.data)
                    saved["revision"] += 1
                    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
                    try:
                        with os.fdopen(fd, "w", encoding="utf-8") as stream:
                            yaml.safe_dump(saved, stream, sort_keys=False)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary, path)
                    finally:
                        if os.path.exists(temporary):
                            os.unlink(temporary)
                    self.data["revision"] = saved["revision"]
                    self.legacy = False
                finally:
                    if os.name == "nt":
                        lock_file.seek(0)
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(lock_file, fcntl.LOCK_UN)
