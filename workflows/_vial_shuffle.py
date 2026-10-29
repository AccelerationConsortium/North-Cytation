"""Plan collision-free subset shuffles without importing hardware libraries.

The private filename keeps this helper out of the scheduler's workflow list.
"""
import random

RACK = "main_8mL_rack"
DESTINATIONS = (RACK, "heater", "photoreactor_array", "clamp")


def positions(frame):
    return {row.vial_name: (row.location, int(row.location_index))
            for row in frame.itertuples()}


def eligible_names(frame):
    return [row.vial_name for row in frame.itertuples()
            if row.vial_type == "8_mL" and str(row.capped).lower() == "true"]


def available_slots(robot):
    slots = []
    for name in DESTINATIONS:
        definition = robot.VIAL_POSITIONS.get(name, {})
        if definition.get("rack_present") and definition.get("vials_movable"):
            slots.extend((name, index) for index in range(definition["rack_size"]))
    if not any(slot[0] == RACK for slot in slots):
        raise ValueError("The main 8 mL rack must be present and movable")
    return slots


def validate_layout(frame, slots):
    required = {"vial_name", "vial_type", "capped", "location", "location_index"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Vial file requires columns: {sorted(required)}")
    if frame.vial_name.isna().any() or not frame.vial_name.is_unique:
        raise ValueError("Vial names must be present and unique")
    if any(float(value) != int(value) for value in frame.location_index):
        raise ValueError("Vial position indices must be integers")
    current = positions(frame)
    if len(set(current.values())) != len(current):
        raise ValueError("Multiple vials occupy the same position")
    names = eligible_names(frame)
    if not names:
        raise ValueError("No capped 8 mL vials are available")
    if any(current[name] not in slots for name in names):
        raise ValueError("Participating vials must start in an enabled shuffle destination")
    if not set(slots) - set(current.values()):
        raise ValueError("Shuffling requires at least one empty destination for temporary moves")
    return names


def plan_shuffle(current, names, slots, min_moves, max_moves, min_off, max_off, rng=None):
    """Return new positions for exactly N distinct participating vials.

    Bounds on off-rack occupancy count participating vials. Fixed vials still
    block slots. Temporary cycle-breaking moves may add one robot transfer.
    """
    rng = rng or random.Random()
    n = rng.randint(min(min_moves, len(names)), min(max_moves, len(names)))
    # Vary both the subset and reachable off-rack target; never silently violate bounds.
    for _ in range(500):
        selected = rng.sample(names, n)
        fixed = {slot for name, slot in current.items() if name not in selected}
        rack = [slot for slot in slots if slot[0] == RACK and slot not in fixed]
        off = [slot for slot in slots if slot[0] != RACK and slot not in fixed]
        fixed_off = sum(current[name][0] != RACK for name in names if name not in selected)
        low = max(0, min_off - fixed_off, n - len(rack))
        high = min(n, max_off - fixed_off, len(off))
        if low > high:
            continue
        off_count = rng.randint(low, high)
        targets = rng.sample(off, off_count) + rng.sample(rack, n - off_count)
        # A derangement avoids counting unchanged vials toward the move limit.
        for _ in range(30):
            rng.shuffle(targets)
            if all(current[name] != target for name, target in zip(selected, targets)):
                return dict(zip(selected, targets))
    raise ValueError("Cannot find a shuffle within the move/off-rack limits; adjust the limits")


def plan_transfers(current, targets, slots):
    """Resolve swaps and longer cycles through an empty buffer before moving hardware."""
    current = dict(current)
    pending = {name: slot for name, slot in targets.items() if current[name] != slot}
    transfers = []
    while pending:
        occupied = set(current.values())
        ready = next((name for name, slot in pending.items() if slot not in occupied), None)
        if ready is not None:
            destination = pending.pop(ready)
        else:
            buffer = next((slot for slot in slots if slot not in occupied), None)
            if buffer is None:
                raise ValueError("No empty buffer position to resolve vial swap")
            ready = next(iter(pending))
            destination = buffer
        transfers.append((ready, destination))
        current[ready] = destination
    return transfers


def apply_transfers(robot, targets, slots):
    transfers = plan_transfers(positions(robot.VIAL_DF), targets, slots)
    for name, destination in transfers:
        current = positions(robot.VIAL_DF)
        if destination in current.values():
            raise RuntimeError(f"Destination became occupied: {destination}")
        robot.move_vial_to_location(name, *destination)
        if positions(robot.VIAL_DF)[name] != destination:
            raise RuntimeError(f"Robot did not record the expected move for {name}")
        if robot.GRIPPER_STATUS is not None:
            raise RuntimeError("Robot still holds an object after vial transfer")
    return len(transfers)
