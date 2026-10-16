"""Two runs per plate, remembered in a separate workflow-owned YAML file."""
from pathlib import Path

import yaml


STATE_FILE = "fluorescence_plate.yaml"


def read_marker(root):
    path = Path(root) / STATE_FILE
    return yaml.safe_load(path.read_text()) if path.exists() else None


def ready_for_second_run(track_state, marker):
    return (
        isinstance(marker, dict)
        and marker.get("status") == "second_half_ready"
        and marker.get("plate_type") == track_state.get("wellplate_type")
        and marker.get("num_in_source") == track_state.get("num_in_source")
        and marker.get("num_in_waste") == track_state.get("num_in_waste")
        and track_state.get("active_wellplate_position") == "pipetting_area"
    )


def _marker(lash):
    if "_two_run_plate_marker" not in lash.__dict__:
        lash._two_run_plate_marker = read_marker(Path(lash.nr_track.TRACK_STATUS_FILE).parent)
    return lash._two_run_plate_marker


def _save(lash, marker):
    lash._two_run_plate_marker = marker
    track = lash.nr_track
    path = Path(track.TRACK_STATUS_FILE).parent / STATE_FILE
    if lash.simulate:
        if "_scheduler_state_root" not in track.__dict__:
            return
        from scheduler.simulation_state import can_save
        if not can_save(track, path):
            return
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(marker))
    temporary.replace(path)


def next_half(lash):
    track = lash.nr_track
    marker = _marker(lash)
    if track.ACTIVE_WELLPLATE_POSITION in (None, "None"):
        return 0
    if ready_for_second_run({"wellplate_type": track.CURRENT_WP_TYPE,
                             "num_in_source": track.NUM_SOURCE,
                             "num_in_waste": track.NUM_WASTE,
                             "active_wellplate_position": track.ACTIVE_WELLPLATE_POSITION}, marker):
        return 1
    raise ValueError("Cannot reuse plate: unfinished run or plate state mismatch. Retire the plate before restarting.")


def begin(lash, half, run_id):
    if half == 0:
        lash.grab_new_wellplate()
    # Save before dispensing; an interrupted run must not reuse partially filled wells.
    _save(lash, {"status": "in_progress", "half": half,
                 "plate_type": lash.nr_track.CURRENT_WP_TYPE, "run_id": run_id,
                 "num_in_source": lash.nr_track.NUM_SOURCE,
                 "num_in_waste": lash.nr_track.NUM_WASTE})


def finish(lash, half):
    if half == 0:
        marker = dict(_marker(lash), status="second_half_ready")
        _save(lash, marker)
        lash.logger.info("First half complete; leaving plate for the next fluorescence run")
    else:
        lash.discard_used_wellplate()
        _save(lash, None)
