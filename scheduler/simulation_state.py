"""Private state copies for scheduler simulations; never restore them to live files."""

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_ROOT = Path(__file__).resolve().parent / "state"
SESSION_ENV = "NORTH_SCHEDULER_SIMULATION_STATE"
_controllers = []
_diagnostic_handler = None


def attach_diagnostics(logger):
    if _diagnostic_handler is not None and _diagnostic_handler not in logger.handlers:
        logger.addHandler(_diagnostic_handler)


def _session_root():
    value = os.environ.get(SESSION_ENV)
    if not value:
        return None
    root = Path(value).resolve()
    if not root.is_relative_to(STATE_ROOT.resolve()) or root == STATE_ROOT.resolve():
        raise ValueError("Scheduler simulation state must be inside scheduler/state/<session>.")
    manifest = json.loads((root / "session.json").read_text(encoding="utf-8"))
    if manifest["simulation"] is not True:
        raise ValueError("Not a scheduler simulation session.")
    return root


def _checked_file(root, value):
    path = Path(value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"State path is not an existing private simulation file: {path}")
    return path


def validate_launch(simulate, vial_file):
    root = _session_root()
    if root is None:
        return
    if simulate is not True:
        raise ValueError("Scheduler simulation cannot initialize live hardware.")
    if vial_file is None:
        return
    manifest = json.loads((root / "session.json").read_text(encoding="utf-8"))
    vial = _checked_file(root, vial_file)
    if str(vial) not in manifest["vial_files"].values():
        raise ValueError("Vial file is not registered in this simulation session.")


def configure_controller(controller, kind):
    root = _session_root()
    if root is None:
        return
    if controller.simulate is not True:
        raise ValueError("Private scheduler state is only available during simulation.")
    if kind == "robot":
        validate_launch(controller.simulate, controller.VIAL_FILE)
        controller.ROBOT_STATUS_FILE = str(_checked_file(root, root / "robot_status.yaml"))
    elif kind == "track":
        controller.TRACK_STATUS_FILE = str(_checked_file(root, root / "track_status.yaml"))
    else:
        raise ValueError(f"Unknown controller kind: {kind}")
    controller._scheduler_state_root = str(root)


def can_save(controller, *paths):
    """Default simulation remains nonpersistent; opted-in writes stay private."""
    value = getattr(controller, "_scheduler_state_root", None)
    if value is None:
        return not controller.simulate
    root = _session_root()
    if root is None or root != Path(value).resolve() or controller.simulate is not True:
        raise ValueError("Simulation session changed; refusing state write.")
    for path in paths:
        if path is not None:
            _checked_file(root, path)
    return True


def register_coordinator(coordinator):
    if _session_root() is not None:
        _controllers.append(coordinator)


def create_session(vial_files):
    """Copy reviewed starting inputs once. Later jobs use updated copies."""
    sources = {str(Path(value).resolve()) for value in vial_files if value is not None}
    for path in sources:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    state_files = [REPO_ROOT / "robot_state" / name for name in ("robot_status.yaml", "track_status.yaml")]
    for path in state_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    root = STATE_ROOT / uuid4().hex
    root.mkdir(parents=True)
    try:
        for path in state_files:
            shutil.copy2(path, root / path.name)
        (root / "vials").mkdir()
        mapping = {}
        for index, source in enumerate(sorted(sources)):
            target = root / "vials" / f"{index}_{Path(source).name}"
            shutil.copy2(source, target)
            mapping[source] = str(target.resolve())
        (root / "session.json").write_text(
            json.dumps({"simulation": True, "vial_files": mapping}, indent=2), encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(root)
        raise
    return root


def simulated_config(root, config):
    """Copy saved config and substitute only simulation mode and private vial path."""
    from copy import deepcopy
    root = Path(root).resolve()
    manifest = json.loads((root / "session.json").read_text(encoding="utf-8"))
    selected = deepcopy(config)
    if selected["INPUT_VIAL_STATUS_FILE"] is not None:
        source = str(Path(selected["INPUT_VIAL_STATUS_FILE"]).resolve())
        selected["INPUT_VIAL_STATUS_FILE"] = manifest["vial_files"][source]
    selected["SIMULATE"] = True
    return selected


@contextmanager
def workflow_state(root, job_id):
    """Activate private state for one child job, then save its END state.

    A future runner wraps execute(config=simulated_config(...), show_gui=False)
    with this context. Reuse root across jobs, never create_session between jobs.
    Snapshots marked failed are diagnostics, not approved starting points.
    """
    if os.environ.get(SESSION_ENV) or _controllers:
        raise RuntimeError("A scheduler simulation job is already active.")
    if not job_id or Path(job_id).name != job_id or job_id in {".", ".."}:
        raise ValueError("Job ID must be a simple directory name.")
    os.environ[SESSION_ENV] = str(Path(root).resolve())
    succeeded = False
    try:
        _session_root()
        yield
        succeeded = True
    finally:
        try:
            if _controllers:
                for coordinator in _controllers:
                    robot = getattr(coordinator, "nr_robot", None)
                    track = getattr(coordinator, "nr_track", None)
                    if robot is not None:
                        robot.save_robot_status()
                    if track is not None:
                        track.save_track_status()
                session = _session_root()
                snapshot = session / "end_states" / job_id
                snapshot.mkdir(parents=True, exist_ok=False)
                for name in ("robot_status.yaml", "track_status.yaml"):
                    shutil.copy2(session / name, snapshot / name)
                shutil.copytree(session / "vials", snapshot / "vials")
                (snapshot / "result.json").write_text(
                    json.dumps({"completed": succeeded}), encoding="utf-8"
                )
        except Exception:
            if succeeded:
                raise
            if _controllers:
                _controllers[-1].logger.exception("Could not save failed simulation end state")
        finally:
            _controllers.clear()
            os.environ.pop(SESSION_ENV, None)