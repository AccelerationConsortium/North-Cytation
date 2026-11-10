"""Read ordered workflow/config pairs without importing or executing workflows."""

from pathlib import Path
import hashlib
import json
import os
from uuid import uuid4

import yaml


def write_config_list(path, jobs):
    """Save ordered row snapshots and publish an Import Queue compatible list.

    Each save owns fresh copies, so later row edits and repeated saves cannot
    change an earlier saved queue. Resource paths keep repository semantics.
    """
    path = Path(path).resolve()
    snapshots = []
    for index, (name, config_path) in enumerate(jobs, 1):
        if not isinstance(name, str) or not name:
            raise ValueError(f"Select a workflow for row {index}.")
        if not config_path:
            raise ValueError(f"Row {index} has no configuration. Review Setup first.")
        content = Path(config_path).read_bytes()
        if not isinstance(yaml.safe_load(content), dict):
            raise ValueError(f"Row {index}: workflow config must be a YAML mapping.")
        snapshots.append((name, content))
    if not snapshots:
        raise ValueError("Add at least one workflow before saving the queue.")
    folder = path.parent / f"{path.stem}_configs_{uuid4().hex}"
    folder.mkdir()
    entries = []
    for index, (name, content) in enumerate(snapshots, 1):
        config = folder / f"{index:03d}.yaml"
        config.write_bytes(content)
        entries.append({"workflow": name, "config_file": config.relative_to(path.parent).as_posix()})
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


PLANNED_KEYS = ("experiment_id", "config_sha256", "label")
# Keys the lab agent leaves out of a config's fingerprint (see its scheduling/queue_index.py).
UNHASHED_KEYS = {"INPUT_VIAL_STATUS_FILE", "SIMULATE"}


def read_config_entries(path, workflow_names):
    """Validate every entry, then return one dict per job.

    Each dict has workflow, config_path, and the optional planner keys
    experiment_id, config_sha256 and label (None when absent). Relative config
    paths are resolved beside the list file. Config contents retain the normal
    workflow convention: their own paths are repository-relative.
    """
    path = Path(path).resolve()
    entries = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Config list must be a nonempty YAML list of workflow/config_file pairs.")
    allowed = {"workflow", "config_file", *PLANNED_KEYS}
    jobs = []
    for index, entry in enumerate(entries, 1):
        prefix = f"Entry {index}"
        if not isinstance(entry, dict) or not {"workflow", "config_file"} <= set(entry) or set(entry) - allowed:
            raise ValueError(f"{prefix} must contain workflow and config_file, plus optionally "
                             f"{', '.join(PLANNED_KEYS)}.")
        name, config_file = entry["workflow"], entry["config_file"]
        if not isinstance(name, str) or name not in workflow_names:
            raise ValueError(f"{prefix}: unknown workflow {name!r}.")
        if not isinstance(config_file, str) or not config_file.strip():
            raise ValueError(f"{prefix}: config_file must be a nonempty path.")
        for key in PLANNED_KEYS:
            if entry.get(key) is not None and not isinstance(entry[key], str):
                raise ValueError(f"{prefix}: {key} must be text or null.")
        config_path = (path.parent / config_file).resolve()
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"{prefix}: cannot read config {config_path}: {error}") from error
        if not isinstance(config, dict):
            raise ValueError(f"{prefix}: workflow config must be a YAML mapping: {config_path}")
        jobs.append({"workflow": name, "config_path": config_path,
                     **{key: entry.get(key) for key in PLANNED_KEYS}})
    return jobs


def read_config_list(path, workflow_names):
    """Validate every entry before returning (workflow name, config path) pairs."""
    return [(job["workflow"], job["config_path"]) for job in read_config_entries(path, workflow_names)]


def config_fingerprint(config):
    """Identify an experiment's settings exactly as the lab agent does.

    Must match lab_slack_agent/scheduling/queue_index.py config_fingerprint.
    """
    payload = {k: v for k, v in (config or {}).items() if k not in UNHASHED_KEYS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def live_experiment_id(planned_id, planned_sha256, config):
    """Return the planned ID while the config is unchanged, otherwise a derived or random ID."""
    if not planned_id:
        return uuid4().hex
    fingerprint = config_fingerprint(config)
    if fingerprint == planned_sha256:
        return planned_id
    return f"{planned_id}-changed-{fingerprint[:8]}"


def default_queue_index_path():
    override = os.environ.get("LAB_AGENT_QUEUE_INDEX")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "lab_slack_agent" / "experiment_queue" / "queues" / "index.json"


def read_planned_consumables(queue_file):
    """Summarize wellplates and tips from the manifest.json the lab agent writes beside a queue.

    Returns None when there is no readable manifest; this is only an operator reminder.
    """
    try:
        manifest = json.loads((Path(queue_file).parent / "manifest.json").read_text(encoding="utf-8-sig"))
        jobs = manifest["jobs"]
        plates = [int(job.get("plates") or 0) for job in jobs]
        tips = {size: sum(int((job.get("tips") or {}).get(size) or 0) for job in jobs) for size in ("small", "large")}
        plate_types = [str(kind) for kind in manifest.get("plate_types") or []]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return {"plates_per_job": plates, "plate_types": plate_types, "tips": tips}


def read_planned_queues(index_path=None):
    """Return the lab agent's per-setup-day queue entries, or [] if it has published none."""
    path = Path(index_path) if index_path is not None else default_queue_index_path()
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return []
    try:
        data = json.loads(text)
    except ValueError as error:
        raise ValueError(f"Cannot read lab agent queue index {path}: {error}") from error
    queues = data.get("queues") if isinstance(data, dict) else None
    if not isinstance(queues, list):
        raise ValueError(f"Lab agent queue index {path} has no queues list.")
    return [entry for entry in queues if isinstance(entry, dict)]
