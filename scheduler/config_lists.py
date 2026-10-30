"""Read ordered workflow/config pairs without importing or executing workflows."""

from pathlib import Path
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


def read_config_list(path, workflow_names):
    """Validate every entry before returning (workflow name, config path) pairs.

    Relative config paths are resolved beside the list file. Config contents
    retain the normal workflow convention: their own paths are repository-relative.
    """
    path = Path(path).resolve()
    entries = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Config list must be a nonempty YAML list of workflow/config_file pairs.")
    jobs = []
    for index, entry in enumerate(entries, 1):
        prefix = f"Entry {index}"
        if not isinstance(entry, dict) or set(entry) != {"workflow", "config_file"}:
            raise ValueError(f"{prefix} must contain exactly workflow and config_file.")
        name, config_file = entry["workflow"], entry["config_file"]
        if not isinstance(name, str) or name not in workflow_names:
            raise ValueError(f"{prefix}: unknown workflow {name!r}.")
        if not isinstance(config_file, str) or not config_file.strip():
            raise ValueError(f"{prefix}: config_file must be a nonempty path.")
        config_path = (path.parent / config_file).resolve()
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"{prefix}: cannot read config {config_path}: {error}") from error
        if not isinstance(config, dict):
            raise ValueError(f"{prefix}: workflow config must be a YAML mapping: {config_path}")
        jobs.append((name, config_path))
    return jobs
