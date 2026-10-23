"""Read ordered workflow/config pairs without importing or executing workflows."""

from pathlib import Path

import yaml


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
