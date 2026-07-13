"""
YAML round-trip helper.

Uses ruamel.yaml so that writing back a loaded config file preserves
comments, key order, and formatting exactly as authored.

Usage:
    from sdl_pipette_calibration.yaml_io import load_yaml, dump_yaml

    data = load_yaml("experiment_config.yaml")
    data["experiment"]["liquid"] = "ethanol"
    dump_yaml(data, "experiment_config.yaml")
"""

from pathlib import Path
from typing import Union

from ruamel.yaml import YAML

_yaml = YAML()
_yaml.preserve_quotes = True
# Indent 2 spaces for mappings, 4 for block sequences (matches PyYAML default look)
_yaml.best_map_flow_style = False
_yaml.default_flow_style = False


def load_yaml(path: Union[str, Path]):
    """Load a YAML file and return a CommentedMap (preserves comments and key order)."""
    with open(path, "r", encoding="utf-8") as f:
        return _yaml.load(f)


def dump_yaml(data, path: Union[str, Path]) -> None:
    """Write data back to a YAML file, preserving comments and key order."""
    with open(path, "w", encoding="utf-8") as f:
        _yaml.dump(data, f)
