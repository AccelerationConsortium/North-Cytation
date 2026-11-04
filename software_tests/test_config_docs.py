"""Every scheduler job's config keys have a planning annotation.

The lab Slack agent reads _CONFIG_DOCS with ast (no import), so entries must be
plain literals and cover the workflow's config keys exactly.
"""

import ast
import logging
from pathlib import Path

import pytest

from scheduler.workflow_modules import workflow_paths
from workflows.helpers._planning_markers import mark_clock_start, mark_wait, mark_wait_until

ROOT = Path(__file__).resolve().parents[1]


def _entrypoints():
    """Scheduler jobs (workflows plus selectable diagnostics) and the template."""
    paths = {**workflow_paths(ROOT), "workflow_template": ROOT / "workflows" / "workflow_template.py"}
    for name, path in sorted(paths.items()):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(isinstance(node, ast.FunctionDef) and node.name == "execute" for node in tree.body):
            yield name, path


ENTRYPOINTS = dict(_entrypoints())
ENTRYPOINTS = dict(_entrypoints())
RESOURCES = {"duration", "tips", "plates", "vials", "liquids"}
SHARES = {"vial_layout", "input_file", "dye", "plate_type"}
FIELDS = {"description", "unit", "choices", "affects", "effect", "shares", "ask"}


def _assignments(path):
    """Module-level assignments, plus constants imported from repository modules.

    `from workflows.x import NAME` puts NAME in the workflow's globals, so
    ConfigManager's fallback treats it as a config key (as the Slack agent does).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            source = ROOT.joinpath(*node.module.split(".")).with_suffix(".py")
            if source.is_file():
                imported = dict(_assignments(source))
                for alias in node.names:
                    if alias.name in imported:
                        yield alias.asname or alias.name, imported[alias.name]
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            if isinstance(target, ast.Name) and node.value is not None:
                yield target.id, node.value


def _assigned(name, path):
    for target, value in _assignments(path):
        if target == name:
            return ast.literal_eval(value)
    return None


def config_keys(path):
    """_CONFIG_KEYS, or ConfigManager's fallback: public UPPERCASE non-None constants."""
    declared = _assigned("_CONFIG_KEYS", path)
    if declared is not None:
        return declared
    keys = {}
    for name, value in _assignments(path):
        if name.isupper() and not name.startswith("_"):
            is_none = isinstance(value, ast.Constant) and value.value is None
            keys[name] = name == "INPUT_VIAL_STATUS_FILE" or not is_none
    return [name for name, included in keys.items() if included]


@pytest.mark.parametrize("workflow", sorted(ENTRYPOINTS))
def test_config_docs_cover_config_keys(workflow):
    path = ENTRYPOINTS[workflow]
    keys = config_keys(path)
    docs = _assigned("_CONFIG_DOCS", path)
    assert keys, f"{workflow} has no detectable config keys"
    assert isinstance(docs, dict), f"{workflow} must declare _CONFIG_DOCS as a literal dict"
    assert set(docs) == set(keys), {"undocumented": sorted(set(keys) - set(docs)),
                                    "unknown": sorted(set(docs) - set(keys))}
    for key, entry in docs.items():
        assert set(entry) <= FIELDS, (key, sorted(set(entry) - FIELDS))
        assert isinstance(entry.get("description"), str) and entry["description"].strip(), key
        assert "unit" in entry, key
        assert isinstance(entry.get("affects"), list) and set(entry["affects"]) <= RESOURCES, key
        if "shares" in entry:
            assert entry["shares"] in SHARES, key
        if "choices" in entry:
            assert isinstance(entry["choices"], list) and entry["choices"], key
        if "ask" in entry:
            assert isinstance(entry["ask"], bool), key
    products = _assigned("_PRODUCT_VIALS", path)
    assert isinstance(products, list) and all(isinstance(p, str) and p for p in products), \
        f"{workflow} must declare _PRODUCT_VIALS as a literal list of vial-name patterns"


def test_planning_markers_log_parseable_lines(caplog):
    logger = logging.getLogger("planning-marker-test")
    with caplog.at_level(logging.INFO, logger="planning-marker-test"):
        mark_clock_start(logger, "plate1_block1")
        mark_wait_until(logger, "plate1_block1", 600)
        mark_wait(logger, "reaction", 30)
    assert caplog.messages == [
        "PLANNING clock_start id=plate1_block1",
        "PLANNING wait_until id=plate1_block1 offset_s=600.000",
        "PLANNING clock_start id=reaction",
        "PLANNING wait_until id=reaction offset_s=30.000",
    ]
    with pytest.raises(ValueError):
        mark_clock_start(logger, "has space")
    with pytest.raises(ValueError):
        mark_wait_until(logger, "ok", -1)
