"""Every documented workflow config key has a planning annotation.

The lab Slack agent reads _CONFIG_DOCS with ast (no import), so entries must be
plain literals and cover _CONFIG_KEYS exactly.
"""

import ast
import logging
from pathlib import Path

import pytest

from workflows._planning_markers import mark_clock_start, mark_wait, mark_wait_until

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"
# Workflows that predate _CONFIG_DOCS (2026-10-31). Every other workflow with an
# execute() entrypoint - including every new one - must be annotated. Remove a
# name from this list when you annotate it; never add one.
UNDOCUMENTED_LEGACY = {
    "ailsa_clamp_test",
    "cytation_labware_test",
    "Degradation_serena",
    "SQ_Peptoid_LNP_test",
    "surfactant_grid_adaptive_concentrations",
    "test_vortex_scheduler",
    "vial_shuffle_capture_workflow",
}


def _entrypoints():
    for path in sorted(WORKFLOWS.glob("*.py")):
        if path.name.startswith("_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(isinstance(node, ast.FunctionDef) and node.name == "execute" for node in tree.body):
            yield path.stem


DOCUMENTED = [name for name in _entrypoints() if name not in UNDOCUMENTED_LEGACY]
RESOURCES = {"duration", "tips", "plates", "vials", "liquids"}
SHARES = {"vial_layout", "input_file", "dye", "plate_type"}
FIELDS = {"description", "unit", "choices", "affects", "effect", "shares", "ask"}


def _assignments(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
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


@pytest.mark.parametrize("workflow", DOCUMENTED)
def test_config_docs_cover_config_keys(workflow):
    path = WORKFLOWS / f"{workflow}.py"
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


def test_legacy_list_only_shrinks():
    entrypoints = set(_entrypoints())
    stale = sorted(UNDOCUMENTED_LEGACY - entrypoints)
    assert not stale, f"Remove deleted or renamed workflows from UNDOCUMENTED_LEGACY: {stale}"
    annotated = sorted(name for name in UNDOCUMENTED_LEGACY if _assigned("_CONFIG_DOCS", WORKFLOWS / f"{name}.py"))
    assert not annotated, f"These now have _CONFIG_DOCS; remove them from UNDOCUMENTED_LEGACY: {annotated}"


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
