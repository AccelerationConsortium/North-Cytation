"""Resolve scheduler job names without importing operator-run programs."""

from pathlib import Path


SCHEDULER_DIAGNOSTICS = (
    "ailsa_clamp_test",
    "cytation_labware_test",
    "test_vortex_scheduler",
)


def workflow_paths(root):
    root = Path(root)
    paths = {
        path.stem: path
        for path in (root / "workflows").glob("*.py")
        if not path.name.startswith("_") and path.stem != "workflow_template"
    }
    for name in SCHEDULER_DIAGNOSTICS:
        path = root / "hardware_tests" / f"{name}.py"
        if path.is_file():
            if name in paths:
                raise ValueError(f"Duplicate scheduler job name: {name}")
            paths[name] = path
    return paths


def workflow_module(name, root):
    paths = workflow_paths(root)
    if not isinstance(name, str) or name not in paths:
        raise ValueError(f"Invalid selected workflow: {name!r}")
    return ".".join(paths[name].relative_to(root).with_suffix("").parts)
