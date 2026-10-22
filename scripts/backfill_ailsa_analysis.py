"""Backfill current CMC fits and scoring for the ten historical Ailsa studies.

Run with --apply after inspecting the default read-only preflight. Analysis
functions are loaded verbatim from the workflow AST to avoid robot imports.
Existing analysis files are backed up under temp before replacement.
"""

import argparse
import ast
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight():
    runs = sorted((ROOT / "output/surfactant_grid/experimental").glob("SDS_BDDAC_study*"))
    assert len(runs) == 10, f"Expected ten historical studies, found {len(runs)}"
    inputs = []
    measurements = ["turbidity_600", "fluorescence_334_373", "fluorescence_334_384",
                    "ratio", "fluorescence_metric"]
    keys = ["execution_block", "wellplate_index"]
    for run in runs:
        raw_path = run / "complete_experiment_results.csv"
        raw = pd.read_csv(raw_path)
        assert len(raw) == 288 and raw.dye.eq("pyrene").all(), run
        assert raw.surf_A.eq("SDS").all() and raw.surf_B.eq("BDDAC").all(), run
        assert sorted(raw.measurement_replicate.unique()) == [1, 2, 3], run
        assert "timepoint_min" not in raw or raw.timepoint_min.nunique() == 1, run
        inputs.append(raw_path)
        for label in ["measurement_replicate_1", "measurement_replicate_2",
                      "measurement_replicate_3", "measurement_mean"]:
            path = run / "measurement_replicate_analysis" / f"{label}_results.csv"
            table = pd.read_csv(path)
            assert len(table) == 96 and not table.duplicated(keys).any(), path
            expected = (raw.groupby(keys)[measurements].mean().sort_index()
                        if label == "measurement_mean" else
                        raw[raw.measurement_replicate == int(label[-1])]
                        .set_index(keys)[measurements].sort_index())
            actual = table.set_index(keys)[measurements].sort_index()
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
            assert np.isfinite(actual.to_numpy()).all(), path
            metadata = ["well_type", "control_type", "surf_A", "surf_B",
                        "surf_A_conc_mm", "surf_B_conc_mm"]
            expected_meta = raw.drop_duplicates(keys).set_index(keys)[metadata].sort_index()
            pd.testing.assert_frame_equal(table.set_index(keys)[metadata].sort_index(),
                                          expected_meta, check_dtype=False)
            for name, column in [("SDS", "surf_A_conc_mm"), ("BDDAC", "surf_B_conc_mm")]:
                controls = table[table.control_type.str.startswith(f"cmc_{name}_")]
                assert len(controls) == 10 and controls[column].nunique() == 10, path
            inputs.append(path)
        print(f"READY {run.name}", flush=True)
    return runs, inputs


def load_plot_function():
    # The legacy module imports seaborn for unrelated plots. Load its exact
    # numerical functions without that optional plotting dependency.
    from scipy.optimize import curve_fit
    curve_path = ROOT / "analysis/control_cmc_analysis.py"
    curve_tree = ast.parse(curve_path.read_text(encoding="utf-8"))
    curve_nodes = [node for node in curve_tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"boltzmann", "fit_cmc_curve"}]
    curve_module = ModuleType("analysis.control_cmc_analysis")
    curve_module.__dict__.update(np=np, curve_fit=curve_fit, warnings=warnings)
    exec(compile(ast.Module(body=curve_nodes, type_ignores=[]), str(curve_path), "exec"),
         curve_module.__dict__)
    sys.modules[curve_module.__name__] = curve_module
    path = ROOT / "workflows/surfactant_grid_ailsa.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {"get_dye_protocol", "get_volume_scale", "generate_study_1d_plots"}
    constants = {"DYE", "DYE_SOLVENT", "DYE_VIAL", "DYE_VOLUME_UL", "WELLPLATE_TYPE",
                 "VOLUME_SCALE_BY_PLATE", "_DYE_PROTOCOLS"}
    nodes = [node for node in tree.body if
             (isinstance(node, ast.FunctionDef) and node.name in functions) or
             (isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id in constants for target in node.targets))]
    namespace = {"os": os, "pd": pd}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["generate_study_1d_plots"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    runs, inputs = preflight()
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in inputs}
    if not args.apply:
        print("Preflight passed: 10 studies, 40 analysis tables, 80 control curves.")
        return
    from analysis.surfactant_grid_scoring import score_surfactant_grid_2d

    generate = load_plot_function()
    backup = ROOT / "temp" / f"ailsa_analysis_backfill_backup_{datetime.now():%Y%m%d_%H%M%S}"
    backup.mkdir(exist_ok=False)
    (backup / "input_sha256.json").write_text(json.dumps(hashes, indent=2))
    logger = logging.getLogger(__name__)
    summaries = []
    for run in runs:
        for path in (run / "one_dimensional").rglob("*"):
            if path.is_file():
                target = backup / run.name / path.relative_to(run)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        for source in sorted((run / "measurement_replicate_analysis").glob("*_results.csv")):
            label = source.stem.removesuffix("_results")
            one_d = run / "one_dimensional" / label
            paths = generate(str(source), str(one_d), logger, "SDS", "BDDAC")
            assert len(paths) == 6 and all(Path(path).is_file() for path in paths)
            summary = score_surfactant_grid_2d(source, one_d, run / "scoring" / label,
                                              "SDS", "BDDAC")
            assert summary["status"] == "success", (run, label, summary)
            assert summary["n_experiment_points"] == 73, (run, label)
            summaries.append({"run": run.name, "measurement": label, **summary})
            print(f"DONE {run.name} / {label}: {summary['calibration_source']}", flush=True)
    assert all(digest(ROOT / path) == sha for path, sha in hashes.items()), "Input changed"
    pd.DataFrame(summaries).to_csv(backup / "backfill_report.csv", index=False)
    print("Verified: 40 scoring outputs; original results and measurement tables unchanged.")


if __name__ == "__main__":
    main()
