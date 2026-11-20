"""Run Gen5 protocols on the Cytation and save exactly what comes back.

Use it to see the data shape of a new protocol before a workflow relies on it.
Run: python -m hardware_tests.cytation_protocol_preview (or from the scheduler).

No robot or track moves: the reader reads whatever is in its tray (an empty plate
is fine; the values are meaningless, only the layout matters). For each protocol:
- raw_<protocol>.csv: the DataFrame the reader wrapper returns, unchanged
- procedure_<protocol>.xml: the Gen5 procedure, to see its read steps
- summary.json / summary.txt: read type, read steps, shape, index, columns, and
  whether surfactant_dye_cmc can turn it into results.csv columns as it is.
"""
import sys
sys.path.append("../utoronto_demo")

import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd

from master_usdl_coordinator import Lash_E

SIMULATE = True
INPUT_VIAL_STATUS_FILE = None
PROTOCOL_FILES = [
    r"C:\Protocols\Coumarin_96.prt",               # fluorescence 485/530 (used by the runs)
    r"C:\Protocols\Coumarin_Abs_96.prt",           # absorbance 460 and 750 nm (used by the runs)
    r"C:\Protocols\Coumarin_FluorSpectrum_96.prt", # optional: ex 450, em 480-650, 5 nm
    r"C:\Protocols\Coumarin_AbsSpectrum_96.prt",   # optional: 400-750, 5 nm
]  # a file that does not exist is skipped
WELLS = [0, 1, 12]                  # well indices: A1, A2 and B1 (two rows, to see how rows are joined)
CYTATION_PLATE_TYPE = "96 WELL PLATE"
OUTPUT_ROOT = "output/cytation_protocol_preview"

_WORKFLOW_NAME = Path(__file__).stem
_CONFIG_KEYS = ["SIMULATE", "INPUT_VIAL_STATUS_FILE", "PROTOCOL_FILES", "WELLS",
                "CYTATION_PLATE_TYPE", "OUTPUT_ROOT"]
_CONFIG_DOCS = {
    "SIMULATE": {"description": "Simulation returns no reader data; run live to see real output.",
                 "unit": None, "affects": []},
    "INPUT_VIAL_STATUS_FILE": {"description": "Not used (no vials).", "unit": None, "affects": []},
    "PROTOCOL_FILES": {"description": "Gen5 protocol files to run, in order.", "unit": None,
                       "affects": ["duration"]},
    "WELLS": {"description": "Well indices to read (0 = A1, 12 = B1 on a 96-well plate).",
              "unit": None, "affects": ["duration"]},
    "CYTATION_PLATE_TYPE": {"description": "Plate name passed to the reader.", "unit": None,
                            "choices": ["96 WELL PLATE", "48 WELL PLATE", "24 WELL PLATE"], "affects": []},
    "OUTPUT_ROOT": {"description": "Folder for the saved output.", "unit": None, "affects": []},
}
_PRODUCT_VIALS = []


def read_steps(procedure_xml):
    """Each read step in a Gen5 procedure: read type, detection and its text settings."""
    steps = []
    for step in ET.fromstring(procedure_xml).iter("ReadStep"):
        info = {"ReadType": step.findtext("ReadType"), "Detection": step.findtext("Detection")}
        for child in step:
            if child.text and child.text.strip() and child.tag not in info:
                info[child.tag] = child.text.strip()
        steps.append(info)
    return steps


def describe(data, protocol_path):
    """Shape and layout of one protocol's data, and whether the dye CMC workflow can use it."""
    if data is None:
        return {"data": None}
    kind = "spectrum (rows = wavelengths, columns = wells)" if "Wavelengths" in data.columns \
        else "endpoint (rows = wells, columns = readings)"
    info = {"layout": kind, "shape": list(data.shape),
            "index_first": [str(i) for i in list(data.index[:5])],
            "columns": [str(c) for c in data.columns],
            "dtypes": {str(k): str(v) for k, v in data.dtypes.items()}}
    # Same steps as the workflow: label columns like measure_wellplate, then raw_to_columns.
    from workflows.surfactant_dye_cmc import raw_to_columns
    labelled = data.copy()
    labelled.columns = pd.MultiIndex.from_tuples(
        [(f"rep1_{Path(protocol_path).stem}", c) for c in labelled.columns])
    try:
        parsed = raw_to_columns(labelled, "reading")
        info["workflow_parse"] = {"ok": True, "columns": list(parsed.columns), "rows": len(parsed)}
    except Exception as error:
        info["workflow_parse"] = {"ok": False, "error": repr(error)}
    return info


def execute(config=None, show_gui=True):
    lash = Lash_E(workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
                  config=config, show_gui=show_gui)
    if not lash._workflow_should_continue:
        return None
    cfg = lash.workflow_config
    folder = Path(cfg["OUTPUT_ROOT"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(folder / "preview.log", encoding="utf-8")
    lash.logger.addHandler(handler)
    summary = {"simulation": lash.simulate, "wells": cfg["WELLS"], "protocols": {}}
    try:
        for path in cfg["PROTOCOL_FILES"]:
            stem = Path(path).stem
            entry = summary["protocols"].setdefault(stem, {"file": path})
            if not lash.simulate and not Path(path).is_file():
                entry["error"] = "file not found"
                lash.logger.error(f"{path} not found; skipped")
                continue

            def save_procedure(plate, stem=stem, entry=entry):
                xml = plate.get_procedure()
                (folder / f"procedure_{stem}.xml").write_text(xml, encoding="utf-8")
                entry["read_steps"] = read_steps(xml)

            lash.logger.info(f"Running {path} on wells {cfg['WELLS']}")
            try:
                data = lash._run_plate_protocol(path, cfg["WELLS"], plate_type=cfg["CYTATION_PLATE_TYPE"],
                                                plate_setup=save_procedure)
            except Exception as error:
                entry["error"] = repr(error)
                lash.logger.exception(f"{path} failed")
                continue
            if data is not None:
                data.to_csv(folder / f"raw_{stem}.csv")
                lash.logger.info(f"{stem} returned:\n{data}")
            entry.update(describe(data, path))
        return summary
    finally:
        (folder / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        lines = [f"Cytation protocol preview ({'SIMULATION: no reader data' if lash.simulate else 'live'})",
                 f"Wells: {cfg['WELLS']}", ""]
        for stem, entry in summary["protocols"].items():
            lines.append(f"== {stem}")
            for key in ("error", "read_steps", "layout", "shape", "index_first", "columns", "workflow_parse"):
                if key in entry:
                    lines.append(f"  {key}: {entry[key]}")
            if "data" in entry:
                lines.append("  data: none returned")
            lines.append("")
        (folder / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
        lash.logger.info(f"Preview saved in {folder}")
        lash.logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    execute()
