"""Run Gen5 protocols on the Cytation and check the dye CMC workflow handles the data.

Run: python -m hardware_tests.cytation_protocol_preview (or from the scheduler), with
SIMULATE false. No robot or track moves: the reader reads whatever is in its tray.
An empty plate shows the layout; different liquids in the wells (e.g. water, dye in
ethanol) also show that each well gets its own values.

For each protocol:
- raw_<protocol>.csv: the DataFrame the reader wrapper returns, unchanged
- procedure_<protocol>.xml: the Gen5 procedure (its read steps)
- read time in seconds

Then, as a real run does, the protocols' data are joined into one table
(measure_wellplate's labels) and passed through surfactant_dye_cmc's own processing:
- parsed_results.csv: what results.csv would get (fixed reads, one row per well)
- spectra.csv: what spectra.csv would get (spectra, one row per well x wavelength)
- summary.txt / summary.json: read type, steps, shape, timing, and OK or the error for
  each protocol. Protocol files that do not exist are skipped.
"""
import sys
sys.path.append("../utoronto_demo")

import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd

from master_usdl_coordinator import Lash_E

SIMULATE = True
INPUT_VIAL_STATUS_FILE = None
PROTOCOL_FILES = [
    r"C:\Protocols\Coumarin_96.prt",               # fluorescence 485/530
    r"C:\Protocols\Coumarin_Abs_96.prt",           # absorbance 460 and 750 nm
    r"C:\Protocols\Coumarin_FluorSpectrum_96.prt", # ex 450, em 480-650, 5 nm
    r"C:\Protocols\Coumarin_AbsSpectrum_96.prt",   # 400-750, 5 nm
]
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


def describe(data):
    """Layout of one protocol's data as the reader wrapper returns it."""
    spectrum = "Wavelengths" in data.columns
    return {"kind": "spectrum" if spectrum else "endpoint",
            "layout": ("rows = wavelengths, columns = wells" if spectrum
                       else "rows = wells, columns = readings"),
            "shape": list(data.shape),
            "index_first": [str(i) for i in list(data.index[:5])],
            "columns": [str(c) for c in data.columns]}


def process_like_a_run(frames, wells_labels, folder):
    """Join the protocols' data as measure_wellplate does, then process each protocol with
    surfactant_dye_cmc's own functions. Returns {stem: result} and writes the outputs."""
    from workflows.surfactant_dye_cmc import raw_to_columns, save_spectra, spectrum_to_long
    labelled = []
    for stem, data in frames.items():
        data = data.copy()
        data.columns = pd.MultiIndex.from_tuples([(f"rep1_{stem}", c) for c in data.columns])
        labelled.append(data)
    raw = pd.concat(labelled, axis=1)
    raw.to_csv(folder / "raw_joined.csv")
    wells = pd.DataFrame({"well_position": wells_labels, "block": "preview", "replicate": 1,
                          "matrix": None, "surfactant_mM": None, "dye_uM": None})
    results, outcome = wells[["well_position"]].copy(), {}
    for stem, data in frames.items():
        part = raw[[col for col in raw.columns if col[0] == f"rep1_{stem}"]]
        try:
            if "Wavelengths" in data.columns:
                long = spectrum_to_long(part, stem)
                save_spectra(long, wells, 1, 0, "preview", folder)
                outcome[stem] = {"ok": True, "saved_to": "spectra.csv", "rows": len(long),
                                 "wavelengths": [float(long.wavelength_nm.min()), float(long.wavelength_nm.max())],
                                 "wells": sorted(long.well_position.unique())}
            else:
                columns = raw_to_columns(part, stem)
                results = results.merge(columns, on="well_position", how="left", validate="one_to_one")
                outcome[stem] = {"ok": True, "saved_to": "parsed_results.csv",
                                 "columns": [c for c in columns.columns if c != "well_position"],
                                 "wells": list(columns.well_position)}
        except Exception as error:
            outcome[stem] = {"ok": False, "error": repr(error)}
    results.to_csv(folder / "parsed_results.csv", index=False)
    return outcome


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
    columns = 12 if cfg["CYTATION_PLATE_TYPE"].startswith("96") else 8 if cfg["CYTATION_PLATE_TYPE"].startswith("48") else 6
    labels = [f"{chr(65 + i // columns)}{i % columns + 1}" for i in cfg["WELLS"]]
    summary = {"simulation": lash.simulate, "wells": labels, "protocols": {}, "processing": {}}
    frames = {}
    try:
        for path in cfg["PROTOCOL_FILES"]:
            stem = Path(path).stem
            entry = summary["protocols"].setdefault(stem, {"file": path})
            if not lash.simulate and not Path(path).is_file():
                entry["error"] = "file not found (skipped)"
                lash.logger.warning(f"{path} not found; skipped")
                continue

            def save_procedure(plate, stem=stem, entry=entry):
                xml = plate.get_procedure()
                (folder / f"procedure_{stem}.xml").write_text(xml, encoding="utf-8")
                entry["read_steps"] = read_steps(xml)

            lash.logger.info(f"Running {path} on wells {labels}")
            start = time.monotonic()
            try:
                data = lash._run_plate_protocol(path, cfg["WELLS"], plate_type=cfg["CYTATION_PLATE_TYPE"],
                                                plate_setup=save_procedure)
            except Exception as error:
                entry["error"] = repr(error)
                lash.logger.exception(f"{path} failed")
                continue
            entry["read_seconds"] = round(time.monotonic() - start, 1)
            if data is None:
                entry["data"] = "none returned"
                continue
            data.to_csv(folder / f"raw_{stem}.csv")
            lash.logger.info(f"{stem} returned:\n{data}")
            entry.update(describe(data))
            frames[stem] = data
        if frames:
            summary["processing"] = process_like_a_run(frames, labels, folder)
        return summary
    finally:
        (folder / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        lines = [f"Cytation protocol preview ({'SIMULATION: no reader data' if lash.simulate else 'live'})",
                 f"Wells: {labels}", ""]
        for stem, entry in summary["protocols"].items():
            lines.append(f"== {stem}")
            for key in ("error", "data", "read_seconds", "read_steps", "kind", "layout", "shape",
                        "index_first", "columns"):
                if key in entry:
                    lines.append(f"  {key}: {entry[key]}")
            if stem in summary["processing"]:
                lines.append(f"  workflow processing: {summary['processing'][stem]}")
            lines.append("")
        (folder / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
        lash.logger.info(f"Preview saved in {folder}")
        lash.logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    execute()
