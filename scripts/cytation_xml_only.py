"""Inspect Gen5 procedure changes without loading a physical plate or reading.

Run from the repo root: python -m scripts.cytation_xml_only
Then, if the 96-well test passes: python -m scripts.cytation_xml_only --plate-type "48 WELL PLATE"
"""
import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
import logging
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from biotek_new import Biotek_Wrapper


class ProcedureOnlyReader(Biotek_Wrapper):
    def run_plate(self, plate, wells, prot_type):
        # run_protocol still loads the experiment and invokes plate_setup normally.
        # Replace only the final read stage: no partial selection or read here.
        self._log("info", "XML-only: skipping run_plate; no StartRead or carrier calls")
        return pd.DataFrame()


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.errors = []

    def emit(self, record):
        if record.levelno >= logging.ERROR:
            self.errors.append(record.getMessage())


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _pretty(root):
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def candidate_xml(original, plate_type, step_wells=None):
    root = ET.fromstring(original)
    name = root.find("Labware/Name")
    if name is None:
        raise ValueError("Original procedure has no Labware/Name")
    name.text = plate_type
    # Installed Gen5 explicitly rejected both top-level tokens. The existing
    # absorbance builder also omits them. Preserve Labware and all read steps.
    for tag in ("WellSelection", "Options"):
        for element in root.findall(tag):
            root.remove(element)
    if step_wells is not None:
        reads = root.findall(".//ReadStep")
        if not reads:
            raise ValueError("Procedure has no ReadStep")
        for read in reads:
            wells = read.find("Wells")
            if wells is None:
                raise ValueError("ReadStep has no Wells field")
            wells.text = step_wells
    return _pretty(root)


def inspect_plate(plate, reader, plate_type, folder, summary, capture, step_wells=None):
    logger = reader.logger
    summary["phase"] = "get_original"
    original = plate.get_procedure()
    (folder / "original.xml").write_text(original, encoding="utf-8")
    if not original.strip():
        raise RuntimeError("GetProcedure returned empty XML; see diagnostic.log")
    before = ET.fromstring(original)
    summary["original_labware"] = before.findtext("Labware/Name")
    requested = candidate_xml(original, plate_type, step_wells)
    (folder / "requested.xml").write_text(requested, encoding="utf-8")
    diff = "".join(difflib.unified_diff(
        _pretty(before).splitlines(True), requested.splitlines(True),
        fromfile="original", tofile="requested"))
    (folder / "requested.diff").write_text(diff, encoding="utf-8")
    logger.info("Exact requested XML changes:\n%s", diff)
    summary["phase"] = "set_procedure"
    checkpoint = len(capture.errors)
    plate.set_procedure(requested)
    summary["setter_errors"] = capture.errors[checkpoint:]
    summary["phase"] = "readback"
    readback = plate.get_procedure()
    (folder / "readback.xml").write_text(readback, encoding="utf-8")
    if not readback.strip():
        raise RuntimeError("No procedure returned after SetProcedure; see setter_errors and diagnostic.log")
    after = ET.fromstring(readback)
    summary["returned_labware"] = after.findtext("Labware/Name")
    summary["runtime_well_selection"] = after.findtext("WellSelection")
    if summary["setter_errors"]:
        raise RuntimeError("Driver logged a SetProcedure error; see setter_errors")
    if summary["returned_labware"] != plate_type:
        raise RuntimeError("Readback labware does not match requested plate")
    summary["labware_accepted"] = True
    summary["phase"] = "check_dimensions"
    # These read-only COM properties are documented on guide p.48. The Python
    # driver has no convenience properties; use its existing OLE accessor.
    summary["rows"] = int(plate._invoke_property_get("MaxRows", result_wanted=True))
    summary["columns"] = int(plate._invoke_property_get("MaxColumns", result_wanted=True))
    expected = {"96 WELL PLATE": (8, 12), "48 WELL PLATE": (6, 8), "24 WELL PLATE": (4, 6)}
    if (summary["rows"], summary["columns"]) != expected[plate_type]:
        raise RuntimeError("Gen5 dimensions do not match requested plate format")
    summary["dimensions_match"] = True
    before_steps, after_steps = before.find("StepList"), after.find("StepList")
    if before_steps is None or after_steps is None:
        raise RuntimeError("Procedure is missing StepList")
    summary["steps_unchanged"] = (
        ET.canonicalize(ET.tostring(before_steps, encoding="unicode"), strip_text=True)
        == ET.canonicalize(ET.tostring(after_steps, encoding="unicode"), strip_text=True))
    expected_steps = ET.fromstring(requested).find("StepList")
    summary["steps_match_requested"] = (
        ET.canonicalize(ET.tostring(expected_steps, encoding="unicode"), strip_text=True)
        == ET.canonicalize(ET.tostring(after_steps, encoding="unicode"), strip_text=True))
    if not summary["steps_match_requested"]:
        raise RuntimeError("Gen5 changed requested procedure steps; inspect requested.xml and readback.xml")
    if step_wells is None:
        summary["phase"] = "select_A1"
        partial = reader.build_bti_partial_plate_xml(single_block=False, wells=["A1"])
        (folder / "partial_plate.xml").write_text(partial, encoding="utf-8")
        checkpoint = len(capture.errors)
        plate.set_partial_plate(partial)
        if len(capture.errors) != checkpoint:
            raise RuntimeError("Driver rejected A1 partial-plate selection; see diagnostic.log")
        summary["partial_selection_accepted"] = True
    else:
        summary["phase"] = "check_embedded_wells"
        summary["requested_step_wells"] = step_wells
        summary["returned_step_wells"] = [r.findtext("Wells") for r in after.findall(".//ReadStep")]
        if not summary["returned_step_wells"] or any(w != step_wells for w in summary["returned_step_wells"]):
            raise RuntimeError("Gen5 did not retain requested wells in every read step")
        summary["embedded_selection_accepted"] = True
    summary["phase"] = "validate"
    status, reason = plate.validate_procedure(False)
    summary["validation"] = {"status": status, "reason": reason}
    if status != 1:
        raise RuntimeError(f"Gen5 validation rejected procedure: {status}: {reason}")
    summary.update(passed=True, phase="complete")


def write_report(folder, summary):
    checks = [
        ("Requested labware returned", "labware_accepted"),
        ("Row/column counts match", "dimensions_match"),
        ("Procedure steps match requested XML", "steps_match_requested"),
    ]
    lines = ["# Gen5 XML-only plate test", "",
             f"Outcome: {'PASS' if summary.get('passed') else 'STOPPED'}",
             f"Requested plate: {summary['plate_type']}",
             f"Last phase: {summary['phase']}", ""]
    for label, key in checks:
        verdict = "PASS" if summary.get(key) else "NOT ESTABLISHED"
        lines.append(f"- {label}: {verdict}")
    if summary.get("step_wells") is not None:
        lines += [f"- Requested wells inside read steps: {summary['step_wells']}",
                  f"- Returned wells: {summary.get('returned_step_wells', 'not reached')}",
                  "- SetPartialPlate: intentionally not called in embedded-well mode"]
    else:
        lines += [f"- A1 runtime selection accepted: {summary.get('partial_selection_accepted', False)}"]
    lines += ["", f"Validation: {summary.get('validation', 'not reached')}"]
    if summary.get("error"):
        lines += ["", f"Error: {summary['error']}"]
    lines += ["", "Review requested.diff for exactly what we sent, original.xml and",
              "readback.xml for what Gen5 retained, and diagnostic.log for Gen5 errors.", "",
              "This test does not start a read, move a carrier, initialize the robot,",
              "or change the .prt file. It cannot establish actual measurement behavior",
              "or fluorescence compatibility. Validation assumes a full plate (guide p.66).",
              "No session close, abort, retry, or restart is added."]
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=r"C:\Protocols\CMC_Absorbance_96.prt")
    parser.add_argument("--plate-type", choices=["96 WELL PLATE", "48 WELL PLATE", "24 WELL PLATE"], default="96 WELL PLATE")
    parser.add_argument("--step-wells", choices=["A1", "A1..A3"],
                        help="Test wells embedded in ReadStep instead of runtime selection. A1..A3 comes from a native Gen5 export.")
    args = parser.parse_args()
    folder = Path("output/cytation_xml_only") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True)
    summary = {"plate_type": args.plate_type, "protocol": args.protocol, "phase": "initialize", "passed": False,
               "step_wells": args.step_wells}
    logger = logging.getLogger("cytation_xml_only")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(folder / "diagnostic.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    capture = Capture()
    driver_logger = logging.getLogger("biotek_driver.utils.logger")
    previous_level = driver_logger.level
    driver_logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    driver_logger.addHandler(file_handler)
    driver_logger.addHandler(capture)
    print(f"XML-only artifacts: {folder.resolve()}")
    try:
        summary["protocol_sha256"] = hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest()
        # Same reader initialization and run_protocol as normal; no Lash_E/robot.
        reader = ProcedureOnlyReader(simulate=False, logger=logger)
        reader.run_protocol(args.protocol, [0], plate_type=args.plate_type,
                            plate_setup=lambda p: inspect_plate(p, reader, args.plate_type, folder, summary, capture, args.step_wells))
    except (Exception, KeyboardInterrupt) as error:
        summary["error"] = str(error)
        logger.exception("XML-only test stopped during %s", summary["phase"])
    finally:
        summary["driver_errors"] = capture.errors
        _json(folder / "summary.json", summary)
        write_report(folder, summary)
        logger.removeHandler(file_handler)
        driver_logger.removeHandler(file_handler)
        driver_logger.removeHandler(capture)
        driver_logger.setLevel(previous_level)
        file_handler.close()
    print(f"{'PASS' if summary['passed'] else 'STOPPED'}: review {folder / 'report.md'}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
