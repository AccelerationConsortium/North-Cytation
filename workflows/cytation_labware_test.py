"""One-well Gen5 labware diagnostic. Run: python -m workflows.cytation_labware_test.

Defaults to simulation and a 96-well baseline. No dispensing or shake protocol.
Starts with the active plate in the pipetting area and returns it there.
"""

import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

from datetime import datetime, timezone
import difflib
import hashlib
import json
import logging
from pathlib import Path
import xml.etree.ElementTree as ET

SIMULATE = True
INPUT_VIAL_STATUS_FILE = None
WELLPLATE_TYPE = "96 WELL PLATE"
PROTOCOL_FILE = r"C:\Protocols\CMC_Absorbance_96.prt"
TEST_MODE = "baseline"  # baseline, roundtrip, replace_labware, modifiable_labware
OUTPUT_ROOT = "output/cytation_labware_test"

_WORKFLOW_NAME = Path(__file__).stem
_CONFIG_KEYS = ["SIMULATE", "INPUT_VIAL_STATUS_FILE", "WELLPLATE_TYPE",
                "PROTOCOL_FILE", "TEST_MODE", "OUTPUT_ROOT"]


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _steps(xml):
    root = ET.fromstring(xml)
    steps = root.find("StepList")
    if steps is None:
        raise ValueError("Procedure has no StepList")
    return ET.canonicalize(ET.tostring(steps, encoding="unicode"), strip_text=True)


def _labware(xml):
    root = ET.fromstring(xml)
    name = root.findtext("Labware/Name")
    if not name:
        raise ValueError("Procedure has no Labware/Name")
    return {"name": name, "use_lid": root.findtext("Labware/UseLid")}


def prepare_plate(plate, reader_name, mode, folder, logger):
    """Change only labware, require readback and validation before any read."""
    original = plate.get_procedure()
    (folder / "original.xml").write_text(original, encoding="utf-8")
    before = _labware(original)
    logger.info("Original labware: %s", before)
    modifiable = plate.get_modifiable_procedure()
    (folder / "modifiable.xml").write_text(modifiable, encoding="utf-8")
    status, reason = plate.validate_procedure(False)
    logger.info("Original validation: status=%s reason=%s", status, reason)
    requested = original
    if mode in ("roundtrip", "replace_labware", "modifiable_labware"):
        source = modifiable if mode == "modifiable_labware" else original
        root = ET.fromstring(source)
        name = root.find("Labware/Name")
        if name is None:
            raise ValueError("Selected XML has no Labware/Name")
        if mode == "modifiable_labware" and name.get("Modifiable") != "Yes":
            raise ValueError("Gen5 does not mark Labware/Name as modifiable")
        if mode != "roundtrip":
            name.text = reader_name
        requested = original if mode == "roundtrip" else ET.tostring(root, encoding="unicode")
        (folder / "requested.xml").write_text(requested, encoding="utf-8")
        diff = "".join(difflib.unified_diff(
            source.splitlines(True), requested.splitlines(True),
            fromfile="before", tofile="requested"))
        (folder / "requested.diff").write_text(diff, encoding="utf-8")
        logger.info("Applying %s; XML diff:\n%s", mode, diff or "(unchanged XML)")
        if mode == "modifiable_labware":
            plate.set_modifiable_procedure(requested)
        else:
            plate.set_procedure(requested)
    else:
        (folder / "requested.xml").write_text(requested, encoding="utf-8")
    readback = plate.get_procedure()
    (folder / "readback.xml").write_text(readback, encoding="utf-8")
    after = _labware(readback)
    logger.info("Readback labware: %s", after)
    if after["name"] != reader_name:
        raise RuntimeError(f"Labware readback {after['name']!r} != requested {reader_name!r}")
    if _steps(original) != _steps(readback):
        raise RuntimeError("Procedure StepList changed; refusing to read")
    if after["use_lid"] != before["use_lid"]:
        raise RuntimeError("Procedure lid setting changed; refusing to read")
    if (after["use_lid"] or "").lower() != "false":
        raise RuntimeError("This test requires an explicitly lidless procedure and plate")
    status, reason = plate.validate_procedure(False)
    logger.info("Readback validation: status=%s reason=%s", status, reason)
    if status != 1:
        raise RuntimeError(f"Gen5 rejected procedure: status={status}, reason={reason}")
    return {"before": before, "after": after, "validation": status, "reason": reason}


def execute(config=None, show_gui=True):
    lash = Lash_E(workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
                  config=config, show_gui=show_gui)
    if not lash._workflow_should_continue:
        return None
    cfg = lash.workflow_config
    if cfg["TEST_MODE"] not in ("baseline", "roundtrip", "replace_labware", "modifiable_labware"):
        raise ValueError("Unknown TEST_MODE")
    folder = Path(cfg["OUTPUT_ROOT"]) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(folder / "diagnostic.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    lash.logger.addHandler(handler)
    summary = {"config": dict(cfg), "simulation": lash.simulate, "phase": "preflight",
               "completed": False, "well": "A1", "final_location": "not checked"}
    try:
        inv = lash.nr_track.inventory
        plate_type = cfg["WELLPLATE_TYPE"]
        plate_id = inv.plate_at("pipetting_area", plate_type)
        summary["plate_id"] = plate_id
        if len(inv.contents("pipetting_area")) != 1:
            raise ValueError("Start with exactly one active plate in the pipetting area")
        if any(inv.contents(p) for p in ("cytation_tray", "gripper")):
            raise ValueError("Cytation tray and track gripper must be empty")
        if plate_id in inv.attachments:
            raise ValueError("Use a lidless plate")
        reader_name = inv.definitions[plate_type].get("name_in_cytation") or plate_type
        summary["reader_name"] = reader_name
        lash.logger.info("Config: %s; CarrierIn plate_type=%r use_lid=False", cfg, reader_name)
        if not lash.simulate:
            protocol = Path(cfg["PROTOCOL_FILE"])
            summary["protocol_sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
        def inspect_procedure(plate):
            summary["phase"] = "procedure_xml"
            summary["procedure"] = prepare_plate(
                plate, reader_name, cfg["TEST_MODE"], folder, lash.logger)
            summary["phase"] = "normal_read"

        summary["phase"] = "measure_wellplate"
        if lash.simulate:
            lash.logger.info("SIMULATION: procedure XML and Gen5 read are NOT tested")
        data = lash.measure_wellplate(
            cfg["PROTOCOL_FILE"], [0], plate_type=plate_type,
            use_lid=False, plate_id=plate_id, plate_setup=inspect_procedure)
        if data is not None:
            data.to_csv(folder / "results.csv")
            lash.logger.info("A1 result:\n%s", data)
        if inv.top("pipetting_area") != plate_id:
            raise RuntimeError("Plate did not return to pipetting area")
        summary.update(completed=True, phase="complete")
        lash.logger.info("Diagnostic complete; plate returned to pipetting area. Artifacts: %s", folder)
        return summary
    except (Exception, KeyboardInterrupt) as error:
        summary["error"] = repr(error)
        lash.logger.exception("Stopped during %s; no automatic recovery motion", summary["phase"])
        raise
    finally:
        summary["locations"] = {
            location: list(lash.nr_track.inventory.contents(location))
            for location in ("source_stack", "pipetting_area", "cytation_tray", "gripper")}
        summary["final_location"] = next(
            (location for location, contents in summary["locations"].items()
             if summary.get("plate_id") in contents), "unknown")
        _save_json(folder / "summary.json", summary)
        lash.logger.info("Final location=%s; artifacts=%s", summary["final_location"], folder)
        lash.logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    execute()
