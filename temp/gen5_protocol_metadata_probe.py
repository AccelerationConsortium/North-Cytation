"""One documented NewExperimentEx metadata test; no reading or carrier movement."""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biotek_new import Biotek_Wrapper
from scripts.cytation_xml_only import Capture

folder = Path("output/cytation_xml_only") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_new_experiment_ex")
folder.mkdir(parents=True)
handler = logging.FileHandler(folder / "diagnostic.log", encoding="utf-8")
logger = logging.getLogger("metadata_probe")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
driver_logger = logging.getLogger("biotek_driver.utils.logger")
driver_logger.addHandler(handler)
capture = Capture()
driver_logger.addHandler(capture)
summary = {"phase": "initialize", "passed": False}
try:
    reader = Biotek_Wrapper(simulate=False, logger=logger)
    # Use the known original getter output, retaining Runtime and Options.
    original_path = Path("output/cytation_xml_only/20261030T165605_902570Z/original.xml")
    original = original_path.read_text(encoding="utf-8")
    (folder / "original.xml").write_text(original, encoding="utf-8")
    procedure = ET.fromstring(original)
    procedure.find("Labware/Name").text = "48 WELL PLATE"
    root = ET.Element("BTIProtocol")
    ET.SubElement(root, "ProtocolType", Type="Standard")
    method = ET.SubElement(ET.SubElement(root, "BTIMethods"), "BTIMethod")
    method.append(procedure)
    xml = ET.tostring(root, encoding="unicode")
    (folder / "requested_protocol.xml").write_text(xml, encoding="utf-8")
    summary["phase"] = "new_experiment_ex"
    experiment = reader.biotek.app.new_experiment_ex(definition_type=1, definition_str=xml)
    if experiment is None:
        raise RuntimeError("NewExperimentEx returned None; see diagnostic.log")
    plate = experiment.plates.add()
    summary["phase"] = "readback"
    readback = plate.get_procedure()
    (folder / "readback.xml").write_text(readback, encoding="utf-8")
    after = ET.fromstring(readback)
    summary["labware"] = after.findtext("Labware/Name")
    summary["well_selection"] = after.findtext("WellSelection")
    summary["rows"] = plate._invoke_property_get("MaxRows", result_wanted=True)
    summary["columns"] = plate._invoke_property_get("MaxColumns", result_wanted=True)
    summary["phase"] = "partial_selection"
    partial = reader.build_bti_partial_plate_xml(single_block=False, wells=["A1"])
    (folder / "partial.xml").write_text(partial, encoding="utf-8")
    plate.set_partial_plate(partial)
    if capture.errors:
        raise RuntimeError("Gen5 logged errors; see diagnostic.log")
    summary["phase"] = "validate"
    summary["validation"] = plate.validate_procedure(False)
    summary["passed"] = (summary["labware"] == "48 WELL PLATE" and summary["well_selection"] == "Runtime"
                         and summary["validation"][0] == 1)
except Exception as error:
    summary["error"] = str(error)
    logger.exception("Metadata probe failed during %s", summary["phase"])
finally:
    summary["driver_errors"] = capture.errors
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.removeHandler(handler)
    driver_logger.removeHandler(handler)
    driver_logger.removeHandler(capture)
    handler.close()
print(folder.resolve())
print(json.dumps(summary, indent=2))
