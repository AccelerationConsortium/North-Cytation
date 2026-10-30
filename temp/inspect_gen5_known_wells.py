"""Read-only metadata inspection of existing selected-well protocols; no reads/motion."""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biotek_new import Biotek_Wrapper

folder = Path("output/cytation_xml_only") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_known_wells")
folder.mkdir(parents=True)
handler = logging.FileHandler(folder / "diagnostic.log", encoding="utf-8")
logger = logging.getLogger("known_well_metadata")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
driver_logger = logging.getLogger("biotek_driver.utils.logger")
driver_logger.addHandler(handler)
results = {}
try:
    reader = Biotek_Wrapper(simulate=False, logger=logger)
    for name in ("Serena_Test_A1A3", "SQ_peroxide_A1A3", "SQ_peroxide_A4A6"):
        path = Path("C:/Protocols") / (name + ".prt")
        experiment = reader.biotek.new_experiment(str(path))
        if experiment is None:
            results[name] = {"error": "new_experiment returned None"}
            continue
        plate = experiment.plates.add()
        xml = plate.get_procedure()
        (folder / (name + ".xml")).write_text(xml, encoding="utf-8")
        results[name] = {"xml_saved": bool(xml)}
finally:
    (folder / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.removeHandler(handler)
    driver_logger.removeHandler(handler)
    handler.close()
print(folder.resolve())
