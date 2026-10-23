"""Fill missing master temperature and humidity from the nearest valid MQTT reading.

Dry run by default. --apply backs up both masters before replacing either one.
"""

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


MASTERS = (
    Path("pipetting_data/master_pipetting_sessions.csv"),
    Path("pipetting_data/master_pipetting_measurements.csv"),
)
FIELDS = ("temp_c", "humidity_pct")
MAX_OFFSET_SECONDS = 3600


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sensor_readings(path):
    readings = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["data_quality"] != "valid":
                continue
            try:
                stamp = datetime.fromisoformat(row["Timestamp"])
                temp = float(row["sht_temp_c"])
                humidity = float(row["sht_rh"])
            except (ValueError, TypeError):
                continue
            if math.isfinite(temp) and math.isfinite(humidity):
                readings.append((stamp, row["sht_temp_c"], row["sht_rh"]))
    readings.sort(key=lambda item: item[0])
    if not readings:
        raise ValueError("No valid sensor readings")
    return readings


def update_rows(path, readings):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns or not {"timestamp", *FIELDS}.issubset(columns):
            raise ValueError(f"Missing required columns in {path}")
        rows = list(reader)
    stamps = [reading[0] for reading in readings]
    counts = {field: 0 for field in FIELDS}
    offsets = []
    for row in rows:
        missing = [field for field in FIELDS if not row[field].strip()]
        if not missing:
            continue
        try:
            stamp = datetime.strptime(row["timestamp"], "%Y%m%d_%H%M%S")
        except ValueError as error:
            raise ValueError(f"Unparseable timestamp in {path}: {row['timestamp']}") from error
        index = bisect.bisect_left(stamps, stamp)
        candidates = readings[max(0, index - 1):min(len(readings), index + 1)]
        nearest = min(candidates, key=lambda item: abs((item[0] - stamp).total_seconds()))
        offset = abs((nearest[0] - stamp).total_seconds())
        if offset > MAX_OFFSET_SECONDS:
            continue
        for field, value in (("temp_c", nearest[1]), ("humidity_pct", nearest[2])):
            if field in missing:
                row[field] = value
                counts[field] += 1
        offsets.append(offset)
    return columns, rows, {"rows": len(rows), "filled": counts,
                           "largest_match_offset_seconds": max(offsets, default=None)}


def stage_csv(path, columns, rows):
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}_env_", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        with Path(name).open(newline="", encoding="utf-8") as handle:
            check = list(csv.DictReader(handle))
        if check != rows:
            raise ValueError(f"Staged CSV differs from intended rows: {path}")
        return Path(name)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-log", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    readings = sensor_readings(args.sensor_log)
    prepared = {path: update_rows(path, readings) for path in MASTERS}
    report = {"sensor_log": str(args.sensor_log), "sensor_sha256": sha256(args.sensor_log),
              "sensor_rows_used": len(readings), "max_offset_seconds": MAX_OFFSET_SECONDS,
              "files": {str(path): details for path, (_, _, details) in prepared.items()}}
    if not args.apply:
        print(json.dumps(report, indent=2))
        return
    backup = Path("pipetting_data/backups") / f"environment_{datetime.now():%Y%m%d_%H%M%S_%f}"
    backup.mkdir(parents=True, exist_ok=False)
    staged = {}
    try:
        for path, (columns, rows, _) in prepared.items():
            shutil.copy2(path, backup / path.name)
            if sha256(path) != sha256(backup / path.name):
                raise ValueError(f"Backup verification failed: {path}")
            report["files"][str(path)]["before_sha256"] = sha256(path)
            staged[path] = stage_csv(path, columns, rows)
        for path, temporary in staged.items():
            os.replace(temporary, path)
            report["files"][str(path)]["after_sha256"] = sha256(path)
        (backup / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    print(json.dumps({"backup": str(backup), **report}, indent=2))


if __name__ == "__main__":
    main()
