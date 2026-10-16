"""Validated vial CSV input and atomic GUI saves, independent of Qt."""
import csv
from decimal import Decimal, InvalidOperation
import io
import os
from pathlib import Path
import tempfile


REQUIRED_FIELDS = {"vial_index", "vial_name", "location", "location_index", "vial_volume"}


def parse_vial_csv(text):
    """Choose a separator from the schema header, never a truncated data row."""
    text = text.removeprefix("\ufeff")
    header = text.splitlines()[0] if text else ""
    candidates = []
    for delimiter in (",", "\t", ";", "|"):
        try:
            fields = next(csv.reader([header], delimiter=delimiter, strict=True), [])
        except csv.Error:
            continue
        if REQUIRED_FIELDS.issubset(fields):
            candidates.append(delimiter)
    if len(candidates) != 1:
        raise ValueError("Invalid vial CSV header: expected columns "
                         + ", ".join(sorted(REQUIRED_FIELDS))
                         + " separated by commas, tabs, semicolons or pipes.")
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=candidates[0], strict=True)
    fields = next(reader)
    if any(not field.strip() for field in fields) or len(set(fields)) != len(fields):
        raise ValueError("Vial CSV header contains empty or duplicate column names.")
    records = []
    names, identities = set(), set()
    for values in reader:
        if not values:  # Match csv.DictReader's handling of blank lines.
            continue
        line = reader.line_num
        if len(values) != len(fields):
            raise ValueError(f"CSV row ending on line {line}: expected {len(fields)} fields, got {len(values)}.")
        record = dict(zip(fields, values))
        for field, seen in (("vial_name", names), ("vial_index", identities)):
            value = record[field].strip()
            if not value or value in seen:
                raise ValueError(f"CSV line {line}: missing or duplicate {field}: {value!r}.")
            seen.add(value)
        try:
            volume = Decimal(record["vial_volume"])
        except InvalidOperation as error:
            raise ValueError(f"CSV line {line}: vial_volume must be a finite, nonnegative number.") from error
        if not volume.is_finite() or volume < 0:
            raise ValueError(f"CSV line {line}: vial_volume must be a finite, nonnegative number.")
        records.append(record)
    if not records:
        raise ValueError("No vial data found in file.")
    return fields, records


def read_vial_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return parse_vial_csv(stream.read())


def save_vial_csv(path, fields, records):
    """Validate serialization and its on-disk round trip before replacing a file."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    text = output.getvalue()
    expected = parse_vial_csv(text)
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=destination.parent, prefix=destination.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if read_vial_csv(temporary) != expected:
            raise ValueError("Vial CSV failed verification; original file was preserved.")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
