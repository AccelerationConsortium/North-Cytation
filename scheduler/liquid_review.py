"""Read-only liquid review. Workflow usage comes exclusively from execution logs."""
from dataclasses import dataclass
from pathlib import Path
import ast
import csv
import hashlib
import json
import re

import numpy as np
import pandas as pd
import yaml

from pipetting_data.pipetting_parameters import PipettingParameters


@dataclass
class Dispense:
    liquid: str
    volume_ul: float
    destination: str
    line: int
    tip: str = ""
    method: str = "pipette"


def parse_log(path):
    """Count destination operations once, never both aspiration and dispensing."""
    events = []
    liquid = "Unidentified"
    pending = None
    tip = ""
    pumps = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        message = re.split(r" - (?:DEBUG|INFO|WARNING|ERROR) - ", line, maxsplit=1)[-1]
        if "Liquid review context: " in message:
            try:
                context = json.loads(message.split("Liquid review context: ", 1)[1])
                pending = context.get("liquid") or "Unidentified"
            except (ValueError, TypeError):
                pending = "Unidentified"
        match = re.search(r"Parameters for (.+?) [\d.eE+-]+mL from ", message)
        if match:
            pending = match[1]
        match = re.search(r"for '([^']+)' in optimal_conditions", message)
        if match:
            pending = match[1]
        match = re.search(r"Using calibrated \w+=\S+ for (.+?) at ", message)
        if match:
            pending = match[1]
        match = re.search(r"Selected (\w+) for volume", message)
        if match:
            tip = match[1]
        if "Pipetting from vial " in message or "Pipetting aspirating " in message:
            liquid = pending or "Unidentified"
            pending = None
        match = re.search(r"Pipetting into vial (.+?), amount: ([\d.eE+-]+) mL", message)
        if match:
            events.append(Dispense(liquid, float(match[2]) * 1000, match[1], number, tip))
        match = re.search(r"Pipetting dispensing ([\d.eE+-]+) mL from well plate index (\d+)", message)
        if match:
            events.append(Dispense(liquid, float(match[1]) * 1000, "well " + match[2], number, tip))
        match = re.search(r"Transferring ([\d.eE+-]+) mL into well #(\d+) of (\S+)", message)
        if match:
            events.append(Dispense(liquid, float(match[1]) * 1000, f"{match[3]} well {match[2]}", number, tip))
        match = re.search(r"Loaded pump (\d+): .*liquid=(\S+)", message)
        if match:
            pumps[match[1]] = match[2]
        match = re.search(r"Dispensing into vial (.+?) from reservoir (\d+): ([\d.eE+-]+) mL", message)
        if match:
            name = pumps.get(match[2], "Unidentified")
            events.append(Dispense(name if name not in {"None", "none"} else "Unidentified",
                                   float(match[3]) * 1000, match[1], number, method="reservoir"))
    return events


def calibration_files(root, liquid):
    # Exact class or a suffix variant; never include backups as active calibration.
    prefix = "optimal_conditions_" + liquid.casefold()
    return sorted(p for p in (Path(root) / "pipetting_data").glob("optimal_conditions_*.csv")
                  if (p.stem.casefold() == prefix or p.stem.casefold().startswith(prefix + "_"))
                  and "backup" not in p.stem.casefold())


def fingerprint(root):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(root) / "pipetting_data").glob("optimal_conditions_*.csv")}


def load_calibration(path):
    frame = pd.read_csv(path)
    frame["volume_target"] = pd.to_numeric(frame["volume_target"], errors="raise")
    if not np.isfinite(frame["volume_target"]).all():
        raise ValueError("Calibration contains non-finite target volumes")
    return frame.sort_values("volume_target").drop_duplicates("volume_target", keep="last")


def regime(volume):
    return "small_tip" if volume < 200 else "large_tip"


def coverage(frame, volume, tip=""):
    if frame is None or frame.empty:
        return "Missing calibration"
    if tip and tip != regime(volume):
        return "Tip differs from calibration regime"
    values = frame.loc[(frame.volume_target < 200) == (volume < 200), "volume_target"]
    if values.empty or not values.min() <= volume <= values.max():
        return "Outside tip-specific coverage"
    return "Measured point" if np.isclose(values, volume).any() else "Interpolated coverage"


def current_parameters(frame, volume):
    if coverage(frame, volume) not in {"Measured point", "Interpolated coverage"}:
        return None
    subset = frame[(frame.volume_target < 200) == (volume < 200)]
    values = vars(PipettingParameters()).copy()
    for name in values:
        if name in subset:
            column = pd.to_numeric(subset[name], errors="coerce")
            if not np.isfinite(column).all():
                return None
            value = float(np.interp(volume, subset.volume_target, column))
            values[name] = round(value) if name in {"aspirate_speed", "dispense_speed", "retract_speed"} else value
    # Runtime blowout speed falls back to aspirate speed.
    if values["blowout_speed"] is None:
        values["blowout_speed"] = values["aspirate_speed"]
    return values


def matching_measurements(root, liquid, calibration):
    path = Path(root) / "pipetting_data/master_pipetting_measurements.csv"
    if not path.exists():
        return pd.DataFrame(), {"Measurement file missing": 1}
    data = pd.read_csv(path)
    data = data[data.liquid_type.str.casefold() == liquid.casefold()].copy()
    accepted, excluded = [], {}
    duplicates = data.duplicated(keep="first")
    if duplicates.any():
        excluded["Exact duplicate measurement"] = int(duplicates.sum())
        data = data.loc[~duplicates]
    settings = {volume: current_parameters(calibration, volume * 1000)
                for volume in data.target_volume_ml.unique()}
    for index, row in data.iterrows():
        expected = settings[row.target_volume_ml]
        reason = None
        if row.validation_type != "vial_to_vial":
            reason = "Different dispensing method"
        elif expected is None:
            reason = "No current tip-specific calibration"
        else:
            for key, value in expected.items():
                if key == "overaspirate_vol":
                    continue
                if key not in row or pd.isna(row[key]):
                    reason = "Missing settings"
                    break
                if not np.isclose(float(row[key]), value, rtol=1e-6, atol=1e-8):
                    reason = f"Different {key}"
                    break
        if reason is None and not all(np.isfinite(row[k]) for k in
                                     ("target_volume_ml", "measured_volume_ml", "overaspirate_vol")):
            reason = "Invalid measurement"
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
        else:
            accepted.append(index)
    return data.loc[accepted].copy(), excluded


def estimate_optimum(group):
    """Local zero-error regression; session bootstrap preserves replicate dependence."""
    grouped = group.groupby(["session_id", "overaspirate_vol"], as_index=False).agg(
        measured_volume_ml=("measured_volume_ml", "mean"), target_volume_ml=("target_volume_ml", "first"))
    x = grouped.overaspirate_vol.to_numpy() * 1000
    y = (grouped.measured_volume_ml - grouped.target_volume_ml).to_numpy() * 1000
    if len(np.unique(x)) < 2:
        return None, "One tested V_OA: observed performance available; optimum not identifiable"
    slope, intercept = np.polyfit(x, y, 1)
    if slope <= 1e-6:
        return None, "No reliable positive response to V_OA"
    optimum = -intercept / slope
    if not min(x) <= optimum <= max(x):
        return None, "Zero-error estimate is outside tested V_OA values"
    sessions = grouped.session_id.unique()
    if len(sessions) < 3:
        return (optimum, None), "Too few independent sessions for an uncertainty interval"
    rng = np.random.default_rng(42)
    roots = []
    session_indices = [np.flatnonzero(grouped.session_id.to_numpy() == session) for session in sessions]
    for _ in range(300):
        indices = np.concatenate([session_indices[i] for i in rng.integers(0, len(sessions), len(sessions))])
        sx, sy = x[indices], y[indices]
        if np.ptp(sx) < 1e-8:
            continue
        a, b = np.polyfit(sx, sy, 1)
        if a > 1e-6:
            roots.append(-b / a)
    if len(roots) < 270:
        return (optimum, None), "Unstable response; uncertainty not identifiable"
    return (optimum, tuple(np.percentile(roots, [2.5, 97.5]))), "Exploratory 95% session-bootstrap interval"


def environmental_history(root):
    """Support historical correction rows whose environmental fields lack headers."""
    path = Path(root) / "pipetting_data/parameter_corrections.csv"
    if not path.exists():
        return pd.DataFrame()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if "temp_c" not in header:
            header += ["temp_c", "humidity_pct", "pressure_pa"]
        rows = [dict(zip(header, row)) for row in reader]
    return pd.DataFrame(rows)


def original_runs(root, liquid):
    folder = Path(root) / "sdl_pipette_calibration/output"
    matches = []
    for path in sorted(folder.glob("*/raw_measurements.csv")):
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                first = next(csv.DictReader(handle), {})
            if first.get("liquid", "").casefold() == liquid.casefold():
                matches.append(path)
        except (OSError, csv.Error, UnicodeError):
            continue
    return matches


@dataclass(frozen=True)
class PastResult:
    kind: str
    path: Path
    label: str


def _past_result_liquid(path, kind):
    if kind == "calibration":
        suffix = path.stem.removeprefix("optimal_conditions_")
        if suffix != path.stem:
            return suffix
        raw = path.parent / "raw_measurements.csv"
        if raw.exists():
            with raw.open(newline="", encoding="utf-8-sig") as handle:
                return next(csv.DictReader(handle), {}).get("liquid", "")
        # Legacy output/calibration_runs directories encode the liquid in the run name.
        if path.parent.parent.name == "calibration_runs":
            candidates = path.parent.name.lower().split("_")
            candidates = [part for part in candidates if part not in {"simulate", "calibration", "simplified"}
                          and not part.isdigit()]
            return "_".join(candidates)
        return ""
    if path.parent.name.startswith("validation_run_"):
        return path.parent.name.rsplit("_", 1)[-1]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        first = next(csv.DictReader(handle), {})
    return first.get("liquid", first.get("_liquid", ""))


def _past_result_is_real(path):
    """Execution summaries outrank requested mode; unknown runs are omitted."""
    parent = path.parent
    for name in ("validation_analysis.json", "experiment_summary.json"):
        source = parent / name
        if source.exists():
            try:
                value = json.loads(source.read_text(encoding="utf-8")).get("simulation_mode")
                if isinstance(value, bool):
                    return not value
            except (OSError, ValueError, TypeError):
                return False
    for name in ("experiment_config_used.yaml", "experiment_config.yaml"):
        source = parent / name
        if source.exists():
            try:
                config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
                value = (config.get("experiment", config) or {}).get("simulate")
                if isinstance(value, bool):
                    return not value
            except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError):
                return False
    return False


def past_results(root, liquid):
    """Saved outcomes only: optimal calibration rows and validation measurements."""
    root = Path(root)
    files = list((root / "sdl_pipette_calibration/output").rglob("optimal_conditions*.csv"))
    files += list((root / "output/calibration_runs").rglob("optimal_conditions*.csv"))
    files += list((root / "output/calibration_v2_runs").rglob("optimal_conditions*.csv"))
    files += list((root / "sdl_pipette_calibration/output").rglob("raw_validation_data.csv"))
    files += list((root / "output/validation_runs").rglob("raw_validation_data.csv"))
    results = []
    for path in files:
        kind = "calibration" if path.name.startswith("optimal_conditions") else "validation"
        try:
            if not _past_result_is_real(path):
                continue
            if _past_result_liquid(path, kind).casefold() != liquid.casefold():
                continue
            parent = path.parent.name
            project = next((p.name for p in path.parents if p.name.startswith("pc_")), "")
            label = f"{project + ' / ' if project else ''}{parent} — {'Optimal conditions' if kind == 'calibration' else 'Validation'}"
            results.append(PastResult(kind, path, label))
        except (OSError, csv.Error, UnicodeError):
            continue
    return sorted(results, key=lambda result: result.label.casefold())


def past_result_measurements(result):
    frame = pd.read_csv(result.path)
    if result.kind == "calibration":
        target = next((frame[key] * (1000 if key.endswith("_ml") or key.endswith("_mL") else 1)
                       for key in ("volume_target_ul", "volume_target", "volume_target_ml", "target_volume_mL")
                       if key in frame), pd.Series(dtype=float))
        measured = next((frame[key] * (1000 if key.endswith("_ml") or key.endswith("_mL") else 1)
                         for key in ("volume_measured_ul", "volume_measured", "volume_measured_ml", "average_obtained_volume_mL")
                         if key in frame), pd.Series(dtype=float))
        frame["target_ul"] = pd.to_numeric(target, errors="coerce")
        frame["measured_ul"] = pd.to_numeric(measured, errors="coerce")
    else:
        frame["target_ul"] = pd.to_numeric(frame.get("volume_target_ul", frame.get("volume", frame.get("volume_ml", pd.Series(dtype=float)))*1000), errors="coerce")
        frame["measured_ul"] = pd.to_numeric(frame.get("volume_measured_ul", frame.get("calculated_volume", pd.Series(dtype=float))*1000), errors="coerce")
    return frame


def past_result_parameters(row, kind):
    """Extract saved parameters without evaluating arbitrary CSV text."""
    if kind == "validation" and "parameters_used" in row and pd.notna(row["parameters_used"]):
        try:
            parameters = ast.literal_eval(row["parameters_used"])
            if isinstance(parameters, dict):
                return {str(k): v for k, v in parameters.items() if k != "volume_ml"}
        except (ValueError, SyntaxError, TypeError):
            pass
    params = {}
    for key, value in row.items():
        if pd.isna(value):
            continue
        if key.startswith("hardware_parameters_"):
            params[key.removeprefix("hardware_parameters_")] = value
        elif key == "calibration_overaspirate_vol":
            params["overaspirate_vol"] = value
        elif key in vars(PipettingParameters()):
            params[key] = value
    return params
