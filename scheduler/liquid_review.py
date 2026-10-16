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


def filter_dates(data, start=None, end=None):
    """Keep rows whose timestamp is in [start, end); rows with unreadable timestamps are dropped."""
    when = pd.to_datetime(data.timestamp, format="%Y%m%d_%H%M%S", errors="coerce")
    keep = when.notna()
    if start is not None:
        keep &= when >= start
    if end is not None:
        keep &= when < end
    return data[keep].copy(), int((~keep).sum())


# Physical response of delivered volume to overaspirate is ~1 uL/uL; outside this range the line fit is noise.
MIN_RESPONSE, MAX_RESPONSE = 0.3, 3.0

UNRECORDED = "Stage V_OA not recorded (older validation saved the stage 1 value on every stage)"
NO_STAGES = "Fewer than three stages recorded (V_OA response unknowable)"


def session_overaspirate_estimates(data, start=None, end=None):
    """Per session and target: the V_OA where a line through the stage means (V_OA, V_M - V_T) crosses zero.

    Every stage is used, not only the accepted one, because the tested V_OA values are what reveal the response."""
    counts = {}
    columns = ["session_id", "target_ul", "when", "oa_ul", "slope", "points"]
    if data.empty:
        return pd.DataFrame(columns=columns), counts
    data = data.copy()
    if "simulation_mode" in data:
        simulated = data.simulation_mode.astype(str).str.casefold().eq("true")
        counts["Simulated measurements"] = int(simulated.sum())
        data = data[~simulated]
    data = data[data.session_id.notna() & data.session_id.astype(str).str.strip().ne("")]
    data, outside = filter_dates(data, start, end)
    if outside:
        counts["Outside date range"] = outside
    rows = []
    for (session, volume), group in data.groupby(["session_id", "target_volume_ml"]):
        when = pd.to_datetime(group.timestamp, format="%Y%m%d_%H%M%S", errors="coerce").max()
        points = group.assign(oa=group.overaspirate_vol.round(9)).groupby("oa").measured_volume_ml.mean()
        if len(points) < 2:
            key = NO_STAGES if set(group.optimization_stage.dropna().astype(int)) != {1, 2, 3} else UNRECORDED
            counts[key] = counts.get(key, 0) + 1
            continue
        x = points.index.to_numpy() * 1000
        y = (points.to_numpy() - volume) * 1000
        slope, intercept = np.polyfit(x, y, 1)
        if not MIN_RESPONSE <= slope <= MAX_RESPONSE:
            counts["Implausible response to V_OA"] = counts.get("Implausible response to V_OA", 0) + 1
            continue
        rows.append(dict(session_id=session, target_ul=volume * 1000, when=when, oa_ul=-intercept / slope,
                         slope=slope, points=len(points)))
    return pd.DataFrame(rows, columns=columns), counts


def summarize_overaspirate(estimates, current_oa_ul):
    """Per target: mean of session estimates (equal session weight) with a t interval, versus the current V_OA."""
    from scipy.stats import t

    rows = []
    for volume, group in estimates.groupby("target_ul", sort=True):
        values = group.oa_ul
        n = len(values)
        mean = values.mean()
        margin = float(t.ppf(.975, n - 1) * values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        rows.append(dict(target_ul=volume, current_ul=current_oa_ul.get(volume, np.nan), sessions=n, mean_ul=mean,
                         sd_ul=values.std(ddof=1) if n > 1 else np.nan,
                         ci_low_ul=mean - margin, ci_high_ul=mean + margin, last_validation=group.when.max()))
    return pd.DataFrame(rows)


def summarize_past_validation(matched, excluded=None, start=None, end=None):
    """Accepted outcomes; equal session weighting with a t interval for the mean.

    start (inclusive) and end (exclusive) bound the validation timestamp."""
    from scipy.stats import t

    counts = dict(excluded or {})
    if matched.empty:
        return matched.copy(), pd.DataFrame(), counts
    data = matched.copy()
    if "accepted" not in data:
        counts["Acceptance unresolved"] = len(data)
        return data.iloc[:0], pd.DataFrame(), counts
    accepted = data.accepted.astype(str).str.casefold()
    counts["Stage not accepted"] = int(accepted.eq("false").sum())
    counts["Acceptance unresolved"] = int((~accepted.isin(["true", "false"])).sum())
    data = data[accepted.eq("true")].copy()
    if "simulation_mode" in data:
        simulated = data.simulation_mode.astype(str).str.casefold().eq("true")
        counts["Simulated measurements"] = int(simulated.sum())
        data = data[~simulated]
    missing_session = data.session_id.isna() | data.session_id.astype(str).str.strip().eq("")
    counts["Missing session identity"] = int(missing_session.sum())
    data = data[~missing_session].copy()
    if start is not None or end is not None:
        data, counts["Outside date range"] = filter_dates(data, start, end)
    if data.empty:
        return data, pd.DataFrame(), {k: v for k, v in counts.items() if v}
    rows = []
    for volume, group in data.groupby("target_volume_ml", sort=True):
        session_means = group.groupby("session_id").measured_volume_ml.mean() * 1000
        n = len(session_means)
        mean = session_means.mean()
        margin = float(t.ppf(.975, n - 1) * session_means.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        dates = pd.to_datetime(group.timestamp, format="%Y%m%d_%H%M%S", errors="coerce")
        last = dates.max()
        rows.append(dict(target_ul=volume * 1000, mean_ul=mean, error_ul=mean - volume * 1000,
                         ci_low_ul=mean - margin, ci_high_ul=mean + margin,
                         sessions=n, measurements=len(group),
                         last_validation=last.strftime("%Y-%m-%d") if pd.notna(last) else "Unknown"))
    return data, pd.DataFrame(rows), {k: v for k, v in counts.items() if v}


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
