"""Liquids tab: each liquid's settings, calibrated volumes and dates; add or edit a liquid.

Settings live in robot_state/liquids.yaml (density_g_ml, refill_pipets, condition_tip).
Calibrations live in pipetting_data/optimal_conditions_<liquid>.csv; the merge the agent runs
(import_calibration_run.py) dates each row in a 'calibrated_on' column. Validation sessions are
in pipetting_data/master_pipetting_sessions.csv.

Editing writes one line of liquids.yaml, keeping its comments and layout, and is checked with
the same loader the robot uses before the file is replaced.
"""

import os
import re
from pathlib import Path

import pandas as pd

from pipetting_data.liquid_registry import LIQUID_REGISTRY_PATH, load_liquids

PIPETTING_DATA = Path(__file__).resolve().parent
SESSIONS_FILE = PIPETTING_DATA / "master_pipetting_sessions.csv"
NAME_RE = re.compile(r"^[A-Za-z0-9_%.\-]+$")


def calibration_files(liquid, directory=PIPETTING_DATA):
    prefix = "optimal_conditions_"
    return sorted(p for p in Path(directory).glob(f"{prefix}*.csv") if p.stem[len(prefix):].casefold() == liquid.casefold())


def _volume_range(volumes):
    volumes = sorted({round(float(v), 2) for v in volumes if pd.notna(v)})
    if not volumes:
        return "—"
    fmt = lambda v: f"{v:g}"
    return fmt(volumes[0]) if len(volumes) == 1 else f"{fmt(volumes[0])}–{fmt(volumes[-1])} µL ({len(volumes)})"


def calibration_summary(liquid, directory=PIPETTING_DATA):
    """{'small_tip': '...', 'large_tip': '...', 'last_calibrated': str or None}."""
    frames = []
    for path in calibration_files(liquid, directory):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue
    if not frames:
        return {"small_tip": "not calibrated", "large_tip": "not calibrated", "last_calibrated": None}
    data = pd.concat(frames, ignore_index=True)
    tips = data["tip_type"] if "tip_type" in data else pd.Series(["unknown"] * len(data))
    summary = {tip: _volume_range(data.loc[tips == tip, "volume_target"]) if "volume_target" in data else "—"
               for tip in ("small_tip", "large_tip")}
    dates = data["calibrated_on"].dropna().astype(str) if "calibrated_on" in data else pd.Series(dtype=str)
    summary["last_calibrated"] = max(dates) if len(dates) else None
    return summary


def last_validated(liquid, sessions_file=SESSIONS_FILE, sessions=None):
    """Date of the most recent real (not simulated) validation session, or None."""
    try:
        if sessions is None:
            sessions = pd.read_csv(sessions_file, low_memory=False)
        rows = sessions[sessions.liquid_type.astype(str).str.casefold() == liquid.casefold()]
        if "simulate" in rows:
            rows = rows[~rows.simulate.astype(str).str.casefold().eq("true")]
        if rows.empty:
            return None
        when = pd.to_datetime(rows.timestamp.astype(str), format="%Y%m%d_%H%M%S", errors="coerce").dropna()
        return when.max().strftime("%Y-%m-%d") if len(when) else None
    except Exception:
        return None


RECENT_DAYS = 90


def recent_use(sessions=None, transfers_file=None, days=RECENT_DAYS, now=None):
    """{liquid (casefolded): count} over the last `days`: dispenses in the transfer log, plus real
    validation sessions (a stand-in until the log has filled up). Approximate by design."""
    now = pd.Timestamp(now or pd.Timestamp.now())
    since = now - pd.Timedelta(days=days)
    counts = {}
    if sessions is not None and len(sessions):
        rows = sessions
        if "simulate" in rows:
            rows = rows[~rows.simulate.astype(str).str.casefold().eq("true")]
        when = pd.to_datetime(rows.timestamp.astype(str), format="%Y%m%d_%H%M%S", errors="coerce")
        for liquid in rows.loc[when >= since, "liquid_type"].astype(str).str.casefold():
            counts[liquid] = counts.get(liquid, 0) + 1
    try:
        from experiment_tracking.sample_tracking import load_transfers
        for row in load_transfers(transfers_file):
            when = pd.to_datetime(row.get("time"), errors="coerce")
            if row.get("liquid") and pd.notna(when) and when >= since:
                key = row["liquid"].casefold()
                counts[key] = counts.get(key, 0) + 1
    except Exception:
        pass
    return counts


def liquids_table(registry=LIQUID_REGISTRY_PATH, directory=PIPETTING_DATA, sessions_file=SESSIONS_FILE,
                  transfers_file=None):
    """One dict per liquid for the tab, most used recently first (then by name)."""
    liquids = load_liquids(Path(registry))
    try:
        sessions = pd.read_csv(sessions_file, low_memory=False)
    except Exception:
        sessions = None
    use = recent_use(sessions, transfers_file)
    rows = []
    for name, props in liquids.items():
        calibration = calibration_summary(name, directory)
        rows.append({"liquid": name, "density": props["density"], "reuse_tip": not props["refill_pipets"],
                     "condition_tip": props["condition_tip"], "small_tip": calibration["small_tip"],
                     "large_tip": calibration["large_tip"], "last_calibrated": calibration["last_calibrated"],
                     "last_validated": last_validated(name, sessions=sessions) if sessions is not None else None,
                     "recent_use": use.get(name.casefold(), 0)})
    return sorted(rows, key=lambda r: (-r["recent_use"], r["liquid"].casefold()))


REPO_ROOT = PIPETTING_DATA.parent


def liquid_usage(name, repo_root=REPO_ROOT, sessions_file=SESSIONS_FILE):
    """What still refers to a liquid, for the Delete window:
    {'files': [workflow and config files naming it], 'calibration_files': [...], 'validation_sessions': n}.
    A file counts when the exact name appears as a whole word (water_2 doesn't count as water)."""
    root = Path(repo_root)
    pattern = re.compile(rf"(?<![A-Za-z0-9_%.\-]){re.escape(name)}(?![A-Za-z0-9_%.\-])")
    candidates = [*root.glob("workflows/*.py"), *root.glob("workflows/helpers/*.py"),
                  *root.glob("workflow_configs/**/*.yaml"), *root.glob("workflow_configs/**/*.yml")]
    files = []
    for path in sorted(set(candidates)):
        try:
            if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                files.append(str(path.relative_to(root)).replace("\\", "/"))
        except OSError:
            continue
    try:
        sessions = pd.read_csv(sessions_file, low_memory=False)
        count = int((sessions.liquid_type.astype(str).str.casefold() == name.casefold()).sum())
    except Exception:
        count = 0
    return {"files": files, "calibration_files": [p.name for p in calibration_files(name, root / "pipetting_data")],
            "validation_sessions": count}


def delete_liquid(name, registry=LIQUID_REGISTRY_PATH):
    """Remove one liquid's line from liquids.yaml; nothing else changes. Its calibration file and
    validation history are kept (adding the liquid back with the same name brings them back)."""
    path = Path(registry)
    if name not in load_liquids(path):
        raise ValueError(f"{name} isn't in the list")
    pattern = re.compile(rf"^\s+{re.escape(name)}\s*:")
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if not pattern.match(l)]
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        if name in load_liquids(temporary):  # the robot's own loader must accept the result
            raise ValueError(f"Could not remove {name}")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)


def _line(name, density, refill_pipets, condition_tip):
    flag = lambda value: "true" if value else "false"
    return (f"  {name}: {{density_g_ml: {float(density):g}, refill_pipets: {flag(refill_pipets)}, "
            f"condition_tip: {flag(condition_tip)}}}")


def save_liquid(name, density, reuse_tip, condition_tip, registry=LIQUID_REGISTRY_PATH, new=False):
    """Add a liquid (new=True) or change one, writing only its line of liquids.yaml.
    reuse_tip is the opposite of the file's refill_pipets. Raises ValueError on bad input."""
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise ValueError("Liquid names may only use letters, numbers, _ % . and -")
    if not (float(density) > 0):
        raise ValueError("Density must be a positive number (g/mL)")
    path = Path(registry)
    lines = path.read_text(encoding="utf-8").splitlines()
    existing = load_liquids(path)
    if new and name in existing:
        raise ValueError(f"{name} is already in the list")
    if not new and name not in existing:
        raise ValueError(f"{name} isn't in the list")
    line = _line(name, density, not reuse_tip, condition_tip)
    pattern = re.compile(rf"^\s+{re.escape(name)}\s*:")
    if new:
        lines.append(line)
    else:
        lines = [line if pattern.match(l) else l for l in lines]
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        check = load_liquids(temporary)  # the robot's own loader must accept it
        if name not in check:
            raise ValueError("The edited file didn't contain the liquid")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)
