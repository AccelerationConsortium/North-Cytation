"""Room environment (temperature, humidity, pressure) saved with each run's data.

At the end of every live run, experiment_run_logger.finalize_run calls export_for_run, which cuts
the run's window (plus padding) out of the sensor log and writes
<output folder>/environmental_metadata/:
    environment.csv   timestamp, one column per sensor value, during_run (False in the padding)
    summary.yaml      source log, run window, reading count, gaps, min/mean/max during the run
    environment.png   one panel per plotted value, run window shaded

No workflow changes are needed; the run only has to call lash_e.save_run_info(output_dir).
Sensor log path, columns and padding: environment_log in robot_state/robot_hardware.yaml.

Earlier runs (or a run that crashed before finishing), from the computer that has the log:
    python -m experiment_tracking.environment_metadata --run-id <run_id>
    python -m experiment_tracking.environment_metadata --all     # live runs without the folder yet

Values at each measurement, inside a workflow or analysis:
    env = read_environment(start, end)
    data = pd.merge_asof(data.sort_values("time"), env, left_on="time", right_on="timestamp",
                         direction="nearest")
"""
import argparse
import csv
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = REPO_ROOT / "robot_state" / "robot_hardware.yaml"
RUNS_CSV = Path(__file__).resolve().parent / "experiment_runs.csv"
FOLDER_NAME = "environmental_metadata"
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"

# Axis labels for the plot; a value not listed is labelled with its column name
_LABELS = {"temp_c": "Temperature (°C)", "humidity_pct": "Relative humidity (%)",
           "pressure_pa": "Pressure (Pa)"}

_logger = logging.getLogger(__name__)


def load_settings(settings_file=SETTINGS_FILE):
    """The environment_log section of robot_hardware.yaml. Raises KeyError if it is missing."""
    with open(settings_file, encoding="utf-8") as f:
        return yaml.safe_load(f)["environment_log"]


def load_log(settings=None):
    """The whole sensor log as a DataFrame: timestamp (local time) plus the renamed value columns,
    sorted by time. Rows whose timestamp cannot be read are dropped; blank values stay NaN.
    Raises FileNotFoundError if the log is not on this computer."""
    import pandas as pd
    settings = settings or load_settings()
    ts_col, columns = settings["timestamp_column"], settings["columns"]
    data = pd.read_csv(settings["log_file"], usecols=[ts_col, *columns])
    when = pd.to_datetime(data[ts_col], format="ISO8601", errors="coerce")
    if when.dt.tz is not None:  # Stored with a UTC offset: compare in this computer's local time
        when = when.dt.tz_convert(datetime.now().astimezone().tzinfo).dt.tz_localize(None)
    out = pd.DataFrame({"timestamp": when})
    for raw, name in columns.items():
        out[name] = pd.to_numeric(data[raw], errors="coerce")
    return out.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable").reset_index(drop=True)


def read_environment(start, end, settings=None, log=None):
    """Sensor readings from start to end (datetimes, local time), as load_log returns them."""
    log = load_log(settings) if log is None else log
    return log[(log.timestamp >= start) & (log.timestamp <= end)].reset_index(drop=True)


def _gaps(times, start, end, limit):
    """Stretches longer than limit with no reading, counting the run's start and end as edges."""
    edges = [start, *times, end]
    return [{"from": a.strftime(DATETIME_FMT), "to": b.strftime(DATETIME_FMT),
             "minutes": round((b - a).total_seconds() / 60, 1)}
            for a, b in zip(edges, edges[1:]) if b - a > limit]


def _number(value):
    import math
    return None if value is None or math.isnan(value) else round(float(value), 3)


def _plot(data, names, start, end, path):
    """One panel per value (never two scales on one axis), run window shaded, saved as PNG."""
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure  # No pyplot: never changes the backend of a running GUI
    figure = Figure(figsize=(9, 2.6 * len(names)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(names), 1, sharex=True, squeeze=False)[:, 0]
    for ax, name in zip(axes, names):
        ax.axvspan(start, end, color="#e8e8e8", linewidth=0, label="Run")
        ax.plot(data.timestamp, data[name], color="#2a6fdb", linewidth=1.5)
        ax.set_ylabel(_LABELS.get(name, name))
        ax.grid(True, color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel("Time")
    locator = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    figure.savefig(path, dpi=120)


def export_environment(output_folder, start, end, run_id=None, settings=None, log=None):
    """Write <output_folder>/environmental_metadata/ for a run from start to end (local datetimes).
    Returns the folder. Raises if the log cannot be read; export_for_run is the never-raising form."""
    settings = settings or load_settings()
    pad = timedelta(minutes=settings["padding_min"])
    data = read_environment(start - pad, end + pad, settings, log)
    data["during_run"] = (data.timestamp >= start) & (data.timestamp <= end)
    run = data[data.during_run]
    names = list(settings["columns"].values())

    folder = Path(output_folder) / FOLDER_NAME
    folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(folder / "environment.csv", index=False, date_format=DATETIME_FMT)

    intervals = run.timestamp.diff().dt.total_seconds().dropna()
    summary = {
        "run_id": run_id,
        "source_log": str(settings["log_file"]),
        "run_start": start.strftime(DATETIME_FMT),
        "run_stop": end.strftime(DATETIME_FMT),
        "padding_min": settings["padding_min"],
        "readings_during_run": int(len(run)),
        "median_interval_s": _number(intervals.median()) if len(intervals) else None,
        "gaps_over_min": settings["gap_warning_min"],
        "gaps": _gaps(list(run.timestamp), start, end, timedelta(minutes=settings["gap_warning_min"])),
        "during_run": {name: {"min": _number(run[name].min()), "mean": _number(run[name].mean()),
                              "max": _number(run[name].max()), "missing": int(run[name].isna().sum())}
                       for name in names},
        "saved": datetime.now().strftime(DATETIME_FMT),
    }
    with open(folder / "summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)

    plotted = [name for name in settings["plotted"] if data[name].notna().any()]
    if plotted:
        _plot(data, plotted, start, end, folder / "environment.png")
    return folder


def export_for_run(tracker, datetime_stopped):
    """Called by experiment_run_logger.finalize_run for live runs with an output folder. Never raises."""
    logger = getattr(tracker, "logger", None) or _logger
    try:
        start = datetime.strptime(tracker.datetime_started, DATETIME_FMT)
        folder = export_environment(tracker.output_folder, start, datetime_stopped, run_id=tracker.run_id)
        logger.info(f"Environment data for run {tracker.run_id} saved to {folder}")
        return folder
    except Exception as error:
        logger.warning(f"Could not save environment data for run {tracker.run_id}: {error!r}. "
                       f"Retry later: python -m experiment_tracking.environment_metadata --run-id {tracker.run_id}")
        return None


def _runs_to_export(rows, run_id, force):
    if run_id:
        chosen = [row for row in rows if row["run_id"] == run_id]
        if not chosen:
            raise SystemExit(f"Run {run_id} is not in {RUNS_CSV}")
        return chosen
    return [row for row in rows
            if row["simulate"] == "False" and row["output_folder"] and row["datetime_stopped"]
            and (force or not (Path(row["output_folder"]) / FOLDER_NAME).exists())]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Save the room environment data for earlier runs into their output folders.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--run-id", help="One run from experiment_runs.csv")
    which.add_argument("--all", action="store_true", help="Every live run whose output folder has no environment data yet")
    parser.add_argument("--force", action="store_true", help="With --all: redo runs that already have it")
    args = parser.parse_args(argv)

    with open(RUNS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    runs = _runs_to_export(rows, args.run_id, args.force)
    if not runs:
        print("Nothing to export.")
        return
    settings = load_settings()
    log = load_log(settings)  # Read once for all runs
    for row in runs:
        if not row["output_folder"] or not row["datetime_stopped"]:
            print(f"{row['run_id']}: skipped, no output folder or stop time recorded")
            continue
        folder = export_environment(row["output_folder"],
                                    datetime.strptime(row["datetime_started"], DATETIME_FMT),
                                    datetime.strptime(row["datetime_stopped"], DATETIME_FMT),
                                    run_id=row["run_id"], settings=settings, log=log)
        print(f"{row['run_id']}: {folder}")


if __name__ == "__main__":
    main()
