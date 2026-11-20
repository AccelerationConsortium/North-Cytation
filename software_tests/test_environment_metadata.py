"""Each live run saves the room environment during it into <output folder>/environmental_metadata/."""
import logging
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import yaml

from experiment_tracking import environment_metadata as env


def make_log(path, rows):
    lines = ["Timestamp,sht_temp_c,sht_rh,bmp_pa,other"] + [f"{t},{c},{h},100000,x" for t, c, h in rows]
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def settings(log_file):
    return {"log_file": str(log_file), "timestamp_column": "Timestamp",
            "columns": {"sht_temp_c": "temp_c", "sht_rh": "humidity_pct", "bmp_pa": "pressure_pa"},
            "plotted": ["temp_c", "humidity_pct"], "padding_min": 30, "gap_warning_min": 5}


def test_settings_in_robot_hardware_yaml():
    s = env.load_settings()
    assert s["columns"]["sht_rh"] == "humidity_pct" and s["padding_min"] >= 0


def test_export_writes_slice_summary_and_plot(tmp_path):
    log = tmp_path / "mqtt_log.csv"
    # Every minute 09:00-12:59, with a 20-minute gap during the run and one blank humidity
    rows = []
    for minute in range(240):
        t = datetime(2026, 11, 4, 9 + minute // 60, minute % 60)
        if datetime(2026, 11, 4, 10, 40) < t < datetime(2026, 11, 4, 11, 0):
            continue
        rows.append((t.strftime("%Y-%m-%dT%H:%M:%S"), 20 + minute / 100, "" if minute == 100 else 40))
    make_log(log, rows)
    folder = env.export_environment(tmp_path / "out", datetime(2026, 11, 4, 10, 0), datetime(2026, 11, 4, 11, 30),
                                    run_id="abc", settings=settings(log))

    assert folder == tmp_path / "out" / "environmental_metadata"
    data = pd.read_csv(folder / "environment.csv")
    assert list(data.columns) == ["timestamp", "temp_c", "humidity_pct", "pressure_pa", "during_run"]
    assert data.timestamp.iloc[0] == "2026-11-04 09:30:00" and data.timestamp.iloc[-1] == "2026-11-04 12:00:00"
    assert data.during_run.sum() == 91 - 19  # 10:00-11:30 inclusive, less the gap

    summary = yaml.safe_load((folder / "summary.yaml").read_text(encoding="utf-8"))
    assert summary["run_id"] == "abc" and summary["readings_during_run"] == 72
    assert summary["gaps"] == [{"from": "2026-11-04 10:40:00", "to": "2026-11-04 11:00:00", "minutes": 20.0}]
    assert summary["during_run"]["humidity_pct"] == {"min": 40.0, "mean": 40.0, "max": 40.0, "missing": 1}
    assert summary["during_run"]["temp_c"]["min"] == 20.6 and summary["median_interval_s"] == 60.0
    assert (folder / "environment.png").stat().st_size > 0


def test_no_readings_during_run_is_reported_not_invented(tmp_path):
    log = tmp_path / "mqtt_log.csv"
    make_log(log, [("2026-11-04T08:00:00", 21, 40)])
    folder = env.export_environment(tmp_path / "out", datetime(2026, 11, 4, 10, 0), datetime(2026, 11, 4, 11, 0),
                                    settings=settings(log))
    summary = yaml.safe_load((folder / "summary.yaml").read_text(encoding="utf-8"))
    assert summary["readings_during_run"] == 0
    assert summary["during_run"]["temp_c"] == {"min": None, "mean": None, "max": None, "missing": 0}
    assert summary["gaps"][0]["minutes"] == 60.0
    assert not (folder / "environment.png").exists()


def test_export_for_run_never_raises_when_the_log_is_missing(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(env, "load_settings", lambda: settings(tmp_path / "missing.csv"))
    tracker = SimpleNamespace(run_id="r1", datetime_started="2026-11-04 10:00:00",
                              output_folder=str(tmp_path / "out"), logger=logging.getLogger("t"))
    with caplog.at_level(logging.WARNING):
        assert env.export_for_run(tracker, datetime(2026, 11, 4, 11, 0)) is None
    assert "--run-id r1" in caplog.text


def test_backfill_picks_live_runs_without_the_folder(tmp_path):
    done = tmp_path / "done"
    (done / env.FOLDER_NAME).mkdir(parents=True)
    rows = [{"run_id": "a", "simulate": "False", "output_folder": str(tmp_path / "new"), "datetime_stopped": "x"},
            {"run_id": "b", "simulate": "False", "output_folder": str(done), "datetime_stopped": "x"},
            {"run_id": "c", "simulate": "True", "output_folder": str(tmp_path / "sim"), "datetime_stopped": "x"},
            {"run_id": "d", "simulate": "False", "output_folder": "", "datetime_stopped": "x"}]
    assert [r["run_id"] for r in env._runs_to_export(rows, None, False)] == ["a"]
    assert [r["run_id"] for r in env._runs_to_export(rows, None, True)] == ["a", "b"]
    assert [r["run_id"] for r in env._runs_to_export(rows, "c", False)] == ["c"]
