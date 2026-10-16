"""Qt liquid review of a scheduler result's immutable log."""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QComboBox, QDialog, QLabel, QPushButton,
                               QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
                               QVBoxLayout, QWidget, QHeaderView)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from scheduler.liquid_review import (calibration_files, coverage, current_parameters,
    environmental_history, estimate_optimum, fingerprint, load_calibration,
    matching_measurements, past_result_measurements, past_result_parameters,
    past_results, parse_log)


class LiquidReviewDialog(QDialog):
    def __init__(self, root, result, parent=None):
        super().__init__(parent)
        self.root, self.result = Path(root), result
        self.setWindowTitle("Liquid dispensing review")
        self.resize(1100, 850)
        layout = QVBoxLayout(self)
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        refresh = QPushButton("Refresh calibration and validation data")
        refresh.clicked.connect(self.refresh)
        layout.addWidget(refresh)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.refresh()

    def refresh(self):
        while self.tabs.count():
            widget = self.tabs.widget(0)
            self.tabs.removeTab(0)
            widget.deleteLater()
        log = self.result.get("log_file") or self.result.get("console_log")
        if not log:
            self.notice.setText("This result has no workflow log.")
            return
        path = Path(log)
        if not path.is_absolute():
            path = self.root / path
        try:
            events = parse_log(path)
            snapshot = self.result.get("liquid_calibration_fingerprint")
            stale = snapshot is not None and snapshot != fingerprint(self.root)
            text = f"Source: {path}\n{len(events)} logged destination operations. Volumes are log-reported targets, not measured delivery."
            if stale:
                text += "\nCalibration changed since simulation. Showing current data; simulate again to refresh workflow settings."
            elif snapshot is None:
                text += "\nHistorical result: calibration settings at simulation time are unknown."
            if not events:
                text += "\nNo recognized dispensing operations. Inspect the workflow log for unsupported formats."
            self.notice.setText(text)
            for liquid in sorted({event.liquid for event in events}):
                tab = LiquidTab(self.root, liquid, [e for e in events if e.liquid == liquid])
                self.tabs.addTab(tab, liquid)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.notice.setText(f"Cannot read liquid review: {error}")


class LiquidTab(QWidget):
    def __init__(self, root, liquid, events):
        super().__init__()
        self.root, self.liquid, self.events = root, liquid, events
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{len(events)} dispenses · {sum(e.volume_ul for e in events):g} µL total"))
        self.selector = QComboBox()
        files = calibration_files(root, liquid)
        self.selector.addItem("Select calibration file" if len(files) > 1 else "No calibration file", None)
        for path in files:
            self.selector.addItem(path.name, path)
        if len(files) == 1:
            self.selector.setCurrentIndex(1)
        layout.addWidget(self.selector)
        self.pages = QTabWidget()
        layout.addWidget(self.pages)
        self.selector.currentIndexChanged.connect(self.render)
        self.render()

    def render(self):
        while self.pages.count():
            widget = self.pages.widget(0)
            self.pages.removeTab(0)
            widget.deleteLater()
        try:
            path = self.selector.currentData()
            frame = load_calibration(path) if path else None
            recommendations = self.build_coverage(frame)
            self.build_past_calibration()
            if frame is not None:
                data = self.build_validation(frame)
                self.build_environment(data)
                for volume in sorted({e.volume_ul for e in self.events if e.method == "pipette"}):
                    if data.empty or not np.isclose(data.target_volume_ml * 1000, volume).any():
                        recommendations.add(f"Run embedded validation at {volume:g} µL to test and adjust current settings.")
            self.build_recommendations(recommendations)
        except (OSError, ValueError, KeyError, TypeError) as error:
            label = QLabel(f"Data could not be analyzed: {error}")
            label.setWordWrap(True)
            self.pages.addTab(label, "Data issue")

    def build_coverage(self, frame):
        page = QScrollArea()
        page.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        figure = Figure(figsize=(9, 2.5), constrained_layout=True)
        ax = figure.subplots()
        if frame is not None:
            for small, y in ((True, 1), (False, 0)):
                values = frame.loc[(frame.volume_target < 200) == small, "volume_target"]
                if not values.empty:
                    ax.plot([values.min(), values.max()], [y, y], color="seagreen", linewidth=8, alpha=.3)
                    ax.scatter(values, np.full(len(values), y), color="seagreen", marker="o", s=25)
        counts = Counter((e.volume_ul, e.tip, e.method) for e in self.events)
        rows, recommendations = [], set()
        for (volume, tip, method), count in sorted(counts.items()):
            status = coverage(frame, volume, tip) if method == "pipette" else "Reservoir: separate calibration required"
            supported = status in {"Measured point", "Interpolated coverage"}
            ax.plot(volume, 1 if volume < 200 else 0, "|", color="#333333" if supported else "crimson", markersize=24)
            rows.append([f"{volume:g}", str(count), tip or "Not logged", method, status])
            if not supported:
                if method == "reservoir":
                    recommendations.add(f"Review reservoir calibration for {volume:g} µL.")
                elif status == "Tip differs from calibration regime":
                    recommendations.add(f"Check tip selection for {volume:g} µL; logged tip is {tip}.")
                else:
                    recommendations.add(f"Calibrate {volume:g} µL in the applicable tip region.")
        ax.axvline(200, color="gray", linestyle="--")
        ax.set(yticks=[0, 1], yticklabels=["Large ≥200 µL", "Small <200 µL"], xlabel="Target volume (µL)", ylim=(-.5, 1.5))
        ax.set_title("Green: calibration points and within-regime interpolation · ticks: workflow volumes · red: unsupported")
        layout.addWidget(FigureCanvasQTAgg(figure))
        calibration_figure = Figure(figsize=(9, 2.5), constrained_layout=True)
        calibration_ax = calibration_figure.subplots()
        if frame is not None and "volume_measured" in frame:
            target = pd.to_numeric(frame.volume_target, errors="coerce")
            measured = pd.to_numeric(frame.volume_measured, errors="coerce")
            valid = np.isfinite(target) & np.isfinite(measured)
            calibration_ax.scatter(target[valid], measured[valid], color="crimson", s=40,
                                   label="Current optimal conditions")
            if valid.any():
                high = max(target[valid].max(), measured[valid].max())
                calibration_ax.plot([0, high], [0, high], color="gray", linestyle="--", label="Ideal V_M = V_T")
                calibration_ax.legend(loc="best")
        calibration_ax.set(xlabel="Target V_T (µL)", ylabel="Measured V_M (µL)",
                           title="Current calibration from the selected optimal conditions file")
        layout.addWidget(FigureCanvasQTAgg(calibration_figure))
        page.setWidget(content)
        self.pages.addTab(page, "Calibration Summary")

        coverage_page = QWidget()
        coverage_layout = QVBoxLayout(coverage_page)
        coverage_layout.addWidget(make_table(["Volume (µL)", "Count", "Logged tip", "Method", "Coverage"], rows))
        if self.liquid == "Unidentified":
            recommendations.add("Liquid class is missing from the log; rerun simulation to capture explicit liquid context.")
        if frame is None and self.selector.count() > 2:
            recommendations.add("Multiple calibration files match; select the applicable file explicitly.")
        details = QTextEdit()
        details.setReadOnly(True)
        details.setPlainText("Log references\n" + "\n".join(f"Line {e.line}: {e.volume_ul:g} µL → {e.destination}" for e in self.events))
        coverage_layout.addWidget(details)
        self.pages.addTab(coverage_page, "Coverage")
        return recommendations

    def build_validation(self, frame):
        data, excluded = matching_measurements(self.root, self.liquid, frame)
        page = QWidget()
        layout = QVBoxLayout(page)
        summary = QLabel("Source: pipetting_data/master_pipetting_measurements.csv (embedded validation measurements).\n"
                         + f"{len(data)} matching measurements across {data.target_volume_ml.nunique() if not data.empty else 0} historical target volumes. "
                         + "The table lists measured validation targets; they may differ from this workflow's dispenses.\n"
                         + "Excluded: " + (", ".join(f"{k}: {v}" for k, v in excluded.items()) or "none"))
        summary.setWordWrap(True)
        layout.addWidget(summary)
        raw_button = QPushButton("Show matching individual measurements")
        raw_button.setEnabled(not data.empty)
        def show_raw():
            dialog = QDialog(self)
            dialog.setWindowTitle(f"{self.liquid}: matching validation measurements")
            dialog.resize(1000, 600)
            raw_layout = QVBoxLayout(dialog)
            detail = data.copy()
            detail["error_ul"] = (detail.measured_volume_ml - detail.target_volume_ml) * 1000
            detail["absolute_error_ul"] = detail.error_ul.abs()
            columns = list(detail.columns)
            raw_layout.addWidget(make_table(columns, [[str(v) for v in row]
                                                     for row in detail.itertuples(index=False, name=None)]))
            dialog.exec()
        raw_button.clicked.connect(show_raw)
        layout.addWidget(raw_button)
        figure = Figure(figsize=(9, 3), constrained_layout=True)
        ax = figure.subplots()
        rows = []
        if not data.empty:
            error = (data.measured_volume_ml - data.target_volume_ml) * 1000
            relative = abs(error) / np.maximum(data.target_volume_ml * 1000, 1e-9) * 100
            ax.scatter(data.target_volume_ml * 1000, data.overaspirate_vol * 1000,
                       s=8 + 70 / (1 + relative), alpha=.25, color="steelblue", label="Observed; smaller = larger relative error")
            for volume, group in data.groupby("target_volume_ml"):
                estimate, message = estimate_optimum(group)
                current = current_parameters(frame, volume * 1000)
                ax.scatter([volume * 1000], [current["overaspirate_vol"] * 1000], marker="x", color="black")
                value = "Not identifiable"
                interval = "Unavailable"
                if estimate:
                    optimum, bounds = estimate
                    value = f"{optimum:.3g}"
                    ax.scatter([volume * 1000], [optimum], color="crimson", zorder=5)
                    if bounds:
                        ax.vlines(volume * 1000, *bounds, color="crimson")
                        interval = f"{bounds[0]:.3g}–{bounds[1]:.3g}"
                signed = (group.measured_volume_ml - group.target_volume_ml) * 1000
                date = pd.to_datetime(group.timestamp, format="%Y%m%d_%H%M%S", errors="coerce").max()
                last = date.strftime("%Y-%m-%d") if pd.notna(date) else "Unknown"
                used = "Yes" if any(np.isclose(e.volume_ul, volume * 1000) for e in self.events if e.method == "pipette") else "No"
                rows.append([f"{volume * 1000:g}", used, str(len(group)), str(group.session_id.nunique()),
                             last, f"{signed.mean():+.2f}", f"{signed.abs().quantile(.95):.2f}", value, interval, message])
        ax.set(xlabel="Target V_T (µL)", ylabel="Overaspirate V_OA (µL)")
        layout.addWidget(FigureCanvasQTAgg(figure))
        layout.addWidget(make_table(["V_T µL", "Workflow uses", "Measurements", "Sessions", "Last validation date",
                                      "Mean error µL", "95th |error| µL", "Optimum V_OA µL", "95% interval µL", "Evidence"], rows))
        label = QLabel("Errors are V_M − V_T. Mean error keeps the sign; 95th |error| shows the spread, including large misses. "
                       "Black ×: current V_OA. Red: estimated zero-error V_OA. Intervals are exploratory.")
        label.setWordWrap(True)
        layout.addWidget(label)
        self.pages.addTab(page, "Overvolume Validation")
        return data

    def build_environment(self, data):
        page = QWidget()
        layout = QVBoxLayout(page)
        env = environmental_history(self.root)
        rows = []
        if not data.empty and not env.empty:
            eligible = set(zip(data.session_id.astype(str), data.target_volume_ml.round(9)))
            for _, row in env.iterrows():
                if (str(row.session_id), round(float(row.target_volume_ml), 9)) in eligible:
                    rows.append([str(row.get(k, "")) for k in ["timestamp", "session_id", "target_volume_ul", "temp_c", "humidity_pct", "new_overaspirate_ml"]])
        label = QLabel("Environmental records linked by session and target volume. These readings were captured at correction time.\n"
                       "Additional environmental history can be cross-referenced later. Causal effects and compensation are not inferred from these records.")
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(make_table(["Timestamp", "Session", "V_T µL", "Temperature °C", "Humidity %", "Corrected V_OA mL"], rows))
        self.pages.addTab(page, "Environment")

    def build_past_calibration(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        runs = past_results(self.root, self.liquid)
        label = QLabel(f"{len(runs)} confirmed non-simulated result(s). "
                       "Select saved optimal conditions or validation measurements from a past run. "
                       "Calibration search trials and runs with unknown execution mode are omitted.")
        label.setWordWrap(True)
        layout.addWidget(label)
        selector = QComboBox()
        selector.addItem("Select past results", None)
        for result in runs:
            selector.addItem(result.label, result)
        layout.addWidget(selector)
        figure = Figure(figsize=(8, 3), constrained_layout=True)
        canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(canvas)
        info = QLabel()
        info.setWordWrap(True)
        layout.addWidget(info)
        volume_selector = QComboBox()
        layout.addWidget(volume_selector)
        parameters = QTextEdit()
        parameters.setReadOnly(True)
        parameters.setMaximumHeight(170)
        layout.addWidget(parameters)
        results_table = make_table(["Target V_T (µL)", "Measured V_M (µL)", "Error (µL)"], [])
        layout.addWidget(results_table)

        def selected_volume():
            frame = getattr(page, "result_frame", None)
            result = selector.currentData()
            if frame is None or result is None:
                parameters.clear()
                return
            target = volume_selector.currentData()
            records = frame[frame.target_ul == target]
            variants = {}
            for _, row in records.iterrows():
                values = past_result_parameters(row, result.kind)
                key = tuple(sorted((name, str(value)) for name, value in values.items()))
                variants[key] = variants.get(key, 0) + 1
            if not variants:
                parameters.setPlainText("No pipetting parameters saved for this target volume.")
                return
            parameters.setPlainText("\n\n".join(
                f"{count} measurement(s) with these settings:\n" +
                "\n".join(f"{name}: {value}" for name, value in key)
                for key, count in variants.items()))

        def selected():
            figure.clear()
            ax = figure.subplots()
            result = selector.currentData()
            page.result_frame = None
            volume_selector.clear()
            results_table.setRowCount(0)
            parameters.clear()
            if result:
                try:
                    frame = past_result_measurements(result)
                    page.result_frame = frame
                    valid = np.isfinite(frame.target_ul) & np.isfinite(frame.measured_ul)
                    ax.scatter(frame.loc[valid, "target_ul"], frame.loc[valid, "measured_ul"],
                               s=40 if result.kind == "calibration" else 18, alpha=.8 if result.kind == "calibration" else .45,
                               color="crimson" if result.kind == "calibration" else "steelblue")
                    if valid.any():
                        high = max(frame.loc[valid, "target_ul"].max(), frame.loc[valid, "measured_ul"].max())
                        ax.plot([0, high], [0, high], color="gray", linestyle="--", label="Ideal V_M = V_T")
                        ax.legend(loc="best")
                    for target in sorted(frame.target_ul.dropna().unique()):
                        volume_selector.addItem(f"{target:g} µL", target)
                    for r, row in frame.iterrows():
                        results_table.insertRow(results_table.rowCount())
                        error = row.measured_ul - row.target_ul
                        for c, value in enumerate((row.target_ul, row.measured_ul, error)):
                            results_table.setItem(results_table.rowCount() - 1, c,
                                                  QTableWidgetItem(f"{value:.3f}" if pd.notna(value) else "Failed"))
                    info.setText(f"{result.path}\n{len(frame)} saved {'optimal-condition rows' if result.kind == 'calibration' else 'validation measurements'}.")
                    selected_volume()
                except (OSError, ValueError, KeyError, TypeError) as error:
                    page.result_frame = None
                    info.setText(f"Could not read selected result: {error}")
            ax.set(xlabel="Target V_T (µL)", ylabel="Measured V_M (µL)")
            canvas.draw_idle()
        selector.currentIndexChanged.connect(selected)
        volume_selector.currentIndexChanged.connect(selected_volume)
        selected()
        self.pages.addTab(page, "Past Calibration Results")

    def build_recommendations(self, recommendations):
        page = QWidget()
        layout = QVBoxLayout(page)
        if not recommendations:
            recommendations.add("No coverage or validation action identified for the logged workflow volumes.")
        recommendations.add("If validation performance varies between sessions, repeat validation, repeat calibration, or investigate environmental impacts.")
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText("\n\n".join(f"• {item}" for item in sorted(recommendations)))
        layout.addWidget(text)
        self.pages.addTab(page, "Recommendations")


def make_table(headers, rows):
    table = QTableWidget(len(rows), len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table.setItem(r, c, QTableWidgetItem(value))
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    return table
