"""Qt liquid review of a scheduler result's immutable log."""
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QDate
from PySide6.QtWidgets import (QComboBox, QDateEdit, QDialog, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
                               QVBoxLayout, QWidget, QHeaderView)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from scheduler.liquid_environment_analysis import analyze_environment
from scheduler.liquid_review import (calibration_files, coverage, current_parameters,
    environmental_sessions, fingerprint, load_calibration,
    matching_measurements, past_result_measurements, past_result_parameters,
    past_results, parse_log, session_overaspirate_estimates, summarize_overaspirate,
    summarize_past_validation)


def scale_volume_axis(ax, volumes):
    """Give small positive target volumes space when targets span a decade."""
    values = np.asarray(volumes, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) and np.all(values > 0) and values.max() / values.min() > 10:
        ax.set_xscale("log")


def scale_volume_comparison(ax, targets, measured):
    """Keep target and measured volumes on matching scales."""
    scale_volume_axis(ax, targets)
    values = np.asarray(measured, dtype=float)
    values = values[np.isfinite(values)]
    if ax.get_xscale() == "log" and len(values) and np.all(values > 0):
        ax.set_yscale("log")


def plot_measured_distribution(ax, measured, target):
    """Show repeated measurements at one target without a degenerate x axis."""
    values = np.asarray(measured, dtype=float)
    values = values[np.isfinite(values)]
    if len(values):
        ax.hist(values, bins="auto", color="steelblue", alpha=.75)
    ax.axvline(target, color="crimson", linestyle="--", label=f"Target {target:g} µL")
    ax.set(xlabel="Measured volume (µL)", ylabel="Measurements")
    ax.legend(loc="best")


class LiquidReviewDialog(QDialog):
    def __init__(self, root, result, parent=None):
        super().__init__(parent)
        self.root, self.result = Path(root), result
        self.setWindowTitle("Liquid dispensing review")
        self.resize(1400, 1000)
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
        self.drift_target_ul = None
        self.validation_end = QDate.currentDate()
        self.validation_start = self.validation_end.addMonths(-3)
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
        self.date_edits = []
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
                matched, excluded = matching_measurements(self.root, self.liquid, frame)
                self.build_past_validation(matched, excluded)
                data = self.build_validation(frame, matched, excluded)
                self.build_environment(matched, frame)
                for volume in sorted({e.volume_ul for e in self.events if e.method == "pipette"}):
                    if data.empty or not np.isclose(data.target_volume_ml * 1000, volume).any():
                        recommendations.add(f"Run embedded validation at {volume:g} µL to test and adjust current settings.")
            else:
                self.pages.addTab(QLabel("Select a current calibration file to review accepted validation results."), "Past Validation Results")
            self.build_recommendations(recommendations)
            self.order_pages()
        except (OSError, ValueError, KeyError, TypeError) as error:
            label = QLabel(f"Data could not be analyzed: {error}")
            label.setWordWrap(True)
            self.pages.addTab(label, "Data issue")

    def order_pages(self):
        order = ["Summary", "Past Validation Results", "Past Calibration Results", "Overvolume Changes",
                 "Environmental Analysis", "Workflow Coverage", "Workflow Recommendations"]
        bar = self.pages.tabBar()
        present = [name for name in order if any(self.pages.tabText(i) == name for i in range(self.pages.count()))]
        for target, name in enumerate(present):
            current = next(i for i in range(self.pages.count()) if self.pages.tabText(i) == name)
            bar.moveTab(current, target)

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
        scale_volume_axis(ax, [e.volume_ul for e in self.events] +
                          ([] if frame is None else frame.volume_target.tolist()))
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
                scale_volume_comparison(calibration_ax, target[valid], measured[valid])
                low = target[valid].min() if calibration_ax.get_xscale() == "log" else 0
                calibration_ax.plot([low, high], [low, high], color="gray", linestyle="--", label="Ideal V_M = V_T")
                calibration_ax.legend(loc="best")
        calibration_ax.set(xlabel="Target V_T (µL)", ylabel="Measured V_M (µL)",
                           title="Current calibration from the selected optimal conditions file")
        layout.addWidget(FigureCanvasQTAgg(calibration_figure))
        page.setWidget(content)
        self.pages.addTab(page, "Summary")

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
        self.pages.addTab(coverage_page, "Workflow Coverage")
        return recommendations

    def add_dated_tab(self, name, content):
        """Tab with From/To pickers shared by every dated tab; content(start, end) builds the body."""
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("From:"))
        start_edit, end_edit = QDateEdit(self.validation_start), QDateEdit(self.validation_end)
        for edit in (start_edit, end_edit):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd")
        controls.addWidget(start_edit)
        controls.addWidget(QLabel("To (inclusive):"))
        controls.addWidget(end_edit)
        controls.addStretch()
        outer_layout.addLayout(controls)
        holder = QVBoxLayout()
        outer_layout.addLayout(holder)
        self.date_edits.append((start_edit, end_edit))

        def redraw(*_):
            while holder.count():
                holder.takeAt(0).widget().deleteLater()
            start, end = self.validation_start, self.validation_end
            holder.addWidget(content(datetime(start.year(), start.month(), start.day()),
                                     datetime(end.year(), end.month(), end.day()) + timedelta(days=1)))

        def changed(*_):
            self.validation_start, self.validation_end = start_edit.date(), end_edit.date()
            for other_start, other_end in self.date_edits:
                other_start.setDate(self.validation_start)
                other_end.setDate(self.validation_end)
            redraw()
        start_edit.dateChanged.connect(changed)
        end_edit.dateChanged.connect(changed)
        redraw()
        self.pages.addTab(outer, name)

    def build_past_validation(self, matched, excluded):
        self.add_dated_tab("Past Validation Results",
                           lambda start, end: self._past_validation_content(matched, excluded, start, end))

    @staticmethod
    def _plot_validation_scatter(ax, data, summary):
        sessions = data.groupby(["target_volume_ml", "session_id"], as_index=False).measured_volume_ml.mean()
        ax.scatter(sessions.target_volume_ml * 1000, sessions.measured_volume_ml * 1000,
                   s=14, alpha=.2, color="steelblue", label="Accepted session means")
        ax.scatter(summary.target_ul, summary.mean_ul, s=40, color="crimson", zorder=4, label="Historical mean")
        with_ci = summary[summary.ci_low_ul.notna()]
        if not with_ci.empty:
            ax.errorbar(with_ci.target_ul, with_ci.mean_ul,
                        yerr=with_ci.mean_ul - with_ci.ci_low_ul, fmt="none", ecolor="crimson",
                        capsize=4, label="95% CI of the mean")
        high = max(summary.target_ul.max(), data.measured_volume_ml.max() * 1000)
        low = summary.target_ul.min() if ax.get_xscale() == "log" else 0
        ax.plot([low, high], [low, high], linestyle="--", color="gray", label="Ideal V_M = V_T")
        ax.legend(loc="best")

    def _past_validation_content(self, matched, excluded, start, end):
        data, summary, exclusions = summarize_past_validation(matched, excluded, start, end)
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel(f"{len(data)} accepted measurements between {start:%Y-%m-%d} and {(end - timedelta(days=1)):%Y-%m-%d} "
                       "match the current dispensing settings (overvolume may vary). "
                       "Each session has equal weight in the historical mean.")
        title.setWordWrap(True)
        layout.addWidget(title)
        figure = Figure(figsize=(9, 3.5), constrained_layout=True)
        ax = figure.subplots()
        if not data.empty:
            targets = summary.target_ul.to_numpy()
            if len(targets) == 1:
                plot_measured_distribution(ax, data.measured_volume_ml * 1000, targets[0])
            else:
                scale_volume_comparison(ax, targets, data.measured_volume_ml * 1000)
                self._plot_validation_scatter(ax, data, summary)
        else:
            ax.text(.5, .5, "No accepted validation measurements match this calibration.",
                    ha="center", va="center", transform=ax.transAxes)
        if data.empty or len(summary) != 1:
            ax.set(xlabel="Target V_T (µL)", ylabel="Measured V_M (µL)")
        canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(NavigationToolbar2QT(canvas, page))
        layout.addWidget(canvas)
        rows = []
        for _, row in summary.iterrows():
            ci = f"{row.ci_low_ul:.3f} – {row.ci_high_ul:.3f}" if pd.notna(row.ci_low_ul) else "Unavailable: one session"
            rows.append([f"{row.target_ul:g}", f"{row.mean_ul:.3f}", f"{row.error_ul:+.3f}", ci,
                         str(row.sessions), str(row.measurements), row.last_validation])
        table = make_table(["Target µL", "Mean measured µL", "Mean error µL", "95% confidence interval µL",
                            "Sessions", "Measurements", "Last validation date"], rows)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        layout.addWidget(table)
        button = QPushButton("Inspect accepted measurements for selected volume")
        button.setEnabled(not summary.empty)
        layout.addWidget(button)

        def inspect(*_):
            index = table.currentRow()
            if index < 0:
                return
            volume = summary.iloc[index].target_ul
            detail = data[np.isclose(data.target_volume_ml * 1000, volume)].copy()
            detail.insert(0, "error_ul", (detail.measured_volume_ml - detail.target_volume_ml) * 1000)
            dialog = QDialog(self)
            dialog.setWindowTitle(f"{self.liquid}: accepted validation at {volume:g} µL")
            dialog.resize(1100, 600)
            inner = QVBoxLayout(dialog)
            inner.addWidget(QLabel("Selected-stage measurements, including reconstructed decisions and results outside tolerance."))
            inner.addWidget(make_table(list(detail.columns), [[str(v) if pd.notna(v) else "Unknown" for v in row]
                                                             for row in detail.itertuples(index=False, name=None)]))
            dialog.exec()
        button.clicked.connect(inspect)
        table.cellDoubleClicked.connect(inspect)
        if rows:
            table.selectRow(0)
        excluded_label = QLabel("Excluded measurements: " + ("; ".join(f"{key}: {value}" for key, value in exclusions.items()) or "none"))
        excluded_label.setWordWrap(True)
        layout.addWidget(excluded_label)
        note = QLabel("Intervals describe uncertainty in the historical mean across sessions, not the range of individual dispenses. "
                      "One session has no historical interval; intervals with few sessions are uncertain. "
                      "Accepted results can remain outside tolerance. Historical tip identity is inferred from volume.")
        note.setWordWrap(True)
        layout.addWidget(note)
        return page

    def build_validation(self, frame, data, excluded):
        self.add_dated_tab("Overvolume Changes", lambda start, end: self._overvolume_content(frame, data, excluded, start, end))
        return data

    def _overvolume_content(self, frame, data, excluded, start, end):
        page = QWidget()
        layout = QVBoxLayout(page)
        estimates, dropped = session_overaspirate_estimates(data, start, end)
        current = {volume * 1000: current_parameters(frame, volume * 1000)["overaspirate_vol"] * 1000
                   for volume in (data.target_volume_ml.unique() if not data.empty else [])}
        summary = summarize_overaspirate(estimates, current)
        everything, _ = session_overaspirate_estimates(data)
        if everything.empty:
            span = "No session in the history recorded more than one V_OA, so no estimate is possible."
        else:
            span = (f"Usable sessions in the whole history: {len(everything)}, from {everything.when.min():%Y-%m-%d} "
                    f"to {everything.when.max():%Y-%m-%d}. Older sessions that saved the same V_OA on every stage "
                    "cannot show how delivery responds to V_OA and are not used.")
        span_label = QLabel(span)
        span_label.setWordWrap(True)
        layout.addWidget(span_label)
        explanation = QLabel(
            "Question: what overaspirate volume (V_OA) is statistically most likely to deliver the target volume?\n"
            "Each embedded validation session tests up to three V_OA values at a target volume (stages 1-3, all stages used here). "
            "For each session, a line through (V_OA, error = V_M - V_T) gives the V_OA where error would be zero: that session's "
            "estimate of the true V_OA. Estimates from all sessions in the date range are averaged (each session counts once).\n"
            "Left: the estimate range per target against the current V_OA (black x). Select a target volume to see how its session estimate differs "
            "from the current V_OA over time (0 = current V_OA is exactly right; a steady offset means recalibrate, a wide scatter means noise).")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        drift_selector = QComboBox()
        drift_selector.setToolTip("Target volume shown in the drift plot")
        for volume in sorted(estimates.target_ul.unique()) if not estimates.empty else []:
            drift_selector.addItem(f"Drift at {volume:g} \u00b5L", float(volume))
        if self.drift_target_ul is not None:
            index = drift_selector.findData(self.drift_target_ul)
            if index >= 0:
                drift_selector.setCurrentIndex(index)
        layout.addWidget(drift_selector)
        figure = Figure(figsize=(11, 3.8), constrained_layout=True)
        left, right = figure.subplots(1, 2)
        if estimates.empty:
            for ax in (left, right):
                ax.text(.5, .5, "No sessions with usable V_OA estimates in this date range.",
                        ha="center", va="center", transform=ax.transAxes)
        else:
            left.scatter(estimates.target_ul, estimates.oa_ul, s=14, alpha=.25, color="steelblue", label="Session estimates")
            left.scatter(summary.target_ul, summary.mean_ul, s=40, color="crimson", zorder=4, label="Mean estimate")
            with_ci = summary[summary.ci_low_ul.notna()]
            if not with_ci.empty:
                left.errorbar(with_ci.target_ul, with_ci.mean_ul, yerr=with_ci.mean_ul - with_ci.ci_low_ul,
                              fmt="none", ecolor="crimson", capsize=4, label="95% CI of the mean")
            left.scatter(summary.target_ul, summary.current_ul, marker="x", s=60, color="black", zorder=5, label="Current V_OA")
            scale_volume_axis(left, estimates.target_ul)
            left.legend(loc="best", fontsize=8)
        left.set(xlabel="Target V_T (\u00b5L)", ylabel="Overaspirate V_OA (\u00b5L)", title="Estimated true V_OA vs current")

        def draw_drift(*_):
            if estimates.empty:
                right.set_xlim(start, end)
                right.set(xlabel="Validation date", ylabel="Session estimate - current V_OA (\u00b5L)",
                          title="Drift of the estimate over time")
                return
            volume = drift_selector.currentData()
            if volume is None:
                return
            self.drift_target_ul = volume
            group = estimates[np.isclose(estimates.target_ul, volume)]
            right.clear()
            right.scatter(group.when.to_numpy(), (group.oa_ul - current[volume]).to_numpy(),
                          s=35, alpha=.8, color="steelblue")
            right.axhline(0, color="gray", linestyle="--")
            right.set_xlim(group.when.min() - timedelta(days=2), group.when.max() + timedelta(days=2))
            right.tick_params(axis="x", rotation=30)
            right.set(xlabel="Validation date", ylabel="Session estimate - current V_OA (\u00b5L)",
                      title=f"Drift at {volume:g} \u00b5L ({len(group)} sessions)")
            canvas.draw_idle()
        canvas = FigureCanvasQTAgg(figure)
        drift_selector.currentIndexChanged.connect(draw_drift)
        draw_drift()
        layout.addWidget(NavigationToolbar2QT(canvas, page))
        layout.addWidget(canvas)
        rows = []
        for _, row in summary.iterrows():
            if row.sessions == 1:
                interval, reading = "Unavailable", "One session: no interval"
            else:
                interval = f"{row.ci_low_ul:.2f} – {row.ci_high_ul:.2f}"
                if row.sessions < 3:
                    reading = "Few sessions: uncertain"
                elif row.ci_low_ul <= row.current_ul <= row.ci_high_ul:
                    reading = "Current V_OA consistent with data"
                else:
                    reading = "Current V_OA outside interval: consider updating"
            used = "Yes" if any(np.isclose(e.volume_ul, row.target_ul) for e in self.events if e.method == "pipette") else "No"
            rows.append([f"{row.target_ul:g}", used, f"{row.current_ul:.2f}", str(row.sessions), f"{row.mean_ul:.2f}", interval,
                         f"{row.sd_ul:.2f}" if pd.notna(row.sd_ul) else "n/a", f"{row.mean_ul - row.current_ul:+.2f}",
                         row.last_validation.strftime("%Y-%m-%d") if pd.notna(row.last_validation) else "Unknown", reading])
        layout.addWidget(make_table(["V_T µL", "Workflow uses", "Current V_OA µL", "Sessions", "Estimated true V_OA µL",
                                      "95% interval µL", "Between-session SD µL", "Suggested change µL",
                                      "Last validation", "Reading"], rows))
        notes = {**excluded, **dropped}
        note = QLabel("Excluded: " + ("; ".join(f"{k}: {v}" for k, v in notes.items()) or "none") + "\n"
                      "The interval is the uncertainty of the mean across sessions, not the spread of single dispenses. "
                      "Suggested change = estimated true V_OA - current V_OA. Estimates assume V_M responds roughly linearly to V_OA; "
                      "sessions with an implausible response are excluded.")
        note.setWordWrap(True)
        layout.addWidget(note)
        raw_button = QPushButton("Show session estimates")
        raw_button.setEnabled(not estimates.empty)

        def show_raw():
            dialog = QDialog(self)
            dialog.setWindowTitle(f"{self.liquid}: per-session V_OA estimates")
            dialog.resize(900, 600)
            inner = QVBoxLayout(dialog)
            inner.addWidget(make_table(["Session", "V_T µL", "Date", "Estimated V_OA µL", "Response slope", "Tested V_OA values"],
                                       [[str(r.session_id), f"{r.target_ul:g}", str(r.when), f"{r.oa_ul:.3f}",
                                         f"{r.slope:.2f}", str(r.points)] for r in estimates.itertuples()]))
            dialog.exec()
        raw_button.clicked.connect(show_raw)
        layout.addWidget(raw_button)
        return page

    def build_environment(self, matched, calibration):
        self.environment_analysis = analyze_environment(matched, calibration)
        self.add_dated_tab("Environmental Analysis", self._environment_content)

    def _environment_content(self, start, end):
        page = QWidget()
        layout = QVBoxLayout(page)
        sessions = environmental_sessions(self.root, self.liquid, start, end)
        temperature = sessions[sessions.temp_c.notna()]
        humidity = sessions[sessions.humidity_pct.notna()]
        label = QLabel(
            f"{len(sessions)} recorded sessions in this date range. "
            f"Temperature available for {len(temperature)}; humidity for {len(humidity)}. "
            "Each point is one session reading. Historical readings are associated with the session timestamp."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        figure = Figure(figsize=(10, 5), constrained_layout=True)
        temperature_ax, humidity_ax = figure.subplots(2, 1, sharex=True)
        temperature_ax.scatter(temperature.when, temperature.temp_c, s=25, alpha=.75, color="firebrick")
        humidity_ax.scatter(humidity.when, humidity.humidity_pct, s=25, alpha=.75, color="steelblue")
        temperature_ax.set(ylabel="Temperature (\u00b0C)", title="Temperature at validation sessions")
        humidity_ax.set(xlabel="Session date", ylabel="Relative humidity (%)", title="Humidity at validation sessions")
        humidity_ax.set_xlim(start, end)
        humidity_ax.tick_params(axis="x", rotation=30)
        if sessions.empty or (temperature.empty and humidity.empty):
            temperature_ax.text(.5, .5, "No environmental readings in this date range.",
                                ha="center", va="center", transform=temperature_ax.transAxes)
        canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(NavigationToolbar2QT(canvas, page))
        layout.addWidget(canvas)
        heading = QLabel("Do temperature or humidity appear related to delivery error?")
        heading.setWordWrap(True)
        layout.addWidget(heading)
        analysis = self.environment_analysis
        if analysis is None:
            layout.addWidget(QLabel("Not enough independent sessions or environmental variation for this analysis."))
        else:
            description = QLabel(
                f"Using {analysis.sessions} past sessions that match today's calibration settings, "
                "we estimate how delivery error changes from a typical low to a typical high reading "
                "(the 10th and 90th percentiles of past session conditions). "
                "The comparison accounts for target volume, overaspirate, the other environmental reading, "
                "and calendar month. Negative values mean less liquid was delivered relative to the target."
            )
            description.setWordWrap(True)
            layout.addWidget(description)
            def finding(name, effect):
                if effect.typical_ci_high < 0:
                    return (f"Higher {name} is associated with less delivered volume: "
                            f"about {abs(effect.typical_change):.1f}% of the target across the typical range "
                            f"(95% interval: {abs(effect.typical_ci_high):.1f}% to "
                            f"{abs(effect.typical_ci_low):.1f}% less).")
                if effect.typical_ci_low > 0:
                    return (f"Higher {name} is associated with more delivered volume: "
                            f"about {effect.typical_change:.1f}% of the target across the typical range "
                            f"(95% interval: {effect.typical_ci_low:.1f}% to "
                            f"{effect.typical_ci_high:.1f}% more).")
                return (f"The {name} pattern is unclear: the plausible change across the typical range "
                        f"runs from {effect.typical_ci_low:+.1f}% to {effect.typical_ci_high:+.1f}% "
                        "of the target volume.")
            takeaway = QLabel("What these sessions suggest: " +
                              finding("temperature", analysis.temperature) + " " +
                              finding("humidity", analysis.humidity))
            takeaway.setWordWrap(True)
            takeaway.setStyleSheet("font-weight: 600")
            layout.addWidget(takeaway)
            effect_figure = Figure(figsize=(9, 2.5), constrained_layout=True)
            effect_ax = effect_figure.subplots()
            effects = [("Warmer", analysis.temperature, "°C", 1, "firebrick"),
                       ("More humid", analysis.humidity, "% RH", 0, "steelblue")]
            for name, effect, unit, row, color in effects:
                effect_ax.errorbar(effect.typical_change, row,
                                   xerr=[[max(0, effect.typical_change - effect.typical_ci_low)],
                                         [max(0, effect.typical_ci_high - effect.typical_change)]],
                                   fmt="o", color=color, capsize=5, markersize=7)
            effect_ax.axvline(0, color="gray", linestyle="--")
            effect_ax.set(yticks=[1, 0],
                          yticklabels=[f"Warmer: {analysis.temperature.typical_low:.1f} to "
                                       f"{analysis.temperature.typical_high:.1f} °C",
                                       f"More humid: {analysis.humidity.typical_low:.0f} to "
                                       f"{analysis.humidity.typical_high:.0f}% RH"],
                          xlabel="Estimated change in delivery error (% of target volume)",
                          title="Dot: estimate    Line: 95% uncertainty interval    Dashed line: no change",
                          ylim=(-.5, 1.5))
            layout.addWidget(FigureCanvasQTAgg(effect_figure))
            for name, effect, unit, _, _ in effects:
                sentence = QLabel(
                    f"{name}: estimated change {effect.typical_change:+.1f}% of target volume "
                    f"(95% uncertainty interval {effect.typical_ci_low:+.1f} to {effect.typical_ci_high:+.1f}%). "
                    f"The direction was the same in {effect.direction_agreement_pct:.0f}% "
                    "of session resamples."
                )
                sentence.setWordWrap(True)
                layout.addWidget(sentence)
            explanation = QLabel(
                "Direction agreement measures how consistently the pattern appears when past sessions are resampled. "
                "It is not the probability that temperature or humidity caused the change. "
                "If an uncertainty line crosses zero, the data also support no change."
            )
            explanation.setWordWrap(True)
            layout.addWidget(explanation)
            prediction = QLabel(
                "Separate question — can these readings help predict another session? We repeatedly hid entire sessions, learned from "
                "the remaining sessions, then measured the average prediction miss on the hidden ones. "
                f"Without temperature and humidity the average miss was {analysis.baseline_mae:.2f}% of target volume; "
                f"with them it was {analysis.environment_mae:.2f}%. Smaller is better. "
                "This prediction check does not erase associations observed in past sessions."
            )
            prediction.setWordWrap(True)
            layout.addWidget(prediction)
        return page

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
            is_distribution = False
            page.result_frame = None
            volume_selector.clear()
            results_table.setRowCount(0)
            parameters.clear()
            if result:
                try:
                    frame = past_result_measurements(result)
                    page.result_frame = frame
                    valid = np.isfinite(frame.target_ul) & np.isfinite(frame.measured_ul)
                    if valid.any():
                        targets = frame.loc[valid, "target_ul"]
                        if targets.nunique() == 1 and result.kind == "validation":
                            is_distribution = True
                            plot_measured_distribution(ax, frame.loc[valid, "measured_ul"], targets.iloc[0])
                        else:
                            ax.scatter(targets, frame.loc[valid, "measured_ul"],
                                       s=40 if result.kind == "calibration" else 18,
                                       alpha=.8 if result.kind == "calibration" else .45,
                                       color="crimson" if result.kind == "calibration" else "steelblue")
                            scale_volume_comparison(ax, targets, frame.loc[valid, "measured_ul"])
                            high = max(targets.max(), frame.loc[valid, "measured_ul"].max())
                            low = targets.min() if ax.get_xscale() == "log" else 0
                            ax.plot([low, high], [low, high], color="gray", linestyle="--", label="Ideal V_M = V_T")
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
            if not is_distribution:
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
        self.pages.addTab(page, "Workflow Recommendations")


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
