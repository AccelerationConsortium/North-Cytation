"""Best overaspirate per calibrated volume, estimated from validation history.

Shared by the robot's validation updates, the liquid review GUI and recalibration, so
all three agree. Within a session, delivered volume responds ~1 uL per uL of
overaspirate, so every replicate implies the overaspirate that would have hit target:

    oa* = overaspirate used - (measured - target)

Only history whose other settings match the current settings for that volume is used.
Sessions are combined with a local-level (Kalman) model: the true best value drifts
slowly (variance Q per session) and each session observes it with noise (variance R).
Q and R are estimated from each volume's own session series, so a noisy volume moves
slowly and a drifting volume follows its drift.
"""

import math

import numpy as np
import pandas as pd

# Settings that must match for history to count; overaspirate is what is being estimated.
MATCHED_SETTINGS = ("aspirate_speed", "dispense_speed", "retract_speed", "aspirate_wait_time",
                    "dispense_wait_time", "post_retract_wait_time", "pre_asp_air_vol",
                    "post_asp_air_vol", "blowout_vol")
MIN_SESSIONS_FOR_MODEL = 6
OUTLIER_MADS = 4.0
INNOVATION_LIMIT_SD = 3.0
FAILED_DISPENSE_FRACTION = 0.10  # below 10% of target is no dispense (no tip, empty vial), not calibration data


def implied_overaspirate_ul(frame):
    """Overaspirate (uL) each measurement implies would have delivered the target."""
    return (frame.overaspirate_vol - (frame.measured_volume_ml - frame.target_volume_ml)) * 1000


def usable_measurements(measurements, liquid, tip):
    """Real, balance-accepted vial-to-vial measurements for one liquid and tip."""
    data = measurements[measurements.liquid_type.astype(str).str.casefold() == liquid.casefold()]
    if "tip_type" in data:
        data = data[data.tip_type == tip]
    if "simulate" in data:
        data = data[~data.simulate.astype(str).str.casefold().eq("true")]
    if "validation_type" in data:
        data = data[data.validation_type == "vial_to_vial"]
    if "measurement_acceptable" in data:
        data = data[~data.measurement_acceptable.astype(str).str.casefold().eq("false")]
    data = data.copy()
    for column in ("target_volume_ml", "measured_volume_ml", "overaspirate_vol") + MATCHED_SETTINGS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[np.isfinite(data.target_volume_ml) & np.isfinite(data.measured_volume_ml)
                & np.isfinite(data.overaspirate_vol)
                & (data.measured_volume_ml >= FAILED_DISPENSE_FRACTION * data.target_volume_ml)]
    return drop_unrecorded_stages(data)


def drop_unrecorded_stages(data):
    """Keep only stage 1 where an old session saved one overaspirate on every stage.

    Older validations recorded the stage 1 value on stages 2 and 3, although those stages
    dispensed with different overaspirate; only the stage 1 rows are then trustworthy.
    """
    if "optimization_stage" not in data or data.empty:
        return data
    stage = pd.to_numeric(data.optimization_stage, errors="coerce")
    groups = data.groupby(["session_id", "target_volume_ml"])
    single = groups.overaspirate_vol.transform("nunique") == 1
    staged = stage.groupby([data.session_id, data.target_volume_ml]).transform("max") > 1
    return data[~(single & staged & (stage > 1))]


def matching_history(data, volume_ul, settings):
    """Measurements at this volume whose non-overaspirate settings equal ``settings``."""
    data = data[np.isclose(data.target_volume_ml * 1000, volume_ul, atol=.05)]
    keep = np.ones(len(data), bool)
    for name, value in settings.items():
        if name not in MATCHED_SETTINGS or value is None or not np.isfinite(value):
            continue
        if name not in data:
            return data.iloc[:0]
        keep &= np.isclose(data[name].to_numpy(dtype=float), value, rtol=1e-6, atol=1e-8)
    return data[keep]


def session_estimates(history):
    """One robust oa* per session (median after dropping replicate outliers), oldest first."""
    columns = ["session_id", "when", "oa_ul", "replicates", "replicate_sd_ul"]
    if history.empty:
        return pd.DataFrame(columns=columns)
    data = history.assign(oa_star=implied_overaspirate_ul(history),
                          when=pd.to_datetime(history.timestamp.astype(str), format="%Y%m%d_%H%M%S",
                                              errors="coerce"))
    rows = []
    for session, group in data.groupby("session_id"):
        values = group.oa_star.to_numpy()
        median = np.median(values)
        mad = 1.4826 * np.median(np.abs(values - median))
        if mad > 0:
            values = values[np.abs(values - median) <= max(OUTLIER_MADS * mad, 1.0)]
        rows.append(dict(session_id=session, when=group.when.min(), oa_ul=float(np.median(values)),
                         replicates=len(values),
                         replicate_sd_ul=float(np.std(values, ddof=1)) if len(values) > 1 else np.nan))
    return pd.DataFrame(rows, columns=columns).sort_values("when", na_position="first").reset_index(drop=True)


def noise_and_drift(sessions):
    """Per-session noise R and drift Q (uL^2) from the session series.

    With y_t = x_t + v_t and x_t = x_(t-1) + w_t, first differences d_t satisfy
    var(d) = Q + 2R and cov(d_t, d_(t-1)) = -R. R is floored at the replicate noise of
    a session median, so a volume is never treated as noise-free (which would chase noise).
    """
    y = sessions.oa_ul.to_numpy()
    d = np.diff(y)
    floor = np.nanmean((sessions.replicate_sd_ul ** 2 / sessions.replicates).to_numpy())
    floor = floor if np.isfinite(floor) else 0.0
    var_d = float(np.var(d, ddof=1))
    lag1 = float(np.mean((d[1:] - d.mean()) * (d[:-1] - d.mean()))) if len(d) > 2 else 0.0
    noise = max(-lag1, floor, 1e-6)
    drift = max(var_d - 2 * noise, 0.0)
    return noise, drift


def estimate_row(sessions):
    """Best overaspirate for one calibrated volume from its session series.

    Returns estimate_ul, sd_ul, gain (steady-state weight given to a new session),
    noise_sd_ul and drift_sd_ul; None when there is no history.
    """
    if sessions.empty:
        return None
    y = sessions.oa_ul.to_numpy()
    if len(y) < MIN_SESSIONS_FOR_MODEL:
        sd = float(np.std(y, ddof=1)) if len(y) > 1 else float("nan")
        return dict(estimate_ul=float(np.median(y)), sd_ul=sd / math.sqrt(len(y)) if len(y) > 1 else sd,
                    gain=float("nan"), noise_sd_ul=sd, drift_sd_ul=float("nan"), method="median (few sessions)")
    noise, drift = noise_and_drift(sessions)
    x, p = y[0], noise
    for value in y[1:]:
        p += drift
        innovation = value - x
        limit = INNOVATION_LIMIT_SD * math.sqrt(p + noise)
        innovation = max(-limit, min(limit, innovation))
        gain = p / (p + noise)
        x += gain * innovation
        p *= 1 - gain
    # Steady-state gain: what one new session would move the estimate by, as a fraction.
    steady_prior = (drift + math.sqrt(drift ** 2 + 4 * drift * noise)) / 2
    steady_gain = steady_prior / (steady_prior + noise)
    return dict(estimate_ul=float(x), sd_ul=math.sqrt(p), gain=steady_gain,
                noise_sd_ul=math.sqrt(noise), drift_sd_ul=math.sqrt(drift), method="local level")


def backtest(sessions):
    """One-step-ahead RMSE (uL) predicting each next session: this model vs jumping to the last session."""
    y = sessions.oa_ul.to_numpy()
    if len(y) < MIN_SESSIONS_FOR_MODEL + 2:
        return float("nan"), float("nan")
    model = [estimate_row(sessions.iloc[:i])["estimate_ul"] for i in range(MIN_SESSIONS_FOR_MODEL, len(y))]
    actual = y[MIN_SESSIONS_FOR_MODEL:]
    last = y[MIN_SESSIONS_FOR_MODEL - 1:-1]
    rmse = lambda predicted: float(np.sqrt(np.mean((actual - np.asarray(predicted)) ** 2)))
    return rmse(model), rmse(last)


SPEED_SETTINGS = ("aspirate_speed", "dispense_speed", "retract_speed")


def row_settings(row):
    """Settings as the robot applies them: speeds are rounded to whole numbers."""
    return {name: float(round(row[name])) if name in SPEED_SETTINGS else float(row[name])
            for name in MATCHED_SETTINGS if name in row and pd.notna(row[name])}


def review_table(measurements, calibration, liquid):
    """Current vs estimated best overaspirate for every calibrated row of one liquid."""
    rows = []
    for _, row in calibration.sort_values(["tip_type", "volume_target"]).iterrows():
        data = usable_measurements(measurements, liquid, row.tip_type)
        sessions = session_estimates(matching_history(data, row.volume_target, row_settings(row)))
        result = estimate_row(sessions) or {}
        measured = row.get("volume_measured")
        effective = (row.overaspirate_vol * 1000
                     + (row.volume_target - measured if pd.notna(measured) else 0.0))
        all_sessions = data[np.isclose(data.target_volume_ml * 1000, row.volume_target, atol=.05)].session_id.nunique()
        model_rmse, last_rmse = backtest(sessions)
        rows.append(dict(liquid=liquid, tip=row.tip_type, volume_ul=row.volume_target,
                         current_ul=row.overaspirate_vol * 1000, effective_ul=effective,
                         sessions=len(sessions), sessions_other_settings=all_sessions - len(sessions),
                         last=sessions.when.max() if len(sessions) else pd.NaT,
                         backtest_model_ul=model_rmse, backtest_current_method_ul=last_rmse, **result))
    return pd.DataFrame(rows)


def uncalibrated_volumes(measurements, calibration, liquid):
    """Volumes with history but no calibration row, using each volume's latest settings."""
    rows = []
    for tip in sorted(calibration.tip_type.unique()):
        data = usable_measurements(measurements, liquid, tip)
        known = calibration[calibration.tip_type == tip].volume_target.to_numpy()
        for volume_ml, group in data.groupby("target_volume_ml"):
            volume_ul = round(volume_ml * 1000, 2)
            if np.isclose(known, volume_ul, atol=.05).any():
                continue
            latest = group.sort_values("timestamp").iloc[-1]
            sessions = session_estimates(matching_history(group, volume_ul, row_settings(latest)))
            result = estimate_row(sessions) or {}
            rows.append(dict(liquid=liquid, tip=tip, volume_ul=volume_ul, sessions=len(sessions),
                             last=sessions.when.max() if len(sessions) else pd.NaT,
                             settings={k: v for k, v in row_settings(latest).items()}, **result))
    return pd.DataFrame(rows)
