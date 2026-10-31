"""Tracked overaspirate corrections, kept separate from the base calibration files.

One row per (liquid, tip, calibrated volume): the tracked best overaspirate and the
model state behind it. The base ``optimal_conditions_*.csv`` files are never edited by
validation; lookup replaces a row's overaspirate with the tracked value (and neutralises
the old measured-volume compensation for that row, so only one correction applies).
Deleting a row here, or the whole file, returns to the base calibration.

Updates follow the local-level model in ``overaspirate_estimation``: the true best value
drifts slowly (Q per session) and each session observes it with noise (R). A session
moves the estimate by the Kalman gain P / (P + R): small for noisy volumes, larger for
volumes that drift. A surprising session is confirmed before it can move the estimate
far (see ``decide``). Nothing here raises into a workflow: callers get None or a
warning and keep the last good value.
"""

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipetting_data import overaspirate_estimation as estimation
from pipetting_data.validation_decisions import append_csv, write_csv

CORRECTIONS_FILE = Path(__file__).resolve().parent / "overaspirate_corrections.csv"
COLUMNS = ["liquid", "tip_type", "volume_target", "overaspirate_ul", "sd_ul", "noise_sd_ul",
           "drift_sd_ul", "sessions", "base_overaspirate_ul", "updated", "last_session_id", "last_verdict"]
SURPRISE_SD = 2.0            # check batch this many SDs from expectation -> confirm first
MIN_DRIFT_FRACTION = 0.01    # Q >= 1% of R: the estimate can always follow real drift
MIN_REPLICATE_SD_UL = 0.5    # floor for replicate scatter when judging confirm agreement
DEFAULT_NOISE_FRACTION = 0.02  # no history: assume session noise of 2% of volume


def _key(frame, liquid, tip, volume_ul):
    return ((frame.liquid.astype(str).str.casefold() == liquid.casefold())
            & (frame.tip_type == tip) & np.isclose(frame.volume_target.astype(float), volume_ul, atol=.05))


def load_corrections(path=None):
    path = Path(path or CORRECTIONS_FILE)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(path)
    for column in COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame


def effective_base_ul(row):
    """Overaspirate (uL) the robot uses for a base row: value plus measured-volume compensation."""
    value = float(row.overaspirate_vol) * 1000
    measured = row.get("volume_measured") if hasattr(row, "get") else None
    if measured is not None and pd.notna(measured) and abs(row.volume_target - measured) > 0.01:
        value += row.volume_target - measured
    return value


def apply_corrections(df, liquid, tip, path=None, logger=None):
    """Return calibration rows with tracked overaspirate substituted; never raises.

    A correction applies only while its base row is unchanged: after a deliberate
    recalibration of that row, the new base value is used and the correction ignored.
    """
    try:
        corrections = load_corrections(path)
        if corrections.empty or tip is None:
            return df
        df = df.copy()
        for index, row in df.iterrows():
            match = corrections[_key(corrections, liquid, tip, row.volume_target)]
            if match.empty or not np.isfinite(match.overaspirate_ul.iloc[-1]):
                continue
            base = match.base_overaspirate_ul.iloc[-1]
            if pd.notna(base) and abs(effective_base_ul(row) - float(base)) > 0.05:
                if logger is not None:
                    logger.warning("Ignoring tracked overaspirate for %s %s %g uL: base calibration "
                                   "changed since tracking began", liquid, tip, row.volume_target)
                continue
            df.at[index, "overaspirate_vol"] = float(match.overaspirate_ul.iloc[-1]) / 1000
            if "volume_measured" in df:
                df.at[index, "volume_measured"] = row.volume_target
        return df
    except Exception as error:
        if logger is not None:
            logger.warning("Ignoring overaspirate corrections for %s (%s): %s", liquid, tip, error)
        return df


def prior_state(liquid, tip, volume_ul, effective_ul, settings, measurements=None, path=None, base_ul=None):
    """Current model state for one row: tracked file, else history, else a cautious default.

    ``base_ul`` is the base row's current value; if it differs from the base the tracked
    state started from, the row was recalibrated and tracking restarts from ``base_ul``.
    """
    corrections = load_corrections(path)
    match = corrections[_key(corrections, liquid, tip, volume_ul)]
    stale = (not match.empty and base_ul is not None and pd.notna(match.base_overaspirate_ul.iloc[-1])
             and abs(float(match.base_overaspirate_ul.iloc[-1]) - base_ul) > 0.05)
    if stale:
        effective_ul = base_ul
    elif not match.empty and np.isfinite(match.overaspirate_ul.iloc[-1]):
        row = match.iloc[-1]
        return dict(estimate_ul=float(row.overaspirate_ul), variance=float(row.sd_ul) ** 2,
                    noise=float(row.noise_sd_ul) ** 2, drift=float(row.drift_sd_ul) ** 2,
                    sessions=int(row.sessions), source="tracked")
    noise = (DEFAULT_NOISE_FRACTION * volume_ul) ** 2
    state = dict(estimate_ul=float(effective_ul), variance=noise, noise=noise,
                 drift=MIN_DRIFT_FRACTION * noise, sessions=0, source="default")
    if measurements is not None:
        history = estimation.session_estimates(estimation.matching_history(
            estimation.usable_measurements(measurements, liquid, tip), volume_ul, settings))
        if len(history) >= estimation.MIN_SESSIONS_FOR_MODEL:
            # Model spread from history, but start from the value in use: the base value
            # was already set from this history, so re-estimating here would double count.
            result = estimation.estimate_row(history)
            noise_h = max(result["noise_sd_ul"] ** 2, 1e-6)
            state.update(variance=result["sd_ul"] ** 2, noise=noise_h,
                         drift=max(result["drift_sd_ul"] ** 2, MIN_DRIFT_FRACTION * noise_h),
                         sessions=len(history), source="history")
    return state


def robust_values(values):
    """Mask of replicates kept after the MAD outlier rule shared with the estimator."""
    values = np.asarray(values, float)
    if len(values) < 3:
        return np.ones(len(values), bool)
    median = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - median))
    if mad == 0:
        return np.ones(len(values), bool)
    return np.abs(values - median) <= max(estimation.OUTLIER_MADS * mad, 1.0)


def is_surprising(state, check_ul):
    """True when the check batch is far enough from expectation to need confirming."""
    spread = math.sqrt(state["variance"] + state["drift"] + state["noise"])
    return abs(float(np.median(check_ul)) - state["estimate_ul"]) > SURPRISE_SD * spread


def decide(state, check_ul, confirm_ul=None, upper_ul=None, lower_ul=None):
    """Update one row's state from this session's implied overaspirate values (uL).

    consistent      check batch matches expectation: normal damped update
    drift_confirmed check was surprising and the confirm batch agreed: full damped update
    transient       check was surprising and the confirm batch disagreed: the step is
                    limited to SURPRISE_SD standard deviations
    """
    check = np.asarray(check_ul, float)
    values = check if confirm_ul is None else np.concatenate([check, np.asarray(confirm_ul, float)])
    observed = float(np.median(values))
    prior = state["variance"] + state["drift"]
    spread = math.sqrt(prior + state["noise"])
    innovation = observed - state["estimate_ul"]
    if confirm_ul is None:
        verdict = "consistent"
    else:
        scatter = max(np.std(values, ddof=1) if len(values) > 1 else 0.0, MIN_REPLICATE_SD_UL)
        limit = 2 * scatter * math.sqrt(1 / len(check) + 1 / len(confirm_ul))
        agree = abs(float(np.median(confirm_ul)) - float(np.median(check))) <= limit
        verdict = "drift_confirmed" if agree else "transient"
        if not agree:
            innovation = max(-SURPRISE_SD * spread, min(SURPRISE_SD * spread, innovation))
    gain = prior / (prior + state["noise"])
    estimate = state["estimate_ul"] + gain * innovation
    if upper_ul is not None:
        estimate = min(estimate, upper_ul)
    if lower_ul is not None:
        estimate = max(estimate, lower_ul)
    return dict(state, estimate_ul=estimate, variance=(1 - gain) * prior, sessions=state["sessions"] + 1,
                previous_ul=state["estimate_ul"], observed_ul=observed, gain=gain, verdict=verdict,
                source="tracked")


def save_state(liquid, tip, volume_ul, state, base_ul, session_id, path=None):
    """Replace this row's tracked state (atomic write). ``base_ul`` None keeps the stored base."""
    path = Path(path or CORRECTIONS_FILE)
    corrections = load_corrections(path)
    if not corrections.empty:
        existing = corrections[_key(corrections, liquid, tip, volume_ul)]
        if base_ul is None and not existing.empty:
            base_ul = existing.base_overaspirate_ul.iloc[-1]
        corrections = corrections[~_key(corrections, liquid, tip, volume_ul)]
    row = dict(liquid=liquid, tip_type=tip, volume_target=volume_ul, overaspirate_ul=state["estimate_ul"],
               sd_ul=math.sqrt(state["variance"]), noise_sd_ul=math.sqrt(state["noise"]),
               drift_sd_ul=math.sqrt(state["drift"]), sessions=state["sessions"], base_overaspirate_ul=base_ul,
               updated=datetime.now().strftime("%Y%m%d_%H%M%S"), last_session_id=session_id,
               last_verdict=state.get("verdict"))
    frame = pd.concat([corrections, pd.DataFrame([row])], ignore_index=True)[COLUMNS]
    write_csv(frame.sort_values(["liquid", "tip_type", "volume_target"]), path)


def log_update(log_path, *, liquid, tip, volume_ul, state, session_id, environment):
    """Append to parameter_corrections.csv so existing history views keep working."""
    record = dict(timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"), session_id=session_id,
                  validation_type="check_confirm", liquid_type=liquid, tip_type=tip,
                  target_volume_ml=volume_ul / 1000, target_volume_ul=volume_ul,
                  old_overaspirate_ml=state["previous_ul"] / 1000, new_overaspirate_ml=state["estimate_ul"] / 1000,
                  correction_ml=(state["estimate_ul"] - state["previous_ul"]) / 1000,
                  correction_ul=state["estimate_ul"] - state["previous_ul"],
                  accepted=True, decision_reason=state["verdict"], decision_source="check_confirm",
                  **{k: (environment or {}).get(k) for k in ("temp_c", "humidity_pct", "pressure_pa")})
    append_csv(pd.DataFrame([record]), log_path)
