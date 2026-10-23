"""Exploratory association of session conditions with pipetting error.

All stages are included. Replicates at one tested overaspirate are averaged first;
each session has equal total weight, regardless of its number of stages or volumes.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scheduler.liquid_review import current_parameters


@dataclass(frozen=True)
class EnvironmentEffect:
    coefficient: float
    ci_low: float
    ci_high: float
    typical_low: float
    typical_high: float
    typical_change: float
    typical_ci_low: float
    typical_ci_high: float
    direction_agreement_pct: float


@dataclass(frozen=True)
class EnvironmentAnalysis:
    sessions: int
    stage_means: int
    volumes: int
    temperature_range: tuple[float, float]
    humidity_range: tuple[float, float]
    temperature: EnvironmentEffect
    humidity: EnvironmentEffect
    baseline_mae: float
    environment_mae: float
    bootstrap_fits: int


def _stage_means(matched, calibration):
    data = matched.copy()
    if data.empty:
        return pd.DataFrame()
    if "simulate" in data:
        data = data[~data.simulate.astype(str).str.casefold().eq("true")]
    for field in ("target_volume_ml", "measured_volume_ml", "overaspirate_vol",
                  "temp_c", "humidity_pct"):
        data[field] = pd.to_numeric(data[field], errors="coerce")
    data["when"] = pd.to_datetime(data.timestamp, format="%Y%m%d_%H%M%S", errors="coerce")
    data = data.dropna(subset=["session_id", "when", "target_volume_ml", "measured_volume_ml",
                               "overaspirate_vol", "temp_c", "humidity_pct"])
    data = data[(data.target_volume_ml > 0) & np.isfinite(data.measured_volume_ml)]
    if data.empty:
        return pd.DataFrame()
    # One environment reading per session; all stages of that session share it.
    groups = data.groupby(["session_id", "target_volume_ml", "overaspirate_vol"], as_index=False)
    stage = groups.agg(measured_volume_ml=("measured_volume_ml", "mean"),
                       when=("when", "first"), temp_c=("temp_c", "first"),
                       humidity_pct=("humidity_pct", "first"))
    # A target seen in fewer than five sessions cannot support a useful target intercept.
    counts = stage.groupby("target_volume_ml").session_id.nunique()
    stage = stage[stage.target_volume_ml.isin(counts[counts >= 5].index)].copy()
    if stage.empty:
        return stage
    reference = {volume: current_parameters(calibration, volume * 1000)["overaspirate_vol"]
                 for volume in stage.target_volume_ml.unique()}
    stage["error_pct"] = 100 * (stage.measured_volume_ml - stage.target_volume_ml) / stage.target_volume_ml
    stage["oa_delta_pct"] = 100 * (stage.overaspirate_vol - stage.target_volume_ml.map(reference)) / stage.target_volume_ml
    return stage.sort_values(["session_id", "target_volume_ml", "overaspirate_vol"]).reset_index(drop=True)


def _fit(design, outcome, weights):
    weighted = np.sqrt(weights)
    return np.linalg.lstsq(design * weighted[:, None], outcome * weighted, rcond=None)[0]


def _session_mae(stage, baseline, environment, weights, folds=5):
    sessions = np.array(sorted(stage.session_id.unique()))
    rng = np.random.default_rng(20261022)
    rng.shuffle(sessions)
    errors = [[], []]
    for held in np.array_split(sessions, min(folds, len(sessions))):
        test = stage.session_id.isin(held).to_numpy()
        train = ~test
        for design, collection in zip((baseline, environment), errors):
            fit = _fit(design[train], stage.error_pct.to_numpy()[train], weights[train])
            collection.extend(zip(np.abs(stage.error_pct.to_numpy()[test] - design[test] @ fit), weights[test]))
    return tuple(float(sum(error * weight for error, weight in collection) /
                       sum(weight for _, weight in collection)) for collection in errors)


def analyze_environment(matched, calibration, bootstrap_samples=400):
    """Return associations in percentage points of delivery error, or None if too sparse."""
    stage = _stage_means(matched, calibration)
    if stage.empty or stage.session_id.nunique() < 20:
        return None
    volumes = sorted(stage.target_volume_ml.unique())
    months = stage.when.dt.to_period("M").to_numpy()
    columns = [np.ones(len(stage))]
    columns.extend((stage.target_volume_ml.to_numpy() == volume).astype(float) for volume in volumes[1:])
    if stage.oa_delta_pct.std() > 1e-8:
        columns.append(stage.oa_delta_pct.to_numpy())
    columns.extend((months == month).astype(float) for month in sorted(set(months))[1:])
    baseline = np.column_stack(columns)
    environment = np.column_stack((baseline, stage.temp_c.to_numpy() - 23,
                                   (stage.humidity_pct.to_numpy() - 50) / 10))
    if np.linalg.matrix_rank(environment) < environment.shape[1]:
        return None
    outcome = stage.error_pct.to_numpy()
    weights = 1 / stage.groupby("session_id").session_id.transform("size").to_numpy()
    fit = _fit(environment, outcome, weights)
    sessions = stage.session_id.unique()
    indices = {session: np.flatnonzero(stage.session_id.to_numpy() == session) for session in sessions}
    rng = np.random.default_rng(20261022)
    boot = []
    for _ in range(bootstrap_samples):
        sample = rng.choice(sessions, size=len(sessions), replace=True)
        index = np.concatenate([indices[session] for session in sample])
        design = environment[index]
        if np.linalg.matrix_rank(design) == design.shape[1]:
            boot.append(_fit(design, outcome[index], weights[index])[-2:])
    if len(boot) < max(100, bootstrap_samples // 2):
        return None
    intervals = np.percentile(boot, [2.5, 97.5], axis=0)
    baseline_mae, environment_mae = _session_mae(stage, baseline, environment, weights)
    unique_sessions = stage.drop_duplicates("session_id")
    boot_coefficients = np.asarray(boot)

    def effect(column, field, scale):
        low, high = np.percentile(unique_sessions[field], [10, 90])
        span = (high - low) / scale
        coefficient = float(fit[column])
        lower, upper = (float(value) for value in intervals[:, column])
        return EnvironmentEffect(
            coefficient=coefficient, ci_low=lower, ci_high=upper,
            typical_low=float(low), typical_high=float(high),
            typical_change=float(coefficient * span),
            typical_ci_low=float(lower * span), typical_ci_high=float(upper * span),
            direction_agreement_pct=float(100 * np.mean(
                np.sign(boot_coefficients[:, column]) == np.sign(coefficient))),
        )

    return EnvironmentAnalysis(
        sessions=len(sessions), stage_means=len(stage), volumes=len(volumes),
        temperature_range=(float(stage.temp_c.min()), float(stage.temp_c.max())),
        humidity_range=(float(stage.humidity_pct.min()), float(stage.humidity_pct.max())),
        temperature=effect(-2, "temp_c", 1),
        humidity=effect(-1, "humidity_pct", 10),
        baseline_mae=baseline_mae, environment_mae=environment_mae,
        bootstrap_fits=len(boot),
    )
