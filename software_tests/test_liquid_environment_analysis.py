"""The same environmental model accepts fixed and varied overaspirate sessions."""

import numpy as np
import pandas as pd
import pytest

from scheduler import liquid_environment_analysis as analysis


@pytest.mark.parametrize("vary_overaspirate", [False, True])
def test_recovers_environmental_effect_without_counting_replicates_as_sessions(monkeypatch, vary_overaspirate):
    monkeypatch.setattr(analysis, "current_parameters", lambda *_: {"overaspirate_vol": 0.002})
    rng = np.random.default_rng(19)
    rows = []
    for session in range(36):
        month = session % 3 + 4
        temperature = 21 + rng.uniform(0, 5)
        humidity = 30 + rng.uniform(0, 30)
        overaspirates = [0.001, 0.002, 0.003] if vary_overaspirate else [0.002]
        for overaspirate in overaspirates:
            error_pct = (1.5 * (temperature - 23) - 0.7 * (humidity - 50) / 10
                         + 0.8 * 100 * (overaspirate - 0.002) / 0.02
                         + rng.normal(0, 0.15))
            for replicate in range(4):
                rows.append(dict(session_id=f"session-{session}", timestamp=f"2026{month:02d}15_120000",
                                 target_volume_ml=0.02, measured_volume_ml=0.02 * (1 + error_pct / 100),
                                 overaspirate_vol=overaspirate, temp_c=temperature,
                                 humidity_pct=humidity, simulate=False))
    result = analysis.analyze_environment(pd.DataFrame(rows), calibration=None, bootstrap_samples=150)
    assert result is not None
    assert result.sessions == 36
    assert result.stage_means == 36 * (3 if vary_overaspirate else 1)
    assert abs(result.temperature.coefficient - 1.5) < 0.2
    assert abs(result.humidity.coefficient + 0.7) < 0.2
    assert result.temperature.typical_low < result.temperature.typical_high
    assert abs(result.temperature.typical_change - result.temperature.coefficient *
               (result.temperature.typical_high - result.temperature.typical_low)) < 1e-9
    assert result.temperature.direction_agreement_pct > 95
    assert result.environment_mae < result.baseline_mae
