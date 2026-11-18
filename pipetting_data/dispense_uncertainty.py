"""How far off one pipette dispense is likely to be, from the validation history.

Uses the same screened history and local-level model as validation and the liquid review
GUI (``overaspirate_estimation``), so all three agree. For one liquid, tip and calibrated
volume (µL of delivered volume):

    random  = typical spread between replicates in one session (differs dispense to dispense)
    shared  = sqrt(estimate SD² + drift² + session noise²): how far one day's dispenses can
              all be off together

Only calibrated volumes with at least MIN_SESSIONS_FOR_MODEL screened sessions under the
current settings count. A volume between two such calibrated volumes is interpolated;
anything else (outside the range, a weak neighbour, no history, reservoir) has no estimate
and returns None. Never raises.
"""

import math

import numpy as np
import pandas as pd

from pipetting_data import overaspirate_estimation as estimation

TOLERANCE_UL = 0.05


class DispenseUncertainty:
    def __init__(self, measurements=None, logger=None):
        self._measurements = measurements
        self._tables = {}
        self.logger = logger

    def measurements(self):
        if self._measurements is None:
            from pipetting_data.embedded_calibration_validation import MASTER_PIPETTING_MEASUREMENTS_DB
            self._measurements = pd.read_csv(MASTER_PIPETTING_MEASUREMENTS_DB, low_memory=False)
        return self._measurements

    def table(self, liquid, tip):
        """Sorted [(volume_ul, random_ul, shared_ul)] for validated calibrated volumes."""
        key = (str(liquid).casefold(), tip)
        if key in self._tables:
            return self._tables[key]
        rows = []
        try:
            from pipetting_data.pipetting_wizard import PipettingWizard
            found = PipettingWizard().find_best_calibration_file(liquid, 0.1, tip_type=tip)
            calibration = found[1] if found else None
            if calibration is not None:
                if "tip_type" in calibration:
                    calibration = calibration[calibration.tip_type == tip]
                data = estimation.usable_measurements(self.measurements(), liquid, tip)
                for _, row in calibration.iterrows():
                    rows.append((float(row.volume_target), *self._estimate(data, row)))
        except Exception as error:
            if self.logger is not None:
                self.logger.warning(f"Dispense uncertainty unavailable for {liquid} {tip}: {error}")
            rows = []
        self._tables[key] = sorted(rows)
        return self._tables[key]

    @staticmethod
    def _estimate(data, row):
        """(random_ul, shared_ul) for one calibrated row, or (None, None) if not validated."""
        sessions = estimation.session_estimates(
            estimation.matching_history(data, row.volume_target, estimation.row_settings(row)))
        if len(sessions) < estimation.MIN_SESSIONS_FOR_MODEL:
            return None, None
        result = estimation.estimate_row(sessions)
        if not result or result.get("method") != "local level":
            return None, None
        kept = sessions[~sessions.session_id.isin([e["session_id"] for e in result["excluded"]])]
        random_ul = float(np.nanmedian(kept.replicate_sd_ul))
        parts = (result["sd_ul"], result["drift_sd_ul"], result["noise_sd_ul"])
        if not all(np.isfinite(parts)) or not np.isfinite(random_ul):
            return None, None
        return random_ul, math.sqrt(sum(p * p for p in parts))

    def lookup(self, liquid, tip, volume_ul):
        """(random_ul, shared_ul) for one dispense, or None when it can't be estimated."""
        try:
            if not liquid or tip not in ("small_tip", "large_tip") or not volume_ul or volume_ul <= 0:
                return None
            table = self.table(liquid, tip)
            for volume, random_ul, shared_ul in table:
                if abs(volume - volume_ul) <= TOLERANCE_UL:
                    return (random_ul, shared_ul) if random_ul is not None else None
            below = [r for r in table if r[0] < volume_ul]
            above = [r for r in table if r[0] > volume_ul]
            if not below or not above:
                return None  # outside the calibrated range
            (v0, r0, s0), (v1, r1, s1) = below[-1], above[0]
            if r0 is None or r1 is None:
                return None  # a neighbouring calibrated volume isn't validated
            weight = (volume_ul - v0) / (v1 - v0)
            return r0 + weight * (r1 - r0), s0 + weight * (s1 - s0)
        except Exception as error:
            if self.logger is not None:
                self.logger.warning(f"Dispense uncertainty lookup failed for {liquid} {tip} {volume_ul}: {error}")
            return None
