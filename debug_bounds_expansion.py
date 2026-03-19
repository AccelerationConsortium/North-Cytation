# Fix for calibration bounds expansion issue
# 
# Problem: During two-point calibration, screening trials (0.5-7.2 μL) are not 
# available for bounds expansion, causing all historical data to be excluded 
# from optimization.
#
# Root cause: `_current_screening_trials` is empty when bounds calculation runs
#
# Solution: Add diagnostic logging and ensure screening trials are preserved
# during the bounds calculation phase.

# The fix needs to be applied to experiment.py around line 1341 where:
# existing_screening_trials = getattr(self, '_current_screening_trials', []) if self.current_volume_index == 0 else []

import logging
logger = logging.getLogger(__name__)

def debug_screening_trials_availability(self):
    """Debug helper to check screening trials availability"""
    if hasattr(self, '_current_screening_trials'):
        trials = self._current_screening_trials
        logger.info(f"DEBUG: _current_screening_trials exists with {len(trials)} trials")
        if trials:
            overaspirates = [trial.parameters.overaspirate_vol * 1000 for trial in trials]
            logger.info(f"DEBUG: Overaspirate range: [{min(overaspirates):.1f}-{max(overaspirates):.1f}] μL")
        else:
            logger.warning("DEBUG: _current_screening_trials is empty!")
    else:
        logger.error("DEBUG: _current_screening_trials attribute missing!")
    
    if self.current_volume_index == 0:
        logger.info("DEBUG: This is volume index 0 (first volume)")
    else:
        logger.info(f"DEBUG: This is volume index {self.current_volume_index}")