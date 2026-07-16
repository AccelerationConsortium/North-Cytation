"""Continuously move a vial to random rack positions (0-47), one after another,
until the user presses Ctrl+C. The vial starts at rack position 0.

Usage:
    python tests/random_vial_shuffle_test.py

Configuration:
    SIMULATE - set to True for dry-run (no hardware)
    VIAL_FILE - path to vial status CSV
    VIAL_NAME - name of the vial that lives at home position 0
"""

import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from master_usdl_coordinator import Lash_E

# ============================================================
# CONFIGURATION
# ============================================================
SIMULATE = False
VIAL_FILE = str(REPO_ROOT / "status" / "random_shuffle_test_vials.csv")
VIAL_NAME = "test_vial"   # vial that starts at main_8mL_rack index 0
RACK_MIN = 0
RACK_MAX = 47
DELAY_BETWEEN_MOVES_S = 1.0  # pause between moves

# ============================================================
# MAIN
# ============================================================

lash_e = Lash_E(VIAL_FILE, simulate=SIMULATE, initialize_biotek=False)
logger = lash_e.logger

logger.info(f"Starting random vial shuffle test. SIMULATE={SIMULATE}")
logger.info(f"Vial: {VIAL_NAME}, rack range: [{RACK_MIN}, {RACK_MAX}]")
logger.info("Press Ctrl+C to stop.")

current_index = 0
iteration = 0
try:
    while True:
        iteration += 1
        # Pick a new index different from the current one
        choices = [i for i in range(RACK_MIN, RACK_MAX + 1) if i != current_index]
        next_index = random.choice(choices)
        logger.info(f"Move {iteration}: {VIAL_NAME} index {current_index} -> {next_index}")

        lash_e.nr_robot.move_vial_to_location(VIAL_NAME, "main_8mL_rack", next_index)
        current_index = next_index
        time.sleep(DELAY_BETWEEN_MOVES_S)

except KeyboardInterrupt:
    logger.info(f"Interrupted after {iteration} move(s). Vial is at index {current_index}.")
