"""Four cumulative 5 mL water transfers into an initially empty BZT_refill vial.

Run from the repository root: python tests/water_to_bzt_refill.py
Uses the multidimensional surfactant workflow's vial-status file. Leave the
water in BZT_refill between rounds: expected totals are 5, 10, 15, and 20 mL.
The startup GUI lets you review simulation/hardware mode and vial positions.
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from master_usdl_coordinator import Lash_E
from workflows.interrupt_cleanup import cleanup_after_keyboard_interrupt
from workflows.surfactant_grid_adaptive_concentrations import fill_water_vial


SIMULATE = False
INPUT_VIAL_STATUS_FILE = str(_REPO_ROOT / 'status' / 'surfactant_multidim_vials.csv')


def run_transfer_test(lash_e):
    robot = lash_e.nr_robot
    destination = 'BZT_refill'
    destination_index = robot.normalize_vial_index(destination)
    input(
        'Place an EMPTY 20 mL vial at BZT_refill (large_vial_rack[3]). '
        'Leave water in it between rounds. Press Enter to start: '
    )
    # The shared CSV describes stock, but the user has installed an empty test vial.
    robot.VIAL_DF.at[destination_index, 'vial_volume'] = 0.0
    robot.save_robot_status()

    for round_number in range(1, 5):
        fill_water_vial(lash_e, 'water')
        lash_e.logger.info('Water transfer %s/4: 5 mL into %s', round_number, destination)
        robot.dispense_from_vial_into_vial(
            source_vial_name='water', dest_vial_name=destination,
            volume=5.0, liquid='water', remove_tip=True,
            return_vial_home=True, record_measurement=False,
        )
        robot.return_vial_home('water')
        robot.move_home()
        expected_ml = round_number * 5
        print(f'Round {round_number}/4 complete: expected cumulative volume {expected_ml} mL.')
        if round_number < 4:
            input('Measure/check the vial; leave the water in it. Press Enter for the next round: ')
        else:
            input('Final 20 mL transfer complete. Press Enter to finish: ')
    return 4


def execute(config=None, show_gui=True):
    lash_e = Lash_E(
        initialize_biotek=False, initialize_track=False,
        workflow_globals=globals(), workflow_name=Path(__file__).stem,
        config=config, show_gui=show_gui,
    )
    if not lash_e._workflow_should_continue:
        return None
    try:
        return run_transfer_test(lash_e)
    except (Exception, KeyboardInterrupt):
        cleanup_after_keyboard_interrupt(lash_e)
        raise


if __name__ == '__main__':
    execute()
