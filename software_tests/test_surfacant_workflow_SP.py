"""Manually exercise the vial positions used by the multidimensional workflow.

Run this file directly with the robot connected. It does not prepare solutions
or move a wellplate. Set MIMIC_PIPETTING to True to aspirate and return a small
volume from each staged vial, then test the configurable pipetting settings
using the water vial.
"""

from contextlib import contextmanager
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CAMERA_CONTROL_FOLDER = Path.home() / "Desktop" / "camera-control"
CAMERA_RECORDING_FOLDER = CAMERA_CONTROL_FOLDER / "recordings"
sys.path.insert(0, str(ROOT))

from master_usdl_coordinator import Lash_E


SIMULATE = False
MOVE_ALL_VIALS = False
MOVE_WATER_VIAL = True
MIMIC_PIPETTING = True
RECORD_CAMERA_FOOTAGE = False
INPUT_VIAL_STATUS_FILE = ROOT / "status" / "surfactant_multidim_vials.csv"
PIPET_TEST_VOLUME_ML = 0.02
TEST_WATER_VIAL = "water"
USE_TEST_WATER_VOLUME_FOR_HEIGHT = True
TEST_WATER_VOLUME_FOR_HEIGHT_ML = 2.0
USE_SPECIFIED_PIPET_TIP = True
PIPET_TEST_TIP_TYPE = "small_tip"

# These are the safe dispensing positions used by
# position_surfactant_vials_by_concentration in the multidimensional workflow.
WORKFLOW_VIAL_POSITIONS = [
    ("main_8mL_rack", 47),
    ("main_8mL_rack", 46),
    ("main_8mL_rack", 45),
    ("main_8mL_rack", 44),
    ("clamp", 0),
    ("main_8mL_rack", 43),
    ("clamp_2", 0),
]


def _surfactant_dilution_groups(vial_df):
    """Return configured 8 mL dilution vials grouped by surfactant."""
    groups = {}
    dilution_rows = vial_df[
        vial_df["vial_name"].str.contains("_dilution_", regex=False)
        & (vial_df["vial_type"] == "8_mL")
    ]
    for row in dilution_rows.itertuples(index=False):
        prefix, _, suffix = row.vial_name.partition("_dilution_")
        if not suffix.isdigit():
            raise ValueError(f"Unexpected dilution vial name: {row.vial_name}")
        groups.setdefault(prefix, []).append((int(suffix), row.vial_name))

    return {
        surfactant: [
            vial_name
            for _, vial_name in sorted(vials, key=lambda item: item[0])
        ]
        for surfactant, vials in sorted(groups.items())
    }


def _aspirate_from_water_for_test(robot):
    """Test aspiration with optional height and tip overrides."""
    water_rows = robot.VIAL_DF.index[
        robot.VIAL_DF["vial_name"] == TEST_WATER_VIAL
    ]
    if len(water_rows) != 1:
        raise ValueError(
            f"Expected exactly one '{TEST_WATER_VIAL}' vial in the vial file"
        )
    if USE_SPECIFIED_PIPET_TIP and PIPET_TEST_TIP_TYPE not in robot.PIPET_TIPS:
        raise ValueError(
            f"Unknown pipette tip '{PIPET_TEST_TIP_TYPE}'. "
            f"Available tips: {list(robot.PIPET_TIPS)}"
        )
    if USE_TEST_WATER_VOLUME_FOR_HEIGHT and (
        not math.isfinite(TEST_WATER_VOLUME_FOR_HEIGHT_ML)
        or TEST_WATER_VOLUME_FOR_HEIGHT_ML <= 0
    ):
        raise ValueError("Test water volume for height must be a positive number")

    original_height_function = robot.get_aspirate_height
    if USE_TEST_WATER_VOLUME_FOR_HEIGHT:
        row_index = water_rows[0]
        original_volume_ml = robot.VIAL_DF.at[row_index, "vial_volume"]

        def height_using_test_volume(
            vial, amount_ml, track_height=True, buffer=1.0
        ):
            robot.VIAL_DF.at[row_index, "vial_volume"] = (
                TEST_WATER_VOLUME_FOR_HEIGHT_ML
            )
            try:
                return original_height_function(
                    vial, amount_ml, track_height=track_height, buffer=buffer
                )
            finally:
                robot.VIAL_DF.at[row_index, "vial_volume"] = original_volume_ml

        robot.get_aspirate_height = height_using_test_volume

    try:
        robot.aspirate_from_vial(
            TEST_WATER_VIAL,
            PIPET_TEST_VOLUME_ML,
            liquid="water",
            specified_tip=(
                PIPET_TEST_TIP_TYPE if USE_SPECIFIED_PIPET_TIP else None
            ),
        )
    finally:
        if USE_TEST_WATER_VOLUME_FOR_HEIGHT:
            robot.get_aspirate_height = original_height_function

    robot.dispense_into_vial(
        TEST_WATER_VIAL,
        PIPET_TEST_VOLUME_ML,
        liquid="water",
        record_measurement=False,
    )
    robot.remove_pipet()


@contextmanager
def _camera_recording(lash_e):
    if not RECORD_CAMERA_FOOTAGE:
        yield None
        return

    if not CAMERA_CONTROL_FOLDER.is_dir():
        raise FileNotFoundError(
            f"Camera-control folder not found: {CAMERA_CONTROL_FOLDER}"
        )
    sys.path.insert(0, str(CAMERA_CONTROL_FOLDER))
    from camera_automation import CameraAutomation

    with CameraAutomation(
        recording_folder=str(CAMERA_RECORDING_FOLDER),
        profile_name="both",
    ) as camera:
        with camera.recording() as session_folder:
            lash_e.logger.info(
                f"Recording both cameras to {session_folder}"
            )
            yield session_folder
        if camera.recording_failure is not None:
            raise RuntimeError(
                f"Camera recording failed: {camera.recording_failure}"
            )


def execute():
    """Stage every configured surfactant dilution vial in workflow positions."""
    lash_e = Lash_E(
        vial_file=str(INPUT_VIAL_STATUS_FILE),
        initialize_track=False,
        initialize_biotek=False,
        simulate=SIMULATE,
        show_gui=True,
        logging_folder=str(ROOT / "logs"),
    )
    if not lash_e._workflow_should_continue:
        return

    robot = lash_e.nr_robot
    vial_groups = _surfactant_dilution_groups(robot.VIAL_DF)
    if MOVE_ALL_VIALS and not vial_groups:
        raise ValueError(
            f"No 8 mL surfactant dilution vials found in {INPUT_VIAL_STATUS_FILE}"
        )
    if MOVE_ALL_VIALS and len(WORKFLOW_VIAL_POSITIONS) == 0:
        raise ValueError("WORKFLOW_VIAL_POSITIONS must not be empty")

    if MOVE_ALL_VIALS or MOVE_WATER_VIAL:
        occupied_positions = [
            (location, index)
            for location, index in WORKFLOW_VIAL_POSITIONS
            if not robot.VIAL_DF.loc[
                (robot.VIAL_DF["location"] == location)
                & (robot.VIAL_DF["location_index"] == index)
            ].empty
        ]
        if occupied_positions:
            raise ValueError(
                "Workflow staging positions must be empty before this test: "
                f"{occupied_positions}"
            )
    if MOVE_ALL_VIALS:
        oversized_groups = {
            surfactant: len(vials)
            for surfactant, vials in vial_groups.items()
            if len(vials) > len(WORKFLOW_VIAL_POSITIONS)
        }
        if oversized_groups:
            raise ValueError(
                "A surfactant has more dilution vials than workflow staging "
                f"positions: {oversized_groups}"
            )

    with _camera_recording(lash_e):
        lash_e.logger.info(
            f"Starting surfactant workflow movement test: simulate={SIMULATE}, "
            f"move_all_vials={MOVE_ALL_VIALS}, "
            f"mimic_pipetting={MIMIC_PIPETTING}, "
            f"record_camera_footage={RECORD_CAMERA_FOOTAGE}, "
            f"surfactant_groups={len(vial_groups)}"
        )

        if MOVE_ALL_VIALS:
            robot.home_robot_components()

            for surfactant, vials in vial_groups.items():
                lash_e.logger.info(
                    f"Positioning {len(vials)} {surfactant} dilution vials"
                )
                staged_vials = []
                for vial, (location, location_index) in zip(
                    vials, WORKFLOW_VIAL_POSITIONS
                ):
                    lash_e.logger.info(
                        f"Moving {vial} to {location}[{location_index}]"
                    )
                    robot.move_vial_to_location(vial, location, location_index)
                    staged_vials.append(vial)

                if MIMIC_PIPETTING:
                    for vial in staged_vials:
                        vial_volume_ml = float(
                            robot.get_vial_info(vial, "vial_volume")
                        )
                        if vial_volume_ml < PIPET_TEST_VOLUME_ML:
                            lash_e.logger.info(
                                f"Skipping pipetting mimic for {vial}: "
                                f"{vial_volume_ml:.3f}mL available"
                            )
                            continue
                        robot.aspirate_from_vial(
                            vial,
                            PIPET_TEST_VOLUME_ML,
                            liquid=surfactant,
                        )
                        robot.dispense_into_vial(
                            vial,
                            PIPET_TEST_VOLUME_ML,
                            liquid=surfactant,
                            record_measurement=False,
                        )
                        robot.remove_pipet()

                for vial in staged_vials:
                    robot.return_vial_home(vial)

        if MOVE_WATER_VIAL:
            water_vial_rows = robot.VIAL_DF.loc[
                robot.VIAL_DF["vial_name"] == TEST_WATER_VIAL
            ]
            if len(water_vial_rows) != 1:
                raise ValueError(
                    f"Expected exactly one '{TEST_WATER_VIAL}' vial in the vial file"
                )
            for location, location_index in WORKFLOW_VIAL_POSITIONS:
                water_volume_ml = float(
                    robot.get_vial_info(TEST_WATER_VIAL, "vial_volume")
                )
                lash_e.logger.info(
                    f"Moving {TEST_WATER_VIAL} to "
                    f"{location}[{location_index}]"
                )
                robot.move_vial_to_location(
                    TEST_WATER_VIAL, location, location_index
                )
                if MIMIC_PIPETTING:
                    if water_volume_ml < PIPET_TEST_VOLUME_ML:
                        lash_e.logger.info(
                            f"Skipping water pipetting test at "
                            f"{location}[{location_index}]: "
                            f"{water_volume_ml:.3f}mL available"
                        )
                        continue
                    lash_e.logger.info(
                        f"Testing water aspiration at "
                        f"{location}[{location_index}]: "
                        f"use_test_volume_for_height="
                        f"{USE_TEST_WATER_VOLUME_FOR_HEIGHT}, "
                        f"test_volume={TEST_WATER_VOLUME_FOR_HEIGHT_ML:.2f}mL, "
                        f"specified_tip="
                        f"{PIPET_TEST_TIP_TYPE if USE_SPECIFIED_PIPET_TIP else 'automatic'}"
                    )
                    _aspirate_from_water_for_test(robot)

            robot.return_vial_home(TEST_WATER_VIAL)

        if MOVE_ALL_VIALS or MOVE_WATER_VIAL:
            robot.home_robot_components()
        else:
            lash_e.logger.info(
                "MOVE_ALL_VIALS and MOVE_WATER_VIAL are False; skipping all "
                "vial movements and pipetting tests."
            )
        lash_e.logger.info("Surfactant workflow movement test complete")


if __name__ == "__main__":
    execute()