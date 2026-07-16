"""Standalone safe-position mover for water vial checks.

This script uses the same safe-position context as the surfactant workflows
without modifying any workflow code.

Sequence:
1) Move selected vial to clamp and return home.
2) Move the same vial to each configured safe position and return home.
3) Optionally run full robot homing between position cycles.
"""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.append(str(REPO_ROOT))

from master_usdl_coordinator import Lash_E


DEFAULT_VIAL_FILE = str(REPO_ROOT / "status" / "surfactant_multidim_vials.csv")

# Mirrors the safe-placement context used by surfactant positioning logic.
# The initial explicit clamp move is handled separately, so clamp is omitted
# from this default traversal list.
DEFAULT_SAFE_POSITIONS = [47, 46, 45, 44, 43, ("heater", 2), ("heater", 4)]

# Set these booleans directly to control homing behavior.
HOME_BEFORE_START = False
HOME_BETWEEN_MOVES = False


def parse_position_token(token):
	token = token.strip().lower()
	if token == "clamp":
		return "clamp"
	if token.startswith("heater:"):
		parts = token.split(":", maxsplit=1)
		return ("heater", int(parts[1]))
	return int(token)


def parse_positions(tokens):
	if not tokens:
		return list(DEFAULT_SAFE_POSITIONS)
	return [parse_position_token(t) for t in tokens]


def describe_position(position):
	if position == "clamp":
		return "clamp[0]"
	if isinstance(position, tuple) and position[0] == "heater":
		return f"heater[{position[1]}]"
	return f"main_8mL_rack[{position}]"


def move_to_position(lash_e, vial_name, position):
	if position == "clamp":
		lash_e.nr_robot.move_vial_to_location(vial_name, "clamp", 0)
		return

	if isinstance(position, tuple) and position[0] == "heater":
		lash_e.nr_robot.move_vial_to_location(vial_name, "heater", int(position[1]))
		return

	lash_e.nr_robot.move_vial_to_location(vial_name, "main_8mL_rack", int(position))


def should_home_before_next_move(cycle_index, total_cycles, always_home_between):
	if cycle_index >= total_cycles - 1:
		return False
	return always_home_between


def run_position_test(args):
	lash_e = Lash_E(
		vial_file=args.vial_file,
		simulate=args.simulate,
		initialize_biotek=False,
	)

	logger = lash_e.logger
	vial_name = args.vial_name
	safe_positions = parse_positions(args.safe_positions)

	logger.info("Starting safe-position vial movement test")
	logger.info(f"Simulation mode: {args.simulate}")
	logger.info(f"Vial: {vial_name}")

	if args.home_before_start:
		logger.info("Homing robot components before starting test")
		lash_e.nr_robot.home_robot_components()

	logger.info("Step 1: Move vial to clamp, then return home")
	lash_e.nr_robot.move_vial_to_location(vial_name, "clamp", 0)
	lash_e.nr_robot.return_vial_home(vial_name)

	logger.info("Step 2: Iterate safe positions and return home after each move")
	for idx, position in enumerate(safe_positions):
		logger.info(f"Cycle {idx + 1}/{len(safe_positions)}: moving {vial_name} -> {describe_position(position)}")
		move_to_position(lash_e, vial_name, position)
		lash_e.nr_robot.return_vial_home(vial_name)

		if should_home_before_next_move(
			cycle_index=idx,
			total_cycles=len(safe_positions),
			always_home_between=args.home_between_moves,
		):
			logger.info("Running full homing step before next move")
			lash_e.nr_robot.home_robot_components()

	logger.info("Safe-position vial movement test complete")


def build_parser():
	parser = argparse.ArgumentParser(
		description="Move a vial through safe positions used by surfactant workflows."
	)
	parser.add_argument(
		"--vial-file",
		default=DEFAULT_VIAL_FILE,
		help="Path to vial status CSV",
	)
	parser.add_argument(
		"--vial-name",
		default="water",
		help="Vial name to move",
	)
	parser.add_argument(
		"--simulate",
		action="store_true",
		help="Run in simulation mode",
	)
	parser.add_argument(
		"--home-before-start",
		action="store_true",
		default=HOME_BEFORE_START,
		help="Run full robot homing before first move (default from HOME_BEFORE_START)",
	)
	parser.add_argument(
		"--home-between-moves",
		action="store_true",
		default=HOME_BETWEEN_MOVES,
		help="Always run full robot homing between safe-position cycles (default from HOME_BETWEEN_MOVES)",
	)
	parser.add_argument(
		"--safe-positions",
		nargs="+",
		help=(
			"Optional safe-position override list. Use rack indices (e.g. 47 46), "
			"'clamp', or heater:index (e.g. heater:2)."
		),
	)
	return parser


if __name__ == "__main__":
	cli_args = build_parser().parse_args()
	run_position_test(cli_args)
