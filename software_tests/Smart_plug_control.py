"""Set selected smart outlets to the same configured state."""

import sys
from pathlib import Path

SMART_PLUG_DIR = Path(__file__).resolve().parents[2] / "Smart plug"
sys.path.insert(0, str(SMART_PLUG_DIR))

from smart_plug import get_status, load_config, set_outlet


OUTLETS = (1, 2, 3, 4)
TURN_ON = False  # True turns the selected outlets on; False turns them off.


def main():
    if type(TURN_ON) is not bool:
        raise ValueError("TURN_ON must be True or False.")
    if not OUTLETS or any(
        type(outlet) is not int or outlet not in range(1, 7) for outlet in OUTLETS
    ):
        raise ValueError("Select smart outlets 1-6.")

    config = load_config()
    current = get_status()
    print("Current outlet states:", current)

    unknown = [outlet for outlet in OUTLETS if current[outlet] is None]
    if unknown:
        raise RuntimeError(f"Unknown status for selected outlets {unknown}; no changes sent.")

    changes = [outlet for outlet in OUTLETS if current[outlet] != TURN_ON]
    if not changes:
        print("Selected outlets already match the requested states.")
        return

    print(f"Requested changes: turn {'on' if TURN_ON else 'off'} outlets {changes}")
    if not config["simulation"]:
        approval = input("LIVE HARDWARE: apply these power changes? Type yes to confirm: ")
        if approval.strip().lower() != "yes":
            print("Cancelled; no power commands sent.")
            return

    for outlet in changes:
        try:
            current = set_outlet(outlet, TURN_ON)
        except (OSError, ValueError, RuntimeError, ImportError):
            print(
                "Power may have changed, including earlier outlets. "
                "Check status before retrying.",
                file=sys.stderr,
            )
            raise
        print(f"Outlet {outlet} set {'on' if TURN_ON else 'off'}; current states: {current}")


if __name__ == "__main__":
    main()
