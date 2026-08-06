#Turn on and off the photoreactor LED and fan

import subprocess
import sys
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E
import time

VIAL_FILE = "../utoronto_demo/status/sample_capped_vial.txt"  # Vials used
COM_PORT = "COM6"


def run_mpremote(command):
    """Run a command directly on the Pico via mpremote and print the result."""
    print(f"\n--- mpremote exec: {command} ---")
    result = subprocess.run(
        ["mpremote", "connect", COM_PORT, "exec", command],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.stdout:
        print("Output:", result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)


def sample_workflow(input_vial_status_file, target_vial):

    # Step 1: List available functions on the Pico to confirm correct function names
    print("\n=== Step 1: List reactor_test functions on Pico ===")
    run_mpremote("import reactor_test; print(dir(reactor_test))")

    # Step 2: Try turn_on_reactor_fan directly via mpremote (bypasses Python controller)
    print("\n=== Step 2: Direct mpremote stir command (10 sec) ===")
    run_mpremote(f"import reactor_test; reactor_test.turn_on_reactor_fan(0, 600)")
    time.sleep(10)
    run_mpremote(f"import reactor_test; reactor_test.turn_off_reactor_fan(0)")

    # Step 3: Try stir_reactor in case firmware uses a different name
    print("\n=== Step 3: Direct mpremote stir_reactor command (10 sec) ===")
    run_mpremote(f"import reactor_test; reactor_test.stir_reactor(0, 600)")
    time.sleep(10)
    run_mpremote(f"import reactor_test; reactor_test.turn_off_reactor_fan(0)")

    # Step 4: Use the Python controller (same path the workflow uses)
    print("\n=== Step 4: Via Python controller (initialize + fan on for 10 sec) ===")
    lash_e = Lash_E(input_vial_status_file, initialize_robot=False, initialize_biotek=False, initialize_track=False)
    lash_e.photoreactor.initialize_photoreactor()
    lash_e.photoreactor.turn_on_reactor_fan(reactor_num=0, rpm=600)
    time.sleep(10)
    lash_e.photoreactor.turn_off_reactor_fan(reactor_num=0)

    # LED test (uncomment to run)
    # print("\n=== LED test ===")
    # lash_e.photoreactor.turn_on_reactor_led(reactor_num=0, intensity=100)
    # time.sleep(5)
    # lash_e.photoreactor.turn_off_reactor_led(reactor_num=0)

    print("\n=== Done ===")


sample_workflow(VIAL_FILE, 0)
