import subprocess

class Photoreactor_Controller:
    def run_command(command):
        try:
            result = subprocess.run(
                ['mpremote', 'connect', 'COM6', 'exec', command],
                stdout=subprocess.PIPE,  # Capture the output of the command
                stderr=subprocess.PIPE,  # Capture any error output
                text=True  # Ensure the output is returned as a string
            )

            # Print the output and error (if any)
            print("Output:")
            print(result.stdout)
            
            if result.stderr:
                print("Errors:")
                print(result.stderr)
        except Exception as e:
            print(f"Error running mpremote: {e}")

    def initialize_photoreactor():
        command = f"import reactor_test; reactor_test.initialize_photoreactor()"
        run_command(command)
    def run_photoreactor(rpm,duration,intensity,reactor_num):
        command = f"import reactor_test; reactor_test.run_reactor({rpm},{duration},{intensity},{reactor_num})"
        run_command(command)
    def turn_on_reactor_led(reactor_num, intensity):
        command = f"import reactor_test; reactor_test.turn_on_reactor_led({reactor_num},{intensity})"
        run_command(command)
    def turn_off_reactor_led(reactor_num):
        command = f"import reactor_test; reactor_test.turn_off_reactor_led({reactor_num})"
        run_command(command)
    def turn_on_reactor_fan(reactor_num, rpm):
        command = f"import reactor_test; reactor_test.turn_on_reactor_fan({reactor_num},{rpm})"
        run_command(command)
    def turn_off_reactor_fan(reactor_num):
        command = f"import reactor_test; reactor_test.turn_off_reactor_fan({reactor_num})"
        run_command(command)
