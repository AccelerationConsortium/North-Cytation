import subprocess

class Photoreactor_Controller:
    def run_command(self,command):
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

    def initialize_photoreactor(self):
        command = f"import reactor_test; reactor_test.initialize_photoreactor()"
        self.run_command(command)
    
    def run_photoreactor(self,rpm,duration,intensity,reactor_num):
        command = f"import reactor_test; reactor_test.run_reactor({rpm},{duration},{intensity},{reactor_num})"
        self.run_command(command)
    
    def turn_on_reactor_led(self,reactor_num, intensity):
        command = f"import reactor_test; reactor_test.turn_on_reactor_led({reactor_num},{intensity})"
        self.run_command(command)
    
    def turn_off_reactor_led(self,reactor_num):
        command = f"import reactor_test; reactor_test.turn_off_reactor_led({reactor_num})"
        self.run_command(command)
    
    def stir_reactor(self,reactor_num, rpm):
        self.turn_on_reactor_fan(reactor_num, rpm)

    def turn_off_stirring(self,reactor_num):
        self.turn_off_reactor_fan(reactor_num)

    def turn_on_reactor_fan(self,reactor_num, rpm):
        command = f"import reactor_test; reactor_test.turn_on_reactor_fan({reactor_num},{rpm})"
        self.run_command(command)
    
    def turn_off_reactor_fan(self,reactor_num):
        command = f"import reactor_test; reactor_test.turn_off_reactor_fan({reactor_num})"
        self.run_command(command)
