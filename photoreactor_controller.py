import subprocess

def run_photoreactor(rpm,duration,intensity,reactor_num):
    try:
        # Run the mpremote command as a subprocess
        command = f"import reactor_test; reactor_test.run_reactor({rpm},{duration},{intensity},{reactor_num})"
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

# # Run the mpremote command
# rpm=600
# duration=10
# intensity = 50
# reactor_num = 2
# run_mpremote_command(rpm,duration,intensity,reactor_num)
# run_mpremote_command(1000,5,100,2)
# run_mpremote_command(400,5,30,1)
# run_mpremote_command(800,5,90,1)