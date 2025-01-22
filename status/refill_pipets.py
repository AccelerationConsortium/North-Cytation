import sys
import numpy as np
sys.path.append("../utoronto_demo")
sys.path.append("..\\utoronto_demo\\status")
PIPET_FILE = "../utoronto_demo/status/pipets.txt"

save_data = "0,0"

with open(PIPET_FILE, "w") as output:
    output.write(save_data)

print('Pipets refilled')
