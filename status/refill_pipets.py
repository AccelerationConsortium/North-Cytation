import sys
import numpy as np
sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo")
sys.path.append("C:\\Users\\Imaging Controller\\Desktop\\utoronto_demo\\status")
PIPET_FILE = "../utoronto_demo/status/pipets.txt"

pipet_full_array = np.linspace(0,47,48, dtype=int)

save_data = ','.join(map(str, pipet_full_array.flatten()))

PIPET_FILE = "C://Users//Imaging Controller//Desktop//utoronto_demo//status//pipets.txt"

with open(PIPET_FILE, "w") as output:
    output.write(save_data)

print('Pipets refilled')
