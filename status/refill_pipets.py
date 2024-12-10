import numpy as np

pipet_full_array = np.linspace(0,47,48, dtype=int)

save_data = ','.join(map(str, pipet_full_array.flatten()))

PIPET_FILE = "pipets.txt" #NEED TO CHANGE FOLDER
#PIPET_FILE = "C:\Users\Imaging Controller\Desktop\utoronto_demo\status\refill_pipets.py"

with open(PIPET_FILE, "w") as output:
    output.write(save_data.replace('\0', ''))

