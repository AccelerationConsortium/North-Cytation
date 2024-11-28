import numpy as np

pipet_full_array = np.linspace(0,47,48)

save_data = ','.join(map(str, pipet_full_array.flatten()))

PIPET_FILE = "/pipets.txt"

with open(PIPET_FILE, "w") as output:
    output.write(save_data.replace('\0', ''))

