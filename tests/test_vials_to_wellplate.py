import numpy as np
import pandas as pd
import math
import sys
import os
import time
sys.path.append("../utoronto_demo")
from master_usdl_coordinator import Lash_E

def generate_random_matrix(rows, cols, row_sum, divisible_by, min_value):
    if row_sum < (cols - 1) * divisible_by:
        raise ValueError("Row sum is too small to meet the divisibility constraint.")

    matrix = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        remaining_sum = row_sum
        values = []

        for j in range(cols - 1):
            # Allow zero as a choice, but enforce minimum value for non-zero entries
            max_val = remaining_sum - (cols - j - 1) * min_value
            max_val = max(max_val, 0)

            if max_val == 0:
                values.append(0)
                continue

            # Generate possible values (including zero)
            choices = np.arange(0, max_val + divisible_by, divisible_by)
            if remaining_sum > min_value:
                choices = np.concatenate(([0], choices[choices >= min_value]))
            value = np.random.choice(choices)
            values.append(value)
            remaining_sum -= value

        # Last column takes the remaining sum (ensure it's divisible by divisible_by)
        if remaining_sum % divisible_by != 0:
            raise ValueError("Row sum constraints cannot be satisfied.")
        values.append(remaining_sum)

        # Shuffle to randomize placement of zeros and values
        np.random.shuffle(values)
        matrix[i, :] = values

    return matrix


# Parameters
rows = 3 #number of samples generated
replicates = 2 #number of replicates
cols = 4 #number of colors

row_sum = 250 #max per well 
divisible_by = 10 #resolution
min_value = 40 #It's actually divis_by + this value

data_colors_uL = generate_random_matrix(rows, cols, row_sum, divisible_by, min_value)/1000

print("Row sums:", np.sum(data_colors_uL * 1000, axis=1))  # Should all equal 250

data_colors_uL = np.repeat(data_colors_uL, replicates, axis=0)

sum_colors = np.sum(data_colors_uL,0)
print("Total volume per vial:", sum_colors) #how much of each volume is used

#needed files: 1. vial_status 2.wellplate_recipe
VIAL_FILE="../utoronto_demo/status/color_matching_vials.txt"
vial_df = pd.read_csv(VIAL_FILE, sep=r'\t', engine='python')
vial_indices = vial_df['vial index'].values
print("Vial indices:", vial_indices[0:cols])

lash_e = Lash_E(VIAL_FILE, initialize_biotek=False)

lash_e.nr_robot.dispense_from_vials_into_wellplate(pd.DataFrame(data_colors_uL), vial_indices[0:cols])
