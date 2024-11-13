import numpy as np

rows = ['A','B','C','D','E','F','G','H']

def convert_wells_into_indices(wells):
    well_indices = []
    for well in wells:
        well_row = rows.index(well[0:1])
        well_col = int(well[1:])
        index = well_row*12 + well_col - 1
        well_indices.append(index)

    return well_indices

wells = ['A4', 'C5', 'H12']

print("well names: ", wells)
print("converted wells:", convert_wells_into_indices(wells))

