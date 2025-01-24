import pandas as pd
import numpy as np
import os

#We will need the file data and the labels

def get_differences(reference_data_file, reference_index, target_data_file, target_index_list):

    ref_data = pd.read_csv(reference_data_file, sep='\t', skiprows=[0,1])
    comp_data = pd.read_csv(target_data_file, sep='\t', skiprows=[0,1])

    wavelengths_ref = ref_data['Wavelength'].values
    wavelengths_comp = comp_data['Wavelength'].values

    ref_spectra = ref_data.iloc[:, reference_index+1].values

    difference = []
    for i in target_index_list:
        if np.all(wavelengths_ref == wavelengths_comp):
            comp_spectra=comp_data.iloc[:,i+1].values
            difference.append(float(np.sum(ref_spectra-comp_spectra)))
        else:
            print("Script failed - spectra not comparable")
            difference.append(None)

    return difference


def get_most_recent_file(folder_path):
    try:
        # Get all files in the folder
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        # Filter only files (not directories)
        files = [f for f in files if os.path.isfile(f)]
        # Get the file with the latest modification time
        most_recent_file = max(files, key=os.path.getmtime)
        return most_recent_file
    except ValueError:
        return "The folder is empty or doesn't contain any files."
    except Exception as e:
        return f"An error occurred: {e}"

