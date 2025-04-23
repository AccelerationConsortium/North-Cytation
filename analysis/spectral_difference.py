import pandas as pd
import numpy as np
import os
import math

#We will need the file data and the labels

#whole and key_wavelengths
COMP_METHOD_A = 0 #Absolute spectral difference
COMP_METHOD_B = 1 #Discrete SS Difference
PEAK_WAVELENGTHS = [410,515,630]

def remove_overflow(data):
    processed_data = []
    last_valid_value = None
    for item in data:
        if item == 'OVRFLW':
            # If we encounter 'OVRFLW', replace it with the last valid number
            processed_data.append(last_valid_value)
        else:
            # Convert the item to a float and add to the list
            value = float(item)
            processed_data.append(value)
            last_valid_value = value  # Update the last valid value
    return processed_data

def get_absolute_spectral_difference(wavelengths_ref, spectra_ref, wavelengths_target, spectra_target):
    result = 0
    if np.all(wavelengths_ref == wavelengths_target):
        result = float(np.sum(np.abs(np.array(spectra_ref,)-np.array(spectra_target))))
    else:
        print("Issue getting Sum Squares Discrete Differences")
        result = None
    return result

def get_sum_squares_discrete_difference(wavelengths_ref, spectra_ref, wavelengths_target, spectra_target, peak_wavelengths=PEAK_WAVELENGTHS):
    result = 0
    if np.all(wavelengths_ref == wavelengths_target):
        for peak_wavelength in peak_wavelengths:
            index = np.abs(wavelengths_ref - peak_wavelength).argmin()
            result += (spectra_ref[index]-spectra_target[index])**2
    else:
        print("Issue getting Sum Squares Discrete Differences")
        return None
    return math.sqrt(result)

def get_spectra_from_df(spec_df, index):
    spectra = spec_df.iloc[:, index+1].values
    spectra = remove_overflow(spec_df)
    return spectra

def get_data_and_wavelengths_from_file(file_name):
    data = pd.read_csv(file_name, sep='\t', skiprows=[0,1])
    wavelengths = data['Wavelength'].values
    return data,wavelengths

#TODO: Let's take the plotter out!
def get_spectral_differences(reference_data_file, reference_index, target_data_file, target_index_list, difference_type=COMP_METHOD_A,plotter=None,color=None):
    print("Processing spectral difference...")

    ref_data,wavelengths_ref = get_data_and_wavelengths_from_file(reference_data_file)
    comp_data,wavelengths_comp = get_data_and_wavelengths_from_file(target_data_file)
    ref_spectra = get_spectra_from_df(ref_data,reference_index)

    #print("Ref spectra", ref_spectra)
    if plotter is not None and reference_data_file == target_data_file:
        plotter.add_data(0,wavelengths_ref,ref_spectra,color='r')
        plotter.add_data(1,wavelengths_ref,ref_spectra,color='r')

    spectral_difference_list = []
    for i in target_index_list:
        comp_spectra = get_spectra_from_df(comp_data,i)
        if difference_type==COMP_METHOD_A:
            spectral_difference_list.append(get_absolute_spectral_difference(wavelengths_ref, ref_spectra, wavelengths_comp, comp_spectra))
        elif difference_type==COMP_METHOD_B:
            spectral_difference_list.append (get_sum_squares_discrete_difference(wavelengths_ref, ref_spectra, wavelengths_comp, comp_spectra))
    
    if plotter is not None:
        best_spectra_index = np.argmin(spectral_difference_list)
        plotter.add_data(1,wavelengths_ref,get_spectra_from_df(comp_data,best_spectra_index),color=color)

    return spectral_difference_list

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

