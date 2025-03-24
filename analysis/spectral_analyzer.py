import pandas as pd
import numpy as np
import os
import math

#We will need the file data and the labels
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

#Get the absolute spectral difference between two spectra
def get_absolute_spectral_difference(wavelengths_ref, spectra_ref, wavelengths_target, spectra_target):
    result = 0
    if np.all(wavelengths_ref == wavelengths_target):
        result = float(np.sum(np.abs(np.array(spectra_ref,)-np.array(spectra_target))))
    else:
        print("Issue getting Sum Squares Discrete Differences")
        result = None
    return result

#Get the absolute wavelength difference
def get_absolute_peak_wavelength_difference(wavelengths_ref, spectra_ref, wavelengths_target, spectra_target):
    if np.all(wavelengths_ref == wavelengths_target):
        # Find the index of the maximum value in each spectrum
        max_index_ref = np.argmax(spectra_ref)
        max_index_target = np.argmax(spectra_target)
        
        # Get the corresponding wavelengths
        peak_wavelength_ref = wavelengths_ref[max_index_ref]
        peak_wavelength_target = wavelengths_target[max_index_target]
        
        # Compute the absolute difference
        result = abs(peak_wavelength_ref - peak_wavelength_target)
    else:
        print("Issue getting difference: Wavelength arrays do not match")
        result = None
    
    return result

#Get the SS difference at a set of specified wavelengths
def get_sum_squares_discrete_difference(wavelengths_ref, spectra_ref, wavelengths_target, spectra_target, peak_wavelengths):
    result = 0
    if np.all(wavelengths_ref == wavelengths_target):
        for peak_wavelength in peak_wavelengths:
            index = np.abs(wavelengths_ref - peak_wavelength).argmin()
            result += (spectra_ref[index]-spectra_target[index])**2
    else:
        print("Issue getting Sum Squares Discrete Differences")
        return None
    return math.sqrt(result)

#Get the spectral data from the most recent file
def get_spectral_data_from_most_recent_file(folder_path):
    file = get_most_recent_file(folder_path)
    data = get_spectral_data_from_file(file)
    return data

#Get a specific spectra from a data set using the index (typically the wavelength)
def get_spectra_from_df_using_index(spec_df, index,offset_columns=1):
    spectra = spec_df.iloc[:, index+offset_columns].values
    spectra = remove_overflow(spec_df)
    return spectra

#Get the wavelengths from a data set (typically the first column)
def get_wavelengths_from_df(spec_df, wavelength_index=0):
    wavelengths = spec_df.iloc[:,wavelength_index].values
    return wavelengths

#Get the spectral data from a file
def get_spectral_data_from_file(file_name):
    data = pd.read_csv(file_name, sep='\t', skiprows=[0,1])
    return data

#Get the file that's most recent in a folder
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

