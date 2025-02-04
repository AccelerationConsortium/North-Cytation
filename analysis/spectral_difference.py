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

#whole and key_wavelengths
COMP_METHOD_A = 0
COMP_METHOD_B = 1
def get_differences(reference_data_file, reference_index, target_data_file, target_index_list,difference_type=COMP_METHOD_A,plotter=None,color=None):

    print("Reference file: ", reference_data_file)
    print("Reference index", reference_index)
    print("Target Data file", target_data_file)
    print("Target index ", target_index_list)

    ref_data = pd.read_csv(reference_data_file, sep='\t', skiprows=[0,1])
    comp_data = pd.read_csv(target_data_file, sep='\t', skiprows=[0,1])

    wavelengths_ref = ref_data['Wavelength'].values
    wavelengths_comp = comp_data['Wavelength'].values

    ref_spectra = ref_data.iloc[:, reference_index+1].values
    ref_spectra = remove_overflow(ref_spectra)

    #print("Ref spectra", ref_spectra)
    if plotter is not None and reference_data_file == target_data_file:
        print(type(wavelengths_ref), type(ref_spectra))
        print(len(wavelengths_ref), len(ref_spectra))

        plotter.add_data(0,wavelengths_ref,ref_spectra,color='r')
        plotter.add_data(1,wavelengths_ref,ref_spectra,color='r')

    difference = []
    comp_spectra_list = []
    for i in target_index_list:
        if np.all(wavelengths_ref == wavelengths_comp):
            comp_spectra=comp_data.iloc[:,i+1].values

            comp_spectra = remove_overflow(comp_spectra)
            comp_spectra_list.append(comp_spectra)

            if difference_type==COMP_METHOD_A:
                difference.append(float(np.sum(np.abs(np.array(ref_spectra)-np.array(comp_spectra)))))
            elif difference_type==COMP_METHOD_B:
                peak_red_wavelength = 425
                peak_yellow_wavelength = 520
                peak_blue_wavelength = 630

                red_index = np.abs(wavelengths_ref - peak_red_wavelength).argmin()
                yellow_index = np.abs(wavelengths_ref - peak_yellow_wavelength).argmin()
                blue_index = np.abs(wavelengths_ref - peak_blue_wavelength).argmin()

                red_dif = ref_spectra[red_index]-comp_spectra[red_index]
                yellow_dif = ref_spectra[yellow_index]-comp_spectra[yellow_index]
                blue_dif = ref_spectra[blue_index]-comp_spectra[blue_index]

                difference.append (math.sqrt(red_dif**2 + yellow_dif**2 + blue_dif**2)) #sum squares error
        else:
            print("Script failed - spectra not comparable")
            difference.append(None)
    
    if plotter is not None:
        best_spectra_index = np.argmin(difference)
        plotter.add_data(1,wavelengths_ref,comp_spectra_list[best_spectra_index],color=color)

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

