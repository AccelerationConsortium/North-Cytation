import pandas as pd
import numpy as np

#We will need the file data and the labels

def get_differences(test_data_file, target_data_file):

    raw_data = pd.read_csv('analysis_data.txt', delimiter='\t')
    target_spectra_label = 'A1' #Potentially should come from another file
    test_spectra_labels = ['A2', 'A3']

    wavelengths_test = raw_data['Wavelength'].values
    wavelengths_target = wavelengths_test #This should come from another file
    spectra_target = raw_data[target_spectra_label].values

    difference = []
    for test_label in test_spectra_labels:

        if np.all(wavelengths_target == wavelengths_test):
            spectra_test=raw_data[test_label].values
            difference.append(float(np.sum(spectra_target-spectra_test)))
        else:
            print("Script failed - spectra not comparable")
            difference.append(None)

    return difference

