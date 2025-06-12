import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Define the Boltzmann sigmoid function
def boltzmann(x, A1, A2, x0, dx):
    return A2 + (A1 - A2) / (1 + np.exp((x - x0) / dx))

def CMC_plot(i1_i3_ratio, conc):
    # Initial guess for parameters [A1, A2, x0, dx]
    p0 = [max(i1_i3_ratio), min(i1_i3_ratio), (max(conc) - min(conc))/2, (max(conc) - min(conc)) / 5]

    # Fit the data to the Boltzmann sigmoid
    popt, pcov = curve_fit(boltzmann, conc, i1_i3_ratio, p0, maxfev=5000)
    A1, A2, x0, dx = popt

    # Compute the second CMC (xCMC2) using the derived formula
#    xCMC2 = x0 + dx * np.log((A1 - A2) / (0.5 * (A1 - A2)))

    # Compute R-squared value for goodness of fit
    residuals = i1_i3_ratio - boltzmann(conc, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((i1_i3_ratio - np.mean(i1_i3_ratio))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Plot the data and fitted curve
    x_fit = np.linspace(min(conc), max(conc), 100)
    y_fit = boltzmann(x_fit, *popt)

    plt.figure(figsize=(8, 6))
    plt.scatter(conc, i1_i3_ratio, label='Experimental Data', color='blue')
    plt.plot(x_fit, y_fit, label='Boltzmann Fit', color='red')
    plt.axvline(x0, linestyle='--', color='green', label=f'(xCMC)1 = {x0:.2f} mM')
#    plt.axvline(xCMC2, linestyle='--', color='purple', label=f'(xCMC)2 = {xCMC2:.2f} mM')
    plt.xlabel('Surfactant Concentration (mM)')
    plt.ylabel('I₁/I₃ Ratio')
    plt.title('CMC Determination using Boltzmann Fit')
    plt.legend()
    plt.grid()
    plt.show()

    # Output the computed CMC values and R-squared
    print(f'Estimated CMC: {x0:.2f} mM')
    print(f'Fit Accuracy (R²): {r_squared:.4f}')

    return A1, A2, x0, dx, r_squared