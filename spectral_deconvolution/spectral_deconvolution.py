# -*- coding: utf-8 -*-
"""
@author: Haruka Ono
"""

# Import necessary libraries
import pandas as pd
import scipy as sp
from scipy.optimize import minimize


# spectral deconvolution function
def deconvolution(data):
    def fun(params):
        a, b, c, d = params
        total_sum = 0
        for j in range(0, 15):
            value = (data[j] - (a*Pdots["F8BT"][j] + b*Pdots["MEH-PPV"][j] + c*Pdots["TPP/PFO"][j] + d*Pdots["LN"][j]))**2
            total_sum += value
        return total_sum
    
    initial_guess = [1,1,1,1]
    bounds = [(0, None), (0, None), (0, None), (0, None)]
    results = minimize(fun, initial_guess, bounds=bounds)
    optimal_params = results.x
    SSE = results.fun
    a, b, c, d = optimal_params
    
    Fitting_results = pd.DataFrame(
        {
            "Wavelength" : Pdots["Wavelength"],
            "Pdot mixture" : data,
            "F8BT Pdot" : a*Pdots["F8BT"],
            "MEH-PPV Pdot" : b*Pdots["MEH-PPV"],
            "TPP-doped PFO Pdot" : c*Pdots["TPP/PFO"],
            "Autofluorescence" : d*Pdots["LN"]
        }
    )

    return Fitting_results

# Load spectral data of Pdots and autofluorescence
Pdots = pd.read_csv("Pdots_profile.csv")

# Example usage
file_name = "Normal_1.csv" # Specify the filename of spectral data of Pdot accumulated SLN
data = pd.read_csv(f"../data/{file_name}") # Load spectral data of Pdot accumulated SLN
deconvolution(data["ROI"]).to_csv(f"results/Deconvolution_results_{file_name}") # Perform spectral deconvolution