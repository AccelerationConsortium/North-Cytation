import numpy as np
import pandas as pd



# surfactant_library
surfactant_library = {
    "SDS": {
        "full_name": "Sodium Dodecyl Sulfate",
        "CAS": "151-21-3",
        "CMC": 8.3,
        "Category": "anionic",
        "MW": 289.39,
        "stock_conc": 50,  # mM
    },


    # "SLS": {
    #     "full_name": "Sodium Lauryl Sulfate",
    #     "CAS": "151-21-3",
    #     "CMC": 7.7,
    #     "Category": "anionic",
    #     "MW": 288.38,
    # },


    "NaDC": {
        "full_name": "Sodium Docusate",
        "CAS": "577-11-7",
        "CMC": 8.2,
        "Category": "anionic",
        "MW": 445.57,
        "stock_conc": 25,  # mM
    },

    
    "NaC": {
        "full_name": "Sodium Cholate",
        "CAS": "361-09-1",
        "CMC": 11,
        "Category": "anionic",
        "MW": 431.56,
        "stock_conc": 50,  # mM
    },


    "CTAB": {
        "full_name": "Hexadecyltrimethylammonium Bromide",
        "CAS": "57-09-0",
        "CMC": 0.93,
        "Category": "cationic",
        "MW": 364.45,
        "stock_conc": 5 # mM
    },


    "DTAB": {
        "full_name": "Dodecyltrimethylammonium Bromide",
        "CAS": "1119-94-4",
        "CMC": 15.85,
        "Category": "cationic",
        "MW": 308.34,
        "stock_conc": 50,  # mM
    },


    "TTAB": {
        "full_name": "Tetradecyltrimethylammonium Bromide",
        "CAS": "1119-97-7",
        "CMC": 3.77,
        "Category": "cationic",
        "MW": 336.39,
        "stock_conc": 50,  # mM
    },


    # "BAC": {
    #     "full_name": "Benzalkonium Chloride",
    #     "CAS": "63449-41-2",
    #     "CMC": 0.42,
    #     "Category": "cationic",
    #     "MW": "unknown",
    # },


#     "T80": {
#         "full_name": "Tween 80",
#         "CAS": "9005-65-6",
#         "CMC": 0.015,
#         "Category": "nonionic",
#         "MW": 1310,
# #        "stock_conc": ,  # mM
#     },

    
#     "T20": {
#         "full_name": "Tween 20",
#         "CAS": "9005-64-5",
#         "CMC": 0.0355,
#         "Category": "nonionic",
#         "MW": 1226,
# #        "stock_conc": ,  # mM
#     },


    "P188": {
        "full_name": "Kolliphor® P 188 Geismar",
        "CAS": "9003-11-6",
        "CMC": 0.325,
        "Category": "nonionic",
        "MW": 8595,
        "stock_conc": 2,  # mM
    },


    "P407": {
        "full_name": "Kolliphor® P 407 Geismar",
        "CAS": "9003-11-6",
        "CMC": 0.1,
        "Category": "nonionic",
        "MW": 12300,
        "stock_conc": 2,  # mM
    },

    "CAPB": {
        "full_name": "Cocamidopropyl Betaine",
        "CAS": "61789-40-0",
        "CMC": 0.627,
        "Category": "zwitterionic",
        "MW": 342.52,
        "stock_conc": 50,  # mM
    },

#     "SBS-12": {
#         "full_name": "Sulfobetaine-12",
#         "CAS": "14933-08-5",
#         "CMC": 3,
#         "Category": "zwitterionic",
#         "MW": 335.55,
# #        "stock_conc": ,  # mM
#     },

#     "SBS-14": {
#         "full_name": "Sulfobetaine-14",
#         "CAS": "14933-09-6",
#         "CMC": 0.16,
#         "Category": "zwitterionic",
#         "MW": 363.60,
# #       "stock_conc": ,  # mM
#     },
    
    "CHAPS": {
        "full_name": "CHAPS",
        "CAS": "75621-03-3",
        "CMC": 8.5,
        "Category": "zwitterionic",
        "MW": 614.88,
        "stock_conc": 30,  # mM
    }
}


# function to estimate the CMC of signle/mixed surfactants
def CMC_estimate(list_of_surfactants, list_of_ratios):

    cmc_inverse_sum = 0.0

    print("This batch of surfactants are: ")
    print(list_of_surfactants)
    print("This batch of ratios are: ")
    print(list_of_ratios)
    print()

    for surfactant, ratio in zip(list_of_surfactants, list_of_ratios):
        if surfactant is not None:
            cmc = surfactant_library[surfactant]['CMC']
            cmc_inverse_sum += ratio / cmc

    if cmc_inverse_sum == 0:
        print("Warning: CMC inverse sum is zero. Check surfactant inputs.")
        return None
    else:
        return 1 / cmc_inverse_sum


# function to generate a series of dilutions around the CMC, with more points closer to the CMC
# and fewer points further away
def generate_cmc_concentrations(cmc):
    """
    Generate 12 concentration points from cmc/2.5 to cmc*2.5.
    """
    # Log-spaced: from cmc/2.5 to cmc/1.5 (3 points)
    below = np.logspace(np.log10(cmc / 2.5), np.log10(cmc / 1.5), 3)

    # Linearly spaced: ±25% around CMC (6 points)
    around = np.linspace(cmc * 0.75, cmc * 1.25, 6)

    # Log-spaced: from cmc*1.5 to cmc*2.5 (3 points)
    above = np.logspace(np.log10(cmc * 1.5), np.log10(cmc * 2.5), 3)

    print(f"CMC estimate: ")
    print(cmc)
    print()
    print(f"Generated concentrations (refined): ")
    print(np.concatenate([below, around, above]))

    return np.concatenate([below, around, above]).tolist()



def rough_generate_cmc_concentrations(estimate_cmc, number_of_points=12 ,scale="log"): # low/high in mM


    low = estimate_cmc / (50**0.5)
    high = estimate_cmc * (50**0.5)

    if scale == "log":
        exponent_low = np.round(np.log10(low), 3)
        exponent_high = np.round(np.log10(high), 3)
        concentrations = np.round(np.logspace(exponent_low, exponent_high, number_of_points), 3)
    
    elif scale == "linear":
        concentrations = np.round(np.linspace(low, high, number_of_points), 3)

    print(f"CMC estimate: ")
    print(estimate_cmc)
    print()
    print(f"Generated concentrations (rough): ")
    print(concentrations)
    return concentrations.tolist()



# function to prepare a surfactant mix as a surfactant/surfactant mixture sub-stock    
def surfactant_substock(cmc_concs, list_of_surfactants, list_of_ratios,
                        probe_volume, sub_stock_volume, CMC_sample_volume, stock_concs):

    # Calculate adjusted mix stock concentration
    max_cmc_conc = max(cmc_concs)
    sub_stock_concentration = max_cmc_conc / ((CMC_sample_volume - probe_volume) / CMC_sample_volume)

    # Total moles needed
    total_mmol = sub_stock_concentration * sub_stock_volume / 1000  # mmol

    result = {}
    total_surfactant_volume = 0

    for i in range(3):
        surf = list_of_surfactants[i]
        ratio = list_of_ratios[i]
        stock = stock_concs[i]

        # Assign placeholder if name is None
        surf_label = surf if surf is not None else f"None_{i+1}"

        # Calculate volume or assign 0
        volume = (total_mmol * ratio) / (stock / 1000) if ratio > 0 else 0

        result[surf_label] = volume
        total_surfactant_volume += volume

    result['water'] = sub_stock_volume - total_surfactant_volume

    print(f"Sub-stock concentration: ")
    print(f"{sub_stock_concentration} mM")
    print()
    # print("result")
    # print(result)
    return sub_stock_concentration, result



def calculate_volumes(concentration_list, sub_stock_concentration, probe_volume, CMC_sample_volume):

    concentrations = []
    surfactant_volumes = []
    water_volumes = []
    probe_volumes = []
    total_volumes = []

    for conc in concentration_list:
        surfactant_volume = (conc * (CMC_sample_volume - probe_volume)) / sub_stock_concentration
        water_volume = CMC_sample_volume - probe_volume - surfactant_volume

        # Round for consistency
        surfactant_volume = round(surfactant_volume, 2)
        water_volume = round(water_volume, 2)

        # Collect values
        concentrations.append(conc)
        surfactant_volumes.append(surfactant_volume)
        water_volumes.append(water_volume)
        probe_volumes.append(probe_volume)
        total_volumes.append(round(surfactant_volume + water_volume + probe_volume, 2))

    df = pd.DataFrame({
        "concentration": concentrations,
        "surfactant volume": surfactant_volumes,
        "water volume": water_volumes,
        "probe volume": probe_volumes,
        "total volume": total_volumes
    })

    return df

def generate_exp(list_of_surfactants, list_of_ratios, probe_volume = 30, sub_stock_volume = 5000, CMC_sample_volume=1000):

    ## to confirm probe_volume = 30, sub_stock_volume = 5000, CMC_sample_volume=1000 ##

    stock_concs = []
    for surf in list_of_surfactants:

        if surf is None:
            stock_concs.append(0)

        elif surf in surfactant_library:
            stock = surfactant_library[surf].get("stock_conc")
            if stock is None:
                raise ValueError(f"Stock concentration missing for {surf}.")
            stock_concs.append(stock)
        else:
            raise KeyError(f"{surf} not found in the surfactant library.")

        # Validations
    # if len(list_of_surfactants) != 3 or len(list_of_ratios) != 3 or len(stock_concs) != 3:
    #     raise ValueError("Inputs must all have 3 elements.")
    if sum(list_of_ratios) != 1:
        raise ValueError("Sum of surfactant ratios must be == 1")



    estimated_CMC = CMC_estimate(list_of_surfactants, list_of_ratios)
    cmc_concs = generate_cmc_concentrations(estimated_CMC)

    sub_stock_concentration, sub_stock_vol = surfactant_substock(cmc_concs = cmc_concs, 
                                                        list_of_surfactants = list_of_surfactants, 
                                                        list_of_ratios = list_of_ratios, 
                                                        stock_concs=stock_concs,
                                                        probe_volume = probe_volume,
                                                        sub_stock_volume = sub_stock_volume,
                                                        CMC_sample_volume = CMC_sample_volume,
                                                        )
    
    df = calculate_volumes(cmc_concs, sub_stock_concentration, probe_volume = probe_volume, CMC_sample_volume = CMC_sample_volume)

    # All info
    exp = {
        "list_of_surfactants": list_of_surfactants,
        "list_of_ratios": list_of_ratios,
        "original_surfactant_stock_concs": stock_concs,
        "surfactant_sub_stock_conc": sub_stock_concentration,
        "surfactant_sub_stock_vols": sub_stock_vol,
        "estimated_CMC": estimated_CMC,
        "df": df,
    }

    # Info needed to generate OTFlex protocol
    small_exp = {

        "surfactant_sub_stock_vols": sub_stock_vol,
        "solvent_sub_vol": df["surfactant volume"].tolist(),
        "water_vol": df["water volume"].tolist(),
        "pyrene_vol": df["probe volume"].tolist(),
    }

    print("Experiment volume info:" )
    print(small_exp)
    return exp, small_exp


def mM_to_mg_per_mL(surfactant, concentration_mM):

    if surfactant not in surfactant_library:
        raise ValueError(f"Surfactant {surfactant} not found in library.")

    MW = surfactant_library[surfactant]['MW']
    concentration_mg_per_mL = (concentration_mM * MW) / 1000  # Convert mM to mg/mL

    return concentration_mg_per_mL