import numpy as np
import pandas as pd



# surfactant_library
surfactant_library = {
    "SDS": {
        "full_name": "Sodium Dodecyl Sulfate",
        "CAS": "151-21-3",
        "CMC": 8.3,
        "Category": "anionic",
    },


    "SLS": {
        "full_name": "Sodium Lauryl Sulfate",
        "CAS": "151-21-3",
        "CMC": 7.7,
        "Category": "anionic",
    },


    "NaDC": {
        "full_name": "Sodium Docusate",
        "CAS": "577-11-7",
        "CMC": 8.2,
        "Category": "anionic",
    },

    
    "NaC": {
        "full_name": "Sodium Cholate",
        "CAS": "361-09-1",
        "CMC": 11,
        "Category": "anionic",
    },


    "CTAB": {
        "full_name": "Hexadecyltrimethylammonium Bromide",
        "CAS": "57-09-0",
        "CMC": 0.93,
        "Category": "cationic",
    },


    "DTAB": {
        "full_name": "Dodecyltrimethylammonium Bromide",
        "CAS": "1119-94-4",
        "CMC": 15.85,
        "Category": "cationic",
    },


    "TTAB": {
        "full_name": "Tetradecyltrimethylammonium Bromide",
        "CAS": "1119-97-7",
        "CMC": 3.77,
        "Category": "cationic",
    },


    "BAC": {
        "full_name": "Benzalkonium Chloride",
        "CAS": "63449-41-2",
        "CMC": 0.42,
        "Category": "cationic",
    },


    "T80": {
        "full_name": "Tween 80",
        "CAS": "9005-65-6",
        "CMC": 0.015,
        "Category": "nonionic",
    },

    
    "T20": {
        "full_name": "Tween 20",
        "CAS": "9005-64-5",
        "CMC": 0.0355,
        "Category": "nonionic",
    },


    "P188": {
        "full_name": "Kolliphor® P 188 Geismar",
        "CAS": "9003-11-6",
        "CMC": 0.325,
        "Category": "nonionic",
    },


    "P407": {
        "full_name": "Kolliphor® P 407 Geismar",
        "CAS": "9003-11-6",
        "CMC": 0.1,
        "Category": "nonionic",
    },

    "CAPB": {
        "full_name": "Cocamidopropyl Betaine",
        "CAS": "61789-40-0",
        "CMC": 0.627,
        "Category": "zwitterionic",
    },

    "SBS-12": {
        "full_name": "Sulfobetaine-12",
        "CAS": "14933-08-5",
        "CMC": 3,
        "Category": "zwitterionic",
    },

    "SBS-14": {
        "full_name": "Sulfobetaine-14",
        "CAS": "14933-09-6",
        "CMC": 0.16,
        "Category": "zwitterionic",
    },
    
    "CHAPS": {
        "full_name": "CHAPS",
        "CAS": "75621-03-3",
        "CMC": 8.5,
        "Category": "zwitterionic",
    }
}


# function to estimate the CMC of signle/mixed surfactants
def CMC_estimate(list_of_surfactants, list_of_ratios):

    cmc_inverse_sum = 0.0

    for surfactant, ratio in zip(list_of_surfactants, list_of_ratios):
        if surfactant is not None:
            cmc = surfactant_library[surfactant]['CMC']
            cmc_inverse_sum += ratio / cmc

    if cmc_inverse_sum == 0:
        return None
    else:
        return 1 / cmc_inverse_sum


# function to generate a series of dilutions around the CMC, with more points closer to the CMC
# and fewer points further away
def generate_cmc_concentrations(cmc):
    """
    Generate 12 concentration points from cmc/3 to cmc*3.
    """
    # Log-spaced: from cmc/10 to cmc/1.5 (3 points)
    below = np.logspace(np.log10(cmc / 3), np.log10(cmc / 1.5), 3)

    # Linearly spaced: ±25% around CMC (6 points)
    around = np.linspace(cmc * 0.75, cmc * 1.25, 6)

    # Log-spaced: from cmc*1.5 to cmc*10 (3 points)
    above = np.logspace(np.log10(cmc * 1.5), np.log10(cmc * 3), 3)

    return np.concatenate([below, around, above]).tolist()


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

def generate_exp(list_of_surfactants, list_of_ratios, stock_concs=[50, 50, 50], probe_volume = 10, sub_stock_volume = 5000, CMC_sample_volume=1000):

        # Validations
    if len(list_of_surfactants) != 3 or len(list_of_ratios) != 3 or len(stock_concs) != 3:
        raise ValueError("Inputs must all have 3 elements.")
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

    return exp, small_exp


