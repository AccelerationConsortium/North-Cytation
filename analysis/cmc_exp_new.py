#Re-defining essentials after kernel reset

import numpy as np
import pandas as pd


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

# Flexible versions of surfactant_substock and generate_exp to handle arbitrary number of surfactants
def surfactant_substock_flexible(cmc_concs, list_of_surfactants, list_of_ratios,
                                 probe_volume, sub_stock_volume, CMC_sample_volume, stock_concs):
    max_cmc_conc = max(cmc_concs)
    sub_stock_concentration = max_cmc_conc / ((CMC_sample_volume - probe_volume) / CMC_sample_volume)
    total_mmol = sub_stock_concentration * sub_stock_volume / 1000  # mmol

    result = {}
    total_surfactant_volume = 0

    for surf, ratio, stock in zip(list_of_surfactants, list_of_ratios, stock_concs):
        if ratio > 0:
            volume = (total_mmol * ratio) / (stock / 1000)
        else:
            volume = 0
        result[surf] = volume
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

        surfactant_volume = round(surfactant_volume, 2)
        water_volume = round(water_volume, 2)

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


def CMC_estimate(list_of_surfactants, list_of_ratios):
    surfactant_library = {
        "SDS": {"CMC": 8.3}, "NaDC": {"CMC": 8.2}, "NaC": {"CMC": 11},
        "CTAB": {"CMC": 0.93}, "DTAB": {"CMC": 15.85}, "TTAB": {"CMC": 3.77},
        "P188": {"CMC": 0.325}, "P407": {"CMC": 0.1}, "CAPB": {"CMC": 0.627}, "CHAPS": {"CMC": 8.5}
    }
    cmc_inverse_sum = 0.0
    for surfactant, ratio in zip(list_of_surfactants, list_of_ratios):
        cmc = surfactant_library[surfactant]['CMC']
        cmc_inverse_sum += ratio / cmc

    if cmc_inverse_sum == 0:
        return None
    return 1 / cmc_inverse_sum


def generate_cmc_concentrations(cmc):
    below = np.logspace(np.log10(cmc / 2.5), np.log10(cmc / 1.5), 3)
    around = np.linspace(cmc * 0.75, cmc * 1.25, 6)
    above = np.logspace(np.log10(cmc * 1.5), np.log10(cmc * 2.5), 3)
    return np.concatenate([below, around, above]).tolist()


def generate_exp_flexible(list_of_surfactants, list_of_ratios, probe_volume=25,
                          sub_stock_volume=6000, CMC_sample_volume=1000, rough_screen=False, estimated_CMC=None):

    surfactant_library = {
        "SDS": {"stock_conc": 50}, "NaDC": {"stock_conc": 25}, "NaC": {"stock_conc": 50},
        "CTAB": {"stock_conc": 5}, "DTAB": {"stock_conc": 50}, "TTAB": {"stock_conc": 50},
        "P188": {"stock_conc": 2}, "P407": {"stock_conc": 2}, "CAPB": {"stock_conc": 50}, "CHAPS": {"stock_conc": 30}
    }

    active = [(s, r) for s, r in zip(list_of_surfactants, list_of_ratios) if r > 0]
    if not active:
        raise ValueError("At least one surfactant must have a ratio > 0.")

    active_surfactants, active_ratios = zip(*active)
    if not np.isclose(sum(active_ratios), 1.0):
        raise ValueError("Sum of surfactant ratios must be 1.")

    stock_concs = [surfactant_library[s]['stock_conc'] for s in active_surfactants]
    
    if estimated_CMC is None:
        estimated_CMC = CMC_estimate(active_surfactants, active_ratios)

    if not rough_screen:
        cmc_concs = generate_cmc_concentrations(estimated_CMC)
    else:
        cmc_concs = rough_generate_cmc_concentrations(estimated_CMC)


    sub_stock_concentration, sub_stock_vol = surfactant_substock_flexible(
        cmc_concs=cmc_concs,
        list_of_surfactants=active_surfactants,
        list_of_ratios=active_ratios,
        stock_concs=stock_concs,
        probe_volume=probe_volume,
        sub_stock_volume=sub_stock_volume,
        CMC_sample_volume=CMC_sample_volume,
    )

    df = calculate_volumes(cmc_concs, sub_stock_concentration, probe_volume, CMC_sample_volume)

    exp = {
        "list_of_surfactants": active_surfactants,
        "list_of_ratios": active_ratios,
        "original_surfactant_stock_concs": stock_concs,
        "surfactant_sub_stock_conc": sub_stock_concentration,
        "surfactant_sub_stock_vols": sub_stock_vol,
        "estimated_CMC": estimated_CMC,
        "df": df,
    }

    small_exp = {
        "surfactant_sub_stock_vols": sub_stock_vol,
        "solvent_sub_vol": df["surfactant volume"].tolist(),
        "water_vol": df["water volume"].tolist(),
        "pyrene_vol": df["probe volume"].tolist(),
    }

    return exp, small_exp
