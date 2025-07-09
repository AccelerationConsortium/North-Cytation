#Re-defining essentials after kernel reset

import numpy as np
import pandas as pd

surfactant_library = {
    "SDS": {
        "full_name": "Sodium Dodecyl Sulfate",
        "CAS": "151-21-3",
        "CMC": 8.5,
        "Category": "anionic",
        "MW": 289.39,
        "stock_conc": 50,  # mM
    },


    "NaDC": {
        "full_name": "Sodium Docusate",
        "CAS": "577-11-7",
        "CMC": 5.3375,
        "Category": "anionic",
        "MW": 445.57,
        "stock_conc": 25,  # mM
    },

    
    "NaC": {
        "full_name": "Sodium Cholate",
        "CAS": "361-09-1",
        "CMC": 14,
        "Category": "anionic",
        "MW": 431.56,
        "stock_conc": 50,  # mM
    },


    "CTAB": {
        "full_name": "Hexadecyltrimethylammonium Bromide",
        "CAS": "57-09-0",
        "CMC": 1.07,
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
        "CMC": 3.985,
        "Category": "cationic",
        "MW": 336.39,
        "stock_conc": 50,  # mM
    },

     "CAPB": {
        "full_name": "Cocamidopropyl Betaine",
        "CAS": "61789-40-0",
        "CMC": 0.627,
        "Category": "zwitterionic",
        "MW": 342.52,
        "stock_conc": 50,  # mM (Changed from 50)
    },
    
    "CHAPS": {
        "full_name": "CHAPS",
        "CAS": "75621-03-3",
        "CMC": 8,
        "Category": "zwitterionic",
        "MW": 614.88,
        "stock_conc": 30,  # mM
    }
}


def rough_generate_cmc_concentrations(list_of_surfactants, list_of_ratios, estimate_cmc, number_of_points=12 ,scale="log", max_conc_factor=0.8): # low/high in mM

    # surfactant_library = {
    #     "SDS": {"stock_conc": 50}, "NaDC": {"stock_conc": 25}, "NaC": {"stock_conc": 50},
    #     "CTAB": {"stock_conc": 5}, "DTAB": {"stock_conc": 50}, "TTAB": {"stock_conc": 50},
    #     "P188": {"stock_conc": 2}, "P407": {"stock_conc": 2}, "CAPB": {"stock_conc": 50}, "CHAPS": {"stock_conc": 30}
    # }

    max_conc = 0

    max_conc = get_max_total_concentration_joint_limit(
    list_of_surfactants,
    list_of_ratios,
    surfactant_library,
    CMC_sample_volume=1000,
    probe_volume=25
)

    low = estimate_cmc / (50**0.5)
    high = estimate_cmc * (50**0.5)

    if high > max_conc:
        print(f"Warning: High concentration {high} exceeds maximum stock concentration {max_conc}. Adjusting high to max stock concentration.")
        high = max_conc

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

def get_max_total_concentration_joint_limit(
    list_of_surfactants,
    list_of_ratios,
    surfactant_library,
    CMC_sample_volume=1000,
    probe_volume=25,
    safety_factor=0.95
):
    usable_volume = CMC_sample_volume - probe_volume

    def total_volume_required(total_conc):
        return sum(
            ((total_conc * ratio) * CMC_sample_volume) / surfactant_library[surf]['stock_conc']
            for surf, ratio in zip(list_of_surfactants, list_of_ratios)
        )

    # Binary search to find maximum total_conc such that total_volume_required ≤ usable_volume
    low = 0
    high = 1000  # arbitrarily high initial guess
    for _ in range(100):
        mid = (low + high) / 2
        if total_volume_required(mid) <= usable_volume:
            low = mid
        else:
            high = mid

    return low * safety_factor  # apply safety margin

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

        #surfactant_volume = round(surfactant_volume, 2)
        #water_volume = round(water_volume, 2)

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
    # surfactant_library = {
    #     "SDS": {"CMC": 8.3}, "NaDC": {"CMC": 8.2}, "NaC": {"CMC": 11},
    #     "CTAB": {"CMC": 0.93}, "DTAB": {"CMC": 15.85}, "TTAB": {"CMC": 3.77},
    #     "P188": {"CMC": 0.48}, "P407": {"CMC": 0.1}, "CAPB": {"CMC": 0.627}, "CHAPS": {"CMC": 8.5}
    # }
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


def scale_substock_to_required_volume(sub_stock_vol_dict, df, buffer_volume=500):
    """
    Scale sub-stock volumes (including water) to match required experimental volume + buffer.
    
    Args:
        sub_stock_vol_dict (dict): Raw volumes from sub-stock prep (includes surfactants + water).
        df (pd.DataFrame): Output from `calculate_volumes`, used to find how much is needed.
        buffer_volume (float): Extra volume to ensure pipetting margin (µL).
    
    Returns:
        scaled_volumes (dict): Same keys as input, but scaled down.
    """
    required_volume = df["surfactant volume"].sum()
    total_needed = required_volume + buffer_volume
    original_total = sum(sub_stock_vol_dict.values())

    scaling_factor = total_needed / original_total

    scaled_volumes = {k: round(v * scaling_factor, 2) for k, v in sub_stock_vol_dict.items()}
    
    print(f"\n Scaling sub-stock volumes to total {total_needed:.2f} µL (including {buffer_volume} µL buffer)")
    print(f"Original total: {original_total:.2f} µL | Scaling factor: {scaling_factor:.3f}")
    print("Scaled volumes:")
    for k, v in scaled_volumes.items():
        print(f"  {k}: {v:.2f} µL")

    return scaled_volumes


def verify_concentrations(df, sub_stock_concentration, probe_volume, CMC_sample_volume):
    dilution_volume = CMC_sample_volume - probe_volume
    df["actual_concentration"] = (sub_stock_concentration * df["surfactant volume"]) / dilution_volume
    df["difference"] = df["actual_concentration"] - df["concentration"]

    mismatches = df[np.abs(df["difference"]) > 0.01]
    if not mismatches.empty:
        print(" Warning: Some wells do not match intended concentrations:")
        print(mismatches[["concentration", "actual_concentration", "difference"]])
    else:
        print(" All concentrations match within ±0.01 mM")

    return df


def generate_exp_flexible(list_of_surfactants, list_of_ratios, probe_volume=25,
                          sub_stock_volume=6000, CMC_sample_volume=1000, rough_screen=False, estimated_CMC=None):

    # surfactant_library = {
    #     "SDS": {"stock_conc": 50}, "NaDC": {"stock_conc": 25}, "NaC": {"stock_conc": 50},
    #     "CTAB": {"stock_conc": 5}, "DTAB": {"stock_conc": 50}, "TTAB": {"stock_conc": 50},
    #     "P188": {"stock_conc": 2}, "P407": {"stock_conc": 2}, "CAPB": {"stock_conc": 50}, "CHAPS": {"stock_conc": 30}
    # }

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
        print("Concentrations (refined): ", cmc_concs)
    else:
        cmc_concs = rough_generate_cmc_concentrations(list_of_surfactants, list_of_ratios,  estimated_CMC)


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

    sub_stock_vol = scale_substock_to_required_volume(sub_stock_vol,df)

    #df = verify_concentrations(df, sub_stock_concentration, probe_volume, CMC_sample_volume)

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


# # Track results
# result = generate_exp_flexible(
#     list_of_surfactants=['NaC', 'SDS'],
#     list_of_ratios=[0.8, 0.2],
#     rough_screen=True  # or False
# )
# print(result)

# from itertools import combinations, product
# import traceback

# # Define the test surfactants and ratios
# test_surfactants = ['SDS', 'NaDC', 'NaC', 'CTAB', 'DTAB', 'TTAB', 'P188',"P407", 'CAPB', 'CHAPS'] 
# test_ratios = [
#     (0.5, 0.5),
#     (0.7, 0.3),
#     (0.3, 0.7),
#     (0.8, 0.2),
#     (0.2, 0.8)
# ]

# # Track results
# failures = []
# counts = 0
# # Try all 2-surfactant combinations and all ratio splits
# for (s1, s2) in combinations(test_surfactants, 2):
#     for (r1, r2) in test_ratios:
#         try:
#             exp, small = generate_exp_flexible(
#                 list_of_surfactants=[s1, s2],
#                 list_of_ratios=[r1, r2],
#                 rough_screen=True
#             )

#             df = exp["df"]

#             if exp['surfactant_sub_stock_vols']['water'] < 0:
#                 print(f"❌ NEGATIVE WATER: {s1} ({r1}) + {s2} ({r2})")
#                 failures.append((s1, s2, r1, r2, "Negative water"))
#             else:
#                 #print(f"✅ PASS: {s1} ({r1}) + {s2} ({r2})")
#                 None

#         except Exception as e:
#             print(f"❌ FAILED: {s1} ({r1}) + {s2} ({r2})")
#             print(traceback.format_exc(limit=1))
#             failures.append((s1, s2, r1, r2, str(e)))

#         counts+=1

# # Summary
# print("\n====== SUMMARY ======")
# if not failures:
#     print("✅ All tests passed.")
# else:
#     for f in failures:
#         print(f"❌ {f[0]} ({f[2]}) + {f[1]} ({f[3]}) → {f[4]}")
#     print("Num failures: ", len(failures))
#     print("Num successes: ", counts- len(failures))