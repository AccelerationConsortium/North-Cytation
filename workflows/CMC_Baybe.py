# --- cmc_baybe_workflow.py ---

import sys
sys.path.append("../North-Cytation")
import pandas as pd
import numpy as np
from baybe.targets import NumericalTarget, TargetMode
from baybe.objectives import SingleTargetObjective
from baybe import Campaign
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.constraints import ContinuousCardinalityConstraint
from baybe.utils.random import set_random_seed
from baybe.constraints import ContinuousLinearConstraint


# --- Config ---
initial_batch_size = 1
random_seed = 42
CMC_target = 5.0 #What CMC are we trying to achieve?
CMC_tolerance = 0.5 #Set this later
surfactants = ['SDS', 'NaDC', 'NaC', 'CTAB', 'DTAB', 'TTAB', 'CAPB', 'CHAPS']

# --- Initialize BayBE ---
set_random_seed(random_seed)
target = NumericalTarget(name='CMC_difference', mode=TargetMode.MATCH, bounds=(0, CMC_tolerance))
objective = SingleTargetObjective(target=target)

parameters = [
    NumericalContinuousParameter(name=s, bounds=(0, 1)) for s in surfactants
]
constraints = [
    ContinuousCardinalityConstraint(parameters=[p.name for p in parameters], min_cardinality=2, max_cardinality=2),
    ContinuousLinearConstraint(
    parameters=[p.name for p in parameters],  # these parameters must exist in the search space
    operator="=",
    coefficients=[1.0]*8,
    rhs=1.0,
),
ContinuousLinearConstraint(
    parameters=["SDS"],
    operator=">=",
    coefficients=[1.0],
    rhs=0.001,  # Minimum concentration threshold to be considered "present"
)
]

searchspace = SearchSpace.from_product(parameters, constraints)
campaign = Campaign(searchspace, objective)

#Read in the data
df = pd.read_csv("../North-Cytation/analysis/selected_surfactant_combinations.csv")

#Fill in dummy data
df["CMC"] = np.random.uniform(0.5, 16, len(df))

# Create columns for each surfactant, filled with 0.0
for surf in surfactants:
    df[surf] = 0.0

# Fill in concentrations in the correct surfactant columns [This assumes that we have the form "Surfactant1, Ratio1, Surfactant2, Ratio2, CMC"]
for i, row in df.iterrows():
    df.at[i, row["Surfactant1"]] = row["Ratio1"]
    df.at[i, row["Surfactant2"]] = row["Ratio2"]

# Compute CMC_difference target for BayBE
df["CMC_difference"] = (df["CMC"] - CMC_target)

# Optional: keep only the required BayBE columns
baybe_input_df = df[surfactants + ["CMC_difference"]]

#Print your input data
print(baybe_input_df)

#Add your input data
campaign.add_measurements(df)

#Get initial recommendations
recommendations = campaign.recommend(batch_size=initial_batch_size)

#Print recs
print(recommendations)

