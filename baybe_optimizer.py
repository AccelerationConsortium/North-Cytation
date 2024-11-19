import baybe
from baybe.targets import NumericalTarget
from baybe.objectives import SingleTargetObjective
from baybe import Campaign
import pandas as pd
from baybe.parameters import NumericalDiscreteParameter, NumericalContinuousParameter
from baybe.searchspace import SearchSpace
import numpy as np
from baybe.constraints import DiscreteSumConstraint, ThresholdCondition

def dummy_function(x, y, z):
    return pow(x, 2) - 20*y + pow(z, 0.15)

target = NumericalTarget(
    name = 'output',
    mode = 'MAX'
)

objective = SingleTargetObjective(target=target)

parameters = [
    NumericalDiscreteParameter(
        name = 'R',
        values = np.linspace(0,200,41),
        #encoding = 'INT'
        ),
    NumericalDiscreteParameter(
        name = 'G',
        values = np.linspace(0,200,41),
        #encoding = 'INT'
        ),
    NumericalDiscreteParameter(
        name = 'B',
        values= np.linspace(0,200,41),
        #encoding = 'INT'
        ),
]

constraints = [DiscreteSumConstraint(
        parameters=["R", "G", "B"],  # these parameters must exist in the search space
        condition=ThresholdCondition(  # set condition that should apply to the sum
        threshold=200,
        operator="="))]

searchspace = SearchSpace.from_product(parameters=parameters, constraints=constraints)

campaign = Campaign(searchspace, objective)

df = campaign.recommend(batch_size=10)
#Get your initial recs

#Add some outputs
df['output'] = 0.0
for i in range(len(df)):
    df.iloc[i, 3] = dummy_function(df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2])

print("Initial Recs!")
print(df)

campaign.add_measurements(df)

new_df = campaign.recommend(batch_size=5)

print("Recommended!")
print(new_df)

#Add some outputs
new_df['output'] = 0.0
for i in range(len(new_df)):
    new_df.iloc[i, 3] = dummy_function(new_df.iloc[i, 0], new_df.iloc[i, 1], new_df.iloc[i, 2])

print("New additions!")
print(new_df)

'''
Questions:
1) How do we explore vs exploit?
'''