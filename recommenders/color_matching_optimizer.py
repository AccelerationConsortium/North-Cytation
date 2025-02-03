import sys
sys.path.append(r"C:\Users\Imaging Controller\anaconda3\Lib\site-packages")
import baybe
from baybe.targets import NumericalTarget, TargetMode, TargetTransformation
from baybe.objectives import SingleTargetObjective
from baybe import Campaign
import pandas as pd
from baybe.parameters import NumericalDiscreteParameter, NumericalContinuousParameter
from baybe.searchspace import SearchSpace
import numpy as np
from baybe.constraints import DiscreteSumConstraint, ThresholdCondition
from baybe.utils.random import set_random_seed


def initialize_campaign(upper_bound, random_seed):
    set_random_seed(random_seed) 

    target = NumericalTarget(
        name = 'output',
        mode = TargetMode.MIN,
        bounds=(0, upper_bound),
    )

    objective = SingleTargetObjective(target=target)

    parameters = [
        NumericalDiscreteParameter(
            name = 'Water',
            values = np.array(range(0, 240, 10))
            #encoding = 'INT'
            ),
        NumericalDiscreteParameter(
            name = 'R',
            values = np.array(range(0, 240, 10)),
            #encoding = 'INT'
            ),
        NumericalDiscreteParameter(
            name = 'Y',
            values= np.array(range(0, 240, 10)),
            #encoding = 'INT'
            ),
        NumericalDiscreteParameter(
            name = 'B',
            values= np.array(range(0, 240, 10)),
            #encoding = 'INT'
            ),
    ]

    constraints = [DiscreteSumConstraint(
            parameters=["Water", "R", "Y", "B"],  # these parameters must exist in the search space
            condition=ThresholdCondition(  # set condition that should apply to the sum
            threshold=240,
            operator="="))]

    searchspace = SearchSpace.from_product(parameters=parameters, constraints=constraints)
    campaign = Campaign(searchspace, objective)
    return campaign

def get_initial_recommendations(campaign,size):
    initial_suggestions = campaign.recommend(batch_size=size)
    return campaign,initial_suggestions

def get_new_recs_from_results(campaign,data,size):
    campaign.add_measurements(data)
    new_suggestions = campaign.recommend(batch_size=size)
    return campaign,new_suggestions
