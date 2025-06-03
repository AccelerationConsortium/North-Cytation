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
from baybe.recommenders import RandomRecommender

def initialize_campaign(upper_bound, random_seed, random_recs=False):
    set_random_seed(random_seed) 

    target = NumericalTarget(
        name = 'output',
        mode = TargetMode.MAX,
        bounds=(0, upper_bound),
    )

    objective = SingleTargetObjective(target=target)

    parameters = [
        NumericalDiscreteParameter(
            name = 'Water',
            values = np.array(range(0, 140, 20))
            #encoding = 'INT'
            ),
        NumericalDiscreteParameter(
            name = 'PVA_1',
            values = np.array(range(0, 100, 20))
            #encoding = 'INT'
            ),
        NumericalDiscreteParameter(
            name = 'PVA_2',
            values= np.array(range(0, 100, 20)),
            #encoding = 'INT'
            ),
        NumericalDiscreteParameter(
            name = 'PVA_3',
            values= np.array(range(0, 100, 20)),
            #encoding = 'INT'
            ),
        NumericalDiscreteParameter(
            name = 'HCl',
            values= np.array(range(0, 60, 20)),
            #encoding = 'INT'
            ),
        NumericalDiscreteParameter(
            name = 'Acid_2',
            values= np.array(range(0, 60, 20)),
            #encoding = 'INT'
            ),
        NumericalDiscreteParameter(
            name = 'Acid_3',
            values= np.array(range(0, 60, 20)),
            #encoding = 'INT'
            ),
        NumericalDiscreteParameter(
            name = 'NaOH',
            values= np.array(range(0, 60, 20)),
            #encoding = 'INT'
            ),    
    ]

    constraints = [DiscreteSumConstraint(
            parameters=["Water", "PVA_1", "PVA_2", "PVA_3", "HCl", "Acid_2", "Acid_3", "NaOH"],  # these parameters must exist in the search space
            condition=ThresholdCondition(  # set condition that should apply to the sum
            threshold=120,
            operator="="))]

    searchspace = SearchSpace.from_product(parameters=parameters, constraints=constraints)
    
    if not random_recs:
        campaign = Campaign(searchspace, objective)
    else:
        recommender = RandomRecommender()
        campaign = Campaign(searchspace, objective, recommender)
    return campaign,searchspace

def get_initial_recommendations(campaign,size):
    initial_suggestions = campaign.recommend(batch_size=size)
    return campaign,initial_suggestions

def get_new_recs_from_results(campaign,data,size):
    campaign.add_measurements(data)
    new_suggestions = campaign.recommend(batch_size=size)
    return campaign,new_suggestions
