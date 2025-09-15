import sys
sys.path.append(r"C:\Users\Imaging Controller\anaconda3\Lib\site-packages")
from baybe.targets import NumericalTarget, TargetMode
from baybe.objectives import SingleTargetObjective
from baybe import Campaign
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.utils.random import set_random_seed
from baybe.recommenders import RandomRecommender
from baybe.constraints import ContinuousCardinalityConstraint

def initialize_campaign(bound, random_seed, random_recs=False):
    set_random_seed(random_seed) 

    target = NumericalTarget(
        name = 'CMC_difference',
        mode = TargetMode.MATCH,
        bounds=(-bound, bound), #The bound depends on how we define the target
    )

    objective = SingleTargetObjective(target=target)

    parameters = [
        NumericalContinuousParameter( #TODO, redefine these bounds
            name = 'SDS',
            bounds=(0.1, 20),
            ),
        NumericalContinuousParameter(
            name = 'NaDC',
            bounds=(0.1, 20),
            ),
        NumericalContinuousParameter(
            name = 'NaC',
            bounds=(0.1, 20),
            ),
        NumericalContinuousParameter(
            name = 'CTAB',
            bounds=(0.1, 20),
            ),
        NumericalContinuousParameter(
            name = 'DTAB',
            bounds=(0.1, 20),
            ),
        NumericalContinuousParameter(
            name = 'TTAB',
            bounds=(0.1, 20),
            ),
        NumericalContinuousParameter(
            name = 'CAPB',
            bounds=(0.1, 20),
            ),
        NumericalContinuousParameter(
            name = 'CHAPS',
            bounds=(0.1, 20),
            ),
    ]

    constraints = [ContinuousCardinalityConstraint(
    parameters=['SDS', 'NaDC', 'NaC', 'CTAB','DTAB','TTAB','CAPB','CHAPS'],
    min_cardinality=2, 
    max_cardinality=2,  
    ) ]

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
