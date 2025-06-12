import sys
sys.path.append("../utoronto_demo")
import recommenders.CMC_optimizer as recommender

#List of input parameters for the experiment
random_recs = False 
seed = 27
bound = 2
starting_samples = 6

# #Get initial recs
campaign,searchspace = recommender.initialize_campaign(bound,seed,random_recs=random_recs) 
campaign,recommendations = recommender.get_initial_recommendations(campaign,starting_samples)
print("Initial Recommendations: ", recommendations)
print(f"Random: {random_recs}, seed #: {seed}")
#print("Searchspace Size: ", searchspace)







