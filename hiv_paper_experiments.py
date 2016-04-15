"""
Experiments for a research paper.
"""
__author__ = "Sean McGregor"
from rlpy.Domains import HIVTreatment as domain_hiv
from rlpy.Domains import Stitching as domain_stitching
import numpy as np
import os
import pickle

def hiv_factory(rti_probability, pi_probability):
    rs = np.random.RandomState(0)
    def policy_reinforce(state, possibleActions, rti_probability=rti_probability, pi_probability=pi_probability, rs=rs):
        uni = rs.uniform(0,1)
        rti = uni < rti_probability
        uni = rs.uniform(0,1)
        pi = uni < pi_probability
        t1 = state[0] # non-infected CD4+ T-lymphocytes [cells / ml]
        t1infected = state[1] # infected CD4+ T-lymphocytes [cells / ml]
        t2 = state[2] # non-infected macrophages [cells / ml]
        t2infected = state[3] # infected macrophages [cells / ml]
        v = state[4] # number of free HI viruses [copies / ml]
        e = state[5] # number of cytotoxic T-lymphocytes [cells / ml]

        spiking = (t1infected > 100 or v > 20000)

        if spiking:
            rti = True
            pi = True

        # Actions
        # *0*: none active
        # *1*: RTI active
        # *2*: PI active
        # *3*: RTI and PI active

        if rti and pi:
            return 3
        elif rti:
            return 1
        elif pi:
            return 2
        else:
            return 0
    return policy_reinforce

databaseProbabilities = [0, .25, 1]
databasePolicies = []
for prob1 in databaseProbabilities:
    for prob2 in databaseProbabilities:
        databasePolicies.append(hiv_factory(prob1, prob2))

# Policies used to normalize the metric in the non-optimized case
targetPolicies = []
targetPolicies.append(hiv_factory(.2, .2))

database_rollouts = 200
database_horizon = 100
targetPoliciesRolloutCount = 200

#database_rollouts = 50
#database_horizon = 5
#targetPoliciesRolloutCount = 50

database_cache_name = "{}-{}".format(database_rollouts, database_horizon)
database_cache_path = "rlpy/Domains/StitchingPackage/databases/hiv/" + database_cache_name
save_database = True
stitching_database_for_optimized = None
stitching_database_for_normalized = None
if os.path.isfile(database_cache_path):
    f = open(database_cache_path, 'r')
    stitching_database_for_optimized = pickle.load(f)
    f.close()
    f = open(database_cache_path, 'r')
    stitching_database_for_normalized = pickle.load(f)
    f.close()
    print "loaded database from pickled file"
    save_database = False
else:
    print "no database fount at {}, generating database...".format(database_cache_path)

# Create a stitching object
optimized_domain = domain_hiv()
stitching_optimized = domain_stitching(
    optimized_domain,
    rolloutCount = database_rollouts,
    horizon = database_horizon,
    databasePolicies = databasePolicies,
    targetPolicies = targetPolicies,
    targetPoliciesRolloutCount = targetPoliciesRolloutCount,
    seed = 0,
    labels = ["t1", "t1infected", "t2", "t2infected", "v", "e"],
    optimizeMetric = False,
    metricFile = "rlpy/Domains/StitchingPackage/metrics/hiv/300",
    database = stitching_database_for_optimized
)

if save_database:
    print "saving database to: " + database_cache_path
    f = open(database_cache_path, 'w')
    pickle.dump(stitching_normalized.database, f)
    print "saved database, re-run script to get results"
    exit()

normalized_domain = domain_hiv()
stitching_normalized = domain_stitching(
    normalized_domain,
    rolloutCount = database_rollouts,
    horizon = database_horizon,
    database = stitching_database_for_normalized,
    databasePolicies = databasePolicies,
    targetPolicies = targetPolicies,
    targetPoliciesRolloutCount = targetPoliciesRolloutCount,
    seed = 0,
    labels = ["t1", "t1infected", "t2", "t2infected", "v", "e"],
    optimizeMetric = False,
    metricFile = None
)

def hiv_paper_graph(normalized=True, targetProb1=None, targetProb2=None):

    global stitching_normalized
    global stitching_optimized

    stitching = None
    if normalized:
        stitching = stitching_normalized
    else:
        stitching = stitching_optimized

    targetPolicies = []
    targetPolicies.append(hiv_factory(targetProb1, targetProb2))
    stitching.targetPolicies = targetPolicies
    stitching.mahalanobis_metric._sampleTargetTrajectories()

    flat_metric = stitching.mahalanobis_metric.flatten(stitching.mahalanobis_metric.distance_metric)

    # Sum over each benchmark and policy
    loss = stitching.mahalanobis_metric.loss(
        stitching.mahalanobis_metric.ceiling_logarithm(flat_metric),
        stitching,
        stitching.mahalanobis_metric.benchmarks,
        benchmark_rollout_count=50
    )

    print "{},{},{}".format(targetProb1, targetProb2, loss)
    return loss

if __name__ == "__main__":
    if False:
        """
        Heat map data
        """
        targetProbabilities = [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
        #targetProbabilities = [.0]
        resultsOptimized = len(targetProbabilities)*[None]
        resultsNormalized = len(targetProbabilities)*[None]
        for idx1, prob1 in enumerate(targetProbabilities):
            resultsOptimized[idx1] = len(targetProbabilities)*[None]
            resultsNormalized[idx1] = len(targetProbabilities)*[None]
            for idx2, prob2 in enumerate(targetProbabilities):
                print "Finding {},{}...".format(prob1, prob2)
                loss = hiv_paper_graph(
                    normalized=False,
                    targetProb1=prob1,
                    targetProb2=prob2)
                resultsOptimized[idx1][idx2] = loss
                loss = hiv_paper_graph(
                    normalized=True,
                    targetProb1=prob1,
                    targetProb2=prob2)
                resultsNormalized[idx1][idx2] = loss
        print resultsOptimized
        print resultsNormalized
