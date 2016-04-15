"""
Experiments for a research paper.
"""

import timeit

setup = """

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

database_rollouts = 30
database_horizon = 100
#database_rollouts = 10
#database_horizon = 10

targetPoliciesRolloutCount = 2

database_cache_name = "{}-{}".format(database_rollouts, database_horizon)
database_cache_path = "rlpy/Domains/StitchingPackage/databases/hiv/" + database_cache_name
save_database = True
stitching_database_for_optimized = None
if os.path.isfile(database_cache_path):
    print "loading database from pickled file"
    f = open(database_cache_path, 'r')
    stitching_database_for_optimized = pickle.load(f)
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
"""
t = timeit.Timer('stitching_optimized.getRollouts(policy=hiv_factory(.6,.6), horizon=100, count=30)', setup=setup)
res = t.repeat(repeat=2, number=5)
print res
print min(res)