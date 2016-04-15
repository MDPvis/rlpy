"""
Experiments for a research paper.
"""
__author__ = "Sean McGregor"
from rlpy.Domains import HIVTreatment as domain_hiv
from rlpy.Domains import Stitching as domain_stitching
import numpy as np
import os
import random

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

def hiv_paper_learn_metric(metricFile):

    databaseProbabilities = [0, .25, 1]
    #targetProbabilities = [.05, .2, .95]
    targetProbabilities = [.2] # metric 200
    #targetProbabilities = [.7] # metric 300

    databasePolicies = []
    for prob1 in databaseProbabilities:
        for prob2 in databaseProbabilities:
            databasePolicies.append(hiv_factory(prob1, prob2))

    targetPolicies = []
    for prob1 in targetProbabilities:
        for prob2 in targetProbabilities:
            targetPolicies.append(hiv_factory(prob1, prob2))


    domain = domain_hiv()

    # Create a stitching object
    stitching = domain_stitching(
      domain,
      rolloutCount = 50,
      horizon = 50,
      databasePolicies = databasePolicies,
      targetPolicies = targetPolicies,
      targetPoliciesRolloutCount = 200,
      stitchingToleranceSingle = .1,
      stitchingToleranceCumulative = .1,
      seed = 0,
      labels = ["t1", "t1infected", "t2", "t2infected", "v", "e"],
      optimizeMetric = True,
      metricFile = metricFile,
      writeNormalizedMetric = "rlpy/Domains/StitchingPackage/metrics/hiv/600"
    )

if __name__ == "__main__":
    hiv_paper_learn_metric(
        #"rlpy/Domains/StitchingPackage/metrics/hiv/200",
        "rlpy/Domains/StitchingPackage/metrics/hiv/500"
    )
