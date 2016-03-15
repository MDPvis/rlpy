"""
Experiments for a research paper.
"""
__author__ = "Sean McGregor"
from rlpy.Domains import MountainCar as domain_mountain_car
from rlpy.Domains import Stitching as domain_stitching
import numpy as np
import os
import random

def mountaincar_factory(reinforce_threshold):
    rs = np.random.RandomState(0)
    def policy_reinforce(state, possibleActions, reinforce_threshold=reinforce_threshold, rs=rs):
        uni = rs.uniform(0,1)
        reinforce = uni < reinforce_threshold
        x = state[0]
        xdot = state[1]
        if not reinforce:
            return rs.choice(possibleActions)
        elif xdot > 0.0:
            return 2 # Right
        elif xdot == 0.0:
            return rs.choice(possibleActions)
        else:
            return 0 # Left
    return policy_reinforce

def mountaincar_paper_learn_metric(metricFile):

    databasePolicies = []
    databasePolicies.append(mountaincar_factory(1.0))
    databasePolicies.append(mountaincar_factory(0.75))
    databasePolicies.append(mountaincar_factory(0.5))

    targetPolicies = []
    targetPolicies.append(mountaincar_factory(.6))
    targetPolicies.append(mountaincar_factory(.9))

    domain = domain_mountain_car(.1)

    # Create a stitching object
    stitching = domain_stitching(
      domain,
      rolloutCount = 200,
      horizon = 50,
      databasePolicies = databasePolicies,
      targetPolicies = targetPolicies,
      targetPoliciesRolloutCount = 200,
      stitchingToleranceSingle = .1,
      stitchingToleranceCumulative = .1,
      seed = 0,
      metricFile = metricFile,
      labels = ["x", "xdot"],
      optimizeMetric = True
    )

if __name__ == "__main__":
    mountaincar_paper_learn_metric(
        "rlpy/Domains/StitchingPackage/metrics/mountaincar/100",
    )
