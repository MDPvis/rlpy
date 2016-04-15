"""
Experiments for a research paper.
"""
__author__ = "Sean McGregor"
from rlpy.Domains import MountainCar as domain_mountain_car
from rlpy.Domains import Stitching as domain_stitching
import numpy as np

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

def mountaincar_paper_learn_metric(targetPolicies):

    databasePolicies = []
    databasePolicies.append(mountaincar_factory(1.0))
    databasePolicies.append(mountaincar_factory(0.75))
    databasePolicies.append(mountaincar_factory(0.5))

    normalizedMetricFile = "rlpy/Domains/StitchingPackage/metrics/mountaincar/normalized-{}".format(len(targetPolicies))
    optimizedMetricFile = "rlpy/Domains/StitchingPackage/metrics/mountaincar/optimized-{}".format(len(targetPolicies))

    domain = domain_mountain_car(.1)

    # Create a stitching object
    stitching = domain_stitching(
      domain,
      rolloutCount = 50,
      horizon = 50,
      databasePolicies = databasePolicies,
      targetPolicies = targetPolicies,
      targetPoliciesRolloutCount = 200,
      seed = 0,
      metricFile = optimizedMetricFile,
      labels = ["x", "xdot"],
      optimizeMetric = True,
      writeNormalizedMetric = normalizedMetricFile
    )

if __name__ == "__main__":
    targetPolicies = []
    targetPolicies.append(mountaincar_factory(.6))
    mountaincar_paper_learn_metric(targetPolicies)
    targetPolicies.append(mountaincar_factory(.9))
    mountaincar_paper_learn_metric(targetPolicies)
