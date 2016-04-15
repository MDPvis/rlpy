"""
Experiments for a research paper.
"""
__author__ = "Sean McGregor"
from rlpy.Domains import MountainCar as domain_mountain_car
from rlpy.Domains import Stitching as domain_stitching
import numpy as np
import pickle

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

metricFiles = ["rlpy/Domains/StitchingPackage/metrics/mountaincar/optimized-2",
               "rlpy/Domains/StitchingPackage/metrics/mountaincar/optimized-1",
               "rlpy/Domains/StitchingPackage/metrics/mountaincar/normalized-2",
               "rlpy/Domains/StitchingPackage/metrics/mountaincar/normalized-1"]

def mountaincar_bar_chart_different_metrics(policyProbability, metricFiles=metricFiles):

    databaseSize = 50

    databasePolicies = []
    databasePolicies.append(mountaincar_factory(1.0))
    databasePolicies.append(mountaincar_factory(0.75))
    databasePolicies.append(mountaincar_factory(0.5))

    targetPolicies = []
    targetPolicies.append(mountaincar_factory(policyProbability))

    domain = domain_mountain_car(.1)

    # Create a stitching object
    stitching = domain_stitching(
      domain,
      rolloutCount = databaseSize,
      horizon = 200,
      databasePolicies = databasePolicies,
      targetPolicies = targetPolicies,
      targetPoliciesRolloutCount = 200,
      seed = 0,
      labels = ["x", "xdot"],
      metricFile = metricFiles[0], # not used since it will be repeatedly overwritten below
      optimizeMetric = False
    )

    print "optimized for two target policies,optimized for one target policy,normalized for target policy 1,normalized for target policy 2"

    metrics = []
    for path in metricFiles:
        f = open(path)
        met = stitching.mahalanobis_metric.flatten(stitching.mahalanobis_metric.distance_metric)
        metrics.append(stitching.mahalanobis_metric.ceiling_logarithm(met))
        f.close()

    # Sum over each benchmark and policy
    losses = [-1,-1,-1,-1]
    for idx, metric in enumerate(metrics):
        loss = stitching.mahalanobis_metric.loss(
            metric,
            stitching,
            stitching.mahalanobis_metric.benchmarks,
            benchmark_rollout_count=30
        )
        losses[idx] = loss
    print losses


def mountaincar_database_size(databaseSize, metricFiles=metricFiles):

    databasePolicies = []
    databasePolicies.append(mountaincar_factory(1.0))
    databasePolicies.append(mountaincar_factory(0.75))
    databasePolicies.append(mountaincar_factory(0.5))

    targetPolicies = []
    targetPolicies.append(mountaincar_factory(.9))

    domain = domain_mountain_car(.1)

    # Create a stitching object
    stitching = domain_stitching(
        domain,
        rolloutCount = databaseSize,
        horizon = 200,
        databasePolicies = databasePolicies,
        targetPolicies = targetPolicies,
        targetPoliciesRolloutCount = 200,
        seed = 0,
        labels = ["x", "xdot"],
        metricFile = metricFiles[0],
        optimizeMetric = False
    )

    print "optimized for two target policies,optimized for one target policy,normalized for target policy 1,normalized for target policy 2"

    metrics = []
    for path in metricFiles:
        f = open(path)
        met = stitching.mahalanobis_metric.flatten(stitching.mahalanobis_metric.distance_metric)
        metrics.append(stitching.mahalanobis_metric.ceiling_logarithm(met))
        f.close()

    # Sum over each benchmark and policy
    losses = [-1,-1,-1,-1]
    for idx, metric in enumerate(metrics):
        loss = stitching.mahalanobis_metric.loss(
            metric,
            stitching,
            stitching.mahalanobis_metric.benchmarks,
            benchmark_rollout_count=50
        )
        losses[idx] = loss
    print losses


def mountaincar_sans_replacement_bias(metricFiles=metricFiles):

    sampleCounts = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150]

    databasePolicies = []
    databasePolicies.append(mountaincar_factory(1.0))
    databasePolicies.append(mountaincar_factory(0.75))
    databasePolicies.append(mountaincar_factory(0.5))

    targetPolicies = []
    targetPolicies.append(mountaincar_factory(.9))

    domain = domain_mountain_car(.1)

    # Create a stitching object
    stitching = domain_stitching(
        domain,
        rolloutCount = 50,
        horizon = 200,
        databasePolicies = databasePolicies,
        targetPolicies = targetPolicies,
        targetPoliciesRolloutCount = 200,
        seed = 0,
        labels = ["x", "xdot"],
        metricFile = metricFiles[0],
        optimizeMetric = False
    )

    print "optimized for two target policies,optimized for one target policy,normalized for target policy 1,normalized for target policy 2"

    metrics = []
    for path in metricFiles:
        f = open(path)
        met = stitching.mahalanobis_metric.flatten(stitching.mahalanobis_metric.distance_metric)
        metrics.append(stitching.mahalanobis_metric.ceiling_logarithm(met))
        f.close()

    # Sum over each benchmark and policy
    for sampleCount in sampleCounts:
        losses = [-1,-1,-1,-1]
        for idx, metric in enumerate(metrics):
            loss = stitching.mahalanobis_metric.loss(
                metric,
                stitching,
                stitching.mahalanobis_metric.benchmarks,
                benchmark_rollout_count=sampleCount
            )
            losses[idx] = loss
        print losses


if __name__ == "__main__":
    targets = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.]
    print "mountaincar_bar_chart_different_metrics"
    for policyProbability in targets:
        mountaincar_bar_chart_different_metrics(policyProbability)

    ## Experiment not generated
    #databaseSizes = [20,25,30,35,40,45,50]
    #print "mountaincar_different_database_sizes"
    #for databaseSize in databaseSizes:
    #    mountaincar_database_size(databaseSize)

    print "mountaincar_bar_chart_different_sample_counts"
    mountaincar_sans_replacement_bias()