"""
This experiment examines how the bias correction affects visual fidelity. We generate results for two
databases, one where we subsample the database trajectories to be half the original size, and another
database in which we remove all the bias correction state transitions.
Record the sum of the stitching distance for each policy and the resulting visual fidelity.
"""
__author__ = "Sean McGregor"
from rlpy.Domains import WildfireData
from experiments.wildfire_policy_functions import wildfirePolicySeverityFactory
import numpy as np
import os
import pickle
import rlpy.Domains.StitchingPackage.benchmark
import rlpy.Domains.StitchingPackage.MahalanobisDistance
import rlpy.Domains.Stitching


def visualFidelityError(
        wildfireData,
        varianceDictionary,
        stitchingDomain,
        outCSVFile,
        benchmarks,
        stitchingVariables,
        policyValues,
        policies,
        horizon=100,
        sampleCount=50):
    """
    :return:
    """

    benchmarkSampleHalved = []
    benchmarkSampleBiased = []
    for idx,policyValue in enumerate(policyValues):
        var_count = len(stitchingVariables)
        mahaMetric = rlpy.Domains.StitchingPackage.MahalanobisDistance.MahalanobisDistance(var_count,
                                                                                           stitchingDomain,
                                                                                           target_policies=[],
                                                                                           normalize_starting_metric=False,
                                                                                           cached_metric=None)
        inverseVariances = []
        for stitchingVariable in stitchingVariables:
            cur = varianceDictionary[stitchingVariable]
            assert cur >= 0
            if cur == 0:
                inverseVariances.append(0.0)
            else:
                inverseVariances.append(1.0/float(cur))
        mahaMetric.updateInverseVariance(inverseVariances)

        stitchingDomain.setMetric(mahaMetric)

        db = wildfireData.getDatabaseWithoutTargetSeverityPolicy(policyValue[0], policyValue[1])

        dbHalved =[]
        for transition in db:
            if transition.additionalState["initial event"] % 2 == 0:
                dbHalved.append(transition)

        dbBiased =[]
        for transition in db:
            if transition.additionalState["on policy"] == 1:
                dbBiased.append(transition)

        stitchingDomain.setDatabase(dbHalved)
        rolloutsHalved = stitchingDomain.getRollouts(
            count=sampleCount,
            horizon=horizon,
            policy=policies[idx],
            domain=None,
            biasCorrected=True,
            actionsInDistanceMetric=False)
        total = 0
        for variable in stitchingDomain.domain.VISUALIZATION_VARIABLES:
            total += benchmarks[idx].benchmark_variable(rolloutsHalved, variable)
        benchmarkSampleHalved.append(total)

        stitchingDomain.setDatabase(dbBiased)
        rolloutsBiased = stitchingDomain.getRollouts(
            count=sampleCount,
            horizon=horizon,
            policy=policies[idx],
            domain=None,
            biasCorrected=False,
            actionsInDistanceMetric=False)
        total = 0
        for variable in stitchingDomain.domain.VISUALIZATION_VARIABLES:
            total += benchmarks[idx].benchmark_variable(rolloutsBiased, variable)
        benchmarkSampleBiased.append(total)
    varianceBiased = rlpy.Domains.StitchingPackage.benchmark.Benchmark.variance(benchmarkSampleBiased)
    varianceHalved = rlpy.Domains.StitchingPackage.benchmark.Benchmark.variance(benchmarkSampleHalved)
    meanBiased = sum(benchmarkSampleBiased)/float(len(benchmarkSampleBiased))
    meanHalved = sum(benchmarkSampleHalved)/float(len(benchmarkSampleHalved))

    outCSVFile.write("biased mean,halved mean,biased variance,halved variance\n")
    outCSVFile.write("{},{},{}\n".format(meanBiased, meanHalved, varianceBiased, varianceHalved))

if __name__ == "__main__":
    assert False # todo: change to the proper paths
    databaseCSVPath = "synthesis_tests/wildfire_data/wildfire_processed.csv"
    wildfireData = WildfireData(databaseCSVPath)
    stitchingVariables = [
        "Fuel Model start",
        "Canopy Closure start",
        "Canopy Height start",
        "Canopy Base Height start",
        "Canopy Bulk Density start",
        "Covertype start",
        "Stand Density Index start",
        "Succession Class start",
        "Maximum Time in State start",
        "Stand Volume Age start"
    ]

    # Update all the transition tuples in the database
    wildfireData.populateDatabase(
        stitchingVariables=stitchingVariables,
        visualizationVariables=wildfireData.VISUALIZATION_VARIABLES)


    inputVariancesPath = "synthesis_tests/wildfire_data/wildfire_variances.pkl"
    inputVariances = file(inputVariancesPath, "rb")
    varianceDictionary = pickle.load(inputVariances)
    outputCSVFilePath = "synthesis_tests/wildfire_data/sample_size_bias.csv"
    outCSVFile = file(outputCSVFilePath, "wb")

    stitchingDomain = rlpy.Domains.Stitching(wildfireData,
                                             rolloutCount = 0,
                                             horizon = 100,
                                             databasePolicies = [],
                                             targetPolicies = [],
                                             targetPoliciesRolloutCount = 0,
                                             stitchingToleranceSingle = .1,
                                             stitchingToleranceCumulative = .1,
                                             seed = None,
                                             database = None,
                                             labels = None,
                                             metricFile = None,
                                             optimizeMetric = False,
                                             writeNormalizedMetric = None,
                                             initializeMetric = False)

    policies = []
    policyValues = []
    # todo: when doing the actual experiments, these values should be associated with the sampled policies:
    for targetPolicyERC in databasePolicyParameters["ercThreshold"]:
        for targetPolicyTime in databasePolicyParameters["timeUntilEndOfFireSeasonThreshold"]:
            policyValues.append([targetPolicyERC, targetPolicyTime])
            policies.append(wildfirePolicySeverityFactory(targetPolicyERC, targetPolicyTime))

    benchmarks = []
    for policyValue in policyValues:
        base_rollouts = wildfireData.getTargetRollouts(policyValue[0], policyValue[1])
        current = rlpy.Domains.StitchingPackage.benchmark.Benchmark(base_rollouts, 2, benchmarkActions=False)
        benchmarks.append(current)

    visualFidelityError(
        wildfireData,
        varianceDictionary,
        stitchingDomain,
        outCSVFile,
        benchmarks,
        stitchingVariables,
        policyValues,
        policies)
