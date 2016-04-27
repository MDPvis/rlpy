"""
Find the decline in visual fidelity for increasing sample sizes. Produce a scatterplot with error bars across
all policies
"""
__author__ = "Sean McGregor"
from rlpy.Domains import WildfireData
from rlpy.Domains import Stitching
from experiments.wildfire_policy_functions import wildfirePolicySeverityFactory
import numpy as np
import os
import pickle
import rlpy.Domains.StitchingPackage.benchmark
import rlpy.Domains.StitchingPackage.MahalanobisDistance
import rlpy.Domains.Stitching



def visualFidelityError(
        sampleCount,
        wildfireData,
        varianceDictionary,
        stitchingDomain,
        outCSVFile,
        benchmarks,
        stitchingVariables,
        policyValues,
        policies,
        horizon=100):
    """
    :return:
    """
    for idx,policyValue in enumerate(policyValues):
        var_count = len(stitchingVariables)
        mahaMetric = rlpy.Domains.StitchingPackage.MahalanobisDistance.MahalanobisDistance(var_count,
                                                                                           stitchingDomain,
                                                                                           target_policies=[],
                                                                                           normalize_starting_metric=False,
                                                                                           cached_metric=None)
        benchmarkSample = []

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
        stitchingDomain.setDatabase(db)
        rollouts = stitchingDomain.getRollouts(
            count=sampleCount,
            horizon=horizon,
            policy=policies[idx],
            domain=None,
            biasCorrected=True,
            actionsInDistanceMetric=False)
        total = 0
        for variable in stitchingDomain.domain.VISUALIZATION_VARIABLES:
            total += benchmarks[idx].benchmark_variable(rollouts, variable)
        benchmarkSample.append(total)
    variance = rlpy.Domains.StitchingPackage.benchmark.Benchmark.variance(benchmarkSample)
    mean = sum(benchmarkSample)/float(len(benchmarkSample))

    for variable in stitchingVariables:
        outCSVFile.write(variable + "|")
    outCSVFile.write(",")
    outCSVFile.write("{},{},{}\n".format(mean, variance, len(stitchingVariables)))
    return mean


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

    for sampleCount in [10,20,30,40,50,60,70,80,90,100,
                        110,120,130,140,150,160,170,180,
                        190,200]:
        visualFidelityError(
            sampleCount,
            wildfireData,
            varianceDictionary,
            stitchingDomain,
            outCSVFile,
            benchmarks,
            stitchingVariables,
            policyValues,
            policies)
