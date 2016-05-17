"""
Produce a dictionary and serialize it containing the variances of all the variables in the database.
Next, produce metrics using the variances and greadily eliminate variables to produce many metrics.
We perform this variable elimination on both the many-variable metric and the theoretically determined metric.
The metrics are labeled according to their variable count, ie 30, 29, 28, etc.
The landscape-only metrics are numbered |variables|*1000.
"""
__author__ = "Sean McGregor"
from rlpy.Domains import WildfireData
from rlpy.Domains import Stitching as Stitching
from experiments.wildfire_policy_functions import wildfirePolicySeverityFactory
import numpy as np
import os
import pickle
import rlpy.Domains.StitchingPackage.benchmark
import rlpy.Domains.StitchingPackage.MahalanobisDistance
import rlpy.Domains.Stitching
from experiments.configurations import clusterConfigurationDict as configDict

def computeVariances(wildfireData, outFilePath):
    """
    Writes a dictionary to a pickled file where:
    {variable: variance}
    :return:
    """
    variableNames = wildfireData.ALL_PRE_TRANSITION_STITCHING_VARIABLES
    varianceDictionary = {}

    for idx,variable in enumerate(variableNames):
        values = []
        for transition in wildfireData.database:
            values.append(transition.preStateDistanceMetricVariables[idx])
        variance = rlpy.Domains.StitchingPackage.benchmark.Benchmark.variance(values)
        varianceDictionary[variableNames[idx]] = variance
    f = open(outFilePath, "wb")
    pickle.dump(varianceDictionary, f)
    f.close()
    return varianceDictionary

def featureSelection(wildfireData, varianceDictionary, csvFilePath):
    """
    Pickle a series of distance metrics and write a CSV file containing each metric's mean visual fidelity and variance
    across all the policies. CSV: metric column list, visual fidelity mean, variance, variable count
    :param wildfireData: An instance of the WildfireData domain.
    :param varianceDictionary: A dictionary containing the variances for all the potential stitching variables.
    :return:
    """

    stitchingDomain = rlpy.Domains.Stitching(wildfireData,
                                             rolloutCount = 0,
                                             horizon = 99,
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

    outFile = file(csvFilePath, "wb")

    policies = []
    policyValues = []
    for targetPolicyERC in configDict["policy parameters ERC"]:
        for targetPolicyTime in configDict["policy parameters startIndex"]:
            policyValues.append([targetPolicyERC, targetPolicyTime])
            policies.append(wildfirePolicySeverityFactory(targetPolicyERC, targetPolicyTime))

    benchmarks = []
    for policyValue in policyValues:
        base_rollouts = wildfireData.getTargetRollouts(policyValue[0], policyValue[1])
        current = rlpy.Domains.StitchingPackage.benchmark.Benchmark(base_rollouts, 2, benchmarkActions=False)
        benchmarks.append(current)

    def benchmark(stitchingVariables,
                  benchmarks=benchmarks,
                  policies=policies,
                  policyValues=policyValues,
                  stitchingDomain=stitchingDomain,
                  outFile=outFile,
                  varianceDictionary=varianceDictionary):
        """
        Benchmark the current set of stitching variables.
        :param stitchingVariables: ["variable 1 name", "variable 2 name", ...]
        :return:
        """
        benchmarkSample = []

        # Update all the transition tuples in the database
        wildfireData.populateDatabase(
            stitchingVariables=stitchingVariables,
            visualizationVariables=wildfireData.VISUALIZATION_VARIABLES)
        for idx,policyValue in enumerate(policyValues):
            db = wildfireData.getDatabaseWithoutTargetSeverityPolicy(policyValue[0], policyValue[1])
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
            stitchingDomain.setDatabase(db)
            rollouts = stitchingDomain.getRollouts(
                count=30,
                horizon=99,
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
            outFile.write(variable + "|")
        outFile.write(",")
        outFile.write("{},{},{}\n".format(mean, variance, len(stitchingVariables)))
        return mean

    def minStep(stitchingVariables):
        """
        Find the minimal benchmark in the next step (removing a feature)).
        :param stitchingVariables:
        :return:
        """
        minBench = float("inf")
        for idx in range(0, len(stitchingVariables)):
            potentialDistanceMetricVariables = stitchingVariables[0:idx] + stitchingVariables[idx+1:len(stitchingVariables)]
            bench = benchmark(potentialDistanceMetricVariables)
            if bench < minBench:
                minBench = bench
                ret = stitchingVariables[0:idx] + stitchingVariables[idx+1:len(currentDistanceMetricVariables)]
        return ret

    def minStepAdd(currentStitchingVariables, potentialStitchingVariables):
        """
        Find the minimal benchmark while adding variables.
        :param: currentStitchingVariables the variables that will always be used
        :param: potentialStitchingVariables the variables we can add
        """
        minBench = float("inf")
        for idx in range(0, len(potentialStitchingVariables)):
            cur = currentStitchingVariables + potentialStitchingVariables[idx]
            bench = benchmark(cur)
            if bench < minBench:
                minBench = bench
                ret = cur
        return ret

    ## REMOVING VARIABLES ##

    currentDistanceMetricVariables = wildfireData.ALL_PRE_TRANSITION_STITCHING_VARIABLES
    nextDistanceMetricVariables = currentDistanceMetricVariables
    benchmark(nextDistanceMetricVariables)
    while len(nextDistanceMetricVariables) > 1:
        nextDistanceMetricVariables = minStep(nextDistanceMetricVariables)

    currentDistanceMetricVariables = wildfireData.BEST_PRE_TRANSITION_STITCHING_VARIABLES
    nextDistanceMetricVariables = currentDistanceMetricVariables
    benchmark(nextDistanceMetricVariables)
    while len(nextDistanceMetricVariables) > 1:
        nextDistanceMetricVariables = minStep(nextDistanceMetricVariables)

    currentDistanceMetricVariables = ["Precipitation start",
        "MaxTemperature start",
        "MinHumidity start",
        "WindSpeed start",
        "ignitionCovertype start",
        "ignitionSlope start",
        "startIndex start",
        "endIndex start",
        "ERC start",
        "SC start"]
    nextDistanceMetricVariables = currentDistanceMetricVariables
    benchmark(nextDistanceMetricVariables)
    while len(nextDistanceMetricVariables) > 1:
        nextDistanceMetricVariables = minStep(nextDistanceMetricVariables)

    ## ADDING VARIABLES ##
    currentDistanceMetricVariables = []
    nextDistanceMetricVariables = []
    benchmark(nextDistanceMetricVariables)
    while len(nextDistanceMetricVariables) > 1:
        potentials = [x for x in wildfireData.ALL_PRE_TRANSITION_STITCHING_VARIABLES if x not in nextDistanceMetric]
        nextDistanceMetricVariables = minStepAdd(nextDistanceMetricVariables, potentials)

    outFile.close()
