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

def featureSelectionFilename(features):
    """
    Each of the features will be recorded into the filename.

    :param features: The features we will want to incorporate into a filename
    :return:
    """
    filename =""
    for feature in features:
        if filename != "":
            filename += "-"
        feature.replace("_", " ").split("-")
        filename += feature.replace(" ", "_")
    return filename

def featureSelectionProcessUnpickled(unpickled):
    """
    Get the mean value of all the policies in the file.
    :param unpickled:
    :return:
    """
    total = 0.0
    for v in unpickled:
        total += v
    return total/float(len(unpickled))

def getNextFeatureToEvaluate(wildfireData):
    """
    Find the next list of features to evaluate based on what is found on the file system.
    :return:
    """
    # All the potential metrics
    bigList = wildfireData.ALL_STITCHING_VARIABLES

    # Get a list of the output files in the directory
    allFiles = os.listdir(configDict["feature selection results directory"])

    # Split each of the files by the metric names they contain
    currentFeatures = []
    for f in allFiles:
        cur = f.replace("_", " ").split("-")
        currentFeatures.append(cur)

    # Find the largest metric being worked on
    if len(currentFeatures) > 0:
        maxLength = max([max(len(k)) for k in currentFeatures])
        currentLayer = [x for x in currentFeatures if len(x) == maxLength]
        countInLayer = len(currentLayer)
    else:
        maxLength = len(bigList)
        currentLayer = []
        countInLayer = 0

    # Find the metric that hasn't been processed for this count,
    # or determine that the last layer is complete and adopt the best results,
    # or if it hasn't completed the last layer, just exit
    if (len(bigList) - maxLength) > countInLayer:
        # The next set of features to evaluate
        currentlyEvaluating = [x[-1] for x in currentLayer]
        nextList = currentLayer[0][:-1]
        nextFeature = (filter(lambda x: x not in nextList and x not in currentlyEvaluating, bigList))[0]
        nextList.append(nextFeature)
        filename = featureSelectionFilename(nextList)
        open(filename, "w") # "touch" the file
        return nextList
    elif (len(bigList) - maxLength) == countInLayer:
        bestValue = float("Inf")
        bestList = []
        for cur in currentLayer:
            try:
                pickledFileName = featureSelectionFilename(cur)
                pickledFile = open(pickledFileName, "r")
                unpickled = pickle.load(pickledFile)
                mean = featureSelectionProcessUnpickled(unpickled)
                if bestValue > mean:
                    bestValue = mean
                    bestList = cur
            except Exception:
                print "file did not exist {}".format()
                exit()
        nextList = bestList
        nextFeature = (filter(lambda x: x not in nextList, bigList))[0]
        nextList.append(nextFeature)
        filename = featureSelectionFilename(nextList)
        open(filename, "w") # "touch" the file
        return nextList
    else:
        print "strange tidings"
        exit(1)


def featureSelection(wildfireData, varianceDictionary):
    """
    Check the outputs folder for output visual fidelity error then conditionally evaluate the next potential
    additions to the metric.
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
                  outFile=None,
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

    while True:
        featureList = getNextFeatureToEvaluate(wildfireData)
        filename = featureSelectionFilename(featureList)
        print "evaluating {}".format(filename)
        outFile = file(filename, "wb")
        stitchingVariables = [x + " start" for x in featureList]
        benchmark(
            stitchingVariables,
            benchmarks=benchmarks,
            policies=policies,
            policyValues=policyValues,
            stitchingDomain=stitchingDomain,
            outFile=outFile,
            varianceDictionary=varianceDictionary
                  )
        outFile.close()
