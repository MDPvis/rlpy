import logging
import os
import csv
import nose.tools
import experiments.wildfire_policy_functions
import experiments.produce_metrics
import experiments.regression
import experiments.sample_size_bias
import experiments.structural_bias
import experiments.other_policies
import rlpy.Domains.StitchingPackage.benchmark
from rlpy.Domains.WildfireData import WildfireData
import pickle

# These variables will be used to benchmark each of the experiments
VISUALIZATION_VARIABLES = WildfireData.VISUALIZATION_VARIABLES

# These variables will be used to stitch between states in each of the experiments
STITCHING_VARIABLES = WildfireData.BEST_STITCHING_VARIABLES

def test_post_process_data():
    """
    Create the CSV file for MFMCi based off of a results file and the associated landscapes.
    :return:
    """
    WildfireData.postProcessData(
        "/nfs/stak/students/m/mcgregse/Projects/rlpy/raw.csv",
        "/nfs/stak/students/m/mcgregse/Projects/rlpy/processed.csv",
        DISTANCE_METRIC_VARIABLES=STITCHING_VARIABLES
    )

def test_wildfire_produce_metrics_variances():
    """
    Find the variances of all the variables in the post-processed database. This will enable
    the creation of distance metrics using any of the database variables.
    :return:
    """
    databaseCSVPath = "experiments/data/processed_database.csv"
    wildfireData = WildfireData(databaseCSVPath)
    outputVariancesPath = "experiments/data/processed_database_variances.pkl"
    if os.path.isfile(outputVariancesPath):
        assert False, "The processed_database_variances.pkl file already exists, remove or rename it."
    experiments.produce_metrics.computeVariances(wildfireData, outputVariancesPath)
    f = open(outputVariancesPath, "r")


def test_wildfire_sample_size_bias():
    """
    Find the bias introduced as the number of samples increases.
    :return:
    """
    databaseCSVPath = "experiments/data/processed_database.csv"
    wildfireData = WildfireData(databaseCSVPath)

    # Update all the transition tuples in the database
    wildfireData.populateDatabase(
        stitchingVariables=DISTANCE_METRIC_VARIABLES,
        visualizationVariables=VISUALIZATION_VARIABLES)

    inputVariancesPath = "experiments/data/processed_database_variances.pkl"
    inputVariances = file(inputVariancesPath, "rb")
    varianceDictionary = pickle.load(inputVariances)
    outputCSVFilePath = "experiments/results/sample_size_bias.csv"
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
    assert False == True, "todo: update the experiments.wildfire_policy_functions.databasePolicyParameters"
    for targetPolicyERC in experiments.wildfire_policy_functions.databasePolicyParameters["ercThreshold"]:
        for targetPolicyTime in experiments.wildfire_policy_functions.databasePolicyParameters["timeUntilEndOfFireSeasonThreshold"]:
            policyValues.append([targetPolicyERC, targetPolicyTime])
            policies.append(experiments.wildfire_policy_functions.wildfirePolicySeverityFactory(targetPolicyERC, targetPolicyTime))

    benchmarks = []
    for policyValue in policyValues:
        base_rollouts = wildfireData.getTargetRollouts(policyValue[0], policyValue[1])
        current = rlpy.Domains.StitchingPackage.benchmark.Benchmark(base_rollouts, 2, benchmarkActions=False)
        benchmarks.append(current)

    for sampleCount in [10,20,30,40,50,60,70,80,90,100,
                        110,120,130,140,150,160,170,180,
                        190,200]:

        experiments.sample_size_bias.visualFidelityError(
            sampleCount,
            wildfireData,
            varianceDictionary,
            stitchingDomain,
            outCSVFile,
            benchmarks,
            stitchingVariables,
            policyValues,
            policies,
            horizon=100)

def test_wildfire_produce_metrics_feature_selection():
    """
    Find the performance of a series of different distance metrics.
    :return:
    """
    databaseCSVPath = "experiments/data/processed_database.csv"
    wildfireData = WildfireData(databaseCSVPath, visualizationVariables=["reward"])
    variancesPath = "experiments/data/processed_database_variances.pkl"
    f = open(variancesPath, "r")
    varianceDictionary = pickle.load(f)
    csvFilePath = "experiments/results/wildfire_metric_performances.csv"
    experiments.produce_metrics.featureSelection(wildfireData, varianceDictionary, csvFilePath)

def test_wildfire_structural_bias():
    """
    Find the performance in the presence or absence of the bias correction.
    :return:
    """
    databaseCSVPath = "experiments/data/processed_database.csv"
    wildfireData = WildfireData(databaseCSVPath)

    # Update all the transition tuples in the database
    wildfireData.populateDatabase(
        stitchingVariables=STITCHING_VARIABLES,
        visualizationVariables=VISUALIZATION_VARIABLES)

    inputVariancesPath = "synthesis_tests/wildfire_data/wildfire_variances_hand_constructed.pkl"
    inputVariances = file(inputVariancesPath, "rb")
    varianceDictionary = pickle.load(inputVariances)
    outputCSVFilePath = "synthesis_tests/wildfire_data/structural_bias.csv"
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
    for targetPolicyERC in experiments.wildfire_policy_functions.databasePolicyParameters["ercThreshold"]:
        for targetPolicyTime in experiments.wildfire_policy_functions.databasePolicyParameters["timeUntilEndOfFireSeasonThreshold"]:
            policyValues.append([targetPolicyERC, targetPolicyTime])
            policies.append(experiments.wildfire_policy_functions.wildfirePolicySeverityFactory(targetPolicyERC, targetPolicyTime))

    benchmarks = []
    for policyValue in policyValues:
        base_rollouts = wildfireData.getTargetRollouts(policyValue[0], policyValue[1])
        current = rlpy.Domains.StitchingPackage.benchmark.Benchmark(base_rollouts, 2, benchmarkActions=False)
        benchmarks.append(current)

    experiments.structural_bias.visualFidelityError(
        wildfireData,
        varianceDictionary,
        stitchingDomain,
        outCSVFile,
        benchmarks,
        STITCHING_VARIABLES,
        policyValues,
        policies,
        sampleCount=50,
        horizon=100)
