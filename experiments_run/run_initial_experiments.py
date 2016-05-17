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
import experiments.policy_space_error
import rlpy.Domains.StitchingPackage.benchmark
from rlpy.Domains.WildfireData import WildfireData
import pickle
from experiments.configurations import clusterConfigurationDict as configDict

def test_post_process_landscapes():
    """
    Generate pickled version of all the state summaries for each of the landscapes in the landscapes directory
    """
    landscapeDirectory = configDict["raw landscape directory"]
    resultsDirectory = configDict["landscape summary directory"]
    files = os.listdir(landscapeDirectory)
    fileNum = 0
    while fileNum < len(files):
        f = files[fileNum]
        print "processing {}".format(f)
        if os.path.isfile(resultsDirectory+f):
            print "skipping forward {} since this landscape is processed".format(configDict["landscape processing jump"])
            fileNum += configDict["landscape processing jump"]
            continue
        try:
            s = WildfireData.lcpStateSummary(landscapeDirectory+f)
            if os.path.isfile(resultsDirectory+f):
                print "skipping forward {} since this landscape is processed".format(configDict["landscape processing jump"])
                fileNum += configDict["landscape processing jump"]
                continue
            out = open(resultsDirectory+f, "wb")
            pickle.dump(s, out)
            out.close()
        except Exception as inst:
            print type(inst)
            print inst.args
            print "failed to summarize: {}".format(f)
        fileNum += 1

def test_post_process_data():
    """
    Create the CSV file for MFMCi based off of a results file and the associated landscapes.
    :return:
    """
    WildfireData.postProcessData(
        configDict["raw CSV path"],
        configDict["processed CSV path"]
    )

def test_validate_database():
    """
    Open the database and check the ranges of all the values.
    :return:
    """
    configDict["processed CSV path"]
    with open(configDict["processed CSV path"]) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            assert float(row["Fuel Model start"]) >= 90.0, "{}".format(row["Fuel Model start"])
            assert float(row["Canopy Closure start"]) >= 0.0, "{}".format(row["Canopy Closure start"])
            assert float(row["Canopy Closure start"]) <= 100.0, "{}".format(row["Canopy Closure start"])
            assert float(row["Canopy Height start"]) >= 0.0, "{}".format(row["Canopy Height start"])
            assert float(row["Canopy Height start"]) < 99999.0, "{}".format(row["Canopy Height start"])
            assert float(row["Canopy Base Height start"]) >= 0.0, "{}".format(row["Canopy Base Height start"])
            assert float(row["Canopy Base Height start"]) < 99999.0, "{}".format(row["Canopy Base Height start"])
            assert float(row["Canopy Bulk Density start"]) >= 0.0, "{}".format(row["Canopy Bulk Density start"])
            assert float(row["Canopy Bulk Density start"]) < 99999.0, "{}".format(row["Canopy Bulk Density start"])
            assert float(row["Covertype start"]) >= 0.0, "{}".format(row["Covertype start"])
            assert float(row["Stand Density Index start"]) >= 0.0, "{}".format(row["Stand Density Index start"])
            assert float(row["Succession Class start"]) >= 0.0, "{}".format(row["Succession Class start"])
            assert float(row["Maximum Time in State start"]) >= 0.0, "{}".format(row["Maximum Time in State start"])
            assert float(row["Stand Volume Age start"]) >= 0.0, "{}".format(row["Stand Volume Age start"])
            assert float(row["Precipitation start"]) >= 0.0, "{}".format(row["fuelModel"])
            assert float(row["MaxTemperature start"]) >= -50.0, "{}".format(row["MaxTemperature start"])
            assert float(row["MinHumidity start"]) >= 0.0, "{}".format(row["MinHumidity start"])
            assert float(row["WindSpeed start"]) >= 0.0, "{}".format(row["WindSpeed start"])
            assert float(row["ignitionCovertype start"]) >= 0.0, "{}".format(row["ignitionCovertype start"])
            assert float(row["startIndex start"]) >= 0.0, "{}".format(row["startIndex start"])
            assert float(row["endIndex start"]) >= 0, "{}".format(row["endIndex start"])
            assert float(row["endIndex start"]) < 190, "{}".format(row["endIndex start"])
            assert float(row["ERC start"]) >= 0.0, "{}".format(row["ERC start"])
            assert float(row["ERC start"]) < 100.1, "{}".format(row["ERC start"])
            assert float(row["SC start"]) >= 0.0, "{}".format(row["SC start"])
            assert float(row["SC start"]) < 100.1, "{}".format(row["SC start"])
            assert float(row["Fuel Model end"]) >= 80.0, "{}".format(row["Fuel Model end"])
            assert float(row["Fuel Model end"]) < 200.1, "{}".format(row["Fuel Model end"])

def test_wildfire_produce_metrics_variances():
    """
    Find the variances of all the variables in the post-processed database. This will enable
    the creation of distance metrics using any of the database variables.
    :return:
    """
    databaseCSVPath = configDict["processed CSV path"]
    wildfireData = WildfireData(databaseCSVPath,
                                BEST_PRE_TRANSITION_STITCHING_VARIABLES=WildfireData.ALL_PRE_TRANSITION_STITCHING_VARIABLES,
                                BEST_POST_TRANSITION_STITCHING_VARIABLES=WildfireData.ALL_PRE_TRANSITION_STITCHING_VARIABLES)
    outputVariancesPath = configDict["variances output path"]
    if os.path.isfile(outputVariancesPath):
        assert False, "The processed_database_variances.pkl file already exists, remove or rename it."
    experiments.produce_metrics.computeVariances(wildfireData, outputVariancesPath)


def test_wildfire_policy_space():
    """
    Test the performance of MFMCi for each of the policies
    :return:
    """
    databaseCSVPath = configDict["processed CSV path"]
    wildfireData = WildfireData(databaseCSVPath)

    # Update all the transition tuples in the database
    wildfireData.populateDatabase()

    inputVariancesPath = configDict["variances output path"]
    inputVariances = file(inputVariancesPath, "rb")
    varianceDictionary = pickle.load(inputVariances)
    outputCSVFilePath = configDict["experimental outputs directory"] + "test_wildfire_policy_space_BEST_METRIC.csv"
    outCSVFile = file(outputCSVFilePath, "wb")

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
            policies.append(experiments.wildfire_policy_functions.wildfirePolicySeverityFactory(targetPolicyERC, targetPolicyTime))

    benchmarks = []
    for policyValue in policyValues:
        base_rollouts = wildfireData.getTargetRollouts(policyValue[0], policyValue[1])
        current = rlpy.Domains.StitchingPackage.benchmark.Benchmark(base_rollouts, 2, benchmarkActions=False)
        benchmarks.append(current)

    experiments.policy_space_error.visualFidelityError(
        wildfireData,
        varianceDictionary,
        stitchingDomain,
        outCSVFile,
        benchmarks,
        wildfireData.BEST_PRE_TRANSITION_STITCHING_VARIABLES,
        policyValues,
        policies,
        sampleCount=configDict["target trajectory count"],
        horizon=configDict["horizon"])

    outputCSVFilePath = configDict["experimental outputs directory"] + "test_wildfire_policy_space_BIG_METRIC.csv"
    outCSVFile2 = file(outputCSVFilePath, "wb")

    # Update all the transition tuples in the database
    wildfireData.populateDatabase(stitchingVariables=wildfireData.ALL_PRE_TRANSITION_STITCHING_VARIABLES)

    experiments.policy_space_error.visualFidelityError(
        wildfireData,
        varianceDictionary,
        stitchingDomain,
        outCSVFile2,
        benchmarks,
        wildfireData.ALL_PRE_TRANSITION_STITCHING_VARIABLES,
        policyValues,
        policies,
        sampleCount=configDict["target trajectory count"],
        horizon=configDict["horizon"])


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
    for targetPolicyERC in configDict["policy parameters ERC"]:
        for targetPolicyTime in configDict["policy parameters startIndex"]:
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
