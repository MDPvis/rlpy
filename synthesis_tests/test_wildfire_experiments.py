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
from experiments.configurations import testingConfigurationDict as configDict

DISTANCE_METRIC_VARIABLES = [
    "Fuel Model",#99-187?, body 3 [187, 99, 130.01128867828544] http://www.landfire.gov/NationalProductDescriptions2.php
    "Canopy Closure",#0-55?, body 4 [55, 0, 27.48716702222056]
    "Canopy Height",#0-1000? body 5 [1000, 0, 475.19681323037685]
    "Canopy Base Height", #0-100? body 6 [100, 0, 29.654911363250267]
    "Canopy Bulk Density", #0-15? body 7 [15, 0, 5.171177481168032]

    "Covertype", #0-99? externals 0 [99, 0, 57.49733523381562]
    "Stand Density Index", #0-99? externals 1 [99, 0, 24.536655402216752]
    "Succession Class", #0-99? externals 2 [99, 0, 25.885857765862585]
    "Maximum Time in State", #0-999? externals 3 [999, 0, 263.8861739885612]
    "Stand Volume Age" #0-999? externals 4 [999, 0, 263.8861739885612]
    # extra state summary currently returned [0, 0, 0.0]
]

def test_wildfire_policy_functions():
    assert False # todo: change to use the proper index of the state variable
    assert False # todo: reimplement these functions exactly as it is states in FireWoman
    state = [75, 100, 0, 0] # currently assumes the state is ERC, daysUntilEndOfFireSeason, ignition x, ignition y
    severityPolicy = experiments.wildfire_policy_functions.wildfirePolicyLocationFactory(50, 50)
    locationpolicy = experiments.wildfire_policy_functions.wildfirePolicyLocationFactory(500, 500)
    assert severityPolicy(state, []) == 1
    assert locationpolicy(state, []) == 0

    state = [1, 100, 501, 501]
    assert severityPolicy(state, []) == 0
    assert locationpolicy(state, []) == 1

    state = [90, 1, 501, 11]
    assert severityPolicy(state, []) == 0
    assert locationpolicy(state, []) == 0

    state = [99, 100, 11, 501]
    assert severityPolicy(state, []) == 0
    assert locationpolicy(state, []) == 0

def test_wildfire_produce_metrics_variances():
    databaseCSVPath = "synthesis_tests/wildfire_data/wildfire_hand_constructed.csv"
    wildfireData = WildfireData(databaseCSVPath)
    outputVariancesPath = "synthesis_tests/wildfire_data/wildfire_variances_hand_constructed.pkl"
    if os.path.isfile(outputVariancesPath):
        os.remove(outputVariancesPath)
    experiments.produce_metrics.computeVariances(wildfireData, outputVariancesPath)
    f = open(outputVariancesPath, "r")
    variances = pickle.load(f)

    assert variances["Fuel Model start"] == 5.25,\
        "Current value {}".format(variances["Fuel Model start"])
    assert variances["Canopy Closure start"] == 0,\
        "Current value {}".format(variances["Canopy Closure start"])
    assert variances["Canopy Height start"] == 0,\
        "Current value {}".format(variances["Canopy Height start"])
    assert variances["Canopy Base Height start"] == 0,\
        "Current value {}".format(variances["Canopy Base Height start"])
    assert variances["Canopy Bulk Density start"] == 0,\
        "Current value {}".format(variances["Canopy Bulk Density start"])
    assert variances["Covertype start"] == 0,\
        "Current value {}".format(variances["Covertype start"])
    assert variances["Stand Density Index start"] == 0,\
        "Current value {}".format(variances["Stand Density Index start"])
    assert variances["Succession Class start"] == 0,\
        "Current value {}".format(variances["Succession Class start"])
    assert variances["Maximum Time in State start"] == 0,\
        "Current value {}".format(variances["Maximum Time in State start"])
    assert variances["Stand Volume Age start"] == 0,\
        "Current value {}".format(variances["Stand Volume Age start"])

    assert variances["column 1"] == 0, "Variance for column is {}".format(variances["column 1"])
    assert variances["column 2"] == 0, "Variance for column is {}".format(variances["column 2"])
    assert variances["column 3"] == 0, "Variance for column is {}".format(variances["column 3"])
    assert variances["column 4"] == 0, "Variance for column is {}".format(variances["column 4"])

def test_wildfire_produce_metrics_feature_selection():
    databaseCSVPath = "synthesis_tests/wildfire_data/wildfire_hand_constructed.csv"
    wildfireData = WildfireData(databaseCSVPath, visualizationVariables=["reward"])
    variancesPath = "synthesis_tests/wildfire_data/wildfire_variances_hand_constructed.pkl"
    f = open(variancesPath, "r")
    varianceDictionary = pickle.load(f)
    csvFilePath = "synthesis_tests/wildfire_data/wildfire_metric_performances.csv"
    experiments.produce_metrics.featureSelection(wildfireData, varianceDictionary, csvFilePath)

    initialMean = 15.2274446578

    means = [
        6.842829302,
        13.97523857,
        13.97523857,
        13.97523857,
        13.97523857,
        15.22744466,
        15.22744466,
        15.22744466,
        15.22744466,
        15.22744466,
        13.97523857,
        15.22744466,
        13.97523857
    ]

    with open(csvFilePath, 'rb') as csvfile:
        results = csv.reader(csvfile, delimiter=',')
        row = results.next()
        mean = float(row[1])
        variance = float(row[2])
        length = int(row[3])
        assert mean == initialMean, "Mean was {}".format(mean)
        assert variance == 0.0, "Mean was {}".format(variance)
        assert length == 14, "Mean was {}".format(length)
        for mean in means:
            row = results.next()
            mean = float(row[1])
            variance = float(row[2])
            length = int(row[3])
            assert mean == mean, "Mean was {}".format(mean)
            assert variance == 0.0, "Variance was {}".format(variance)
            assert length == 13, "Length was {}".format(length)

def test_wildfire_regression():
    assert False

def test_wildfire_sample_size_bias():
    """
    Test the bias introduced as the number of samples increases. This does not currently have any assertions,
    so it is more of a smoke test.
    :return:
    """
    databaseCSVPath = "synthesis_tests/wildfire_data/wildfire_hand_constructed.csv"
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


    inputVariancesPath = "synthesis_tests/wildfire_data/wildfire_variances_hand_constructed.pkl"
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
    for targetPolicyERC in experiments.wildfire_policy_functions.databasePolicyParameters["ercThreshold"]:
        for targetPolicyTime in experiments.wildfire_policy_functions.databasePolicyParameters["timeUntilEndOfFireSeasonThreshold"]:
            policyValues.append([targetPolicyERC, targetPolicyTime])
            policies.append(experiments.wildfire_policy_functions.wildfirePolicySeverityFactory(targetPolicyERC, targetPolicyTime))

    benchmarks = []
    for policyValue in policyValues:
        base_rollouts = wildfireData.getTargetRollouts(policyValue[0], policyValue[1])
        current = rlpy.Domains.StitchingPackage.benchmark.Benchmark(base_rollouts, 2, benchmarkActions=False)
        benchmarks.append(current)

    for sampleCount in [2]:

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
            horizon=2)

def test_wildfire_structural_bias():
    databaseCSVPath = "synthesis_tests/wildfire_data/wildfire_hand_constructed.csv"
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
        stitchingVariables,
        policyValues,
        policies,
        sampleCount=2,
        horizon=2)

def test_wildfire_other_policies():
    assert False

