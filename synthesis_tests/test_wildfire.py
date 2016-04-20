import logging
import os
import csv
import nose.tools
from rlpy.Domains.WildfireData import WildfireData

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

def test_post_processing():
    # These are the variables we will use to perform stitching.
    DISTANCE_METRIC_VARIABLES =  [
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
    csvOutFile = "synthesis_tests/wildfire_processed.csv"
    os.remove(csvOutFile)
    WildfireData.postProcessData("synthesis_tests/wildfire.csv", csvOutFile,
                                 DISTANCE_METRIC_VARIABLES=DISTANCE_METRIC_VARIABLES)
    with open(csvOutFile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        row = reader.next()
        assert row[0] == "Fuel Model start", "the first header was %s" % row[0]
        assert row[1] == "Canopy Closure start", "the second header was %s" % row[1]
        assert row[2] == "Canopy Height start", "the third header was %s" % row[2]
        row = reader.next()
        assert row[0] == "130.011288678", "the first value was %s" % row[0]
        assert row[-2] == "4", "the last value was %s" % row[-1]

def test_builds_state_summary():
    """
    Test the construction of a state summary based on a given compressed LCP
    """

    # These are the averages of the starting compressed landscape
    expectations = [
        130.01128867828544,
        27.48716702222056,
        475.19681323037685,
        29.654911363250267,
        5.171177481168032,
        57.49733523381562,
        24.536655402216752,
        25.885857765862585,
        263.8861739885612,
        263.8861739885612
    ]
    summary = WildfireData.lcpStateSummary("synthesis_tests/compressed.lcpz.bz2",
                                           DISTANCE_METRIC_VARIABLES=DISTANCE_METRIC_VARIABLES)
    for idx, val in enumerate(expectations):
        assert summary[idx] == val, "%s: actual %f for expected: %f " % (DISTANCE_METRIC_VARIABLES[idx], summary[idx], val)
    return

def test_builds_csv():
    """
    Test the construction of a state summary based on a given compressed LCP
    """
    wildfireData = WildfireData("synthesis_tests/wildfire_processed.csv")
    for transition in wildfireData.database:
        assert transition.preStateDistanceMetricVariables[0] == 130.011288678, "actual: %s" % transition.preStateDistanceMetricVariables[0]
        assert transition.preStateDistanceMetricVariables[1] == 27.4871670222, "actual: %s" % transition.preStateDistanceMetricVariables[1]
        assert transition.preStateDistanceMetricVariables[2] == 475.19681323, "actual: %s" % transition.preStateDistanceMetricVariables[2]
        assert transition.preStateDistanceMetricVariables[3] == 29.6549113633, "actual: %s" % transition.preStateDistanceMetricVariables[3]
        assert transition.preStateDistanceMetricVariables[4] == 5.17117748117, "actual: %s" % transition.preStateDistanceMetricVariables[4]
        assert transition.preStateDistanceMetricVariables[5] == 57.4973352338, "actual: %s" % transition.preStateDistanceMetricVariables[5]
        assert transition.preStateDistanceMetricVariables[6] == 24.5366554022, "actual: %s" % transition.preStateDistanceMetricVariables[6]
        assert transition.preStateDistanceMetricVariables[7] == 25.8858577659, "actual: %s" % transition.preStateDistanceMetricVariables[7]
        assert transition.preStateDistanceMetricVariables[8] == 263.886173989, "actual: %s" % transition.preStateDistanceMetricVariables[8]
        assert transition.preStateDistanceMetricVariables[9] == 263.886173989, "actual: %s" % transition.preStateDistanceMetricVariables[9]
    return


def test_get_database():
    wildfireData = WildfireData("synthesis_tests/wildfire_processed.csv")
    db = wildfireData.getDatabase()
    assert db[0].preStateDistanceMetricVariables[0] == wildfireData.database[0].preStateDistanceMetricVariables[0]
    assert db[0].postStateDistanceMetricVariables[0] == wildfireData.database[0].postStateDistanceMetricVariables[0]


def test_initial_state():
    wildfireData = WildfireData("synthesis_tests/wildfire_processed.csv")
    state, isTerminal, possibleActions = wildfireData.s0()
    assert state[0] == 130.011288678, "actual: %s" % state[0]
    assert state[1] == 27.4871670222, "actual: %s" % state[1]
    assert state[2] == 475.19681323, "actual: %s" % state[2]
    assert state[3] == 29.6549113633, "actual: %s" % state[3]
    assert state[4] == 5.17117748117, "actual: %s" % state[4]
    assert state[5] == 57.4973352338, "actual: %s" % state[5]
    assert state[6] == 24.5366554022, "actual: %s" % state[6]
    assert state[7] == 25.8858577659, "actual: %s" % state[7]
    assert state[8] == 263.886173989, "actual: %s" % state[8]
    assert state[9] == 263.886173989, "actual: %s" % state[9]
    assert isTerminal == False, "The initial state returned as terminal"
    assert possibleActions[0] == 0, "The possible actions in the initial state were not 0 and 1"
    assert possibleActions[1] == 1, "The possible actions in the initial state were not 0 and 1"