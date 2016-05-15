from .Domain import Domain
from subprocess import call
from struct import unpack
import numpy as np
import os.path
import csv
import pickle
from rlpy.Domains.StitchingPackage.TransitionTuple import TransitionTuple
from experiments.configurations import clusterConfigurationDict as configDict

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = ["Sean McGregor"]


class WildfireData(Domain):

    """
    This "domain" provides an interface into wildfire data recorded from a computationally expensive simulator.\n
    It defines a limited interface that can be used by the Stitching domain to perform trajectory synthesis.

    **STATE:**        Todo, define [xxxx, yyyy, ...] \n
    **ACTIONS:**      [suppress, let burn] \n
    **TRANSITIONS:**  Burn, grow, and harvest. \n
    **REWARD:**       -real value to +real value. \n

    **REFERENCE:**
    Based on Houtman, R. M., Montgomery, C. A., Gagnon, A. R., Calkin, D. E., Dietterich, T. G., McGregor, S.,
      & Crowley, M. (2013).
      Allowing a Wildfire to Burn: Estimating the Effect on Future Fire Suppression Costs.
      International Journal of Wildland Fire, 22(7), 871-882.
    """

    # All the variables that could be used in stitching
    ALL_PRE_TRANSITION_STITCHING_VARIABLES = [
        "Fuel Model start", # \/ pulled from the landscape summary of the prior time step's onPolicy landscape
        "Canopy Closure start",
        "Canopy Height start",
        "Canopy Base Height start",
        "Canopy Bulk Density start",
        "Covertype start",
        "Stand Density Index start",
        "Succession Class start",
        "Maximum Time in State start",
        "Stand Volume Age start",
        "Precipitation start", # \/ pulled from the current row's state
        "MaxTemperature start",
        "MinHumidity start",
        "WindSpeed start",
        "ignitionCovertype start",
        "ignitionSlope start",
        "startIndex start",
        "endIndex start",
        "ERC start",
        "SC start"
    ]

    # The variables that correspond to the variables in ALL_PRE_TRANSITION_STITCHING_VARIABLES
    ALL_POST_TRANSITION_STITCHING_VARIABLES = [
        "Fuel Model end", # \/ pulled from the landscape summary of the current row
        "Canopy Closure end",
        "Canopy Height end",
        "Canopy Base Height end",
        "Canopy Bulk Density end",
        "Covertype end",
        "Stand Density Index end",
        "Succession Class end",
        "Maximum Time in State end",
        "Stand Volume Age end",
        "Precipitation end", # \/ pulled from the subsequent state
        "MaxTemperature end",
        "MinHumidity end",
        "WindSpeed end",
        "ignitionCovertype end",
        "ignitionSlope end",
        "startIndex end",
        "endIndex end",
        "ERC end",
        "SC end"
    ]

    NON_LANDSCAPE_TRANSITION_STITCHING_VARIABLE_NAMES = [
        "Precipitation",
        "MaxTemperature",
        "MinHumidity",
        "WindSpeed",
        "ignitionCovertype",
        "ignitionSlope",
        "startIndex",
        "endIndex",
        "ERC",
        "SC"
    ]

    # The variables to use for stitching if we care most about performance (landscape variables)
    BEST_PRE_TRANSITION_STITCHING_VARIABLES = [
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

    # The variables that result from stitching if we care most about performance (landscape variables)
    BEST_POST_TRANSITION_STITCHING_VARIABLES = [
        "Fuel Model end",
        "Canopy Closure end",
        "Canopy Height end",
        "Canopy Base Height end",
        "Canopy Bulk Density end",
        "Covertype end",
        "Stand Density Index end",
        "Succession Class end",
        "Maximum Time in State end",
        "Stand Volume Age end"
    ]

    # All the variables we visualize and use for benchmarking
    VISUALIZATION_VARIABLES = [
        "action",
        "CrownFirePixels",
        "SurfaceFirePixels",
        "fireSuppressionCost",
        "timberLoss_IJWF",
        "boardFeetHarvestTotal",
        "boardFeetHarvestPonderosa",
        "boardFeetHarvestLodgepole",
        "boardFeetHarvestMixedConifer"
    ]

    # All the variables we use in the policy function
    POLICY_VARIABLES = [
        "ERC",
        "startIndex"
    ]

    # A list of tuples that are initial states. These will be drawn with uniform probability
    init_state_tuples = []

    actions_num = 2
    state_space_dims = 2
    continuous_dims = [0, 1]

    TIMESTEPS = 100
    INIT_STATE = np.array([
        130.011288678,
        27.4871670222,
        475.19681323,
        29.6549113633,
        5.17117748117,
        57.4973352338,
        24.5366554022,
        25.8858577659,
        263.886173989,
        263.886173989])
    actions = [0, 1] # Let burn, suppress
    state = {"todo":-99999} # todo: specify all the state variables that will be visualized
    database = []

    def __init__(self,
                 databaseCSV,
                 ALL_PRE_TRANSITION_STITCHING_VARIABLES=ALL_PRE_TRANSITION_STITCHING_VARIABLES,
                 ALL_POST_TRANSITION_STITCHING_VARIABLES=ALL_POST_TRANSITION_STITCHING_VARIABLES,
                 NON_LANDSCAPE_TRANSITION_STITCHING_VARIABLE_NAMES=NON_LANDSCAPE_TRANSITION_STITCHING_VARIABLE_NAMES,
                 BEST_PRE_TRANSITION_STITCHING_VARIABLES=BEST_PRE_TRANSITION_STITCHING_VARIABLES,
                 BEST_POST_TRANSITION_STITCHING_VARIABLES=BEST_POST_TRANSITION_STITCHING_VARIABLES,
                 VISUALIZATION_VARIABLES=VISUALIZATION_VARIABLES,
                 POLICY_VARIABLES=POLICY_VARIABLES
                 ):
        """
        Load data from the database's CSV file.
        """

        self.ALL_PRE_TRANSITION_STITCHING_VARIABLES = ALL_PRE_TRANSITION_STITCHING_VARIABLES
        self.ALL_POST_TRANSITION_STITCHING_VARIABLES = ALL_POST_TRANSITION_STITCHING_VARIABLES
        self.NON_LANDSCAPE_TRANSITION_STITCHING_VARIABLE_NAMES = NON_LANDSCAPE_TRANSITION_STITCHING_VARIABLE_NAMES
        self.BEST_PRE_TRANSITION_STITCHING_VARIABLES = BEST_PRE_TRANSITION_STITCHING_VARIABLES
        self.BEST_POST_TRANSITION_STITCHING_VARIABLES = BEST_POST_TRANSITION_STITCHING_VARIABLES
        self.VISUALIZATION_VARIABLES = VISUALIZATION_VARIABLES
        self.POLICY_VARIABLES = POLICY_VARIABLES

        np.random.seed(0)
        self.databaseCSV = databaseCSV
        self.populateDatabase()

        #super(WildfireData, self).__init__()

    def populateDatabase(self):
        """
        Replace the current database with a new database defined on the parameters.

        :return: self.database
        """

        def parseRow(r):
            try:
                ret = float(r)
                return ret
            except Exception:
                return r

        stitchingVariables = self.BEST_PRE_TRANSITION_STITCHING_VARIABLES
        visualizationVariables = self.VISUALIZATION_VARIABLES

        self.database = []
        self.DimNames = []
        self.init_state_tuples = []
        with open(self.databaseCSV, 'rb') as csvfile:
            transitions = csv.reader(csvfile, delimiter=',')
            row = transitions.next()
            header = []
            for headerValue in row:
                if headerValue:
                    self.DimNames.append(headerValue.strip())
                    header.append(headerValue.strip())
            for row in transitions:
                del row[-1]
                parsedRow = map(parseRow, row)
                state = []
                ns = []
                visualizationState = {}
                additionalState = {}
                for idx, headerValue in enumerate(header):
                    if headerValue not in stitchingVariables and headerValue not in visualizationVariables:
                        additionalState[headerValue] = parsedRow[idx]
                additionalState["action"] = parsedRow[header.index("action")]
                additionalState["onPolicy"] = ("onPolicy" in additionalState["lcpFileName"])
                for stitchingVariableIdx, variable in enumerate(stitchingVariables):
                    stateIndex = header.index(variable)
                    state.append(parsedRow[stateIndex])
                    nsIndex = header.index(self.BEST_POST_TRANSITION_STITCHING_VARIABLES[stitchingVariableIdx])
                    ns.append(parsedRow[nsIndex])
                for variable in visualizationVariables:
                    visualizationStateIndex = header.index(variable)
                    visualizationState[variable] = parsedRow[visualizationStateIndex]
                terminal = False # no states are terminal
                isInitial = (additionalState["year"] == 0)
                assert len(state) == len(ns)
                t = TransitionTuple(
                    state,
                    ns,
                    terminal,
                    isInitial,
                    [0,1],
                    visualizationState,
                    additionalState
                )
                if isInitial:
                    self.init_state_tuples.append(t)
                self.database.append(t)

    @staticmethod
    def lcpStateSummary(landscapeFileName):
        """
        Give the summary variables used for stitching based on the landscapes.
        Landscapes are 940X1127X10=10593800 shorts (11653180)
        :param landscapeFileName: The name of the landscape we want to generate a state summary for.
        :return: array of values for distance metric variables
        """

        # always return the init state variables in the testing environment
        if configDict["environment"] == "testing":
            return WildfireData.INIT_STATE
        elif os.path.isfile(configDict["landscape summary directory"] + landscapeFileName):
            filename = configDict["landscape summary directory"] + landscapeFileName
            f = open(filename, "rb")
            summary = pickle.load(f)
            f.close()
            return summary

        print "Warning!!! reprocessing Landscape: {}".format(landscapeFileName)

        distanceMetricVariableCount = 10

        # tmpFileName
        decompressedFilename = "tmp.lcp." + landscapeFileName.split("/")[-1]#landscapeFileName.split(".bz2")[0]
        
        call(["bzip2 " + landscapeFileName + " -dkc > " + decompressedFilename], shell=True)
        
        lcpFile = file(decompressedFilename, "rb")
        print "processing %s" % lcpFile
        layers = []
        for idx in range(0,distanceMetricVariableCount):
            layers.append([])
        layers.append([]) # hack because there is an extra layer

        shortCount = 0 # 0 to 10,593,800
        shortBytes = lcpFile.read(2)
        while shortBytes != "":
            pix = unpack("<h", shortBytes)
            layers[shortCount % len(layers)].append(pix[0])
            shortCount += 1
            shortBytes = lcpFile.read(2)
        lcpFile.close()
        summary = []
        for layerIdx, layer in enumerate(layers):
            #high = float("-inf")
            #low = float("inf")
            #for idx, pixel in enumerate(layers[layerIdx]):
            #    if high < pixel:
            #        high = pixel
            #    if low > pixel:
            #        low = pixel
            average = 0
            for idx, pixel in enumerate(layers[layerIdx]):
                average = float(average * idx + pixel)/(idx + 1.)
            summary.append(average)
            #summary.append([high, low, average])
        print "removing file {}".format(decompressedFilename)
        call(["rm /nfs/stak/students/m/mcgregse/Projects/rlpy/" + decompressedFilename], shell=True) # cleanup decompressed file
        del summary[-1] # remove the last element because it is not needed
        return summary

    @staticmethod
    def postProcessData(infileName, outfileName):
        """
        Create the updated CSV database based on the current landscape summary functions.
        :param infileName: The name of the CSV file that we are going to post process with the landscape files to
          produce the processed database.
        :param outfileName: The name of the CSV file to generate after post-processing the infileName and the landscapes
        :param DISTANCE_METRIC_VARIABLES: The names of the variables that will be produced for the landscapes
        :return:
        """
        print "writing header to file"
        out = file(outfileName, "w")
        for newVar in WildfireData.ALL_PRE_TRANSITION_STITCHING_VARIABLES:
            out.write(newVar + ",")
        for newVar in WildfireData.ALL_POST_TRANSITION_STITCHING_VARIABLES:
            out.write(newVar + ",")

        with open(infileName, 'rb') as csvfile:
            transitionsReader = csv.DictReader(csvfile)
            fieldnames = transitionsReader.fieldnames
            print fieldnames

            # We will be writing the complete row for every raw transition
            for header in fieldnames:
                out.write(header.strip() + ",")
            out.write("\n")

            transitions = list(transitionsReader)
            for idx, transitionDictionary in enumerate(transitions):

                initialFire = int(transitionDictionary["initialFire"])
                year = int(transitionDictionary["year"])
                action = int(transitionDictionary["action"])
                offPolicy = "offPolicy" in transitionDictionary["lcpFileName"]

                # We can't render year 100 because there is no fire experienced at that year
                if year == 99:
                    continue
                
                parts = transitionDictionary["lcpFileName"].split("_")
                ercPolicyThreshold = int(parts[4])
                daysTilEndPolicyThreshold = int(parts[5])

                if offPolicy:
                    policyLabel = "offPolicy"
                else:
                    policyLabel = "onPolicy"

                landscapeEndFileName = "/scratch/eecs-share/rhoutman/FireWoman/results/landscapes/lcp_{}_{}_{}_{}_{}_{}.lcp.bz2".format(
                    initialFire,
                    year,
                    action,
                    ercPolicyThreshold,
                    daysTilEndPolicyThreshold,
                    policyLabel
                )
                assert landscapeEndFileName.split("/")[-1] == transitionDictionary["lcpFileName"].split("/")[-1] + ".bz2", "{} != {}".format(landscapeEndFileName, transitionDictionary["lcpFileName"])

                if year == 0:
                    summary = WildfireData.INIT_STATE
                else:
                    landscapeStartFileName = "/scratch/eecs-share/rhoutman/FireWoman/results/landscapes/lcp_{}_{}_{}_{}_{}_{}.lcp.bz2".format(
                        initialFire,
                        year-1,
                        0,
                        ercPolicyThreshold,
                        daysTilEndPolicyThreshold,
                        "onPolicy"
                    )
                    if not os.path.isfile(landscapeStartFileName):
                        landscapeStartFileName = "/scratch/eecs-share/rhoutman/FireWoman/results/landscapes/lcp_{}_{}_{}_{}_{}_{}.lcp.bz2".format(
                            initialFire,
                            year-1,
                            1,
                            ercPolicyThreshold,
                            daysTilEndPolicyThreshold,
                            "onPolicy"
                        )
                    print "processing {} and {}".format(landscapeStartFileName, landscapeEndFileName)
                    summary = WildfireData.lcpStateSummary(landscapeStartFileName)
                for elem in summary:
                    out.write(str(elem) + ",")

                # The rest of the potential start variables
                for name in WildfireData.NON_LANDSCAPE_TRANSITION_STITCHING_VARIABLE_NAMES:
                    out.write(str(transitionDictionary[name]) + ",")

                # Write the end summary variables
                summary = WildfireData.lcpStateSummary(landscapeEndFileName)
                for elem in summary:
                    out.write(str(elem) + ",")

                # Write the rest of the potential end variables
                for name in WildfireData.NON_LANDSCAPE_TRANSITION_STITCHING_VARIABLE_NAMES:
                    subsequentIndex = idx + 1
                    if int(transitions[subsequentIndex]["year"]) != year + 1:
                        subsequentIndex += 1
                    assert int(transitions[subsequentIndex]["year"]) == year + 1, "{} does not equal {}".format(transitions[subsequentIndex]["year"], year + 1)
                    out.write(str(transitions[subsequentIndex][name]) + ",")

                # Write out the rest of the result file
                for k in fieldnames:
                    cur = transitionDictionary[k]
                    if cur is None:
                        cur = ""
                    else:
                        cur = str(cur)
                    out.write(cur + ",")
                out.write("\n")

    def getInitState(self):
        """
        Get a state that was marked as an initial state in the database. This will not mark the state as being used
        as a transition in the rollout set.
        :return: TransitionTuple
        """
        i = np.random.choice(len(self.init_state_tuples))
        return self.init_state_tuples[i]


    def getTargetRollouts(self, ercPolicyParameter, timePolicyParameter):
        """
        Return all the rollouts generated under a particular policy. This returns rollouts that are correct in
        distribution, but the states that are stitched together
        :param policy:
        :return:
        """
        targetRollouts = []
        def arbitraryAppend(transition, targetRollouts=targetRollouts):
            timeStep = int(transition.additionalState["year"])
            while len(targetRollouts) <= timeStep:
                targetRollouts.append([])
            targetRollouts[timeStep].append(transition)
        for transition in self.database:
            ercParameter = int(transition.additionalState["policyThresholdERC"])
            timeParameter = int(transition.additionalState["policyThresholdDays"])
            if ercParameter == ercPolicyParameter and timeParameter == timePolicyParameter and transition.additionalState["onPolicy"]:
                arbitraryAppend(transition)
        trajectories = []

        # While rollouts remain
        while len(targetRollouts[0]):
            trajectory = []
            for eventsForTimestep in targetRollouts:
                state = eventsForTimestep[-1].visualizationResultState
                trajectory.append(state)
                eventsForTimestep.pop()
            trajectories.append(trajectory)

        return trajectories

    def getDatabase(self):
        """
        :return: The database used for trajectory synthesis. This should be in the format expected by the
          Stitching class.
        """
        return self.database

    def getDatabaseWithoutTargetSeverityPolicy(self, ercPolicyParameter, timePolicyParameter):
        """
        :return: The database used for trajectory synthesis. This should be in the format expected by the
          Stitching class.
        """
        sansTarget = []
        for transition in self.database:
            ercParameter = transition.additionalState["policyThresholdERC"]
            timeParameter = transition.additionalState["policyThresholdDays"]
            if ercParameter != ercPolicyParameter or timeParameter != timePolicyParameter:
                sansTarget.append(transition)
        return sansTarget


    def step(self, a):
        """
        Should not be called since the domain is generated from data.
        """
        assert(False)
        terminal = self.isTerminal()
        r = -9999999999
        ns = np.array([-9999999999, -9999999999])
        self.state = ns.copy()
        return r, ns, terminal, self.possibleActions()

    def s0(self):
        self.state = self.INIT_STATE.copy()
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def isTerminal(self):
        """
        No landscapes are terminal.
        """
        return False

    def showDomain(self, a):
       assert(False)

    def showLearning(self, representation):
        assert(False)
