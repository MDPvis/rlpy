from .Domain import Domain
from subprocess import call
from struct import unpack
import numpy as np
import csv
from rlpy.Domains.StitchingPackage.TransitionTuple import TransitionTuple

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
    ALL_STITCHING_VARIABLES = [
        "Fuel Model start",
        "Canopy Closure start",
        "Canopy Height start",
        "Canopy Base Height start",
        "Canopy Bulk Density start",
        "Covertype start",
        "Stand Density Index start",
        "Succession Class start",
        "Maximum Time in State start",
        "Stand Volume Age start",
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

    # The variables that correspond to the variables in ALL_STITCHING_VARIABLES
    POST_TRANSITION_VARIABLES = [
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

    # The variables to use for stitching if we care most about performance (landscape variables)
    BEST_STITCHING_VARIABLES = [
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

    # All the variables we visualize
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
                 stitchingVariables=ALL_STITCHING_VARIABLES,
                 visualizationVariables=VISUALIZATION_VARIABLES):
        """
        Load data from the database's CSV file.
        """
        np.random.seed(0)
        self.databaseCSV = databaseCSV
        self.populateDatabase(
            stitchingVariables=stitchingVariables,
            visualizationVariables=visualizationVariables)

        #super(WildfireData, self).__init__()

    def populateDatabase(
            self,
            stitchingVariables=ALL_STITCHING_VARIABLES,
            visualizationVariables=VISUALIZATION_VARIABLES):
        """
        Replace the current database with a new database defined on the parameters.

        :return: self.database
        """
        self.database = []
        self.DimNames = []
        self.init_state_tuples = []
        self.VISUALIZATION_VARIABLES = visualizationVariables
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
                parsedRow = map(float, row)
                state = []
                ns = []
                visualizationState = {}
                additionalState = {}
                for idx, headerValue in enumerate(header):
                    if headerValue not in stitchingVariables and headerValue not in visualizationVariables:
                        additionalState[headerValue] = parsedRow[idx]
                for stitchingVariableIdx, variable in enumerate(stitchingVariables):
                    stateIndex = header.index(variable)
                    state.append(parsedRow[stateIndex])
                    nsIndex = header.index(self.POST_TRANSITION_VARIABLES[stitchingVariableIdx])
                    ns.append(parsedRow[nsIndex])
                for variable in visualizationVariables:
                    visualizationStateIndex = header.index(variable)
                    visualizationState[variable] = parsedRow[visualizationStateIndex]
                terminal = False # no states are terminal
                isInitial = (additionalState["time step"] == 1)
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
    def lcpStateSummary(landscapeFileName, DISTANCE_METRIC_VARIABLES=None):
        """
        Give the summary variables used for stitching based on the landscapes.
        Landscapes are 940X1127X10=10593800 shorts (11653180)
        :param landscapeFileName: The name of the landscape we want to generate a state summary for.
        :param DISTANCE_METRIC_VARIABLES: The variables we are going to generate summaries for. These will be used
          in the distance metric.
        :return: array of values for distance metric variables
        """
        assert DISTANCE_METRIC_VARIABLES is not None
        call(["bzip2 " + landscapeFileName + " -dk"], shell=True)
        decompressedFilename = landscapeFileName.split(".bz2")[0]
        lcpFile = file(decompressedFilename, "rb")
        print "processing %s" % lcpFile
        layers = []
        for idx, layer in enumerate(DISTANCE_METRIC_VARIABLES):
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
        call(["rm " + decompressedFilename], shell=True) # cleanup decompressed file
        del summary[-1] # remove the last element because it is not needed
        return summary

    @staticmethod
    def postProcessData(infileName, outfileName, DISTANCE_METRIC_VARIABLES=None):
        """
        Create the updated CSV database based on the current landscape summary functions.
        :param infileName: The name of the CSV file that we are going to post process with the landscape files to
          produce the processed database.
        :param outfileName: The name of the CSV file to generate after post-processing the infileName and the landscapes
        :param DISTANCE_METRIC_VARIABLES: The names of the variables that will be produced for the landscapes
        :return:
        """
        assert DISTANCE_METRIC_VARIABLES is not None
        out = file(outfileName, "w")
        for newVar in DISTANCE_METRIC_VARIABLES:
            out.write(newVar + " start,")
        for newVar in DISTANCE_METRIC_VARIABLES:
            out.write(newVar + " end,")

        with open(infileName, 'rb') as csvfile:
            transitions = csv.DictReader(csvfile)

            for header in transitions.keys():
                out.write(header.strip() + ",")
            out.write("\n")
            for transitionDictionary in transitions:

                initialEvent = int(transitionDictionary["initialEvent"])
                year = int(transitionDictionary["year"])
                action = int(transitionDictionary["action"])
                ercPolicyThreshold = int(transitionDictionary["ercPolicyThreshold"])
                daysTilEndPolicyThreshold = int(transitionDictionary["daysTilEndPolicyThreshold"])
                offPolicy = "offPolicy" in transitionDictionary["lcpFileName"]

                if offPolicy:
                    policyLabel = "offPolicy"
                else:
                    policyLabel = "onPolicy"

                landscapeEndFileName = "/scratch/eecs-share/rhoutman/FireWoman/results/landscapes/lcp_{}_{}_{}_{}_{}_{}.lcp".format(
                    initialEvent,
                    year,
                    action,
                    ercPolicyThreshold,
                    daysTilEndPolicyThreshold,
                    policyLabel
                )

                if year == 0:
                    landscapeStartFileName = "/nfs/stak/students/m/mcgregse/Projects/rlpy/synthesis_tests/compressed.lcpz.bz2"
                else:
                    landscapeStartFileName = "/scratch/eecs-share/rhoutman/FireWoman/results/landscapes/lcp_{}_{}_{}_{}_{}_{}.lcp".format(
                        initialEvent,
                        year-1,
                        action,
                        ercPolicyThreshold,
                        daysTilEndPolicyThreshold,
                        "onPolicy"
                    )

                # Write the start variables
                summary = WildfireData.lcpStateSummary(landscapeStartFileName,
                                                       DISTANCE_METRIC_VARIABLES=DISTANCE_METRIC_VARIABLES)
                for elem in summary:
                    out.write(str(elem) + ",")

                # Write the end variables
                summary = WildfireData.lcpStateSummary(landscapeEndFileName,
                                                       DISTANCE_METRIC_VARIABLES=DISTANCE_METRIC_VARIABLES)
                for elem in summary:
                    out.write(str(elem) + ",")

                # Write out the rest of the result file
                for elem in row:
                    out.write(elem + ",")
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
            timeStep = int(transition.additionalState["time step"])
            while len(targetRollouts) < timeStep:
                targetRollouts.append([])
            targetRollouts[timeStep-1].append(transition)
        for transition in self.database:
            ercParameter = int(transition.additionalState["ercPolicyParameter"])
            timeParameter = int(transition.additionalState["timePolicyParameter"])
            if ercParameter == ercPolicyParameter and timeParameter == timePolicyParameter:
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
            ercParameter = transition.additionalState["ercPolicyParameter"]
            timeParameter = transition.additionalState["timePolicyParameter"]
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
