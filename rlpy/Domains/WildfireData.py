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
    database = {}

    def __init__(self, databaseCSV):
        """
        Load data from the database's CSV file.
        """
        self.database = []
        self.DimNames = []

        with open(databaseCSV, 'rb') as csvfile:
            transitions = csv.reader(csvfile, delimiter=',')
            row = transitions.next()
            for header in row:
                if header:
                    self.DimNames.append(header.strip())
            for row in transitions:
                del row[-1]
                parsedRow = map(float, row)
                state = parsedRow[0:10]
                ns = parsedRow[11:21]
                visualizationState = parsedRow
                terminal = False # no states are terminal
                isInitial = False # there is only one initial state, which we will hard code construct
                t = TransitionTuple(
                    state,
                    ns,
                    terminal,
                    isInitial,
                    [0,1],
                    visualizationState
                )
                self.database.append(t)

        #super(WildfireData, self).__init__()

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
            transitions = csv.reader(csvfile, delimiter=',')

            # Write the header from the file after the additional headers from the lcp summaries
            row = transitions.next()
            for header in row[2:]:
                out.write(header.strip() + ",")
            out.write("\n")
            for row in transitions:
                landscapeStartFileName = row[0]
                landscapeEndFilNamee = row[1]
                summary = WildfireData.lcpStateSummary(landscapeStartFileName,
                                                       DISTANCE_METRIC_VARIABLES=DISTANCE_METRIC_VARIABLES)
                for elem in summary:
                    out.write(str(elem) + ",")
                summary = WildfireData.lcpStateSummary(landscapeEndFilNamee,
                                                       DISTANCE_METRIC_VARIABLES=DISTANCE_METRIC_VARIABLES)
                for elem in summary:
                    out.write(str(elem) + ",")

                for elem in row[2:]:
                    out.write(elem + ",")
                out.write("\n")


    def getDatabase(self):
        """
        :return: The database used for trajectory synthesis. This should be in the format expected by the
          Stitching class.
        """
        return self.database

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
