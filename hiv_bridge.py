"""
This file interfaces MDPvis with a set of domains defined for rlpy.

You do not call this file directly, but it is expected by the various
servers included in the mdpvis code base. If you are re-writing
this file for different domains, you will want to include
MDPvis in a sub directory from this file.
"""
__author__ = "Sean McGregor"
from rlpy.Domains import Stitching as domain_stitching
from rlpy.Domains import HIVTreatment as domain_hiv
import numpy as np
import os
import pickle

print "Loaded HIV Domain"

###################################
#  Domain Initialization Objects  #
###################################
"""
These objects give the UI elements that will appear in MDPvis, as well
as the parameters that will be sent on every request for rollouts,
optimization, or states.
"""
hivInitialization = {
    "reward": [
        {"name": "use defaults",
         "description":"You cannot change the rewards",
         "current_value": 1, "max": 1, "min": 1, "units": ""}
    ],
    "transition": [
        {"name": "Use Synthesis",
         "description": "Non-zero values are true.",
         "current_value": 0, "max": 1, "min": 0, "units": ""},
        {"name": "Metric Version",
         "description": "If synthesis is active, this will either optimize a new distance metric and cache it to metrics/VERSION, or load a pre-existing metric from the cache.",
         "current_value": 200, "max": 1000, "min": 0, "units": ""},
        {"name": "rollouts in database",
         "description": "Number of rollouts to draw from in the database.",
         "current_value": 30, "max": 10000, "min": 10, "units": ""},
        {"name": "database rollouts horizon",
         "description": "How far into the future the database samples should go.",
         "current_value": 100, "max": 10000, "min": 5, "units": ""},
        {"name": "rollout count",
         "description": "Number of rollouts to generate.",
         "current_value": 10, "max": 10000, "min": 1, "units": ""},
        {"name": "horizon",
         "description": "Maximum number of transitions.",
         "current_value": 100, "max": 10000, "min": 1, "units": ""}
    ],
    "policy": [
        {"name": "Policy Probability RTI",
         "description":"The probability of administering RTI",
         "current_value": .4, "max": 1.0, "min":0.0, "units": " "},
        {"name": "Policy Probability PI",
         "description":"The probability of administering PI",
         "current_value": .4, "max": 1.0, "min":0.0, "units": " "}
    ]
}

##################################
#         Get Rollouts           #
##################################

def generateRollouts(domain, labels, count, horizon, policy=None):
    """
        Helper function for generating rollouts from all the domains.
        Args:
            domain (Domain): The domain that will be called to generate rollouts.
            labels (list(String)): A list of the state labels.
            count (integer): The number of rollouts to generate.
            horizon (integer): The maximum length of rollouts.
    """

    # Need to use the getRollouts function defined in the Stitching class because
    # it ensures transitions are drawn from the database without replacement
    if domain.__class__.__name__ == "Stitching":
        return domain.getRollouts(count=count, horizon=horizon, policy=policy)

    rollouts = []
    for rollout_number in range(count):
        rollout = []
        domain.s0() # reset the state
        terminate = False
        while not terminate and len(rollout) < horizon:
            terminate = domain.isTerminal()
            possible_actions = domain.possibleActions()
            action = policy(domain.state, possible_actions)
            state = {}
            for i in range(len(labels)):
                state[labels[i]] = domain.state[i]
            state["action"] = action
            r, ns, terminal, currentPossibleActions = domain.step(action)
            state["reward"] = r
            #state["stitch distance"] = 0.0
            rollout.append(state)
        rollouts.append(rollout)
    return rollouts

def hivRollouts(query):
    """
       Get rollouts for the hiv domain.
       Args:
           query (dictionary): The parameters as post-processed by flask_server.py.
               All numbers will be floats, so you should convert them to integers
               as necessary. Expected dictionary keys are found in the
               mountaincarInitialization object.
    """
    number_rollouts = int(query["transition"]["rollout count"])
    horizon = int(query["transition"]["horizon"])
    database_rollouts = int(query["transition"]["rollouts in database"])
    database_horizon = int(query["transition"]["database rollouts horizon"])
    hiv = domain_hiv()
    domain = hiv

    metricFile = None
    optimizeMetric = True
    if query["transition"]["Metric Version"] != 0:
        directory = "../rlpy/Domains/StitchingPackage/metrics/hiv/"
        metricFile =  directory + str(int(query["transition"]["Metric Version"]))
        if os.path.isfile(metricFile):
            optimizeMetric = False
            print "loading metric from file: {}".format(metricFile)
        else:
            print "WARNING: Metric is optimizing to: {}".format(metricFile)

    def hiv_factory(rti_probability, pi_probability):
        rs = np.random.RandomState(0)
        def policy_reinforce(state, possibleActions, rti_probability=rti_probability, pi_probability=pi_probability, rs=rs):
            uni = rs.uniform(0,1)
            rti = uni < rti_probability
            uni = rs.uniform(0,1)
            pi = uni < pi_probability
            t1 = state[0] # non-infected CD4+ T-lymphocytes [cells / ml]
            t1infected = state[1] # infected CD4+ T-lymphocytes [cells / ml]
            t2 = state[2] # non-infected macrophages [cells / ml]
            t2infected = state[3] # infected macrophages [cells / ml]
            v = state[4] # number of free HI viruses [copies / ml]
            e = state[5] # number of cytotoxic T-lymphocytes [cells / ml]

            spiking = (t1infected > 100 or v > 20000)

            if spiking:
                rti = True
                pi = True

            # Actions
            # *0*: none active
            # *1*: RTI active
            # *2*: PI active
            # *3*: RTI and PI active

            if rti and pi:
                return 3
            elif rti:
                return 1
            elif pi:
                return 2
            else:
                return 0
        return policy_reinforce

    databaseProbabilities = [0, .25, 1]
    databasePolicies = []
    for prob1 in databaseProbabilities:
        for prob2 in databaseProbabilities:
            databasePolicies.append(hiv_factory(prob1, prob2))

    # Defaults to 0.4 at initialization
    policyProbabilityRTI = float(query["policy"]["Policy Probability RTI"])
    policyProbabilityPI = float(query["policy"]["Policy Probability PI"])
    targetPolicies = []
    targetPolicies.append(hiv_factory(policyProbabilityRTI, policyProbabilityPI))

    if int(query["transition"]["Use Synthesis"]) != 0:
        stitching_database = None
        database_cache_name = "{}-{}".format(database_rollouts, database_horizon)
        database_cache_path = "../rlpy/Domains/StitchingPackage/databases/hiv/" + database_cache_name
        save_database = True
        if os.path.isfile(database_cache_path):
            f = open(database_cache_path, 'r')
            stitching_database = pickle.load(f)
            f.close()
            print "loaded database from pickled file"
            save_database = False
        else:
            print "no database fount at {}, generating database...".format(database_cache_path)

        domain = domain_stitching(
            hiv,
            labels=["t1", "t1infected", "t2", "t2infected", "v", "e"],
            database=stitching_database,
            rolloutCount=database_rollouts,
            targetPoliciesRolloutCount=1, # todo, this is probably not necessary
            horizon=database_horizon,
            databasePolicies=databasePolicies,
            metricFile=metricFile,
            optimizeMetric=optimizeMetric)
        if save_database:
            print "saving database to: " + database_cache_path
            f = open(database_cache_path, 'w')
            pickle.dump(domain.database, f)
            f.close()
    print "generating rollouts now"
    return generateRollouts(
        domain,
        ["t1", "t1infected", "t2", "t2infected", "v", "e"],
        number_rollouts,
        horizon,
        policy=targetPolicies[0])

##################################
#        Server Endpoints        #
##################################

def initialize():
    """
    Get the initialization object for the requested domain.
    """
    return hivInitialization

def rollouts(query):
    """
    Return a series of rollouts

    Args:
        query (dictionary): The parameters as post-processed by flask_server.py.
            All numbers will be floats, so you should convert them to integers
            as necessary.
    """
    return hivRollouts(query)

def optimize(query):
    """
    Return a newly optimized policy.

    todo: implement this

    Args:
        query (dictionary): The parameters as post-processed by flask_server.py.
            All numbers will be floats, so you should convert them to integers
            as necessary.
        domain (Optional[str]): Domain specifier. Defaults to empty string.
            Select which domain we want rollouts from. This will default
            to serving a gridworld domain.
    """
    return {}

def state(query):
    """
    Return a series of images up to the requested event number.
    todo, implement this
    """
    return {}

def print_options():
    """
    Prints the names of domains that can be launched for this domain_bridge file.
    These domains could be launched as a positional argument when starting the server.
    """
    print """
    This domain bridge only supports the HIV domain.
    """
