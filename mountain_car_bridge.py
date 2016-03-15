"""
This file interfaces MDPvis with a set of domains defined for rlpy.

You do not call this file directly, but it is expected by the various
servers included in the mdpvis code base. If you are re-writing
this file for different domains, you will want to include
MDPvis in a sub directory from this file.
"""
__author__ = "Sean McGregor"
from rlpy.Domains import GridWorld as domain_gridworld
from rlpy.Domains import MountainCar as domain_mountain_car
from rlpy.Domains import HelicopterHover as domain_helicopter
from rlpy.Domains import Stitching as domain_stitching
from rlpy.Domains import HIVTreatment as domain_hiv
import numpy as np
import sys
import os

print "Loaded Mountain Car Domain"

###################################
#  Domain Initialization Objects  #
###################################
"""
These objects give the UI elements that will appear in MDPvis, as well
as the parameters that will be sent on every request for rollouts,
optimization, or states.
"""

mountainCarInitialization = {
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
         "current_value": 0, "max": 1000, "min": 0, "units": ""},
        {"name": "noise",
         "description": "Probability of a random transition from an action.",
         "current_value": .01, "max": 1, "min": 0, "units": ""},
        {"name": "rollouts in database",
         "description": "Number of rollouts to draw from in the database.",
         "current_value": 1000, "max": 10000, "min": 10, "units": ""},
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
        {"name": "Policy Probability",
         "description":"An integer identier of the current policy to always sample. -1 for random, 0 for always right, 2 for always left",
         "current_value": 0.875, "max": 1, "min":.5, "units": " "}
    ]
}

##################################
#         Get Rollouts           #
##################################

def randomPolicy(s, actions):
    """Default to a random action selection"""
    return np.random.choice(actions)

def generateRollouts(domain, labels, count, horizon, policy=randomPolicy):
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
            rollout.append(state)
        rollouts.append(rollout)
    return rollouts

def gridworldRollouts(query):
    """
       Get rollouts for the gridworld domain.
       Args:
           query (dictionary): The parameters as post-processed by flask_server.py.
               All numbers will be floats, so you should convert them to integers
               as necessary. Expected dictionary keys are found in the
               gridworldInitialization object.
    """
    number_rollouts = int(query["transition"]["rollout count"])
    horizon = int(query["transition"]["horizon"])
    database_rollouts = int(int(query["transition"]["rollouts in database"]))
    database_horizon = int(int(query["transition"]["database rollouts horizon"]))
    noise = query["transition"]["noise"]
    maze = os.path.join(domain_gridworld.default_map_dir, '4x5.txt')
    gridworld = domain_gridworld(maze, noise=noise)
    domain = gridworld
    metricFile = None
    labels = ["x", "y"]
    if query["transition"]["Metric Version"] != 0:
        directory = "../rlpy/Domains/StitchingPackage/metrics/gridworld/"
        metricFile =  directory + str(query["transition"]["Metric Version"])
        if not os.path.exists(directory):
            os.makedirs(directory)
    if int(query["transition"]["Use Synthesis"]) != 0:
        domain = domain_stitching(gridworld, rolloutCount=database_rollouts, horizon=database_horizon, metricFile=metricFile, labels=labels)
    return generateRollouts(domain, labels, number_rollouts, horizon)

def mountainCarRollouts(query):
    """
       Get rollouts for the mountaincar domain.
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
    noise = query["transition"]["noise"]
    mountaincar = domain_mountain_car(noise)
    domain = mountaincar

    metricFile = None
    optimizeMetric = True
    if query["transition"]["Metric Version"] != 0:
        directory = "../rlpy/Domains/StitchingPackage/metrics/mountaincar/"
        metricFile =  directory + str(int(query["transition"]["Metric Version"]))
        if not os.path.exists(directory):
            os.makedirs(directory)
        if os.path.isfile(metricFile):
            optimizeMetric = False

    databasePolicies = []
    targetPolicies = []

    def mountaincar_factory(reinforce_threshold):
        rs = np.random.RandomState(0)
        def policy_reinforce(state, possibleActions, reinforce_threshold=reinforce_threshold, rs=rs):
            uni = rs.uniform(0,1)
            reinforce = uni < reinforce_threshold
            x = state[0]
            xdot = state[1]
            if not reinforce:
                return rs.choice(possibleActions)
            elif xdot > 0.0:
                return 2 # Right
            elif xdot == 0.0:
                return rs.choice(possibleActions)
            else:
                return 0 # Left
        return policy_reinforce

    databasePolicies.append(mountaincar_factory(1.0))
    databasePolicies.append(mountaincar_factory(0.75))
    databasePolicies.append(mountaincar_factory(0.5))

    # Defaults to 0.875 at initialization
    policyProbability = float(query["policy"]["Policy Probability"])
    targetPolicies.append(policyProbability)

    if int(query["transition"]["Use Synthesis"]) != 0:
        domain = domain_stitching(
            mountaincar,
            labels=["x", "xdot"],
            rolloutCount=database_rollouts,
            targetPoliciesRolloutCount=50,
            horizon=database_horizon,
            databasePolicies=databasePolicies,
            metricFile=metricFile,
            optimizeMetric=optimizeMetric)
    print "generating rollouts now"
    return generateRollouts(
        domain,
        ["x", "xdot"],
        number_rollouts,
        horizon,
        policy=targetPolicies[0])

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
    noise = query["transition"]["noise"]
    hiv = domain_hiv()
    domain = hiv

    metricFile = None
    optimizeMetric = True
    if query["transition"]["Metric Version"] != 0:
        directory = "rlpy/Domains/StitchingPackage/metrics/hiv/"
        metricFile =  directory + str(int(query["transition"]["Metric Version"]))
        if not os.path.exists(directory):
            os.makedirs(directory)
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
        domain = domain_stitching(
            hiv,
            rolloutCount=database_rollouts,
            targetPoliciesRolloutCount=50,
            horizon=database_horizon,
            databasePolicies=databasePolicies,
            metricFile=metricFile,
            optimizeMetric=optimizeMetric)
    print "generating rollouts now"
    return generateRollouts(
        domain,
        ["t1", "t1infected", "t2", "t2infected", "v", "e"],
        number_rollouts,
        horizon,
        policy=targetPolicies[0])

def helicopterRollouts(query):
    """
    Get rollouts for the helicopter domain.
    Args:
        query (dictionary): The parameters as post-processed by flask_server.py.
            All numbers will be floats, so you should convert them to integers
            as necessary. Expected dictionary keys are found in the
            helicopterInitialization object.
    """
    number_rollouts = int(query["transition"]["rollout count"])
    horizon = int(query["transition"]["horizon"])
    discount = query["reward"]["discount_factor"]
    database_rollouts = int(query["transition"]["rollouts in database"])
    database_horizon = int(query["transition"]["database rollouts horizon"])
    noise = query["transition"]["noise"]
    metricFile = None
    if query["transition"]["Metric Version"] != 0:
        directory = "../rlpy/Domains/StitchingPackage/metrics/helicopter/"
        metricFile =  directory + str(query["transition"]["Metric Version"])
        if not os.path.exists(directory):
            os.makedirs(directory)
    helicopter = domain_helicopter(noise, discount)
    domain = helicopter
    if int(query["transition"]["Use Synthesis"]) != 0:
        domain = domain_stitching(helicopter, rolloutCount=database_rollouts, horizon=database_horizon, metricFile=metricFile)
    # Other state variables may be available
    return generateRollouts(domain, ["xerr", "yerr", "zerr", "u", "v", "w", "p", "q", "r"], number_rollouts, horizon)

##################################
#        Server Endpoints        #
##################################

def initialize():
    """
    Get the initialization object for the requested domain.

    Args:
        domain (Optional[str]): Domain specifier. Defaults to empty string.
            Select which domain we want rollouts from. This will default
            to serving a gridworld domain.
    """
    if domain == "gridworld":
        return gridworldInitialization
    elif domain == "mountaincar":
        return mountainCarInitialization
    elif domain == "helicopter":
        return helicopterInitialization
    elif domain == "hiv":
        return hivInitialization
    else:
        return gridworldInitialization

def rollouts(query):
    """
    Return a series of rollouts

    Args:
        query (dictionary): The parameters as post-processed by flask_server.py.
            All numbers will be floats, so you should convert them to integers
            as necessary.
        domain (Optional[str]): Domain specifier. Defaults to empty string.
            Select which domain we want rollouts from. This will default
            to serving a gridworld domain.
    """
    if domain == "gridworld":
        rollouts = gridworldRollouts(query)
    elif domain == "mountaincar":
        rollouts = mountainCarRollouts(query)
    elif domain == "helicopter":
        rollouts = helicopterRollouts(query)
    elif domain == "hiv":
        rollouts = hivRollouts(query)
    else:
        rollouts = gridworldRollouts(query)
    return rollouts

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
    This domain_bridge supports multiple MDP domains. You can specify one of the options
    given below as a positional argument.

    gridworld, mountaincar, helicopter, hiv
    """
