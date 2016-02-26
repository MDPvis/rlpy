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
import numpy as np
import os
import random
from flask_server import domain

###################################
#  Domain Initialization Objects  #
###################################
"""
These objects give the UI elements that will appear in MDPvis, as well
as the parameters that will be sent on every request for rollouts,
optimization, or states.
"""

gridworldInitialization = {
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
                            {"name": "Place holder",
                             "description":"todo, define logistic policies",
                             "current_value": 0, "max": 99999, "min":0, "units": " "}
                          ]
            }
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
                           {"name": "Policy Identifier",
                            "description":"An integer identier of the current policy to always sample. -1 for random, 0 for always right, 2 for always left",
                            "current_value": -1, "max": 99999, "min":-1, "units": " "}
                         ]
           }
helicopterInitialization = {
              "reward": [
                          {"name": "discount_factor",
                            "description":"How much to reward the present over the future",
                            "current_value": .95, "max": 1, "min": .001, "units": ""}
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
                          {"name": "Place holder",
                           "description":"todo, define logistic policies",
                           "current_value": 0, "max": 99999, "min":0, "units": " "}
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
        return domain.getRollouts(count=count, horizon=horizon, policies=[policy])

    rollouts = []
    for rollout_number in range(count):
        rollout = []
        domain.s0() # reset the state
        while not domain.isTerminal() and len(rollout) < horizon:
            possible_actions = domain.possibleActions()
            action = policy(domain.state, possible_actions) # todo, make better, make this an actual policy
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
    if query["transition"]["Metric Version"] != 0:
        directory = "../rlpy/Domains/StitchingPackage/metrics/mountaincar/"
        metricFile =  directory + str(query["transition"]["Metric Version"])
        if not os.path.exists(directory):
            os.makedirs(directory)

    policyNumber = int(query["policy"]["Policy Identifier"])

    def policy(state, possibleActions):
        return possibleActions[policyNumber]

    if int(query["transition"]["Use Synthesis"]) != 0:
        domain = domain_stitching(
          mountaincar,
          rolloutCount=database_rollouts,
          horizon=database_horizon,
          databasePolicies=[policy],
          metricFile=metricFile)
    return generateRollouts(domain, ["x", "xdot"], number_rollouts, horizon, policy = policy)

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
    
    gridworld, mountaincar, helicopter
    """
