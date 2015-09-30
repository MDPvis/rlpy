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
                             {"name": "noise",
                              "description": "Probability of a random transition from an action.",
                              "current_value": .01, "max": 1, "min": 0, "units": ""},
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
                            {"name": "noise",
                             "description": "Probability of a random transition from an action.",
                             "current_value": .01, "max": 1, "min": 0, "units": ""},
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
                           {"name": "noise",
                            "description": "Probability of a random transition from an action.",
                            "current_value": .01, "max": 1, "min": 0, "units": ""},
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

def generateRollouts(domain, labels, count, horizon):
    """
        Helper function for generating rollouts from all the domains.
        Args:
            domain (Domain): The domain that will be called to generate rollouts.
            labels (list(String)): A list of the state labels.
            count (integer): The number of rollouts to generate.
            horizon (integer): The maximum length of rollouts.
    """
    rollouts = []
    for rollout_number in range(count):
        rollout = []
        domain.s0() # reset the state
        while not domain.isTerminal() and len(rollout) < horizon:
            possible_actions = domain.possibleActions()
            action = random.choice(possible_actions) # todo, make better, make this an actual policy
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
    noise = query["transition"]["noise"]
    maze = os.path.join(domain_gridworld.default_map_dir, '4x5.txt')
    gridworld = domain_gridworld(maze, noise=noise)
    domain = gridworld
    if int(query["transition"]["Use Synthesis"]) != 0:
        domain = domain_stitching(gridworld, rolloutCount = 100, horizon = 20)
    return generateRollouts(domain, ["x", "y"], number_rollouts, horizon)

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
    noise = query["transition"]["noise"]
    mountaincar = domain_mountain_car(noise)
    domain = mountaincar
    if int(query["transition"]["Use Synthesis"]) != 0:
        domain = domain_stitching(mountaincar, rolloutCount = 50, horizon = 20)
    return generateRollouts(domain, ["x", "xdot"], number_rollouts, horizon)

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
   noise = query["transition"]["noise"]
   helicopter = domain_helicopter(noise, discount)
   domain = helicopter
   if int(query["transition"]["Use Synthesis"]) != 0:
       domain = domain_stitching(helicopter, rolloutCount = 50, horizon = 20)
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
        return gridworldRollouts(query)
    elif domain == "mountaincar":
        return mountainCarRollouts(query)
    elif domain == "helicopter":
        return helicopterRollouts(query)
    else:
        return gridworldRollouts(query)

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
