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

def gridworldRollouts(query):
    """
       Get rollouts for the gridworld domain.
       Args:
           query (dictionary): The parameters as post-processed by flask_server.py.
               All numbers will be floats, so you should convert them to integers
               as necessary. Expected dictionary keys are found in the
               gridworldInitialization object.
    """
    noise = query["transition"]["noise"]

    maze = os.path.join(domain_gridworld.default_map_dir, '4x5.txt')
    gridworld = domain_gridworld(maze, noise=noise)

    number_rollouts = int(query["transition"]["rollout count"])
    horizon = int(query["transition"]["horizon"])

    rollouts = []
    for rollout_number in range(number_rollouts):
        rollout = []
        gridworld.s0() # reset the state
        while not gridworld.isTerminal() and len(rollout) < horizon:
            action = gridworld.possibleActions()[0] # Always do the first action, todo, make better
            rollout.append(
              {"x": gridworld.state[0],
               "y": gridworld.state[1],
               "a": action}) # todo, make this an actual policy
            gridworld.step(action)
        rollouts.append(rollout)
    return rollouts

def mountainCarRollouts(query):
    """
       Get rollouts for the mountaincar domain.
       Args:
           query (dictionary): The parameters as post-processed by flask_server.py.
               All numbers will be floats, so you should convert them to integers
               as necessary. Expected dictionary keys are found in the
               mountaincarInitialization object.
    """
    noise = query["transition"]["noise"]

    mountaincar = domain_mountain_car(noise)

    number_rollouts = int(query["transition"]["rollout count"])
    horizon = int(query["transition"]["horizon"])

    rollouts = []
    for rollout_number in range(number_rollouts):
        rollout = []
        mountaincar.s0() # reset the state
        while not mountaincar.isTerminal() and len(rollout) < horizon:
            possible_actions = mountaincar.possibleActions() # todo, make better
            action = random.choice(possible_actions)
            stateX = mountaincar.state[0]
            stateXdot = mountaincar.state[1]
            r, ns, terminal, possibleActions = mountaincar.step(action)
            rollout.append(
              {"x": stateX,
               "xdot": stateXdot,
               "reward": r,
               "a": action})
        rollouts.append(rollout)
    return rollouts

def helicopterRollouts(query):
   """
   Get rollouts for the helicopter domain.
   Args:
       query (dictionary): The parameters as post-processed by flask_server.py.
           All numbers will be floats, so you should convert them to integers
           as necessary. Expected dictionary keys are found in the
           helicopterInitialization object.
   """
   discount = query["reward"]["discount_factor"]
   noise = query["transition"]["noise"]

   helicopter = domain_helicopter(noise, discount)

   number_rollouts = int(query["transition"]["rollout count"])
   horizon = int(query["transition"]["horizon"])
   
   rollouts = []
   for rollout_number in range(number_rollouts):
       rollout = []
       helicopter.s0() # reset the state
       while not helicopter.isTerminal() and len(rollout) < horizon:
           action = helicopter.possibleActions()[0] # Always do the first action, todo, make better
           state = helicopter.state
           r, ns, terminal, possibleActions = helicopter.step(action)
           rollout.append(
             # Other state variables may be available
             {"xerr": state[0],
              "yerr": state[1],
              "zerr": state[2],
              "u": state[3],
              "v": state[4],
              "w": state[5],
              "p": state[6],
              "q": state[7],
              "r": state[8]})
       rollouts.append(rollout)
   return rollouts

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
