"""
Cross-origin MDPVis Server
===================
This is a minimal server for integrating MDPVis with an RLPy domain.

:author: Sean McGregor.
"""
from flask import Flask, jsonify, request, redirect, json
from flask.ext.cors import cross_origin
import argparse
import os.path
import random
import numpy as np

from rlpy.Domains import GridWorld as domain_gridworld
from rlpy.Domains import MountainCar as domain_mountain_car
from rlpy.Domains import HIVTreatment as domain_HIV

print "Starting Flask Server at http://localhost:8938"
app = Flask('rlpy', static_folder='.', static_url_path='')
parser = argparse.ArgumentParser(description='Start the RLPy server.')
parser.add_argument('domain', metavar='D', type=str,
                    help='the domain to synthesize trajectories for',
                    default='mountaincar')
args = vars(parser.parse_args())
domain_name = args["domain"] # mountaincar, hiv, gridworld

def policy(possibleActions, state):
    """
    Default random policy
    """
    return random.choice(possibleActions)
def policy_factory(parameters):
    """
    @override
    """
    return policy

if domain_name == "mountaincar":
    domain = domain_mountain_car
    def domain_constructor(parameters):
        return domain(float(parameters["noise"]))
    def label_state(state, action):
        return {"x": state[0], "xdot": state[1]}
    def domain_policy_factory(parameters):
        def policy(possibleActions, state):
            if state[1] > 0: # xdot
                return 2 # right
            else:
                return 0 # left
        return policy
    policy_factory = domain_policy_factory
    print policy_factory("")
elif domain_name == "hiv":
    domain = domain_HIV
    def domain_constructor(parameters):
        return domain()
    def label_state(state, action):
        return {"T1": state[0], "T1*": state[1], "T2": state[2], "T2*": state[3], "V": state[4], "E": state[5]}
    rs = np.random.RandomState(0)
    def domain_policy_factory(parameters, rs=rs):
        rti_probability = float(parameters["RTI"])
        pi_probability = float(parameters["PI"])
        def policy_reinforce(possibleActions, state, rti_probability=rti_probability, pi_probability=pi_probability, rs=rs):
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
            if rti and pi:
                return 3 # *3*: RTI and PI active
            elif rti:
                return 1 # *1*: RTI active
            elif pi:
                return 2 # *2*: PI active
            else:
                return 0 # *0*: none active
        return policy_reinforce
    policy_factory = domain_policy_factory
elif domain_name ==  "gridworld":
    domain = domain_gridworld
    def domain_constructor(parameters):
        maze = os.path.join(domain_gridworld.default_map_dir, '4x5.txt')
        return domain(maze, noise=float(parameters["noise"]))
    def label_state(state, action):
        up = 0
        down = 0
        left = 0
        right = 0
        if action == 0:
            up = 1
        elif action == 1:
            down = 1
        elif action == 2:
            left = 1
        elif action == 3:
            right = 1
        return {"x": state[0], "y": state[1], "up": up, "down": down, "left": left, "right": right}
    def domain_policy_factory(parameters):
        def policy(possibleActions, state):
            rv = random.random()
            if rv > .4:
                return 0 # up
            elif rv > .3:
                return 1 # down
            elif rv > .2:
                return 2 # left
            else:
                return 3 # right
        return policy
    policy_factory = domain_policy_factory

def _get_rollouts(parameters):
    number_rollouts = int(parameters["Sample Count"])
    horizon = int(parameters["Horizon"])
    domain_object = domain_constructor(parameters)
    policy = policy_factory(parameters)

    rollouts = []
    for rollout_number in range(number_rollouts):
        rollout = []
        domain_object.s0() # reset the state
        while not domain_object.isTerminal() and len(rollout) < horizon:
            action = policy(domain_object.possibleActions(), domain_object.state)
            state_dictionary = label_state(domain_object.state, action)
            rollout.append(state_dictionary)
            domain_object.step(action)
        rollouts.append(rollout)
    return rollouts

def get_domain_specific_parameters():
    # The initialization object for MDPvis
    mdpvis_initialization_object = {

        # The control panels that appear at the top of the screen
        "parameter_collections": [
            {
                "panel_title": "Sampling Effort",
                "panel_icon": "glyphicon-retweet",
                "panel_description": "Define how many trajectories you want to generate, and to what time horizon.",
                "quantitative": [  # Real valued parameters
                                   {
                                       "name": "Sample Count",
                                       "description": "Specify how many trajectories to generate",
                                       "current_value": 10,
                                       "max": 1000,
                                       "min": 1,
                                       "step": 10,
                                       "units": "#"
                                   },
                                   {
                                       "name": "Horizon",
                                       "description": "The time step at which simulation terminates",
                                       "current_value": 10,
                                       "max": 10000,
                                       "min": 1,
                                       "step": 10,
                                       "units": "Time Steps"
                                   }
                ]
            }
        ]
    }
    if domain_name == "mountaincar" or domain_name == "gridworld":
        domain_controls = {
            "panel_title": "Domain Parameters",
            "panel_icon": "glyphicon-adjust",
            "panel_description": "Define the noise parameter for Mountain Car.",
            "quantitative": [  # Real valued parameters
                               {
                                   "name": "noise",
                                   "description": "Specify the percentile of a random action",
                                   "current_value": 20,
                                   "max": 100,
                                   "min": 0,
                                   "step": 1,
                                   "units": "#"
                               }
            ]
        }
        mdpvis_initialization_object["parameter_collections"].append(domain_controls)
    elif domain_name == "hiv":
        domain_controls = {
            "panel_title": "Policy Parameters",
            "panel_icon": "glyphicon-adjust",
            "panel_description": "Define the policy parameter for HIV.",
            "quantitative": [  # Real valued parameters
                               {
                                   "name": "RTI",
                                   "description": "The probability of administering RTI",
                                   "current_value": .2,
                                   "max": 1,
                                   "min": 0,
                                   "step": .1,
                                   "units": "#"
                               },
                               {
                                   "name": "PI",
                                   "description": "The probability of administering PI",
                                   "current_value": .2,
                                   "max": 1,
                                   "min": 0,
                                   "step": .1,
                                   "units": "#"
                                }
            ]
        }
        mdpvis_initialization_object["parameter_collections"].append(domain_controls)
    if domain_name == "mountaincar":
        domain_controls = {
            "panel_title": "Policy Parameters",
            "panel_icon": "glyphicon-adjust",
            "panel_description": "Define the policy parameter for Mountain Car.",
            "quantitative": [  # Real valued parameters
                               {
                                   "name": "reinforce direction",
                                   "description": "The probability of reinforcing the direciton of travel",
                                   "current_value": .2,
                                   "max": 1,
                                   "min": 0,
                                   "step": .1,
                                   "units": "#"
                               }
            ]
        }
        mdpvis_initialization_object["parameter_collections"].append(domain_controls)
    return mdpvis_initialization_object

print """
Starting Flask Server...
"""

app = Flask('RLPy', static_folder='.', static_url_path='')

@app.route("/initialize", methods=['GET'])
@cross_origin(allow_headers=['Content-Type'])
def cross_origin_initialize():
    '''
        Asks the domain for the parameters to seed the visualization.
    '''
    return jsonify(get_domain_specific_parameters())

@app.route("/trajectories", methods=['GET'])
@cross_origin(allow_headers=['Content-Type'])
def cross_origin_rollouts():
    '''
        Asks the domain for the rollouts generated by the
        requested parameters.
    '''
    rollouts = _get_rollouts(request.args)
    json_obj = {"trajectories": rollouts}
    resp = jsonify(json_obj)
    return resp

@app.route("/optimize", methods=['GET'])
@cross_origin(allow_headers=['Content-Type'])
def cross_origin_optimize():
    '''
        Asks the domain to optimize for the current parameters, then return the
        new set of parameters for the optimized policy.
    '''
    raise NotImplementedError # Hook this into your optimizer

@app.route("/state", methods=['GET'])
@cross_origin(allow_headers=['Content-Type'])
def cross_origin_state():
    '''
        Ask the domain for the referenced state and state snapshots.
    '''
    raise NotImplementedError # Create your own rollout visualizations here

# Binds the server to port 8938 and listens to all IP addresses.
if __name__ == "__main__":
    print("Starting server...")
    app.run(host='0.0.0.0', port=8938, debug=True, use_reloader=False, threaded=True)
    print("...started")
