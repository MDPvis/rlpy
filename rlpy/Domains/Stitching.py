"""Construct a surrogate model from elementary state transitions
of another problem domain.


**REFERENCE:**
Based on `Batch Mode Reinforcement Learning based on the
Synthesis of Artificial Trajectories <https://goo.gl/1yveeS>`_
"""

from sklearn.neighbors import BallTree
from scipy.optimize import minimize
import numpy as np
from .Domain import Domain
import math
import os.path
import sys
from rlpy.Domains.StitchingPackage.benchmark import benchmark

__copyright__ = "Copyright 2015, Sean McGregor"
__credits__ = ["Sean McGregor"]
__license__ = "BSD 3-Clause"
__author__ = ["Sean McGregor"]

class MahalanobisDistance(object):
    """
    A class for optimizing a Mahalanobis distance metric.
    The metric is initialized to the identity matrix, which is equivalent to Euclidean distance.
    Calling the "optimize" function with sets of rollouts attempts to update the
    distance metric so that the objective function is minimized
    http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.mahalanobis.html
    """
    def __init__(self, var_count, stitching, target):
        """
        :param var_count: The number of variables in the distance metric.
        :param stitching: The Stitching class whose distance metric we are attempting to update.
        :param target: The rollouts whose distribution we are attempting to approximate.
          These will be used to repeatedly evaluate the visual fidelity objective.
        """
        self.distance_metric = np.identity(var_count)
        self.stitching = stitching
        self.target = target

    prng_for_policy = np.random.RandomState(0)

    @staticmethod
    def random_policy(s, actions):
        """Default to a random action selection"""
        return MahalanobisDistance.prng_for_policy.choice(actions)

    @staticmethod
    def loss(flat_metric, stitching, target):
        """
        The function we are trying to minimize when updating the distance metric.
        """
        old_tree = stitching.tree
        length = int(math.sqrt(len(flat_metric)))
        matrix = []
        for i in range(length):
            matrix.append([])
        for idx, val in enumerate(flat_metric):
            matrix[int(idx/length)].append(val)

        stitching.tree = BallTree(stitching.database, metric="mahalanobis", VI=np.array(matrix))

        labels = ["x", "xdot"] # todo, get the labels another way
        count = min(20, len(stitching.database)) # todo, pick more sensible value
        horizon = min(5, int(len(stitching.database)/count))  # todo, pick more sensible value
        policy = MahalanobisDistance.random_policy
        target = stitching.getRollouts(labels, count, horizon, policy = policy, domain=stitching.domain)
        synthesized_rollouts = stitching.getRollouts(labels, count, horizon, policy = policy, domain=stitching)

        total_benchmark = 0
        x_bench = benchmark(target, synthesized_rollouts, "x") # todo, iterate over the dictionary keys
        total_benchmark += x_bench
        stitching.tree = old_tree
        return total_benchmark

    def optimize(self):
        """
        Update the distance metric to optimize the 
        """
        res = minimize(MahalanobisDistance.loss, self.distance_metric, args=(self.stitching, self.target))

        # The result was flattened, need to make square
        length = int(math.sqrt(len(res.x)))
        matrix = []
        for i in range(length):
            matrix.append([])
        for idx, val in enumerate(res.x):
            matrix[int(idx/length)].append(val)
        self.distance_metric = matrix

    def get_matrix(self):
        """
        Return the current distance metric.
        """
        return np.array(self.distance_metric)

class TransitionTuple(tuple):
    """
    A Simple tuple class for storing state transitions in the KD tree [0].
    The object holds the tuple for the pre-transition state that will be stitched to
    in the current post-transition state. The class contains properties not in the
    tuple:

    * preState: The state we might stitch to, this is also represented as a tuple.
    * postState: What state resulted from the pre-transition state.
    * isTerminal: An indicator for whether the transitioned to state is terminal.
    * isInitial: An indicator for whether the pre-transition state is an initial state.
    * reward: The reward for taking the action.
    * possibleActions: What actions can be taken in the resulting state.

    [0] http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree
    """
    def __new__(cls, preState, postState, isTerminal, isInitial, reward, possibeActions):
        t = tuple.__new__(cls, tuple(preState))
        t.preState = preState
        t.postState = postState
        t.isTerminal = isTerminal
        t.isInitial = isInitial
        t.reward = reward
        t.possibleActions = possibeActions
        t.last_accessed_iteration = -1 # determines whether it is available for stitching
        return t

class Stitching(Domain):
    """
    This "domain" produces a surrogate for arbitrary MDP domains using trajectory synthesis.
    The domain is constructed by sampling an MDP domain and using its state
    transitions as components of synthesized transitions. This class takes
    a domain as input, but its transitions are only called in the construction of the
    database and distance metric. All transitions from this domain are from
    stitching.\n

    **STATE:**        Inherited from domain. \n
    **ACTIONS:**      Selected according to the current policy. \n
    **TRANSITIONS:**  Sampled according to the distance metric. \n
    **REWARD:**       Sampled according to the distance metric. \n
    **DATABASE:**     A set of state transitions that are sampled according to the
                      generating policy. \n
    """

    def __init__(self,
      domain,
      rolloutCount = 100,
      horizon = 100,
      generatingPolicies = [],
      trainingPolicies = [],
      stitchingToleranceSingle = .1,
      stitchingToleranceCumulative = .1,
      searchDistance = 0, # The ball from which a random point will be returned
      seed = None,
      database = None
    ):
        """
        :param domain: The domain used to generate MC rollouts
        :param rolloutCount: The number of rollouts to generate for each policy
        :param horizon: The maximum number of transitions for each rollout
        :param generatingPolicies: The policies that are used to populate the transition
          database.
        :param trainingPolicies: The policies that are used to learn a distance metric.
        :param stitchingToleranceSingle:How much distance a transition
          can stitch before it fails.
        :param stitchingToleranceCumulative: How much distance the trajectory
          can stitch before the entire trajectory is a failure.
        :param seed: The random seed. Defaults to random.
        """
        self.domain = domain
        self.rolloutCount = rolloutCount
        self.horizon = horizon
        self.database = []

        # Counter used to determine which set of rollouts are being generated.
        # This ensures states are stitched without replacement only for the
        # current set of rollouts.
        self.rolloutSetCounter = -1

        def randomPolicy(s, actions):
            """Default to a random action selection"""
            return self.random_state.choice(actions)
        self.generatingPolicies = generatingPolicies
        if not generatingPolicies:
            self.generatingPolicies = [randomPolicy]
        self.trainingPolicies = trainingPolicies
        if not trainingPolicies:
            self.trainingPolicies = [randomPolicy] 
        self.stitchingToleranceSingle = stitchingToleranceSingle
        self.stitchingToleranceCumulative = stitchingToleranceCumulative

        self.searchDistance = searchDistance
        self.random_state = np.random.RandomState(seed)
        
        # http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.spatial.KDTree.html
        sys.setrecursionlimit(10000)
        if database:
            database = self.database
        else:
            self._populateDatabase()

        # todo, change these data to target a different distribution than the database
        # todo, the metric is probably not optimizing because it can exactly reconstruct rollouts.
        #       There is no signal to change the distance when everything is exact.
        mahalanobis_metric = MahalanobisDistance(3, self, self.database)
        self.tree = BallTree(self.database, metric="mahalanobis", VI=mahalanobis_metric.get_matrix()) # Create the Ball-tree
        mahalanobis_metric.optimize()
        self.tree = BallTree(self.database, metric="mahalanobis", VI=mahalanobis_metric.get_matrix())

    def _populateDatabase(self):
        """
        Load many transitions into the database then create the KD-Tree.
        """
        for policy in self.generatingPolicies:
            for rolloutNumber in range(self.rolloutCount):
                self.domain.s0() # reset the state
                currentDepth = 0
                while not self.domain.isTerminal() and currentDepth < self.horizon:
                    state = self.domain.state
                    possible_actions = self.domain.possibleActions()
                    action = policy(state, possible_actions)# todo, make this an actual policy
                    state = np.append(state, action)
                    r, ns, terminal, nextStatePossibleActions = self.domain.step(action)
                    ns = self.domain.state # The helicopter domain is partially observable so we need to grab the full state
                    t = TransitionTuple(state, ns, terminal, (currentDepth == 0), r, nextStatePossibleActions)
                    self.database.append(t)
                    currentDepth += 1

    def _getClosest(self, state, a, k=1):
        """
        returns (at random) one of the closest k point from the KD tree.
        :param state: The current state of the world that we want the closest transition for.
        :param a: The selected action for the current state.
        :return: ``TransitionTuple`` The selected transition from the database.
        """
        q = list(state)
        q.append(a)
        k = min(k, len(self.database))
        (distances_array, indices_array) = self.tree.query(q,
          k=k,
          return_distance=True,
          sort_results=True)
        indices = indices_array[0]
        for i in indices:
            if self.database[i].last_accessed_iteration != self.rolloutSetCounter:
                self.database[i].last_accessed_iteration = self.rolloutSetCounter
                return self.database[i]
        if k < 1000 and k < len(self.database):
            return self._getClosest(state, a, k=k*10)
        raise Exception("There were no valid points within {} points".format(k))

    prng_for_policy = np.random.RandomState(0)
    def random_policy(self, s, actions):
        """Default to a random action selection"""
        return prng_for_policy.choice(actions)

    def getRollouts(self, labels, count, horizon, policy=None, domain=None):
        """
            Helper function for generating rollouts from all the domains.
            Args:
                labels (list(String)): A list of the state labels.
                count (integer): The number of rollouts to generate.
                horizon (integer): The maximum length of rollouts.
                policy (function(state, actions)): The function used to select an action.
                domain (Domain): The domain that will be called to generate rollouts.
        """
        if not policy:
            policy = self.random_policy
        if not domain:
            domain = self

        self.rolloutSetCounter += 1

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

    def possibleActions(self):
        """
        :return: ``List`` of the currently available actions.
        """
        return self.currentPossibleActions

    def step(self, a):
        """
        Take the action *a*, update the state variable and return the reward,
        state, whether it is terminal, and the set of possible actions.
        """
        postTransitionObject = self._getClosest(self.state, a)
        r = postTransitionObject.reward
        self.currentPossibleActions = postTransitionObject.possibleActions
        self.terminal = postTransitionObject.isTerminal
        self.state = postTransitionObject.postState
        return r, self.state.copy(), self.terminal, self.currentPossibleActions

    def s0(self):
        """
        Get a starting state from the domain. This gets an actual starting state from the
        domain under the assumption that these states are efficiently accessible.
        If the starting state is not efficiently accessible from the true domain simulator,
        then they could be cached for repeated use under many different policies.
        :return: ``state`` for the starting state.
        """
        self.partially_observed_state, self.terminal, self.currentPossibleActions = self.domain.s0()
        self.state = self.domain.state # in the helicopter domain, s0 doesn't return the full state
        return self.state.copy(), self.terminal, self.currentPossibleActions

    def isTerminal(self):
        """
        :return: ``True`` if the agent has reached or exceeded the goal position.
        """
        return self.terminal
