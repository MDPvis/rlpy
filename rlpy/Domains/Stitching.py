"""Construct a surrogate model from elementary state transitions
of another problem domain.


**REFERENCE:**
Based on `Batch Mode Reinforcement Learning based on the
Synthesis of Artificial Trajectories <https://goo.gl/1yveeS>`_
"""

from sklearn.neighbors import BallTree
from scipy.optimize import minimize
from scipy import linalg
import numpy as np
from .Domain import Domain
import math
import os.path
import sys
from rlpy.Domains.StitchingPackage.benchmark import Benchmark

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
    def __init__(self, var_count, stitching, target_rollouts, normalize_starting_metric=False):
        """
        :param var_count: The number of variables in the distance metric.
        :param stitching: The Stitching class whose distance metric we are attempting to update.
        :param target: The rollouts whose distribution we are attempting to approximate.
          These will be used to repeatedly evaluate the visual fidelity objective.
        """
        self.distance_metric = np.identity(var_count)
        self.stitching = stitching
        self.target_rollouts = target_rollouts
        if normalize_starting_metric:
            for idx, variable in enumerate(stitching.labels):
                total = 0.
                count = 0.
                for rollout in target_rollouts:
                    for event in rollout:
                        total += event[variable]
                        count += 1.
                average = abs(total/count)
                rooted = math.sqrt(1./average)
                self.distance_metric[idx][idx] = rooted

        # The Powell optimizer needs a non-zero value to find the emprical gradient in log space
        for idx, row in enumerate(self.distance_metric):
            for idx2, column in enumerate(row):
                if idx != idx2:
                    pass
                    # todo: additional analysis of this
                    #self.distance_metric[idx][idx2] = .00000000000000001

    @staticmethod
    def flatten(matrix):
        """
        Return the current distance metric as a single list of values.
        """
        flattened = []
        for row in matrix:
            for item in row:
                flattened.append(item)
        return flattened

    @staticmethod
    def unflatten(flat_metric):
        """
        Return the current distance metric as a list of lists.
        """
        length = int(math.sqrt(len(flat_metric))) # Get the size of the matrix
        matrix = []
        for i in range(length):
            matrix.append([])
        for idx, val in enumerate(flat_metric):
            matrix[int(idx/length)].append(val)
        return matrix

    @staticmethod
    def is_upper_triangular(matrix):
        """
        Checks whether all lower triangular values are zero.
        Return True iff all lower triangular values are zero.
        """
        assert type(matrix[0]) == list
        for row_idx, row in enumerate(matrix):
            for col_idx, val in enumerate(row):
                if row_idx > col_idx and val != 0:
                    return False
        return True

    @staticmethod
    def is_psd(matrix):
        """
        Checks whether the current matrix is positive semi-definite
        by taking the Cholesky decomposition.
        Returns True iff SciPy succeeds in taking the Cholesky decomp.
        """
        if type(matrix) == list:
            matrix = np.array(matrix)
        try:
            L = linalg.cholesky(matrix, check_finite=True)
        except linalg.LinAlgError:
            return False
        else:
            return True

    @staticmethod
    def ceiling_exponentiate(flat_metric):
        """
        A list exponentiation function that maxes out at sys.float_info.max.
        """
        def new_exp(x):
            try:
                return math.exp(x)
            except Exception:
                if x < 0:
                    return 0
                else:
                    return sys.float_info.max
        return map(new_exp, flat_metric)

    @staticmethod
    def ceiling_logarithm(flat_metric):
        """
        Take the natural logarithm and allow zero values (give negative inf for zero values)
        """
        def new_log(x):
            assert x >= 0
            try:
                return math.log(x)
            except Exception:
                if x == 0:
                    return -sys.float_info.max
                else:
                    assert False # There should never be an under/over flow for this input
        return map(new_log, flat_metric)

    @staticmethod
    def loss(flat_metric,
             stitching,
             target_rollouts,
             self=None,
             favor_normalized_euclidean=False,
             benchmark_rollout_count=50):
        """
        The function we are trying to minimize when updating the distance metric.
        :param flat_metric: The metric represented as a list of values. This will be converted to
          a matrix when computing distances.
        :param stitching: The Stitching class whose distance metric we are attempting to update.
        :param target_rollouts: The rollouts whose distribution we are attempting to approximate.
          These will be used to repeatedly evaluate the visual fidelity objective.
        :param self: A hack to make this staticmethod behave more like the MahalanobisDistance class.
          Loss needs to be a static method for the minimization library, but we can still pass in
          the MahalanobisDistance object as self.
        """
        old_tree = stitching.tree
        matrix = MahalanobisDistance.unflatten(MahalanobisDistance.ceiling_exponentiate(flat_metric))
        stitching.tree = BallTree(stitching.database, metric="mahalanobis", VI=np.array(matrix))

        # Benchmark against the horizon and target policies of the stitching domain
        rolloutCount = min(stitching.rolloutCount, benchmark_rollout_count)
        horizon = stitching.horizon
        policies = stitching.targetPolicies
        synthesized_rollouts = stitching.getRollouts(
          count=rolloutCount,
          horizon=horizon,
          policies=policies,
          domain=stitching)

        total_benchmark = 0
        quantiles = [0,10,20,30,40,50,60,70,80,90,100]
        for label in stitching.labels + ["reward"]:
            bench = Benchmark.benchmark_variable(target_rollouts, synthesized_rollouts, label, quantiles=quantiles)
            total_benchmark += bench
        action_benchmark = len(quantiles) * Benchmark.benchmark_actions(target_rollouts, synthesized_rollouts, stitching.action_count)
        total_benchmark += action_benchmark
        stitching.tree = old_tree

        return total_benchmark

    def optimize(self):
        """
        Optimize and save the distance metric in non-exponentiated form.
        Improvements:
          todo: visualize optimized and non-optimized stitching
          todo: --- Regularize the objective? --- Maybe square loss as well?
          todo: add metrics for stitching distance and count and add tests for them
          todo: potentially reduce the number of actions in the distance metric by basing all the actions on a single action
          todo: run experiments where states can only be stitched if they have the same action
          todo: look up older papers on log-Cholesky and Cholesky, read them
          todo: Log cholesky \cite{Pinheiro1988} with powell, then maybe worry about lowe
        """

        # The loss function will exponentiate the solution, so our starting point should
        # be the natural log of the solution.
        inverse_exponentiated = MahalanobisDistance.ceiling_logarithm(
          MahalanobisDistance.flatten(self.distance_metric))

        def print_loss(vec):
            print "==Optimization iteration complete=="
            print vec
            print "LOSS:"
            print MahalanobisDistance.loss(
                vec,
                self.stitching, self.target_rollouts, self)

        res = minimize(
          MahalanobisDistance.loss,
          inverse_exponentiated,
          args=(self.stitching, self.target_rollouts, self),
          method="Powell",
          tol=.000000001,
          options={"disp": True},
          callback=print_loss)

        print res

        # The result was flattened, need to make square
        matrix = MahalanobisDistance.unflatten(MahalanobisDistance.ceiling_exponentiate(res.x))

        assert MahalanobisDistance.is_psd(matrix)
        self.distance_metric = matrix

    def get_matrix_as_np_array(self, matrix=None):
        """
        Return the current distance metric as a NumPy array.
        """
        if matrix:
            return np.array(matrix)
        else:
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
      databasePolicies = [],
      targetPolicies = [],
      targetPoliciesRolloutCount = 50,
      stitchingToleranceSingle = .1,
      stitchingToleranceCumulative = .1,
      seed = None,
      database = None,
      labels = ["x", "xdot"] # default to the MountainCar domain
    ):
        """
        :param domain: The domain used to generate MC rollouts
        :param rolloutCount: The number of rollouts to generate for each policy
        :param horizon: The maximum number of transitions for each rollout
        :param databasePolicies: The policies that are used to populate the transition
          database.
        :param targetPolicies: The policies that are used to learn a distance metric.
        :param targetPoliciesRolloutCount: The number of rollouts to generate for each of the
          target policies.
        :param stitchingToleranceSingle:How much distance a transition
          can stitch before it fails.
        :param stitchingToleranceCumulative: How much distance the trajectory
          can stitch before the entire trajectory is a failure.
        :param seed: The random seed. Defaults to random.
        """
        self.domain = domain
        self.rolloutCount = rolloutCount
        self.targetPoliciesRolloutCount = targetPoliciesRolloutCount
        self.horizon = horizon
        self.database = []
        self.labels = labels

        # Counter used to determine which set of rollouts are being generated.
        # This ensures states are stitched without replacement only for the
        # current set of rollouts.
        self.rolloutSetCounter = -1

        def randomPolicy(s, actions):
            """Default to a random action selection"""
            return self.random_state.choice(actions)
        self.databasePolicies = databasePolicies
        if not databasePolicies:
            self.databasePolicies = [randomPolicy]
        self.targetPolicies = targetPolicies
        if not targetPolicies:
            self.targetPolicies = [randomPolicy]
        self.stitchingToleranceSingle = stitchingToleranceSingle
        self.stitchingToleranceCumulative = stitchingToleranceCumulative

        self.random_state = np.random.RandomState(seed)

        # Action Space
        self.action_count = self.domain.actions_num

        # This might not be necessary with the Ball Tree
        # http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.spatial.KDTree.html
        sys.setrecursionlimit(10000)
        if database:
            database = self.database
        else:
            self._populateDatabase()

        # todo: this would be better put in the MahalanobisDistance class since it is only used for
        # optimizing the metric.
        target_rollouts = self.getRollouts(
              count=self.targetPoliciesRolloutCount,
              horizon=self.horizon,
              policies=self.targetPolicies,
              domain=self.domain)

        # Count the total number of state variables and discrete actions
        metric_size = self.action_count
        for key in target_rollouts[0][0]:
            if key != "action" and key != "reward":
                metric_size += 1

        self.mahalanobis_metric = MahalanobisDistance(metric_size, self, target_rollouts)
        self.tree = BallTree(
          self.database,
          metric="mahalanobis",
          VI=self.mahalanobis_metric.get_matrix_as_np_array())

    def optimize_metric(self):
        """
        Update the mahalanobis metric by attempting to optimize it
        for the already defined target rollouts, database, and policies.
        You should call this after initializing the stitching domain if you
        want better performance, but expect it to take a while to find a
        better metric.
        """
        self.mahalanobis_metric.optimize()
        self.tree = BallTree(
          self.database,
          metric="mahalanobis",
          VI=self.mahalanobis_metric.get_matrix_as_np_array())

    def _populateDatabase(self):
        """
        Load many transitions into the database then create the KD-Tree.
        """
        for policy in self.databasePolicies:
            for rolloutNumber in range(self.rolloutCount):
                self.domain.s0() # reset the state
                currentDepth = 0
                while not self.domain.isTerminal() and currentDepth < self.horizon:
                    state = self.domain.state
                    possible_actions = self.domain.possibleActions()
                    action = policy(state, possible_actions)
                    action_indicator = [0] * self.action_count
                    action_indicator[action] = 1
                    state = np.append(state, action_indicator)
                    r, ns, terminal, nextStatePossibleActions = self.domain.step(action)

                    # The helicopter domain is partially observable so we need to grab the full state
                    ns = self.domain.state
                    t = TransitionTuple(
                      state,
                      ns,
                      terminal,
                      (currentDepth == 0),
                      r,
                      nextStatePossibleActions)
                    self.database.append(t)
                    currentDepth += 1

    def _getClosest(self, state, a, k=1):
        """
        returns (at random) one of the closest k point from the KD tree.
        :param state: The current state of the world that we want the closest transition for.
        :param a: The selected action for the current state.
        :return: ``(TransitionTuple, distance)`` The selected transition from the database and the
          distance to that transition.
        """
        q = list(state)
        action_indicator = [0] * self.action_count
        action_indicator[a] = 1
        q = np.append(q, action_indicator)

        k = min(k, len(self.database))
        (distances_array, indices_array) = self.tree.query(q,
          k=k,
          return_distance=True,
          sort_results=True)
        indices = indices_array[0]
        for index, i in enumerate(indices):
            if self.database[i].last_accessed_iteration != self.rolloutSetCounter:
                self.database[i].last_accessed_iteration = self.rolloutSetCounter
                return (self.database[i], distances_array[0][index])
        if k < 1000 and k < len(self.database):
            return self._getClosest(state, a, k=k*10)
        raise Exception("There were no valid points within {} points".format(k))

    prng_for_policy = np.random.RandomState(0)
    def random_policy(self, s, actions):
        """Default to a random action selection"""
        return prng_for_policy.choice(actions)

    def getRollouts(self, count=10, horizon=10, policies=None, domain=None):
        """
            Helper function for generating rollouts from all the domains.
            Args:
                count (integer): The number of rollouts to generate.
                horizon (integer): The maximum length of rollouts.
                policy (function(state, actions)): The function used to select an action.
                domain (Domain): The domain that will be called to generate rollouts.
        """
        if not policies:
            policy = [self.random_policy]
        if not domain:
            domain = self

        self.rolloutSetCounter += 1

        self.totalStitchingDistance = 0

        rollouts = []
        for policy in policies:
            for rollout_number in range(count):
                rollout = []
                domain.s0() # reset the state
                while not domain.isTerminal() and len(rollout) < horizon:
                    possible_actions = domain.possibleActions()
                    action = policy(domain.state, possible_actions)
                    state = {}
                    for i in range(len(self.labels)):
                        state[self.labels[i]] = domain.state[i]
                    state["action"] = action
                    r, ns, terminal, currentPossibleActions = domain.step(action)
                    state["reward"] = r

                    # record stitching distance
                    if type(domain).__name__ == "Stitching":
                        self.totalStitchingDistance += self.lastStitchDistance
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
        (postTransitionObject, stitchDistance) = self._getClosest(self.state, a)
        self.lastStitchDistance = stitchDistance # because we can't change the return signature
        assert stitchDistance >= 0
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
