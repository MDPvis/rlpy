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
import pickle
from rlpy.Domains.StitchingPackage.benchmark import Benchmark
from rlpy.Domains.StitchingPackage.TransitionTuple import TransitionTuple

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
    def __init__(self,
        var_count,
        stitching,
        target_policies=[],
        normalize_starting_metric=True,
        cached_metric=None):
        """
        :param var_count: The number of variables in the distance metric.
        :param stitching: The Stitching class whose distance metric we are attempting to update.
        :param target_rollouts: The rollouts whose distribution we are attempting to approximate.
          These will be used to repeatedly evaluate the visual fidelity objective.
        :param normalize_starting_metric: Determines whether we scale the metric by the magnitude of the variable's mean.
        :param cached_metric: A pre-computed metric, probably loaded from a file.
        """
        if not (cached_metric is None):
            self.distance_metric = cached_metric
            #return # todo, am I supposed to remove this?
        else:
            self.distance_metric = np.identity(var_count)

        self.stitching = stitching

        self.target_policies = target_policies
        self._sampleTargetTrajectories()

        if normalize_starting_metric:
            for idx, variable in enumerate(stitching.labels):
                l = []
                for rollout_set in self.target_rollouts:
                    for rollout in rollout_set:
                        for event in rollout:
                            l.append(event[variable])
                variance = Benchmark.variance(l)
                if variance == 0:
                    variance = 1.0
                self.distance_metric[idx][idx] = 1.0/variance
            if self.stitching.writeNormalizedMetric is not None:
                f = open(self.stitching.writeNormalizedMetric, "wb")
                met = pickle.dump(self.distance_metric, f)
                f.close()

        # The Powell optimizer needs a non-zero value to find the emprical gradient in log space
        for idx, row in enumerate(self.distance_metric):
            for idx2, column in enumerate(row):
                if idx != idx2:
                    pass
                    # todo: additional analysis of this
                    #self.distance_metric[idx][idx2] = .00000000000000001

    def _sampleTargetTrajectories(self):
        """
        Assign the target trajectories and build the benchmark.
        :param policies:
        :return:
        """
        self.target_policies = self.stitching.targetPolicies
        self.target_rollouts = []
        for policy in self.target_policies:
            t = self.stitching.getRollouts(
                count=self.stitching.targetPoliciesRolloutCount,
                horizon=self.stitching.horizon,
                policy=policy,
                domain=self.stitching.domain)
            self.target_rollouts.append(t)

        self.benchmarks = []
        for idx, rollouts in enumerate(self.target_rollouts):
            benchmark = Benchmark(rollouts, self.stitching.action_count, seed=0)
            self.benchmarks.append(benchmark)

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
             benchmarks,
             benchmark_rollout_count=50):
        """
        The function we are trying to minimize when updating the distance metric.
        :param flat_metric: The metric represented as a list of values. This will be converted to
          a matrix when computing distances.
        :param stitching: The Stitching class whose distance metric we are attempting to update.
        :param benchmarks: Instances of the Benchmark class.
        :param self: A hack to make this staticmethod behave more like the MahalanobisDistance class.
          Loss needs to be a static method for the minimization library, but we can still pass in
          the MahalanobisDistance object as self.
        """
        old_tree = stitching.tree
        matrix = MahalanobisDistance.unflatten(MahalanobisDistance.ceiling_exponentiate(flat_metric))
        stitching.tree = BallTree(stitching.database, metric="mahalanobis", VI=np.array(matrix))

        total_benchmark = 0

        # Benchmark against the horizon and target policies of the stitching domain
        rolloutCount = benchmark_rollout_count
        if stitching.rolloutCount < benchmark_rollout_count:
            pass
            #rolloutCount = stitching.rolloutCount
            #print "WARNING!!! You are attempting to find the loss for more trajectories than each DB policy generated"
        horizon = stitching.horizon
        policies = stitching.targetPolicies
        for idx, policy in enumerate(policies):
            benchmark = benchmarks[idx]
            current_benchmark = 0.0
            synthesized_rollouts = stitching.getRollouts(
              count=rolloutCount,
              horizon=horizon,
              policy=policy,
              domain=stitching)
            for label in stitching.labels:
                variable_benchmark = benchmark.benchmark_variable(synthesized_rollouts, label, square=True)
                current_benchmark += variable_benchmark
            action_benchmark = benchmark.benchmark_actions(synthesized_rollouts, square=True)
            current_benchmark += action_benchmark
            total_benchmark += current_benchmark # Square the loss from this policy
        stitching.tree = old_tree

        return total_benchmark

    def optimize(self):
        """
        Optimize and save the distance metric in non-exponentiated form.
        """

        # The loss function will exponentiate the solution, so our starting point should
        # be the natural log of the solution.
        inverse_exponentiated = MahalanobisDistance.ceiling_logarithm(
          MahalanobisDistance.flatten(self.distance_metric))

        # todo: investigate whether saving in this manner is necessary by removing
        # the save and running the tests
        def print_and_save(vec):
            loss = MahalanobisDistance.loss(
                vec,
                self.stitching, self.benchmarks)
            print "==Optimization iteration complete=="
            print vec
            print "LOSS:"
            print loss
            if loss < print_and_save.best_loss:
                print_and_save.best_loss = loss
                print_and_save.best_parameters = vec

        print_and_save.best_loss = float("Inf")
        print_and_save.best_parameters = []

        res = minimize(
          MahalanobisDistance.loss,
          inverse_exponentiated,
          args=(self.stitching, self.benchmarks),
          method="Powell",
          tol=.000000001,
          options={"disp": True},
          callback=print_and_save)

        print res

        # The result was flattened, need to make square
        matrix = MahalanobisDistance.unflatten(MahalanobisDistance.ceiling_exponentiate(print_and_save.best_parameters))

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
      labels = None,
      metricFile = None,
      optimizeMetric = True,
      writeNormalizedMetric = None
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

        self.writeNormalizedMetric = writeNormalizedMetric

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
            self.database = database
        else:
            self._populateDatabase()

        # Count the total number of state variables and discrete actions
        metric_size = self.action_count + len(self.labels)  # actions and state variables

        # Check the cache for the metric file
        if metricFile and os.path.isfile(metricFile):
            print "Using cached metric, delete file or change metric version to optimize a new one"
            if optimizeMetric:
                print "WARNING: You loaded a pre-existing metric that will not be optimized"
            f = open(metricFile, "r")
            met = pickle.load(f)
            self.mahalanobis_metric = MahalanobisDistance(
                metric_size,
                self,
                target_policies=self.targetPolicies,
                normalize_starting_metric=False,
                cached_metric=met
            )
            f.close()
        else:

            self.mahalanobis_metric = MahalanobisDistance(metric_size, self, self.targetPolicies)

            self.tree = BallTree(
              self.database,
              metric="mahalanobis",
              VI=self.mahalanobis_metric.get_matrix_as_np_array())

            if optimizeMetric:
                self.optimize_metric()

            # Cache the learned metric
            if metricFile:
                f = open(metricFile, "wb")
                met = pickle.dump(self.mahalanobis_metric.distance_metric, f)
                f.close()

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
        Load many transitions into the database. This assumes all the state variables (minus reward) are included
        in the distance metric.
        """
        for policy in self.databasePolicies:
            for rolloutNumber in range(self.rolloutCount):
                self.domain.s0() # reset the state
                currentDepth = 0
                terminating = False
                while not terminating and currentDepth < self.horizon:
                    terminating = self.domain.isTerminal()
                    state = self.domain.state
                    possible_actions = self.domain.possibleActions()
                    action = policy(state, possible_actions)
                    action_indicator = [0] * self.action_count
                    action_indicator[action] = 1
                    preState = np.append(state, action_indicator)
                    r, ns, terminal, nextStatePossibleActions = self.domain.step(action)

                    visualizationState = {"reward": r, "action":action}
                    for idx, label in enumerate(self.labels):
                        visualizationState[label] = state[idx]

                    # The helicopter domain is partially observable so we need to grab the full state
                    ns = self.domain.state
                    t = TransitionTuple(
                        preState,
                        ns,
                        terminal,
                        (currentDepth == 0),
                        nextStatePossibleActions,
                        visualizationState # this needs to include all the state variables, including the actions taken
                    )
                    self.database.append(t)
                    currentDepth += 1


    def _getClosest(self, preStateDistanceMetricVariables, k=1):
        """
        returns (at random) one of the closest k point from the KD tree.
        :param preStateDistanceMetricVariables: The current state of the world that we want the closest transition for.
        :return: ``(TransitionTuple, distance)`` The selected transition from the database and the
          distance to that transition.
        """
        q = list(preStateDistanceMetricVariables)

        k = min(k, len(self.database))
        (distances_array, indices_array) = self.tree.query(q,
          k=k,
          return_distance=True,
          sort_results=True)
        indices = indices_array[0]
        for index, i in enumerate(indices):
            if self.database[i].last_accessed_iteration != self.rolloutSetCounter:
                return (self.database[i], distances_array[0][index])
        if k < 10000 and k < len(self.database):
            return self._getClosest(preStateDistanceMetricVariables, k=k*10)
        raise Exception("There were no valid points within "\
        "{} points in a database of {} points. This failure occured when "\
        "attempting to generate rollout set {}".format(k, len(self.database), self.rolloutSetCounter))

    def _getEqualTransitions(self, preStateDistanceMetricVariables, k):
        """
        Mark the K closest tuples as being expended after using the state from a different action.
        :param postStateDistanceMetricVariables: The current state of the world that we want the closest transition for.
        :param k: The number of Tuples to return, should equal the number of actions.
        :return: ``[TransitionTuple,...]`` The selected transitions from the database.
        """
        q = list(preStateDistanceMetricVariables)

        k = min(k, len(self.database))
        (distances_array, indices_array) = self.tree.query(q,
                                                           k=k,
                                                           return_distance=True,
                                                           sort_results=True)

        for distance in distances_array:
            assert distance == 0.0

        nearbyActions = []
        for idx in indices_array:
            assert self.database[idx].visualizationStateVariables["action"] not in nearbyActions
            nearbyActions.append(self.database[idx].visualizationStateVariables["action"])

        indices = indices_array[0]
        transitions = []
        for index, i in enumerate(indices):
            assert self.database[i].last_accessed_iteration != self.rolloutSetCounter
            self.database[i].last_accessed_iteration = self.rolloutSetCounter
            transitions.append(self.database[i])
        return transitions


    def getRollouts(self, count=10, horizon=10, policy=None, domain=None):
        """
            Helper function for generating rollouts from all the domains.
            Args:
                count (integer): The number of rollouts to generate.
                horizon (integer): The maximum length of rollouts.
                policy (function(state, actions)): The function used to select an action.
                domain (Domain): The domain that will be called to generate rollouts.
        """
        if not domain:
            domain = self

        self.rolloutSetCounter += 1

        self.totalStitchingDistance = 0
        self.totalNonZeroStitches = 0

        totalTransitions = 0

        rollouts = []
        for rollout_number in range(count):
            rollout = []
            domain.s0() # reset the state
            terminate = False
            while not terminate and len(rollout) < horizon:
                terminate = domain.isTerminal()
                totalTransitions += 1
                if type(domain).__name__ == "Stitching":
                    self.totalNonZeroStitches += 1
                    r, ns, terminal, currentPossibleActions = domain.step(policy)

                    #state["stitch distance"] = self.lastStitchDistance
                    self.totalStitchingDistance += self.lastStitchDistance

                    state = {"action": self.lastEvaluatedAction, "reward": r}
                    assert state["action"] >= 0
                    for label in self.labels:
                        state[label] = ns[label]
                    rollout.append(state)
                else:
                    action = policy(self.domain.state, self.domain.possibleActions(self.domain.state))
                    state = {}
                    for i in range(len(self.labels)):
                        state[self.labels[i]] = self.domain.state[i]
                    state["action"] = action
                    assert action >= 0
                    #state["stitch distance"] = 0.0
                    r, ns, terminal, currentPossibleActions = domain.step(action)
                    state["reward"] = r
                    rollout.append(state)
            rollouts.append(rollout)
        print "Returning rollouts with {} lossy stitched transitions for {} total transitions".format(self.totalNonZeroStitches, totalTransitions)
        return rollouts

    def possibleActions(self):
        """
        :return: ``List`` of the currently available actions.
        """
        return self.currentPossibleActions

    def step(self, policy, biasCorrected=False, actionsInDistanceMetric=True):
        """
        Find the closest transition matching the policy, then find all the transitions matching the pre-state
        and remove them from the database.

        biasCorrected (bool): Whether the database has sampled every action at every state in the database.
            This will determine whether we remove all matching states when we sample one.
        actionsInDistanceMetric (bool): Indicates whether the distance metric needs indicator variables for each of the
            actions. This will force the policy to be evaluated before stitching.
        """
        # We can't correct the bias if the actions are in the distance metric
        assert biasCorrected != actionsInDistanceMetric
        pre = self.preStateDistanceMetricVariables.copy()

        possibleActions = []
        for idx in range(0,self.domain.actions_num):
            possibleActions.append(idx)
        if actionsInDistanceMetric:
            #action = policy(pre, possibleActions)
            action = policy(pre, self.currentPossibleActions)
            assert action >= 0
            for idx, possibleAction in enumerate(possibleActions):
                if action == idx:
                    pre = np.append(pre, 1.)
                else:
                    pre = np.append(pre, 0.)
        (postStitchObject, stitchDistance) = self._getClosest(pre)
        self.lastStitchDistance = stitchDistance # because we can't change the return signature
        assert stitchDistance >= 0

        if biasCorrected:
            transitions = self._getEqualTransitions(postStitchObject, len(possibleActions))# remove all exactly matching states to avoid bias
            action = policy(postStitchObject.visualizationStateVariables, possibleActions)
            assert action >= 0
            for candidate in transitions:
                if candidate.visualizationStateVariables["action"] == action:
                    transition = candidate
                    break
        else:
            transition = postStitchObject
        self.lastEvaluatedAction = action
        self.preStateDistanceMetricVariables = transition.postStateDistanceMetricVariables
        r = transition.visualizationResultState["reward"]
        self.currentPossibleActions = transition.possibleActions
        self.terminal = transition.isTerminal
        self.state = transition.visualizationResultState
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
        self.state = {} # There is no state until it is stitched and the complete state is recovered
        if hasattr(self.domain, "INIT_STATE"):
            self.preStateDistanceMetricVariables = self.domain.INIT_STATE
        else:
            self.preStateDistanceMetricVariables = self.domain.start_state.copy()
        return self.state.copy(), self.terminal, self.currentPossibleActions

    def isTerminal(self):
        """
        :return: ``True`` if the agent has reached or exceeded the goal position.
        """
        return self.terminal
