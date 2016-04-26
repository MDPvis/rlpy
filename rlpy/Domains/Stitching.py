"""Construct a surrogate model from elementary state transitions
of another problem domain.


**REFERENCE:**
Based on `Batch Mode Reinforcement Learning based on the
Synthesis of Artificial Trajectories <https://goo.gl/1yveeS>`_
"""

from sklearn.neighbors import BallTree
import numpy as np
from .Domain import Domain
import os.path
import sys
import pickle
from rlpy.Domains.StitchingPackage.TransitionTuple import TransitionTuple
from rlpy.Domains.StitchingPackage.MahalanobisDistance import MahalanobisDistance

__copyright__ = "Copyright 2015, Sean McGregor"
__credits__ = ["Sean McGregor"]
__license__ = "BSD 3-Clause"
__author__ = ["Sean McGregor"]

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
      writeNormalizedMetric = None,
      initializeMetric = True
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
        if initializeMetric:
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
            self.setDatabase(self.database)
        elif initializeMetric:

            self.mahalanobis_metric = MahalanobisDistance(metric_size, self, self.targetPolicies)

            self.setDatabase(self.database)

            if optimizeMetric:
                self.optimize_metric()

                # Cache the learned metric
                if metricFile:
                    f = open(metricFile, "wb")
                    met = pickle.dump(self.mahalanobis_metric.distance_metric, f)
                    f.close()

            self.setDatabase(self.database)

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

    def setDatabase(self, db):
        """
        Set the database and build the ball tree associated with it.
        :param db:
        :return:
        """
        self.database = db
        self.tree = BallTree(
            self.database,
            metric="mahalanobis",
            VI=self.mahalanobis_metric.get_matrix_as_np_array())

    def setMetric(self, mahaMetric):
        """
        Set the metric associated with the database. You must rebuild the ball tree after setting the metric.
        :param mahaMetric: An instance of the mahalanobis distance metric we want to assign.
        :return:
        """
        self.mahalanobis_metric = mahaMetric

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
        q = np.array(preStateDistanceMetricVariables)
        q = q.reshape(1,-1)
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

    def _getBiasSet(self, preStateDistanceMetricVariables, k):
        """
        Mark the K closest tuples as being expended after using the state from a different action.
        :param postStateDistanceMetricVariables: The current state of the world that we want the closest transition for.
        :param k: The number of Tuples to return, should equal the number of actions.
        :return: ``[TransitionTuple,...]`` The selected transitions from the database.
        """
        q = list(preStateDistanceMetricVariables)

        k = min(k, len(self.database))
        (indices_array, distances_array) = self.tree.query_radius(q,
                                                           r=0.0,
                                                           return_distance=True,
                                                           sort_results=False,
                                                           count_only=False)

        for i in range(0,len(indices_array[0])):
            selectedTransitionTuple = self.database[indices_array[0][i]]
            if selectedTransitionTuple.last_accessed_iteration != self.rolloutSetCounter:
                selectedTransitionTupleIndex = i
                break
        assert selectedTransitionTuple.last_accessed_iteration != self.rolloutSetCounter
        initialEventID = selectedTransitionTuple.additionalState["initial event"]
        action = selectedTransitionTuple.additionalState["action"]
        timeStep = selectedTransitionTuple.additionalState["time step"]
        for distances in distances_array:
            for distance in distances:
                assert distance == 0.0, "Distance was {}".format(distance)

        # The set of transitions that were generated for this transition
        biasCorrectedTransitionSet = [selectedTransitionTuple]
        for idx in range(0,len(indices_array[0])):
            if selectedTransitionTupleIndex == idx:
                continue
            cur = self.database[idx]
            if cur.additionalState["initial event"] == initialEventID and cur.additionalState["time step"] == timeStep:
                assert cur.additionalState["action"] != action
                biasCorrectedTransitionSet.append(self.database[idx])

        for transition in biasCorrectedTransitionSet:
            assert transition.last_accessed_iteration != self.rolloutSetCounter
        return biasCorrectedTransitionSet


    def getRollouts(self,
                    count=10,
                    horizon=10,
                    policy=None,
                    domain=None,
                    biasCorrected=False,
                    actionsInDistanceMetric=True):
        """
            Helper function for generating rollouts from all the domains.
            Args:
                count (integer): The number of rollouts to generate.
                horizon (integer): The maximum length of rollouts.
                policy (function(state, actions)): The function used to select an action.
                domain (Domain): The domain that will be called to generate rollouts.
                biasCorrected (boolean): Indicates whether the database has additional samples to remove bias.
                actionsInDistanceMetric (boolean): Indicates whether the actions are used in the distance metric.
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
                    r, ns, terminal, currentPossibleActions = domain.step(
                        policy,
                        biasCorrected=biasCorrected,
                        actionsInDistanceMetric=actionsInDistanceMetric)

                    #state["stitch distance"] = self.lastStitchDistance
                    self.totalStitchingDistance += self.lastStitchDistance

                    state = {"action": self.lastEvaluatedAction, "reward": r}
                    assert state["action"] >= 0
                    for label in ns.keys():
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

    def _stepWithActionsInDistanceMetric(self, policy):
        """
        This will find the most related state+action in the database
        :param policy:
        :return:
        """
        pre = self.preStateDistanceMetricVariables.copy()
        possibleActions = []
        for idx in range(0,self.domain.actions_num):
            possibleActions.append(idx)
        action = policy(pre, self.currentPossibleActions)
        self.lastEvaluatedAction = action
        assert action >= 0
        for idx, possibleAction in enumerate(possibleActions):
            if action == idx:
                pre = np.append(pre, 1.)
            else:
                pre = np.append(pre, 0.)
        (postStitchObject, stitchDistance) = self._getClosest(pre)
        self.lastStitchDistance = stitchDistance # because we can't change the return signature
        postStitchObject.last_accessed_iteration = self.rolloutSetCounter # prohibit its use in this rollout set
        assert stitchDistance >= 0
        return postStitchObject

    def _stepWithBiasCorrection(self, policy):
        """
        Generate a state transition and mark all states in the transition database that were added to correct its
        bias. You can only use this step function if actions are not included in the distance metric.
        :return:
        """
        pre = self.preStateDistanceMetricVariables
        (postStitchObject, stitchDistance) = self._getClosest(pre)
        self.lastStitchDistance = stitchDistance # because we can't change the return signature
        assert stitchDistance >= 0

        transitions = self._getBiasSet(postStitchObject, self.domain.actions_num)# remove bias corrected states
        action = policy(None, [], transitionTuple=transitions[0])
        assert action >= 0
        for transition in transitions:
            transition.last_accessed_iteration = self.rolloutSetCounter
        for candidate in transitions:
            if candidate.additionalState["action"] == action:
                transition = candidate
                break
        self.lastEvaluatedAction = action
        self.preStateDistanceMetricVariables = transition.postStateDistanceMetricVariables
        r = transition.visualizationResultState["reward"]
        self.currentPossibleActions = transition.possibleActions
        self.terminal = transition.isTerminal
        self.state = transition.visualizationResultState
        return transition

    def step(self, policy, biasCorrected=False, actionsInDistanceMetric=True):
        """
        Find the closest transition matching the policy.

        biasCorrected (bool): Whether the database has sampled every action at every state in the database.
            This will determine whether we remove all matching states when we sample one.
        actionsInDistanceMetric (bool): Indicates whether the distance metric needs indicator variables for each of the
            actions. This will force the policy to be evaluated before stitching.
        """
        # We can't correct the bias if the actions are in the distance metric
        assert biasCorrected != actionsInDistanceMetric

        if actionsInDistanceMetric:
            transition = self._stepWithActionsInDistanceMetric(policy)
        elif biasCorrected:
            transition = self._stepWithBiasCorrection(policy)
        else:
            assert False # Will eventually implement other cases

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
        if hasattr(self.domain, "getInitState"):
            self.preStateDistanceMetricVariables = self.domain.getInitState().preStateDistanceMetricVariables
        elif hasattr(self.domain, "INIT_STATE"):
            self.preStateDistanceMetricVariables = self.domain.INIT_STATE
        else:
            self.preStateDistanceMetricVariables = self.domain.start_state.copy()
        return self.state.copy(), self.terminal, self.currentPossibleActions

    def isTerminal(self):
        """
        :return: ``True`` if the agent has reached or exceeded the goal position.
        """
        return self.terminal
