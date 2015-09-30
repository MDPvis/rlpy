"""Construct a surrogate model from elementary state transitions
of another problem domain.


**REFERENCE:**
Based on `Batch Mode Reinforcement Learning based on the
Synthesis of Artificial Trajectories <https://goo.gl/1yveeS>`_

**TODOS (things I am working on)**

todo: pickle database so it isn't always re-generated.

todo: use multinomial logistic regression for a generic, interpretable policy supporting more than one action?
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

todo: use generatorPolicies, targetPolicies, and evaluationPolicies for generating database and metric

todo: learn metric

"""

from scipy.spatial import KDTree
import numpy as np
from .Domain import Domain
import math

__copyright__ = "Copyright 2015, Sean McGregor"
__credits__ = ["Sean McGregor"]
__license__ = "BSD 3-Clause"
__author__ = ["Sean McGregor"]

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

    def __init__(self, domain, database = [], rolloutCount = 100, horizon = 100):
        """
        :param domain: The domain used to generate MC rollouts
        :param database: A set of pre-computed state transitions
        :param rolloutCount: The number of rollouts to generate for each policy
        :param horizon: The maximum number of transitions for each rollout
        """
        self.domain = domain
        self.database = database
        self.rolloutCount = rolloutCount
        self.horizon = horizon

        self._populateDatabase()

    def _populateDatabase(self):
        """
        Load many transitions into the database then create the KD-Tree.
        """
        for rolloutNumber in range(self.rolloutCount):
            self.domain.s0() # reset the state
            currentDepth = 0
            while not self.domain.isTerminal() and currentDepth < self.horizon:
                state = self.domain.state
                possible_actions = self.domain.possibleActions()
                action = np.random.choice(possible_actions)# todo, make this an actual policy
                state = np.append(state, action)
                r, ns, terminal, nextStatePossibleActions = self.domain.step(action)
                ns = self.domain.state # The helicopter domain is partially observable so we need to grab the full state
                t = TransitionTuple(state, ns, terminal, (currentDepth == 0), r, nextStatePossibleActions)
                self.database.append(t)
                currentDepth += 1
        self.tree = KDTree(self.database) # Create the KD-tree

    def _getClosest(self, state, a):
        """
        returns (at random) one of the closest k point from the KD tree.
        :param state: The current state of the world that we want the closest transition for.
        :param a: The selected action for the current state.
        :return: ``TransitionTuple`` The selected transition from the database.
        """
        q = list(state)
        q.append(a)
        d, index = self.tree.query(q, 5)

        # if the values are the same, we want to draw one of them at random
        distance = d[0]
        i = 0
        while len(d) < i and distance == d[i]:
            i += 1
        r = np.random.rand(1,1)[0][0]
        selection = math.floor(r*i)

        return self.database[index[selection]]

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
