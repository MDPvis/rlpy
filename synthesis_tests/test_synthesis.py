import logging
from rlpy.Domains import MountainCar as domain_mountain_car
from rlpy.Domains import Stitching as domain_stitching
import numpy as np
import os
import random
import nose.tools
from rlpy.Domains.StitchingPackage.benchmark import benchmark

prng_for_policy = np.random.RandomState(0)
def random_policy(s, actions):
    """Default to a random action selection"""
    return prng_for_policy.choice(actions)

def generate_rollouts(domain, labels, count, horizon, policy=random_policy):
    """
        Helper function for generating rollouts from all the domains.
        Args:
            domain (Domain): The domain that will be called to generate rollouts.
            labels (list(String)): A list of the state labels.
            count (integer): The number of rollouts to generate.
            horizon (integer): The maximum length of rollouts.
            policy (function(state, actions)): The function used to select an action.
    """
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

def test_near_exact_reproduction_of_rollouts_under_same_policy():
    """
    Generate rollouts on a domain and then synthesize rollouts
    for the domain using the same policy. The synthesized
    rollouts should have a close, but not exact distribution
    because the same policy is used. Variation is introduced
    because the initial stitched state is selected uniformly
    at random from a common starting state.
    """
    number_rollouts = 50
    horizon = 5
    database_rollouts = 500
    database_horizon = 5
    noise = .1
    mountaincar = domain_mountain_car(noise)
    mountaincar.random_state = np.random.RandomState(0)
    labels = ["x", "xdot"]

    policyNumber = 0

    def policy(state, possibleActions):
        return possibleActions[policyNumber]

    synthesis_domain = domain_stitching(mountaincar, rolloutCount = database_rollouts, horizon = database_horizon, generatingPolicies = [policy], seed = 0, searchDistance = 0)
    true_rollouts = generate_rollouts(mountaincar, labels, number_rollouts, horizon, policy = policy)
    synthesized_rollouts = generate_rollouts(synthesis_domain, labels, number_rollouts, horizon, policy = policy)
    x_bench = benchmark(true_rollouts, synthesized_rollouts, "x")
    assert x_bench < .03, "x is not synthesized within tolerance, current: %f" % x_bench
    xdot_bench = benchmark(true_rollouts, synthesized_rollouts, "xdot")
    assert xdot_bench < .01, "xdot is not synthesized within tolerance, current: %f" % xdot_bench
    reward_bench = benchmark(true_rollouts, synthesized_rollouts, "reward")
    assert reward_bench < .01, "reward is not synthesized within tolerance, current: %f" % reward_bench
    action_bench = benchmark(true_rollouts, synthesized_rollouts, "action")
    assert action_bench < .01, "action is not synthesized within tolerance, current: %f" % action_bench
    return

def test_starting_state_distribution_is_exact():
    """
    Generate rollouts on a domain and then synthesize rollouts
    for the domain using the same policy. The synthesized
    rollouts should have the exact same state distribution
    in the initial state since it is a consistent policy.
    """
    number_rollouts = 5
    horizon = 5
    database_rollouts = 5
    database_horizon = 5
    noise = .1
    mountaincar = domain_mountain_car(noise)
    mountaincar.random_state = np.random.RandomState(0)
    labels = ["x", "xdot"]

    policyNumber = 0

    def policy(state, possibleActions):
        return possibleActions[policyNumber]

    synthesis_domain = domain_stitching(mountaincar, rolloutCount = database_rollouts, horizon = database_horizon, generatingPolicies = [policy], seed = 0, searchDistance = 0)
    true_rollouts = generate_rollouts(mountaincar, labels, number_rollouts, horizon, policy = policy)
    synthesized_rollouts = generate_rollouts(synthesis_domain, labels, number_rollouts, horizon, policy = policy)
    x_bench = benchmark(true_rollouts, synthesized_rollouts, "x", event_numbers=[0])
    
    assert x_bench == 0, "x is not synthesized within tolerance, current: %f" % x_bench
    xdot_bench = benchmark(true_rollouts, synthesized_rollouts, "xdot", event_numbers=[0])
    assert xdot_bench == 0, "xdot is not synthesized within tolerance, current: %f" % xdot_bench
    reward_bench = benchmark(true_rollouts, synthesized_rollouts, "reward", event_numbers=[0])
    assert reward_bench == 0, "reward is not synthesized within tolerance, current: %f" % reward_bench
    action_bench = benchmark(true_rollouts, synthesized_rollouts, "action", event_numbers=[0])
    assert action_bench == 0, "action is not synthesized within tolerance, current: %f" % action_bench
    return

def test_consistency_in_random_numbers():
    """
    Experiments should have consistent random seeding such that they
    do not change every time they are run. This requires manual initialization
    when creating the RLPy domains.
    """
    number_rollouts = 2
    horizon = 2
    database_rollouts = 2
    database_horizon = 2
    noise = .5
    mountaincar = domain_mountain_car(noise)
    mountaincar.random_state = np.random.RandomState(0)
    labels = ["x", "xdot"]

    policyNumber = 0

    def policy(state, possibleActions):
        return possibleActions[policyNumber]

    synthesis_domain = domain_stitching(mountaincar, rolloutCount = database_rollouts, horizon = database_horizon, generatingPolicies = [policy], seed = 0)
    true_rollouts = generate_rollouts(mountaincar, labels, number_rollouts, horizon, policy = policy)
    synthesized_rollouts = generate_rollouts(synthesis_domain, labels, number_rollouts, horizon, policy = policy)
    first_benchmark = benchmark(true_rollouts, synthesized_rollouts, "x")
    repeated_first_benchmark = benchmark(true_rollouts, synthesized_rollouts, "x")
    assert first_benchmark == repeated_first_benchmark # No stochasticity in benchmark

    mountaincar = domain_mountain_car(noise)
    mountaincar.random_state = np.random.RandomState(0)
    synthesis_domain = domain_stitching(mountaincar, rolloutCount = database_rollouts, horizon = database_horizon, generatingPolicies = [policy], seed = 0)
    true_rollouts = generate_rollouts(mountaincar, labels, number_rollouts, horizon, policy = policy)
    synthesized_rollouts = generate_rollouts(synthesis_domain, labels, number_rollouts, horizon, policy = policy)
    second_benchmark = benchmark(true_rollouts, synthesized_rollouts, "x")
    assert first_benchmark == second_benchmark
