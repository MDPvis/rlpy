import logging
from rlpy.Domains import MountainCar as domain_mountain_car
from rlpy.Domains import Stitching as domain_stitching
import numpy as np
import os
import random
import nose.tools
from rlpy.Domains.Stitching import MahalanobisDistance
from rlpy.Domains.StitchingPackage.benchmark import Benchmark

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

    policyNumber = 0

    def policy(state, possibleActions):
        return possibleActions[policyNumber]

    synthesis_domain = domain_stitching(
      mountaincar,
      rolloutCount=database_rollouts,
      horizon=database_horizon,
      databasePolicies=[policy],
      seed = 0)
    true_rollouts = synthesis_domain.getRollouts(
      number_rollouts,
      horizon,
      policies=[policy],
      domain=mountaincar)
    synthesized_rollouts = synthesis_domain.getRollouts(
      number_rollouts,
      horizon,
      policies=[policy],
      domain=synthesis_domain)
    x_bench = Benchmark.benchmark_variable(true_rollouts, synthesized_rollouts, "x")
    assert x_bench < .2, "x is not synthesized within tolerance, current: %f" % x_bench
    xdot_bench = Benchmark.benchmark_variable(true_rollouts, synthesized_rollouts, "xdot")
    assert xdot_bench < .2, "xdot is not synthesized within tolerance, current: %f" % xdot_bench
    reward_bench = Benchmark.benchmark_variable(true_rollouts, synthesized_rollouts, "reward")
    assert reward_bench < .2, "reward is not synthesized within tolerance, current: %f" % reward_bench
    action_bench = Benchmark.benchmark_variable(true_rollouts, synthesized_rollouts, "action")
    assert action_bench < .2, "action is not synthesized within tolerance, current: %f" % action_bench
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

    policyNumber = 0

    def policy(state, possibleActions):
        return possibleActions[policyNumber]

    synthesis_domain = domain_stitching(
      mountaincar,
      rolloutCount=database_rollouts,
      horizon=database_horizon,
      databasePolicies=[policy],
      seed=0)
    true_rollouts = synthesis_domain.getRollouts(
      number_rollouts,
      horizon,
      policies=[policy],
      domain=mountaincar)
    synthesized_rollouts = synthesis_domain.getRollouts(
      number_rollouts,
      horizon,
      policies=[policy],
      domain=synthesis_domain)
    x_bench = Benchmark.benchmark_variable(
      true_rollouts,
      synthesized_rollouts,
      "x",
      event_numbers=[0])
    
    assert x_bench == 0, "x is not synthesized within tolerance, current: %f" % x_bench
    xdot_bench = Benchmark.benchmark_variable(true_rollouts, synthesized_rollouts, "xdot", event_numbers=[0])
    assert xdot_bench == 0, "xdot is not synthesized within tolerance, current: %f" % xdot_bench
    reward_bench = Benchmark.benchmark_variable(true_rollouts, synthesized_rollouts, "reward", event_numbers=[0])
    assert reward_bench == 0, "reward is not synthesized within tolerance, current: %f" % reward_bench
    action_bench = Benchmark.benchmark_variable(true_rollouts, synthesized_rollouts, "action", event_numbers=[0])
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

    policyNumber = 0

    def policy(state, possibleActions):
        return possibleActions[policyNumber]

    synthesis_domain = domain_stitching(
      mountaincar,
      rolloutCount=database_rollouts,
      horizon=database_horizon,
      databasePolicies=[policy],
      seed = 0)
    true_rollouts = synthesis_domain.getRollouts(
      number_rollouts,
      horizon,
      policies=[policy],
      domain=mountaincar)
    synthesized_rollouts = synthesis_domain.getRollouts(
      number_rollouts,
      horizon,
      policies=[policy],
      domain=synthesis_domain)
    first_benchmark = Benchmark.benchmark_variable(true_rollouts, synthesized_rollouts, "x")
    repeated_first_benchmark = Benchmark.benchmark_variable(true_rollouts, synthesized_rollouts, "x")
    assert first_benchmark == repeated_first_benchmark, "Repeated application of benchmark gave different results"

    mountaincar = domain_mountain_car(noise)
    mountaincar.random_state = np.random.RandomState(0)
    synthesis_domain = domain_stitching(
      mountaincar,
      rolloutCount=database_rollouts,
      horizon=database_horizon,
      databasePolicies=[policy],
      seed=0)
    true_rollouts = synthesis_domain.getRollouts(
      number_rollouts,
      horizon,
      policies=[policy],
      domain=mountaincar)
    synthesized_rollouts = synthesis_domain.getRollouts(
      number_rollouts,
      horizon,
      policies=[policy],
      domain=synthesis_domain)
    second_benchmark = Benchmark.benchmark_variable(
      true_rollouts,
      synthesized_rollouts,
      "x")
    assert first_benchmark == second_benchmark, "Repeated generation of rollouts produced different benchmarks"

def test_benchmark_degenerates_if_benchmark_count_is_too_large_relative_to_database():
    """
    When benchmarking synthesized trajectories, it is important that the benchmark is not generated
    from more trajectories than we have samples for.
    """
    target_rollout_count = 10
    target_horizon = 4
    database_rollout_count = 100
    database_horizon = 4

    prior_loss = 0

    rs = np.random.RandomState(0)
    def generating_policy(state, possibleActions):
        return rs.choice(possibleActions)

    def target_policy(state, possibleActions):
        return possibleActions[0]

    target_domain = domain_mountain_car(noise=.1)
    target_domain.random_state = np.random.RandomState(0)
    synthesis_domain = domain_stitching(
      target_domain,
      rolloutCount=database_rollout_count,
      horizon=database_horizon,
      databasePolicies=[generating_policy],
      seed = 0)

    target_rollouts = synthesis_domain.getRollouts(
      target_rollout_count,
      target_horizon,
      policies=[target_policy],
      domain=target_domain)
    matrix_variable_count=5
    mahalanobis_distance = MahalanobisDistance(matrix_variable_count, synthesis_domain, target_rollouts)
    matrix_metric = mahalanobis_distance.get_matrix_as_np_array()
    for bench_count in [45,50,70,100]:
        current_loss = MahalanobisDistance.loss(
          MahalanobisDistance.ceiling_logarithm(MahalanobisDistance.flatten(matrix_metric)),
          synthesis_domain,
          target_rollouts,
          benchmark_rollout_count=bench_count)
        assert prior_loss < current_loss, "Curent loss ({}) did not degenerate from prior loss ({})".format(current_loss, prior_loss)
        prior_loss = current_loss


def test_stitching_distance():
    """
    The stitching distance should equal the flipping of the action value.
    """
    number_rollouts = 2
    horizon = 100
    database_rollouts = 2
    database_horizon = 100
    noise = .5
    mountaincar = domain_mountain_car(noise)
    mountaincar.random_state = np.random.RandomState(0)

    # The policy in the database
    def policy(state, possibleActions):
        return 0
    synthesis_domain = domain_stitching(
      mountaincar,
      rolloutCount=database_rollouts,
      horizon=database_horizon,
      databasePolicies=[policy],
      seed = 0)

    # The policy we are trying to approximate
    def policy2(state, possibleActions):
      return 1
    synthesized_rollouts = synthesis_domain.getRollouts(
      number_rollouts,
      horizon,
      policies=[policy2],
      domain=synthesis_domain)

    assert synthesis_domain.totalStitchingDistance < 283, "Stitching distance increased to: {}".format(synthesis_domain.totalStitchingDistance)
