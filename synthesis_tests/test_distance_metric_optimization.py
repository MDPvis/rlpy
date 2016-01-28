import logging
from rlpy.Domains import MountainCar as domain_mountain_car
from rlpy.Domains import GridWorld as domain_gridworld
from rlpy.Domains import Stitching as domain_stitching
from rlpy.Domains.Stitching import MahalanobisDistance
import numpy as np
import os
import random
import sys
import nose.tools
from mocks import mock_target_rollouts
from rlpy.Domains.StitchingPackage.benchmark import Benchmark

def test_initialization_of_mahalanobis_distance():
    """
    Make sure the initialized distance metric is sensical.
    """
    number_rollouts = 2
    horizon = 2
    database_rollout_count = 2
    database_horizon = 2
    noise = .1
    mountaincar = domain_mountain_car(noise)
    mountaincar.random_state = np.random.RandomState(0)

    def policy(state, possibleActions):
        return possibleActions[0]

    synthesis_domain = domain_stitching(
      mountaincar,
      rolloutCount=database_rollout_count,
      horizon=database_horizon,
      databasePolicies=[policy],
      seed = 0)
    true_rollouts = synthesis_domain.getRollouts(number_rollouts, horizon, policies=[policy], domain=mountaincar)
    synthesized_rollouts = synthesis_domain.getRollouts(number_rollouts, horizon, policies=[policy], domain=synthesis_domain)

    # get the metric
    distance = MahalanobisDistance(5, synthesis_domain, true_rollouts)

    assert distance.get_matrix_as_np_array()[0][0] == 1, "The initialized distance matrix is identity"
    assert distance.get_matrix_as_np_array()[1][1] == 1, "The initialized distance matrix is identity"
    assert distance.get_matrix_as_np_array()[2][2] == 1, "The initialized distance matrix is identity"

def test_updated_distance_metric_does_not_have_worse_performance():
    """
    When we optimize the distance metric, we expect the benchmark of the synthesized
    rollouts to not get worse.
    """
    number_rollouts = 2
    horizon = 2
    database_rollout_count = 2
    database_horizon = 2
    noise = .1
    mountaincar = domain_mountain_car(noise)
    mountaincar.random_state = np.random.RandomState(0)

    policyNumber = 0

    def policy(state, possibleActions):
        return possibleActions[policyNumber]

    synthesis_domain = domain_stitching(
      mountaincar,
      rolloutCount=database_rollout_count,
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

    # get the metric
    distance = MahalanobisDistance(5, synthesis_domain, true_rollouts)

    matrix_metric_1 = distance.get_matrix_as_np_array()
    flat_metric = []
    for row in matrix_metric_1:
        for val in row:
            flat_metric.append(val)
    starting_loss = MahalanobisDistance.loss(flat_metric, synthesis_domain, true_rollouts)
    distance.optimize()
    matrix_metric_2 = distance.get_matrix_as_np_array()
    flat_metric = []
    for row in matrix_metric_2:
        for val in row:
            flat_metric.append(val)
    optimized_loss = MahalanobisDistance.loss(flat_metric, synthesis_domain, true_rollouts)

    assert optimized_loss <= starting_loss, "Optimized loss is worse than starting loss"

def test_has_non_zero_loss():
    """
    There should be a loss when trying to reconstruct the rollouts from the database
    since the transition function has stochasticity, ie, there is noise in the domain.
    """
    target_rollout_count = 60
    target_horizon = 2
    horizon = 2
    database_rollout_count = 2
    database_horizon = 2
    mountaincar = domain_mountain_car(noise=.1)
    mountaincar.random_state = np.random.RandomState(0)

    rs = np.random.RandomState(0)
    def generating_policy(state, possibleActions):
        return rs.choice([-1,1]) # random direction

    synthesis_domain = domain_stitching(
      mountaincar,
      rolloutCount=database_rollout_count,
      horizon=database_horizon,
      databasePolicies=[generating_policy],
      seed=0)

    target_rollouts = synthesis_domain.getRollouts(
      target_rollout_count,
      target_horizon,
      policies=[generating_policy],
      domain=mountaincar)

    # get the metric
    distance = MahalanobisDistance(5, synthesis_domain, target_rollouts)

    matrix_metric_1 = distance.get_matrix_as_np_array()
    flat_metric = []
    for row in matrix_metric_1:
        for val in row:
            flat_metric.append(val)
    loss = MahalanobisDistance.loss(flat_metric, synthesis_domain, target_rollouts)
    assert loss > 0, "Loss is zero or negative"

    gridworld = domain_gridworld(noise=.1)
    gridworld.random_state = np.random.RandomState(0)

    synthesis_domain = domain_stitching(
      gridworld,
      labels=["x", "y"],
      rolloutCount=database_rollout_count,
      horizon=database_horizon,
      databasePolicies=[generating_policy],
      seed = 0)

    target_rollouts = synthesis_domain.getRollouts(
      target_rollout_count,
      target_horizon,
      policies=[generating_policy],
      domain=gridworld)

    # get the metric
    distance = MahalanobisDistance(6, synthesis_domain, target_rollouts)

    matrix_metric_1 = distance.get_matrix_as_np_array()
    flat_metric = []
    for row in matrix_metric_1:
        for val in row:
            flat_metric.append(val)
    loss = MahalanobisDistance.loss(flat_metric, synthesis_domain, target_rollouts)
    assert loss > 0, "Loss is zero or negative"

def test_has_zero_loss():
    """
    There should be no loss when trying to reconstruct the rollouts from the database
    since the transition function has no stochasticity, ie, there is no noise in the domain.
    """
    target_rollout_count = 50
    target_horizon = 2
    database_rollout_count = 50
    database_horizon = 2
    mountaincar = domain_mountain_car(noise=0)
    mountaincar.random_state = np.random.RandomState(0)

    rs = np.random.RandomState(0)
    def generating_policy(state, possibleActions):
        return rs.choice(possibleActions) # random direction

    synthesis_domain = domain_stitching(
      mountaincar,
      rolloutCount=database_rollout_count,
      horizon = database_horizon,
      databasePolicies=[generating_policy],
      targetPolicies=[generating_policy],
      seed = 0)
    rs = np.random.RandomState(0)
    target_rollouts = synthesis_domain.getRollouts(
      target_rollout_count,
      target_horizon,
      policies=[generating_policy],
      domain=mountaincar)
    rs = np.random.RandomState(0)
    distance = MahalanobisDistance(5, synthesis_domain, [])
    matrix_metric_1 = distance.get_matrix_as_np_array()
    flat_metric = MahalanobisDistance.flatten(matrix_metric_1)
    flat_metric = MahalanobisDistance.ceiling_logarithm(flat_metric)
    loss = MahalanobisDistance.loss(flat_metric,
      synthesis_domain,
      target_rollouts,
      benchmark_rollout_count=target_rollout_count)
    assert loss == 0, "Loss is not zero: {}".format(loss)

    gridworld = domain_gridworld(noise=0)
    gridworld.random_state = np.random.RandomState(0)

    rs = np.random.RandomState(0)
    synthesis_domain = domain_stitching(
      gridworld,
      labels=["x", "y"],
      rolloutCount=database_rollout_count,
      horizon = database_horizon,
      databasePolicies = [generating_policy],
      targetPolicies=[generating_policy],
      seed = 0)

    rs = np.random.RandomState(0)
    target_rollouts = synthesis_domain.getRollouts(
      target_rollout_count,
      target_horizon,
      policies=[generating_policy],
      domain=gridworld)

    distance = MahalanobisDistance(6, synthesis_domain, [])

    matrix_metric_1 = distance.get_matrix_as_np_array()
    flat_metric = MahalanobisDistance.flatten(matrix_metric_1)
    flat_metric = MahalanobisDistance.ceiling_logarithm(flat_metric)
    loss = MahalanobisDistance.loss(flat_metric, synthesis_domain, target_rollouts)
    assert loss == 0, "Loss is not zero: {}".format(loss)



def test_updated_distance_metric_improves_performance():
    """
    The optimizer should improve performance of the benchmark on the target policies.
    """
    target_rollout_count = 60
    target_horizon = 5
    database_rollout_count = 60 # todo: figure out why increasing this count can lead to test failure
    database_horizon = 5

    def expectation(target_domain):

        rs = np.random.RandomState(0)
        def generating_policy(state, possibleActions):
            return rs.choice(possibleActions)

        def target_policy(state, possibleActions):
            return possibleActions[0]

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

        mahalanobis_distance = MahalanobisDistance(3, synthesis_domain, target_rollouts)

        matrix_metric_not_optimized = mahalanobis_distance.get_matrix_as_np_array()

        starting_loss = MahalanobisDistance.loss(
          MahalanobisDistance.ceiling_logarithm(MahalanobisDistance.flatten(matrix_metric_not_optimized)),
          synthesis_domain,
          target_rollouts,
          benchmark_rollout_count=target_rollout_count)
        mahalanobis_distance.optimize()
        matrix_metric_optimized = mahalanobis_distance.get_matrix_as_np_array()
        optimized_loss = MahalanobisDistance.loss(
          MahalanobisDistance.ceiling_logarithm(MahalanobisDistance.flatten(matrix_metric_optimized)),
          synthesis_domain,
          target_rollouts,
          benchmark_rollout_count=target_rollout_count)

        print "metric: " + str(MahalanobisDistance.flatten(matrix_metric_not_optimized))
        print "starting_loss: " + str(starting_loss)
        print "metric: " + str(MahalanobisDistance.flatten(matrix_metric_optimized))
        print "optimized_loss: " + str(optimized_loss)

        assert optimized_loss < starting_loss, "Optimized loss ({}) is not better than starting loss ({})".format(optimized_loss, starting_loss)

    # todo: compare this one to the one that has zero loss
    # todo: this has the same loss, why?
    # It is stochastically selecting a state transition of all the
    # closest state transitions

    print "\n\n\nMountain Car\n\n\n"
    mountaincar = domain_mountain_car(noise=.1)
    mountaincar.random_state = np.random.RandomState(0) 
    expectation(target_domain=mountaincar)

    print "\n\n\nGRID WORLD\n\n\n"
    gridworld = domain_gridworld(noise=.1)
    gridworld.random_state = np.random.RandomState(0)
    expectation(target_domain=gridworld)

def test_updates_distance_metric_to_not_be_identity():
    """
    The updated distance metric should move away from Euclidean distance.
    """
    number_rollouts = 2
    horizon = 2
    database_rollout_count = 2
    database_horizon = 2
    noise = .1
    mountaincar = domain_mountain_car(noise)
    mountaincar.random_state = np.random.RandomState(0)

    # The policy we are trying to synthesize
    def synthesized_policy(state, possibleActions):
        return possibleActions[len(possibleActions) -1]

    # The policy we are attempting to approximate
    rs = np.random.RandomState(0)
    def target_policy(state, possibleActions):
        return rs.choice(possibleActions)

    synthesis_domain = domain_stitching(
      mountaincar,
      rolloutCount=database_rollout_count,
      horizon=database_horizon,
      databasePolicies=[synthesized_policy],
      seed=0)
    true_rollouts = synthesis_domain.getRollouts(
      number_rollouts,
      horizon,
      policies=[target_policy],
      domain=mountaincar)
    synthesized_rollouts = synthesis_domain.getRollouts(
      number_rollouts,
      horizon,
      policies=[synthesized_policy],
      domain=synthesis_domain)

    distance = MahalanobisDistance(5, synthesis_domain, true_rollouts)

    distance.optimize()

    matrix = distance.get_matrix_as_np_array()

    assert matrix[0][0] != 1, "The optimized distance matrix may be identity"
    assert matrix[1][1] != 1, "The optimized distance matrix may be identity"
    assert matrix[2][2] != 1, "The optimized distance matrix may be identity"

def test_flattening_and_unflattenin_matrices():
    """
    The distance matrix should flatten and unflatten properly.
    """
    matrix = [[1,2],[3,4]]
    flat_matrix = MahalanobisDistance.flatten(matrix)
    unflattened_matrix = MahalanobisDistance.unflatten(flat_matrix)
    message = "The matrix does not flatten properly"
    assert flat_matrix[0] == 1, message
    assert flat_matrix[1] == 2, message
    assert flat_matrix[2] == 3, message
    assert flat_matrix[3] == 4, message
    message = "The matrix does not unflattened properly"
    assert matrix[0][0] == unflattened_matrix[0][0], message
    assert matrix[0][1] == unflattened_matrix[0][1], message
    assert matrix[1][0] == unflattened_matrix[1][0], message
    assert matrix[1][1] == unflattened_matrix[1][1], message

def test_finds_psd():
    """
    The test for PSD should work.
    """
    matrix = [[1,0],[0,1]]
    res = MahalanobisDistance.is_psd(matrix)
    message = "A PSD matrix returned false"
    assert res, message
    matrix = [[2,-1,0],[-1,2,-1],[0,-1,2]]
    res = MahalanobisDistance.is_psd(matrix)
    assert res, message

def test_finds_not_psd():
    """
    The test for PSD should work.
    """
    matrix = [[1,2],[2,1]]
    res = MahalanobisDistance.is_psd(matrix)
    message = "A non-PSD matrix returned true"
    assert not res, message
    matrix = [[0,0,0],[0,0,0],[0,0,0]]
    res = MahalanobisDistance.is_psd(matrix)
    assert not res, message
    matrix = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    res = MahalanobisDistance.is_psd(matrix)
    assert not res, message
    matrix = [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]
    res = MahalanobisDistance.is_psd(matrix)
    assert not res, message

def test_finds_upper_triangular():
    """
    The test for upper_triangular should work.
    """
    matrix = [[1,0],[0,1]]
    res = MahalanobisDistance.is_upper_triangular(matrix)
    message = "An upper triangular matrix returned false"
    assert res, message
    matrix = [[1,1],[0,1]]
    res = MahalanobisDistance.is_upper_triangular(matrix)
    message = "An upper triangular matrix returned false"
    assert res, message
    matrix = [[2,-1,99],[0,2,-1],[0,0,1]]
    res = MahalanobisDistance.is_upper_triangular(matrix)
    assert res, message

def test_finds_not_upper_triangular():
    """
    The test for upper_triangular should work.
    """
    matrix = [[1,0],[1,1]]
    res = MahalanobisDistance.is_upper_triangular(matrix)
    message = "An non-upper triangular matrix returned true"
    assert not res, message
    matrix = [[2,-1,99],[0,2,-1],[0,-1,1]]
    res = MahalanobisDistance.is_upper_triangular(matrix)
    assert not res, message

def test_exponentiate_and_logarithm():
    """
    The bounded logarithm and exponentiation should work properly.
    """
    flat_metric = [0,1,9]
    exp = MahalanobisDistance.ceiling_exponentiate(flat_metric)
    flat_metric2 = MahalanobisDistance.ceiling_logarithm(exp)
    message = "Original matrix was not recovered"
    assert flat_metric[0] == flat_metric2[0], message
    assert flat_metric[1] == flat_metric2[1], message
    assert flat_metric[2] == flat_metric2[2], message

    flat_metric = [-1,-.1,-.01]
    exp = MahalanobisDistance.ceiling_exponentiate(flat_metric)
    flat_metric2 = MahalanobisDistance.ceiling_logarithm(exp)
    assert flat_metric[0] == flat_metric2[0], message
    assert flat_metric[1] - flat_metric2[1] < 0.00000000000001, message
    assert flat_metric[2] - flat_metric2[2] < 0.00000000000001, message

def test_action_benchmarking():
    """
    The benchmark reported on actions should
    be expressed as the average shift of the
    percentiles among all the action plots.
    """
    bench = Benchmark.benchmark_actions(
      mock_target_rollouts["simplistic"],
      mock_target_rollouts["simplistic"],
      2,
      event_numbers=range(0, len(mock_target_rollouts["simplistic"])),
      bootstrap_weight=True
    )
    assert bench == 0, "bench was {}".format(bench)

    bench = Benchmark.benchmark_actions(
      mock_target_rollouts["simplistic"],
      mock_target_rollouts["simplistic opposite action"],
      2,
      event_numbers=range(0, len(mock_target_rollouts["simplistic"])),
      bootstrap_weight=True
    )
    assert bench == 2, "bench was {}".format(bench)
