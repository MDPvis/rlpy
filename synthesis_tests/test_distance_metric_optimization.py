import logging
from rlpy.Domains import MountainCar as domain_mountain_car
from rlpy.Domains import Stitching as domain_stitching
from rlpy.Domains.Stitching import MahalanobisDistance
import numpy as np
import os
import random
import nose.tools
from rlpy.Domains.StitchingPackage.benchmark import benchmark


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

    policyNumber = 0

    def policy(state, possibleActions):
        return possibleActions[policyNumber]

    synthesis_domain = domain_stitching(
      mountaincar,
      rolloutCount=database_rollout_count,
      horizon=database_horizon,
      databasePolicies=[policy],
      seed = 0)
    true_rollouts = synthesis_domain.getRollouts(number_rollouts, horizon, policies=[policy], domain=mountaincar)
    synthesized_rollouts = synthesis_domain.getRollouts(number_rollouts, horizon, policies=[policy], domain=synthesis_domain)

    # get the metric
    distance = MahalanobisDistance(3, synthesis_domain, true_rollouts)

    assert distance.get_matrix()[0][0] == 1, "The initialized distance matrix is identity"
    assert distance.get_matrix()[1][1] == 1, "The initialized distance matrix is identity"
    assert distance.get_matrix()[2][2] == 1, "The initialized distance matrix is identity"

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
    distance = MahalanobisDistance(3, synthesis_domain, true_rollouts)

    matrix_metric_1 = distance.get_matrix()
    flat_metric = []
    for row in matrix_metric_1:
        for val in row:
            flat_metric.append(val)
    starting_loss = MahalanobisDistance.loss(flat_metric, synthesis_domain, true_rollouts)
    distance.optimize()
    matrix_metric_2 = distance.get_matrix()
    flat_metric = []
    for row in matrix_metric_2:
        for val in row:
            flat_metric.append(val)
    optimized_loss = MahalanobisDistance.loss(flat_metric, synthesis_domain, true_rollouts)

    for row_number, row in enumerate(matrix_metric_1):
        for col_number, val in enumerate(matrix_metric_1):
            assert matrix_metric_1[row_number][col_number] != matrix_metric_2[row_number][col_number], "The optimized distance matrix is unchanged"

    assert optimized_loss <= starting_loss, "Optimized loss is worse than starting loss"

def test_has_non_zero_loss():
    """
    There should be a loss when trying to reconstruct the rollouts from the database.
    """
    target_rollout_count = 60
    target_horizon = 2
    number_rollouts = 2
    horizon = 2
    database_rollout_count = 2
    database_horizon = 2
    mountaincar = domain_mountain_car(noise=.1)
    mountaincar.random_state = np.random.RandomState(0)

    rs = np.random.RandomState(0)
    def generating_policy(state, possibleActions):
        return rs.choice([-1,1]) # random direction

    def target_policy(state, possibleActions):
        return 1 # right

    synthesis_domain = domain_stitching(
      mountaincar,
      rolloutCount=database_rollout_count,
      horizon = database_horizon,
      databasePolicies = [generating_policy],
      seed = 0)

    target_rollouts = synthesis_domain.getRollouts(
      target_rollout_count,
      target_horizon,
      policies=[target_policy],
      domain=mountaincar)

    # get the metric
    distance = MahalanobisDistance(3, synthesis_domain, target_rollouts)

    matrix_metric_1 = distance.get_matrix()
    flat_metric = []
    for row in matrix_metric_1:
        for val in row:
            flat_metric.append(val)
    loss = MahalanobisDistance.loss(flat_metric, synthesis_domain, target_rollouts)
    assert loss > 0, "Loss is zero or negative"

def test_updated_distance_metric_improves_performance():
    """
    The optimizer should improve performance of the benchmark on the target policies.
    """
    target_rollout_count = 60
    target_horizon = 2
    database_rollout_count = 20
    database_horizon = 2
    mountaincar = domain_mountain_car(noise=.1)
    mountaincar.random_state = np.random.RandomState(0)

    rs = np.random.RandomState(0)
    def generating_policy(state, possibleActions):
        #return 1 # right
        return rs.choice([-1,1]) # left

    def target_policy(state, possibleActions):
        return 1 # right

    synthesis_domain = domain_stitching(
      mountaincar,
      rolloutCount=database_rollout_count,
      horizon = database_horizon,
      databasePolicies = [generating_policy],
      seed = 0)
    target_rollouts = synthesis_domain.getRollouts(
      target_rollout_count,
      target_horizon,
      policies=[target_policy],
      domain=mountaincar)

    mahalanobis_distance = MahalanobisDistance(3, synthesis_domain, target_rollouts)

    matrix_metric_not_optimized = mahalanobis_distance.get_matrix()

    def flatten(matrix):
        flat_metric = []
        for row in matrix:
            for val in row:
                flat_metric.append(val)
        return flat_metric

    starting_loss = MahalanobisDistance.loss(
      flatten(matrix_metric_not_optimized),
      synthesis_domain,
      target_rollouts)
    mahalanobis_distance.optimize()
    matrix_metric_optimized = mahalanobis_distance.get_matrix()
    optimized_loss = MahalanobisDistance.loss(
      flatten(matrix_metric_optimized),
      synthesis_domain,
      target_rollouts)

    print "target_rollouts: " + str(target_rollouts)
    print "metric: " + str(flatten(matrix_metric_not_optimized))
    print "starting_loss: " + str(starting_loss)
    print "metric: " + str(flatten(matrix_metric_optimized))
    print "optimized_loss: " + str(optimized_loss)

    assert optimized_loss < starting_loss, "Optimized loss is not better than starting loss"

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

    distance = MahalanobisDistance(3, synthesis_domain, true_rollouts)

    distance.optimize()

    matrix = distance.get_matrix()

    assert matrix[0][0] != 1, "The optimized distance matrix may be identity"
    assert matrix[1][1] != 1, "The optimized distance matrix may be identity"
    assert matrix[2][2] != 1, "The optimized distance matrix may be identity"
