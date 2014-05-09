#!/usr/bin/env python

__author__ = "William Dabney"

from rlpy.Domains import GridWorld
from rlpy.Agents import NaturalActorCritic
from rlpy.Representations import Tabular
from rlpy.Policies import GibbsPolicy
from rlpy.Experiments import Experiment
import os


def make_experiment(
        id=1, path="./Results/Temp/{domain}/{agent}/{representation}/"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """

    # Experiment variables
    max_steps = 10000
    num_policy_checks = 10

    # Domain:
    # MAZE                = '/Domains/GridWorldMaps/1x3.txt'
    maze = os.path.join(GridWorld.default_map_dir, '4x5.txt')
    domain = GridWorld(maze, noise=0.3)

    # Representation
    representation = Tabular(domain, discretization=20)

    # Policy
    policy = GibbsPolicy(representation)

    # Agent
    agent = NaturalActorCritic(representation, policy, domain,
                               0.3, 100, 1000, .7, 0.1)

    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    experiment = make_experiment()
    experiment.run_from_commandline()
