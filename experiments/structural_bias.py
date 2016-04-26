"""
Record the sum of the stitching distance for each policy and the resulting visual fidelity.
"""
__author__ = "Sean McGregor"
from rlpy.Domains import WildfireData
from rlpy.Domains import Stitching as Stitching
from experiments.wildfire_policy_functions import wildfirePolicySeverityFactory
import numpy as np
import os


def visualFidelityError():
    """
    :return:
    """
    policy = wildfirePolicySeverityFactory("todo", "todo")

    # todo
    # 1. create the benchmarks for all the database policies.
    # 2. generate trajectories and measure the visual fidelity for each of the policies and the total stitching
    #    distance across all the synthesized policies, generate a scatterplot and a linear regression based on this data

if __name__ == "__main__":
    visualFidelityError()
