"""
Find the decline in visual fidelity for increasing sample sizes. Produce a scatterplot with error bars across
all policies
"""
__author__ = "Sean McGregor"
from rlpy.Domains import WildfireData
from rlpy.Domains import Stitching as Stitching
from experiments.wildfire_policy_functions import wildfirePolicySeverityFactory
import numpy as np
import os


def visualFidelityError(sampleCount):
    """
    :return:
    """
    policy = wildfirePolicySeverityFactory("todo", "todo")

    # todo
    # 1. create the benchmarks for all the database policies.
    # 2. generate trajectories and measure the visual fidelity for each of the sample counts, record mean and variance
    # 3. write to CSV

if __name__ == "__main__":
    for sampleCount in [10,20,30,40,50,60,70,80,90,100,
                        110,120,130,140,150,160,170,180,
                        190,200]:
        visualFidelityError(sampleCount)
