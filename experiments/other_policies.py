"""
Find the visual fidelity for policies types not related to the weather, including fire location.
"""
__author__ = "Sean McGregor"
from rlpy.Domains import WildfireData
from rlpy.Domains import Stitching as Stitching
from experiments.wildfire_policy_functions import wildfirePolicyLocationFactory
import numpy as np
import os


def visualFidelityError():
    """
    :return:
    """
    policy = wildfirePolicyLocationFactory("todo", "todo")

    # todo
    # 1. create the benchmark from the special database file
    # 2. find the visual fidelity for the policy under the best metric and the entire database
    # 3. compare to the visual fidelity of an intensity-based policy (found in prior experiment)

if __name__ == "__main__":
    visualFidelityError()
