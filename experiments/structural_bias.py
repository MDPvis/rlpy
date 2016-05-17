"""
This experiment examines how the bias correction affects visual fidelity. We generate results for two
databases, one where we subsample the database trajectories to be half the original size, and another
database in which we remove all the bias correction state transitions.
Record the sum of the stitching distance for each policy and the resulting visual fidelity.
"""
__author__ = "Sean McGregor"
from rlpy.Domains import WildfireData
from experiments.wildfire_policy_functions import wildfirePolicySeverityFactory
import numpy as np
import os
import pickle
import rlpy.Domains.StitchingPackage.benchmark
import rlpy.Domains.StitchingPackage.MahalanobisDistance
import rlpy.Domains.Stitching


def visualFidelityError(
        wildfireData,
        varianceDictionary,
        stitchingDomain,
        outCSVFile,
        benchmarks,
        stitchingVariables,
        policyValues,
        policies,
        horizon=99,
        sampleCount=30):
    """
    :return:
    """

    benchmarkSampleHalved = []
    benchmarkSampleBiased = []
    for idx,policyValue in enumerate(policyValues):
        var_count = len(stitchingVariables)
        mahaMetric = rlpy.Domains.StitchingPackage.MahalanobisDistance.MahalanobisDistance(var_count,
                                                                                           stitchingDomain,
                                                                                           target_policies=[],
                                                                                           normalize_starting_metric=False,
                                                                                           cached_metric=None)
        inverseVariances = []
        for stitchingVariable in stitchingVariables:
            cur = varianceDictionary[stitchingVariable]
            assert cur >= 0
            if cur == 0:
                inverseVariances.append(0.0)
            else:
                inverseVariances.append(1.0/float(cur))
        mahaMetric.updateInverseVariance(inverseVariances)

        stitchingDomain.setMetric(mahaMetric)

        db = wildfireData.getDatabaseWithoutTargetSeverityPolicy(policyValue[0], policyValue[1])

        dbHalved =[]
        for transition in db:
            if transition.additionalState["initialFire"] % 2 == 0:
                dbHalved.append(transition)

        dbBiased =[]
        for transition in db:
            if transition.additionalState["onPolicy"] == 1:
                dbBiased.append(transition)

        stitchingDomain.setDatabase(dbHalved)
        rolloutsHalved = stitchingDomain.getRollouts(
            count=sampleCount,
            horizon=horizon,
            policy=policies[idx],
            domain=None,
            biasCorrected=True,
            actionsInDistanceMetric=False)
        total = 0
        for variable in stitchingDomain.domain.VISUALIZATION_VARIABLES:
            total += benchmarks[idx].benchmark_variable(rolloutsHalved, variable)
        benchmarkSampleHalved.append(total)

        stitchingDomain.setDatabase(dbBiased)
        rolloutsBiased = stitchingDomain.getRollouts(
            count=sampleCount,
            horizon=horizon,
            policy=policies[idx],
            domain=None,
            biasCorrected=False,
            actionsInDistanceMetric=False)
        total = 0
        for variable in stitchingDomain.domain.VISUALIZATION_VARIABLES:
            total += benchmarks[idx].benchmark_variable(rolloutsBiased, variable)
        benchmarkSampleBiased.append(total)
    outCSVFile.write("biased then halved\n")
    outCSVFile.write("{}\n{}\n".format(benchmarkSampleBiased, benchmarkSampleHalved))
