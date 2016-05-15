"""
Find the decline in visual fidelity for increasing sample sizes. Produce a scatterplot with error bars across
all policies
"""
__author__ = "Sean McGregor"
from rlpy.Domains import WildfireData
from rlpy.Domains import Stitching
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
        sampleCount=None,
        horizon=99):
    """
    :return:
    """
    outCSVFile.write("error, ERC policy variable, time policy variable\n")
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
        stitchingDomain.setDatabase(db)
        rollouts = stitchingDomain.getRollouts(
            count=sampleCount,
            horizon=horizon,
            policy=policies[idx],
            domain=None,
            biasCorrected=True,
            actionsInDistanceMetric=False)
        total = 0
        for variable in stitchingDomain.domain.VISUALIZATION_VARIABLES:
            total += benchmarks[idx].benchmark_variable(rollouts, variable)

        outCSVFile.write(str(sampleCount))
        outCSVFile.write(",")
        outCSVFile.write("{},{},{}\n".format(total, policyValue[0], policyValue[1]))
    return
