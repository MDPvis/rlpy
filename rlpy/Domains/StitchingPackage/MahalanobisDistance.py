from scipy.optimize import minimize
from scipy import linalg
import numpy as np
from rlpy.Domains.StitchingPackage.benchmark import Benchmark
import math
import pickle
import sys
from sklearn.neighbors import BallTree

class MahalanobisDistance(object):
    """
    A class for optimizing a Mahalanobis distance metric.
    The metric is initialized to the identity matrix, which is equivalent to Euclidean distance.
    Calling the "optimize" function with sets of rollouts attempts to update the
    distance metric so that the objective function is minimized
    http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.mahalanobis.html
    """
    def __init__(self,
                 var_count,
                 stitching,
                 target_policies=[],
                 normalize_starting_metric=True,
                 cached_metric=None):
        """
        :param var_count: The number of variables in the distance metric.
        :param stitching: The Stitching class whose distance metric we are attempting to update.
        :param target_rollouts: The rollouts whose distribution we are attempting to approximate.
          These will be used to repeatedly evaluate the visual fidelity objective.
        :param normalize_starting_metric: Determines whether we scale the metric by the magnitude of the variable's mean.
        :param cached_metric: A pre-computed metric, probably loaded from a file.
        """
        if not (cached_metric is None):
            self.distance_metric = cached_metric
            #return # todo, am I supposed to remove this?
        else:
            self.distance_metric = np.identity(var_count)

        self.stitching = stitching

        self.target_policies = target_policies
        if target_policies:
            self._sampleTargetTrajectories()

        if normalize_starting_metric:
            for idx, variable in enumerate(stitching.labels):
                l = []
                for rollout_set in self.target_rollouts:
                    for rollout in rollout_set:
                        for event in rollout:
                            l.append(event[variable])
                variance = Benchmark.variance(l)
                if variance == 0:
                    variance = 1.0
                self.distance_metric[idx][idx] = 1.0/variance
            if self.stitching.writeNormalizedMetric is not None:
                f = open(self.stitching.writeNormalizedMetric, "wb")
                met = pickle.dump(self.distance_metric, f)
                f.close()

        # The Powell optimizer needs a non-zero value to find the emprical gradient in log space
        for idx, row in enumerate(self.distance_metric):
            for idx2, column in enumerate(row):
                if idx != idx2:
                    pass
                    # todo: additional analysis of this
                    #self.distance_metric[idx][idx2] = .00000000000000001

    def updateInverseVariance(self, inverseVariances):
        """
        Update the Mahalanobis metric for a new set of variances.
        :param inverseVariances: The diagonal of the matrix.
        :return:
        """
        for idx, inverseVariance in enumerate(inverseVariances):
            self.distance_metric[idx][idx] = inverseVariance

    def _sampleTargetTrajectories(self):
        """
        Assign the target trajectories and build the benchmark.
        :param policies:
        :return:
        """
        self.target_policies = self.stitching.targetPolicies
        self.target_rollouts = []
        for policy in self.target_policies:
            t = self.stitching.getRollouts(
                count=self.stitching.targetPoliciesRolloutCount,
                horizon=self.stitching.horizon,
                policy=policy,
                domain=self.stitching.domain)
            self.target_rollouts.append(t)

        self.benchmarks = []
        for idx, rollouts in enumerate(self.target_rollouts):
            benchmark = Benchmark(rollouts, self.stitching.action_count, seed=0)
            self.benchmarks.append(benchmark)

    @staticmethod
    def flatten(matrix):
        """
        Return the current distance metric as a single list of values.
        """
        flattened = []
        for row in matrix:
            for item in row:
                flattened.append(item)
        return flattened

    @staticmethod
    def unflatten(flat_metric):
        """
        Return the current distance metric as a list of lists.
        """
        length = int(math.sqrt(len(flat_metric))) # Get the size of the matrix
        matrix = []
        for i in range(length):
            matrix.append([])
        for idx, val in enumerate(flat_metric):
            matrix[int(idx/length)].append(val)
        return matrix

    @staticmethod
    def is_upper_triangular(matrix):
        """
        Checks whether all lower triangular values are zero.
        Return True iff all lower triangular values are zero.
        """
        assert type(matrix[0]) == list
        for row_idx, row in enumerate(matrix):
            for col_idx, val in enumerate(row):
                if row_idx > col_idx and val != 0:
                    return False
        return True

    @staticmethod
    def is_psd(matrix):
        """
        Checks whether the current matrix is positive semi-definite
        by taking the Cholesky decomposition.
        Returns True iff SciPy succeeds in taking the Cholesky decomp.
        """
        if type(matrix) == list:
            matrix = np.array(matrix)
        try:
            L = linalg.cholesky(matrix, check_finite=True)
        except linalg.LinAlgError:
            return False
        else:
            return True

    @staticmethod
    def ceiling_exponentiate(flat_metric):
        """
        A list exponentiation function that maxes out at sys.float_info.max.
        """
        def new_exp(x):
            try:
                return math.exp(x)
            except Exception:
                if x < 0:
                    return 0
                else:
                    return sys.float_info.max
        return map(new_exp, flat_metric)

    @staticmethod
    def ceiling_logarithm(flat_metric):
        """
        Take the natural logarithm and allow zero values (give negative inf for zero values)
        """
        def new_log(x):
            assert x >= 0
            try:
                return math.log(x)
            except Exception:
                if x == 0:
                    return -sys.float_info.max
                else:
                    assert False # There should never be an under/over flow for this input
        return map(new_log, flat_metric)

    @staticmethod
    def loss(flat_metric,
             stitching,
             benchmarks,
             benchmark_rollout_count=50):
        """
        The function we are trying to minimize when updating the distance metric.
        :param flat_metric: The metric represented as a list of values. This will be converted to
          a matrix when computing distances.
        :param stitching: The Stitching class whose distance metric we are attempting to update.
        :param benchmarks: Instances of the Benchmark class.
        :param self: A hack to make this staticmethod behave more like the MahalanobisDistance class.
          Loss needs to be a static method for the minimization library, but we can still pass in
          the MahalanobisDistance object as self.
        """
        old_tree = stitching.tree
        matrix = MahalanobisDistance.unflatten(MahalanobisDistance.ceiling_exponentiate(flat_metric))
        stitching.tree = BallTree(stitching.database, metric="mahalanobis", VI=np.array(matrix))

        total_benchmark = 0

        # Benchmark against the horizon and target policies of the stitching domain
        rolloutCount = benchmark_rollout_count
        if stitching.rolloutCount < benchmark_rollout_count:
            pass
            #rolloutCount = stitching.rolloutCount
            #print "WARNING!!! You are attempting to find the loss for more trajectories than each DB policy generated"
        horizon = stitching.horizon
        policies = stitching.targetPolicies
        for idx, policy in enumerate(policies):
            benchmark = benchmarks[idx]
            current_benchmark = 0.0
            synthesized_rollouts = stitching.getRollouts(
                count=rolloutCount,
                horizon=horizon,
                policy=policy,
                domain=stitching)
            for label in stitching.labels:
                variable_benchmark = benchmark.benchmark_variable(synthesized_rollouts, label, square=True)
                current_benchmark += variable_benchmark
            action_benchmark = benchmark.benchmark_actions(synthesized_rollouts, square=True)
            current_benchmark += action_benchmark
            total_benchmark += current_benchmark # Square the loss from this policy
        stitching.tree = old_tree

        return total_benchmark

    def optimize(self, benchmark_rollout_count=50):
        """
        Optimize and save the distance metric in non-exponentiated form.
        """

        # The loss function will exponentiate the solution, so our starting point should
        # be the natural log of the solution.
        inverse_exponentiated = MahalanobisDistance.ceiling_logarithm(
            MahalanobisDistance.flatten(self.distance_metric))

        # todo: investigate whether saving in this manner is necessary by removing
        # the save and running the tests
        def print_and_save(vec):
            loss = MahalanobisDistance.loss(
                vec,
                self.stitching,
                self.benchmarks,
                benchmark_rollout_count=benchmark_rollout_count
            )
            print "==Optimization iteration complete=="
            print vec
            print "LOSS:"
            print loss
            if loss < print_and_save.best_loss:
                print_and_save.best_loss = loss
                print_and_save.best_parameters = vec

        print_and_save.best_loss = float("Inf")
        print_and_save.best_parameters = []

        res = minimize(
            MahalanobisDistance.loss,
            inverse_exponentiated,
            args=(self.stitching, self.benchmarks, benchmark_rollout_count),
            method="Powell",
            tol=.000000001,
            options={"disp": True},
            callback=print_and_save)

        print res

        # The result was flattened, need to make square
        matrix = MahalanobisDistance.unflatten(MahalanobisDistance.ceiling_exponentiate(print_and_save.best_parameters))

        assert MahalanobisDistance.is_psd(matrix)
        self.distance_metric = matrix

    def get_matrix_as_np_array(self, matrix=None):
        """
        Return the current distance metric as a NumPy array.
        """
        if matrix:
            return np.array(matrix)
        else:
            return np.array(self.distance_metric)