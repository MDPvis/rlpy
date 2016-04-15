# benchmark.py
# -----------
# Benchmark trajectory synthesis by computing the visual fidelity.
# These functions are used in the unit test suite to ensure
# the fidelity of synthesis for RLPy domains.

import numpy as np
import bisect
import math
import collections

class Benchmark(object):
    """
    Helper class for computing the objective function defined in the draft paper.
    This helper is defined for a set of Monte Carlo rollouts that defines the
    ground truth the synthesis class is attempting to approximate. You cannot
    update the set of rollouts after instantiating this class because the class
    stores a set of summary statistics that are used to compute the benchmark.
    You should create multiple instances of this benchmark, one for each
    policy that you are evaluating.

    This benchmark computes and stores the following for each visualization:
    (a) the vertical range of the axis displayed by the visualization under the MC rollouts,
    (b) the bootstrap esimate of the variance of each quantile in the visualization,
    (c) the variance correction term for each quantile, e^(-quantile sd/(a))

    When benchmarking a new distance metric or database, find
    for each quantile (known as "pixels" in the paper) and each event:
    (d) = (c)*||MC quantile-MFMC quantile/(a)||
    """

    def __init__(self, base_rollouts, action_count, quantiles=[0,10,20,30,40,50,60,70,80,90,100], seed=0):
        """
        Create data structures for scaling each of the contribution to the objective
        functions according to the visualization size and variance of the values.
        This function also records many intermediate inputs for testing purposes.
        The only values that should be used in future computations here are:

        self.objective_scale_variables

        and

        self.objective_scale_actions

        These is broken into two parts because actions are treated differently from variables.

        1. {variable name{event number{quantile name}}}
        2. {action number{event number}}
        """
        self.base_rollouts = base_rollouts
        self.action_count = action_count
        self.quantiles = quantiles
        self.random_state = np.random.RandomState(seed)

        variable_names = self.base_rollouts[0][0].keys()
        variable_names.remove("action")

        # Store the bootstrap estimates of the variance
        self.bootstrap_variables = collections.defaultdict(lambda : collections.defaultdict(dict))
        max_event_number = len(max(self.base_rollouts, key=len))
        for variable_name in variable_names:
            for quantile_number in quantiles:
                for event_number in range(0, max_event_number):
                    variance = self.bootstrap_variable(variable_name, event_number, quantile_number)
                    self.bootstrap_variables[variable_name][event_number][quantile_number] = variance
        self.bootstrap_actions = collections.defaultdict(dict)
        for action_number in range(0, action_count):
            for event_number in range(0, max_event_number):
                variance = self.bootstrap_action(event_number, action_number)
                self.bootstrap_actions[action_number][event_number] = variance

        # Store the selected quantiles within the MC rollouts
        self.mc_variable_quantiles = collections.defaultdict(lambda : collections.defaultdict(dict))
        for variable_name in variable_names:
            for event_number in range(0, max_event_number):
                filter_function = lambda elem: len(elem) > event_number
                filtered_base_rollouts = filter(filter_function, self.base_rollouts)
                sort_function = lambda elem: elem[event_number][variable_name]
                base = sorted(filtered_base_rollouts, key=sort_function)
                for quantile in quantiles:
                    quantile_value = base[int(quantile/100.*(len(base)-1))][event_number][variable_name]
                    self.mc_variable_quantiles[variable_name][event_number][quantile] = quantile_value
        self.mc_action_quantiles = collections.defaultdict(dict)
        for event_number in range(0, max_event_number):
            filter_function = lambda elem: len(elem) > event_number
            filtered_base_rollouts = filter(filter_function, self.base_rollouts)
            number_of_actions_taken = len(filtered_base_rollouts)
            target_action_counts = [0] * self.action_count
            for rollout in filtered_base_rollouts:
                target_action_counts[rollout[event_number]["action"]] += 1
            for action_idx, action_counter in enumerate(target_action_counts):
                self.mc_action_quantiles[event_number][action_idx] = float(action_counter)/float(number_of_actions_taken)

        # The qauntiles that will be shown to the user
        max_quantile = max(quantiles)
        min_quantile = min(quantiles)

        # Store the height in the variable's units of the variable's visualization
        self.viewport_variables = {}
        for variable_name in variable_names:
            current_max = float("-inf")
            current_min = float("inf")
            for event_number in range(0, max_event_number):
                current_max = max(current_max, self.mc_variable_quantiles[variable_name][event_number][max_quantile])
                current_min = min(current_min, self.mc_variable_quantiles[variable_name][event_number][min_quantile])
            self.viewport_variables[variable_name] = abs(current_min - current_max)
            if self.viewport_variables[variable_name] == 0.0:
                print "Variable: {}, had zero range across all time steps. Defaulting to a window of 1".format(variable_name)
                self.viewport_variables[variable_name] = 1.0
        # not needed, always 1
        #self.viewport_actions = {}

        # Store the variance correction for the quantile,
        # e^(-sd/viewport height)
        self.variance_correction_variables = collections.defaultdict(lambda : collections.defaultdict(dict))
        for variable_name in variable_names:
            for quantile_number in quantiles:
                for event_number in range(0, max_event_number):
                    sd = math.sqrt(self.bootstrap_variables[variable_name][event_number][quantile_number])
                    viewport = self.viewport_variables[variable_name]
                    correction = math.exp(-sd/viewport)
                    self.variance_correction_variables[variable_name][event_number][quantile_number] = correction
        self.variance_correction_actions = collections.defaultdict(dict)
        for action_number in range(0, action_count):
            for event_number in range(0, max_event_number):
                sd = math.sqrt(self.bootstrap_actions[action_number][event_number])
                viewport = 1
                correction = math.exp(-sd/viewport)
                self.variance_correction_actions[action_number][event_number] = correction

        # Store the value to multiply each quantile shift by: variance_correction/viewport
        self.objective_scale_variables = collections.defaultdict(lambda : collections.defaultdict(dict))
        for variable_name in variable_names:
            for quantile_number in quantiles:
                for event_number in range(0, max_event_number):
                    scale = self.variance_correction_variables[variable_name][event_number][quantile_number]/self.viewport_variables[variable_name]
                    self.objective_scale_variables[variable_name][event_number][quantile_number] = scale
        self.objective_scale_actions = collections.defaultdict(dict)
        for action_number in range(0, action_count):
            for event_number in range(0, max_event_number):
                scale = self.variance_correction_actions[action_number][event_number]/1.0
                self.objective_scale_actions[action_number][event_number] = scale

    @staticmethod
    def variance(l):
        """
        Find the variance of a real valued list.
        E(x^2) - E(x)^2
        """
        total = 0.
        total_square = 0.
        for sample in l:
            total += sample
        average = total/len(l)
        for sample in l:
            total_square += math.pow(sample-average, 2)
        ret = total_square/len(l)
        assert not math.isnan(ret)
        assert ret >= 0
        return ret

    def bootstrap_variable(self, variable_name, event_number, quantile_number):
        """
        Compute, save, and return the bootstrap estimate of the variance for the
        variables.
        """
        # The number of complete bootstrap resamples to build for estimating the variance
        number_resamples = 20

        def resampled_quantile(quantile, filtered_base_rollouts):
            resampled = []
            sample_size = len(filtered_base_rollouts)
            for sample_number in range(0, sample_size):
                rollout_id = math.floor(self.random_state.uniform(0,1) * sample_size)
                val = filtered_base_rollouts[int(rollout_id)][int(event_number)][variable_name]
                bisect.insort(resampled, val)
            return resampled[int(quantile/100.*(sample_size-1))]

        filter_function = lambda elem: len(elem) > event_number
        filtered_base_rollouts = filter(filter_function, self.base_rollouts)

        resamples = []
        for resample_number in range(0, number_resamples):
            resamples.append(resampled_quantile(quantile_number, filtered_base_rollouts))
        variance = Benchmark.variance(resamples)
        assert not math.isnan(variance)
        assert variance >= 0
        return variance

    def bootstrap_action(self, event_number, action_number):
        """
        Compute, save, and return the bootstrap estimate of the variance for actions.
        """
        # The number of complete bootstrap resamples to build for estimating the variance
        number_resamples = 20

        def resampled_proportion(action_number, filtered_base_rollouts):
            action_count = 0.
            resampled = []
            sample_size = len(filtered_base_rollouts)
            for sample_number in range(0, sample_size):
                rollout_id = math.floor(self.random_state.uniform(0,1) * sample_size)
                current_action = filtered_base_rollouts[int(rollout_id)][int(event_number)]["action"]
                if current_action == action_number:
                    action_count += 1
            # todo: this is a binomial distribution(?) can just sample directly
            return action_count/float(sample_size)

        filter_function = lambda elem: len(elem) > event_number
        filtered_base_rollouts = filter(filter_function, self.base_rollouts)

        resamples = []
        for resample_number in range(0, number_resamples):
            resamples.append(resampled_proportion(action_number, filtered_base_rollouts))
        variance = Benchmark.variance(resamples)
        assert not math.isnan(variance)
        assert variance >= 0
        return variance

    # todo: update the visualization to render actions in a way reflected in this benchmark.
    def benchmark_actions(
      self,
      synthesized_rollouts,
      event_numbers=[],
      weight_objective=True,
      square=False
    ):
        """
        Benchmark the rendering of actions within the visualization as the average shift of the
        proportion of the states in each time step that selects each action.
        """

        if len(max(synthesized_rollouts, key=len)) > len(max(self.base_rollouts, key=len)):
            pass
            #print "Synthesized Length: {} Target Length: {}".format(len(max(synthesized_rollouts, key=len)), len(max(self.base_rollouts, key=len)))
            #raise Exception, "The base rollouts in the benchmark are shorter than the synthesized rollouts"
            #print "WARNING!!!! The base rollouts in the benchmark are shorter than the synthesized rollouts"

        accumulated_distance = 0.

        # sum the infidelity across all events in both sets
        if not event_numbers:
            events_count_in_base_rollouts = len(max(self.base_rollouts, key=len))
            events_count_in_synthesized_rollouts = len(max(synthesized_rollouts, key=len))
            event_numbers = range(0, min(events_count_in_base_rollouts, events_count_in_synthesized_rollouts))

        for event_number in event_numbers:
            filter_function = lambda elem: len(elem) > event_number
            filtered_synthesized_rollouts = filter(filter_function, synthesized_rollouts)
            synthesized_action_counts = [0] * self.action_count
            for rollout in filtered_synthesized_rollouts:
                synthesized_action_counts[rollout[event_number]["action"]] += 1

            fan_max = 1
            fan_min = 0

            if len(filtered_synthesized_rollouts) == 0:
                raise Exception, "The lengths of the rollouts in the two sets are not equal"
            for action_number in range(0, self.action_count):

                scale = 1.
                if weight_objective:
                    scale = self.objective_scale_actions[action_number][event_number]

                # (synthesized current action count / synthesized total action count) - (base current action count / base total action count)
                current_distance = synthesized_action_counts[action_number] / float(len(filtered_synthesized_rollouts)) - \
                    self.mc_action_quantiles[event_number][action_number]
                current_absolute_distance = abs(current_distance)
                accumulated_distance += (current_absolute_distance * scale)

        # Divide by action count because we don't want large action spaces to dominate the objective,
        # but multiply by the quantile count because every variable has multiple quantiles
        ret = accumulated_distance / float(self.action_count) * len(self.quantiles)
        if square:
            ret = math.pow(ret, 2)
        assert not math.isnan(ret)
        assert ret >= 0.0
        return ret

    def benchmark_variable(
      self,
      synthesized_rollouts,
      variable_name,
      event_numbers=[],
      weight_objective=True,
      square=False
    ):
        """
        Get the visual fidelity of a variable. This is taken with respect to the selected event numbers
        and quantiles for a specific variable name. If the selected event numbers are not in the
        base_rollouts or synthesized_rollouts, then an exception is raised.
        Args:
            synthesized_rollouts (list(list(dictionary))): A set of rollouts generated from synthesis.
            variable_name (list(string)): The dictionary keys for the variables we want to benchmark.
            event_numbers (list(int)): The number of events we are evaluating for fidelity.
            Defaults to all the events in both the sets of rollouts.
            quantiles (list(int)): The quantiles to take the visual fidelity of. Defaults to deciles.
            bootstrap_weight (boolean): Indicates whether the benchmark should be weighted by the variance of
            the estimator of the quantile.
        """
        if len(max(synthesized_rollouts, key=len)) > len(max(self.base_rollouts, key=len)):
            pass
            #print "Synthesized Length: {} Target Length: {}".format(len(max(synthesized_rollouts, key=len)), len(max(self.base_rollouts, key=len)))
            #raise Exception, "The base rollouts in the benchmark are shorter than the synthesized rollouts"
            #print "WARNING!!!! The base rollouts in the benchmark are shorter than the synthesized rollouts"

        accumulated_distance = 0.

        # sum the infidelity across all events in both sets
        if not event_numbers:
            events_count_in_base_rollouts = len(max(self.base_rollouts, key=len))
            events_count_in_synthesized_rollouts = len(max(synthesized_rollouts, key=len))
            event_numbers = range(0, min(events_count_in_base_rollouts, events_count_in_synthesized_rollouts))

        for event_number in event_numbers:
            filter_function = lambda elem: len(elem) > event_number
            filtered_synthesized_rollouts = filter(filter_function, synthesized_rollouts)
            sort_function = lambda elem: elem[event_number][variable_name]
            synthesized = sorted(filtered_synthesized_rollouts, key=sort_function)
            if len(synthesized) == 0:
                raise Exception, "The lengths of the rollouts in the two benchmarked sets are not equal"
            for quantile_number in self.quantiles:

                scale = 1.
                if weight_objective:
                    scale = self.objective_scale_variables[variable_name][event_number][quantile_number]

                synthesized_value = synthesized[int(quantile_number/100.*(len(synthesized)-1))][event_number][variable_name]
                current_distance =  self.mc_variable_quantiles[variable_name][event_number][quantile_number] - synthesized_value
                current_absolute_distance = abs(current_distance)
                accumulated_distance += (current_absolute_distance * scale)

        if square:
            accumulated_distance = math.pow(accumulated_distance, 2)

        return accumulated_distance
