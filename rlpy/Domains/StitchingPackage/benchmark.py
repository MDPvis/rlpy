# benchmark.py
# -----------
# Benchmark trajectory synthesis by computing the visual fidelity.
# These functions are used in the unit test suite to ensure
# the fidelity of synthesis for RLPy domains.

import numpy as np
import bisect
import math

class Benchmark(object):

    def __init__(self, base_rollouts, action_count, seed=0):
        """
        Create a data structure for storing bootstrap estimates of variance for the quantile.

        This is broken into two parts because actions are treated differently from variables.

        1. {variable name{event number{quantile name}}}
        2. {action number{event number}}
        """
        self.base_rollouts = base_rollouts
        self.action_count = action_count
        self.bootstrap_variables = {}
        self.bootstrap_actions = {}
        self.random_state = np.random.RandomState(seed)

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
        print "total_square: {}".format(total_square)
        print "total: {}".format(total)
        print "ret: {}".format(ret)
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

        def resampled_quantile(quantile):
            resampled = []
            sample_size = len(self.base_rollouts)
            for sample_number in range(0, sample_size):
                rollout_id = math.floor(self.random_state.uniform(0,1) * sample_size)
                val = self.base_rollouts[int(rollout_id)][int(event_number)][variable_name]
                bisect.insort(resampled, val)
            return resampled[int(quantile/100.*(sample_size-1))]

        if not variable_name in self.bootstrap_variables:
            self.bootstrap_variables[variable_name] = {}
        if not event_number in self.bootstrap_variables[variable_name]:
            self.bootstrap_variables[variable_name][event_number] = {}
        if not quantile_number in self.bootstrap_variables[variable_name][event_number]:
            resamples = []
            for resample_number in range(0, number_resamples):
                resamples.append(resampled_quantile(quantile_number))
            self.bootstrap_variables[variable_name][event_number][quantile_number] = Benchmark.variance(resamples)
        assert not math.isnan(self.bootstrap_variables[variable_name][event_number][quantile_number])
        assert self.bootstrap_variables[variable_name][event_number][quantile_number] >= 0
        return self.bootstrap_variables[variable_name][event_number][quantile_number]

    def bootstrap_action(self, event_number, action_number):
        """
        Compute, save, and return the bootstrap estimate of the variance for actions.
        """
        # The number of complete bootstrap resamples to build for estimating the variance
        number_resamples = 20

        def resampled_proportion(action_number):
            action_count = 0.
            resampled = []
            sample_size = len(self.base_rollouts)
            for sample_number in range(0, sample_size):
                rollout_id = math.floor(self.random_state.uniform(0,1) * sample_size)
                current_action = self.base_rollouts[int(rollout_id)][int(event_number)]["action"]
                if current_action == action_number:
                    action_count += 1
            # todo: this is a binomial distribution(?) can just sample directly
            return action_count/sample_size

        if not event_number in self.bootstrap_actions:
            self.bootstrap_actions[event_number] = {}
        if not action_number in self.bootstrap_actions[event_number]:
            resamples = []
            for resample_number in range(0, number_resamples):
                resamples.append(resampled_proportion(action_number))
            self.bootstrap_actions[event_number][action_number] = Benchmark.variance(resamples)
        assert not math.isnan(self.bootstrap_actions[event_number][action_number])
        assert self.bootstrap_actions[event_number][action_number] >= 0
        return self.bootstrap_actions[event_number][action_number]


    # todo: update the visualization to render actions in a way reflected in this benchmark.
    def benchmark_actions(
      self,
      synthesized_rollouts,
      event_numbers=[],
      bootstrap_weight=True):
        """
        Benchmark the rendering of actions within the visualization as the average shift of the
        proportion of the states in each time step that selects each action.
        """

        if len(max(synthesized_rollouts, key=len)) > len(max(self.base_rollouts, key=len)):
            print "Synthesized Length: {} Target Length: {}".format(len(max(synthesized_rollouts, key=len)), len(max(self.base_rollouts, key=len)))
            raise Exception, "The base rollouts in the benchmark are shorter than the synthesized rollouts"

        accumulated_distance = 0.

        # sum the infidelity across all events in both sets
        if not event_numbers:
            events_count_in_base_rollouts = len(max(self.base_rollouts, key=len))
            events_count_in_synthesized_rollouts = len(max(synthesized_rollouts, key=len))
            event_numbers = range(0, min(events_count_in_base_rollouts, events_count_in_synthesized_rollouts))

        for event_number in event_numbers:
            filter_function = lambda elem: len(elem) > event_number
            filtered_base_rollouts = filter(filter_function, self.base_rollouts)
            filtered_synthesized_rollouts = filter(filter_function, synthesized_rollouts)
            synthesized_action_counts = [0] * self.action_count
            target_action_counts = [0] * self.action_count
            for rollout in filtered_synthesized_rollouts:
                synthesized_action_counts[rollout[event_number]["action"]] += 1
            for rollout in filtered_base_rollouts:
                target_action_counts[rollout[event_number]["action"]] += 1

            fan_max = 1
            fan_min = 0

            if len(filtered_base_rollouts) == 0 or len(filtered_synthesized_rollouts) == 0:
                raise Exception, "The lengths of the rollouts in the two benchmarked sets are not equal"
            for action in range(self.action_count):

                bootstrap_scale = 1.
                if bootstrap_weight:
                    bootstrap_variance = self.bootstrap_action(event_number, action)
                    if bootstrap_variance == 0:
                        print "warning: {} has zero variance, scaling by 1 instead of 0".format(action)
                    else:
                        bootstrap_scale = 1./bootstrap_variance

                # (synthesized current action count / synthesized total action count) - (base current action count / base total action count)
                current_distance = (synthesized_action_counts[action] / len(filtered_synthesized_rollouts)) - (target_action_counts[action] / len(filtered_base_rollouts))
                current_absolute_distance = abs(current_distance)
                accumulated_distance += (current_absolute_distance * bootstrap_scale)
        ret = accumulated_distance / self.action_count
        assert not math.isnan(ret)
        return ret

    def benchmark_variable(
      self,
      synthesized_rollouts,
      variable_name,
      event_numbers=[],
      quantiles=[0,10,20,30,40,50,60,70,80,90,100],
      bootstrap_weight=True):
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
            print "Synthesized Length: {} Target Length: {}".format(len(max(synthesized_rollouts, key=len)), len(max(self.base_rollouts, key=len)))
            raise Exception, "The base rollouts in the benchmark are shorter than the synthesized rollouts"

        accumulated_distance = 0.

        # sum the infidelity across all events in both sets
        if not event_numbers:
            events_count_in_base_rollouts = len(max(self.base_rollouts, key=len))
            events_count_in_synthesized_rollouts = len(max(synthesized_rollouts, key=len))
            event_numbers = range(0, min(events_count_in_base_rollouts, events_count_in_synthesized_rollouts))

        fan_max = float("-inf")
        fan_min = float("inf")

        for event_number in event_numbers:
            filter_function = lambda elem: len(elem) > event_number
            filtered_base_rollouts = filter(filter_function, self.base_rollouts)
            filtered_synthesized_rollouts = filter(filter_function, synthesized_rollouts)
            sort_function = lambda elem: elem[event_number][variable_name]
            base = sorted(filtered_base_rollouts, key=sort_function)
            synthesized = sorted(filtered_synthesized_rollouts, key=sort_function)

            fan_max = max(base[len(base) - 1][event_number][variable_name], fan_max)
            fan_min = min(base[0][event_number][variable_name], fan_min)

            if len(base) == 0 or len(synthesized) == 0:
                raise Exception, "The lengths of the rollouts in the two benchmarked sets are not equal"
            for quantile in quantiles:

                bootstrap_scale = 1.
                if bootstrap_weight:
                    bootstrap_variance = self.bootstrap_variable(variable_name, event_number, quantile)
                    if bootstrap_variance == 0:
                        print "warning: {} has zero variance, scaling by 1 instead of 0".format(variable_name)
                    else:
                        bootstrap_scale = 1./bootstrap_variance

                base_value = base[int(quantile/100.*(len(base)-1))][event_number][variable_name]
                synthesized_value = synthesized[int(quantile/100.*(len(synthesized)-1))][event_number][variable_name]
                current_distance =  base_value - synthesized_value
                current_absolute_distance = abs(current_distance)
                accumulated_distance += (current_absolute_distance * bootstrap_scale)

        # The distance is currently expressed in the units of the variable, which means the cost between
        # different variables is not comparable. Since the lines will shift on the visual interface
        # from between the max and min values of the base_rollouts, we can normalize on this range.
        height = fan_max - fan_min
        if height == 0:
            return 0.
        else:
            ret = accumulated_distance / height
            assert not math.isnan(ret)
            return ret
