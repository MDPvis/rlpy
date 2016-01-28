# benchmark.py
# -----------
# Benchmark trajectory synthesis by computing the visual fidelity.
# These functions are used in the unit test suite to ensure
# the fidelity of synthesis for RLPy domains.

import numpy as np

class Benchmark(object):

    def __init__(self):
        pass

    @staticmethod
    def bootstrap_estimate():
        return 1 #todo, implement this.
        # todo: since we estimate the variance for the target rollouts, we only need to generate this estimator
        #       once. This may require refactoring the benchmark function.
        # todo: I can compute this on a sorted vector using the binomial(?) distribution by sampling across the vector with replacement


    # todo: when benchmarking each of the actions seprately, we are actually scoring the proportion of the
    #       actions taking in that time step. We can update the visualization for mutually exclusive
    #       discrete variables to give a single line of the proportion of rollouts where that action
    #       was taken. So there are two todos for this one:
    #       todo: update the visualization to better render actions.
    @staticmethod
    def benchmark_actions(
      base_rollouts,
      synthesized_rollouts,
      action_count,
      event_numbers=[],
      bootstrap_weight=True):
        """
        Benchmark the rendering of actions within the visualization as the average shift of the
        proportion of the states in each time step that selects each action.
        """

        if len(max(synthesized_rollouts, key=len)) > len(max(base_rollouts, key=len)):
            raise Exception, "The base rollouts in the benchmark are shorter than the synthesized rollouts"

        accumulated_distance = 0

        # sum the infidelity across all events in both sets
        if not event_numbers:
            events_count_in_base_rollouts = len(max(base_rollouts, key=len))
            events_count_in_synthesized_rollouts = len(max(synthesized_rollouts, key=len))
            event_numbers = range(0, min(events_count_in_base_rollouts, events_count_in_synthesized_rollouts))

        for event_number in event_numbers:
            filter_function = lambda elem: len(elem) > event_number
            filtered_base_rollouts = filter(filter_function, base_rollouts)
            filtered_synthesized_rollouts = filter(filter_function, synthesized_rollouts)
            synthesized_action_counts = [0] * action_count
            target_action_counts = [0] * action_count
            for rollout in filtered_synthesized_rollouts:
                synthesized_action_counts[rollout[event_number]["action"]] += 1
            for rollout in filtered_base_rollouts:
                target_action_counts[rollout[event_number]["action"]] += 1

            fan_max = 1
            fan_min = 0

            if len(filtered_base_rollouts) == 0 or len(filtered_synthesized_rollouts) == 0:
                raise Exception, "The lengths of the rollouts in the two benchmarked sets are not equal"
            for action in range(action_count):

                bootstrap_scale = 1
                if bootstrap_weight:
                    bootstrap_scale = Benchmark.bootstrap_estimate()

                # (synthesized current action count / synthesized total action count) - (base current action count / base total action count)
                current_distance = (synthesized_action_counts[action] / len(filtered_synthesized_rollouts)) - (target_action_counts[action] / len(filtered_base_rollouts))
                current_absolute_distance = abs(current_distance)
                accumulated_distance += (current_absolute_distance * bootstrap_scale)
        return accumulated_distance / action_count

    @staticmethod
    def benchmark_variable(
      base_rollouts,
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
            base_rollouts (list(list(dictionary))): A set of rollouts generated from the simulator.
            synthesized_rollouts (list(list(dictionary))): A set of rollouts generated from synthesis.
            variable_name (list(string)): The dictionary keys for the variables we want to benchmark.
            event_numbers (list(int)): The number of events we are evaluating for fidelity.
            Defaults to all the events in both the sets of rollouts.
            quantiles (list(int)): The quantiles to take the visual fidelity of. Defaults to deciles.
            bootstrap_weight (boolean): Indicates whether the benchmark should be weighted by the variance of
            the estimator of the quantile.
        """
        if len(max(synthesized_rollouts, key=len)) > len(max(base_rollouts, key=len)):
            raise Exception, "The base rollouts in the benchmark are shorter than the synthesized rollouts"

        accumulated_distance = 0

        # sum the infidelity across all events in both sets
        if not event_numbers:
            events_count_in_base_rollouts = len(max(base_rollouts, key=len))
            events_count_in_synthesized_rollouts = len(max(synthesized_rollouts, key=len))
            event_numbers = range(0, min(events_count_in_base_rollouts, events_count_in_synthesized_rollouts))

        fan_max = float("-inf")
        fan_min = float("inf")

        for event_number in event_numbers:
            filter_function = lambda elem: len(elem) > event_number
            filtered_base_rollouts = filter(filter_function, base_rollouts)
            filtered_synthesized_rollouts = filter(filter_function, synthesized_rollouts)
            sort_function = lambda elem: elem[event_number][variable_name]
            base = sorted(filtered_base_rollouts, key=sort_function)
            synthesized = sorted(filtered_synthesized_rollouts, key=sort_function)

            fan_max = max(base[len(base) - 1][event_number][variable_name], fan_max)
            fan_min = min(base[0][event_number][variable_name], fan_min)

            if len(base) == 0 or len(synthesized) == 0:
                raise Exception, "The lengths of the rollouts in the two benchmarked sets are not equal"
            for quantile in quantiles:

                bootstrap_scale = 1
                if bootstrap_weight:
                    bootstrap_scale = Benchmark.bootstrap_estimate()

                base_value = base[int(quantile/100*(len(base)-1))][event_number][variable_name]
                synthesized_value = synthesized[int(quantile/100*(len(synthesized)-1))][event_number][variable_name]
                current_distance =  base_value - synthesized_value
                current_absolute_distance = abs(current_distance)
                accumulated_distance += (current_absolute_distance * bootstrap_scale)

        # The distance is currently expressed in the units of the variable, which means the cost between
        # different variables is not comparable. Since the lines will shift on the visual interface
        # from between the max and min values of the base_rollouts, we can normalize on this range.
        height = fan_max - fan_min
        if height == 0:
            return 0
        else:
            return accumulated_distance / height
