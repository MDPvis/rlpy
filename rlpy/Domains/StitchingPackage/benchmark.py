# benchmark.py
# -----------
# Benchmark trajectory synthesis by computing the visual fidelity.
# These functions are used in the unit test suite to ensure
# the fidelity of synthesis for RLPy domains.


# Set the given parameters to obtain the specified policies through
# value iteration.


def benchmark(
  base_rollouts,
  synthesized_rollouts,
  variable_name,
  event_numbers=[],
  quantiles=[0,10,20,30,40,50,60,70,80,90,100]):
    """
    Get the visual fidelity of a variable. This is taken with respect to the selected event numbers
    and quantiles for a specific variable name. If the selected event numbers are not in the
    base_rollouts or synthesized_rollouts, then an exception is raised.
    Args:
        base_rollouts (list(list(dictionary))): A set of rollouts generated from the simulator.
        synthesized_rollouts (list(list(dictionary))): A set of rollouts generated from synthesis.
        variable_name (string): The dictionary key for the variable we want percentiles from.
        event_numbers (list(int)): The number of events we are evaluating for fidelity.
        Defaults to all the events in both the sets of rollouts.
        quantiles (list(int)): The quantiles to take the visual fidelity of. Defaults to deciles.
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
        sort_function = lambda elem: elem[event_number][variable_name]
        base = sorted(filtered_base_rollouts, key=sort_function)
        synthesized = sorted(filtered_synthesized_rollouts, key=sort_function)
        if len(base) == 0 or len(synthesized) == 0:
            raise Exception, "The lengths of the rollouts in the two benchmarked sets are not equal"
        for quantile in quantiles:
            base_value = base[int(quantile/100*(len(base)-1))][event_number][variable_name]
            synthesized_value = synthesized[int(quantile/100*(len(synthesized)-1))][event_number][variable_name]
            current_distance =  base_value - synthesized_value
            current_absolute_distance = abs(current_distance)
            accumulated_distance += current_absolute_distance

    return accumulated_distance
