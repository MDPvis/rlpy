# benchmark.py
# -----------
# Benchmark trajectory synthesis by computing the visual fidelity.
# These functions are used in the unit test suite to ensure
# the fidelity of synthesis for RLPy domains.


# Set the given parameters to obtain the specified policies through
# value iteration.


def benchmark(base_rollouts, synthesized_rollouts, variable_name, event_numbers=[], quantiles=[0,10,20,30,40,50,60,70,80,90,100]):
    """
    Get the visual fidelity of a variable.
    todo: maybe move this function into test_synthesis as a helper?
    Args:
        base_rollouts (list(list(dictionary))): A set of rollouts generated from the simulator.
        synthesized_rollouts (list(list(dictionary))): A set of rollouts generated from synthesis.
        variable_name (string): The dictionary key for the variable we want percentiles from.
        quantiles (list(int)): The quantiles to take the visual fidelity of.
    """
    accumulated_distance = 0

    # sum the infidelity across all events
    if not event_numbers:
        event_numbers = range(0, len(max(base_rollouts, key=len)))

    for event_number in event_numbers:
        sort_function = lambda elem: elem[event_number][variable_name]
        base = sorted(base_rollouts, key=sort_function)
        synthesized = sorted(synthesized_rollouts, key=sort_function)
        for quantile in quantiles:
            current_distance = abs(base[int(quantile/100*(len(base)-1))][event_number][variable_name] - synthesized[int(quantile/100*(len(base)-1))][event_number][variable_name])
            accumulated_distance += current_distance

    return accumulated_distance
