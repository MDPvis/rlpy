import logging
import nose.tools
from mocks import mock_target_rollouts
from rlpy.Domains.StitchingPackage.benchmark import Benchmark

def test_action_benchmarking_zero_variance():
    """
    Check action contribution to objective function when there is zero
    variance in the selected action.
    """
    benchmark = Benchmark(mock_target_rollouts["simplistic"], 2, quantiles=[0,10,20,30,40,50,60,70,80,90,100], seed=0)
    bench = benchmark.benchmark_actions(
      mock_target_rollouts["simplistic"],
      event_numbers=range(0, len(mock_target_rollouts["simplistic"])),
      weight_objective=True
    )
    assert bench == 0, "bench was {}".format(bench)

    bench = benchmark.benchmark_actions(
      mock_target_rollouts["simplistic opposite action"],
      event_numbers=range(0, len(mock_target_rollouts["simplistic"])),
      weight_objective=True
    )
    assert bench == 22.0, "bench was {}".format(bench)

def test_variable_benchmarking_zero_variance():
    """
    Check variable contribution to objective function when there is zero
    variance in the variable.
    """
    benchmark = Benchmark(mock_target_rollouts["simplistic"], 2, quantiles=[0,10,20,30,40,50,60,70,80,90,100], seed=0)
    bench = benchmark.benchmark_variable(
      mock_target_rollouts["simplistic"],
      "x",
      event_numbers=range(0, len(mock_target_rollouts["simplistic"])),
      weight_objective=True
    )
    assert bench == 0, "bench was {}".format(bench)

    bench = benchmark.benchmark_variable(
      mock_target_rollouts["simplistic opposite state"],
      "x",
      event_numbers=range(0, len(mock_target_rollouts["simplistic"])),
      weight_objective=True
    )
    assert bench == 22, "bench was {}".format(bench)

def test_action_benchmarking_window_1():
    """
    Check cases where all the actions move to the most extreme action.
    """
    benchmark = Benchmark(mock_target_rollouts["viewport 1.0"], 2, quantiles=[0,10,20,30,40,50,60,70,80,90,100], seed=0)
    bench = benchmark.benchmark_actions(
      mock_target_rollouts["viewport 1.0"],
      event_numbers=range(0, len(mock_target_rollouts["viewport 1.0"])),
      weight_objective=True
    )
    assert bench == 0, "bench was {}".format(bench)

    bench = benchmark.benchmark_actions(
      mock_target_rollouts["viewport 1.0 extreme action and state"],
      event_numbers=range(0, len(mock_target_rollouts["viewport 1.0 extreme action and state"])),
      weight_objective=True
    )
    assert abs(bench - 7.78781430931) < 0.0000000001, "bench was {}".format(bench)

def test_variable_benchmarking_window_1():
    """
    Check cases where all the variables get the most extreme value under the target variable.
    """
    benchmark = Benchmark(mock_target_rollouts["viewport 1.0"], 2, quantiles=[0,10,20,30,40,50,60,70,80,90,100], seed=0)
    bench = benchmark.benchmark_variable(
      mock_target_rollouts["viewport 1.0"],
      "x",
      event_numbers=range(0, len(mock_target_rollouts["viewport 1.0"])),
      weight_objective=True
    )
    assert bench == 0, "bench was {}".format(bench)

    bench = benchmark.benchmark_variable(
      mock_target_rollouts["viewport 1.0 extreme action and state"],
      "x",
      event_numbers=range(0, len(mock_target_rollouts["viewport 1.0 extreme action and state"])),
      weight_objective=True
    )
    assert abs(bench - 13.2713523351) < 0.0000000001, "bench was {}".format(bench)


def test_low_variance_variables_matter_more():
    """
    Test the re-weighting of objectives by how easy it is to approximate the quantile value.
    """
    benchmark = Benchmark(mock_target_rollouts["high variance state and action"], 2, quantiles=[0,10,20,30,40,50,60,70,80,90,100], seed=0)

    assert benchmark.viewport_variables["low"] == benchmark.viewport_variables["high"], "Viewports were not equal despite equal variable ranges"
    assert benchmark.bootstrap_variables["low"][0][80] < benchmark.bootstrap_variables["high"][0][80]
    assert benchmark.variance_correction_variables["low"][0][80] > benchmark.variance_correction_variables["high"][0][80]
    assert benchmark.objective_scale_variables["low"][0][80] > benchmark.objective_scale_variables["high"][0][80]

    event_numbers = range(0, 2)
    bench = benchmark.benchmark_variable(
      mock_target_rollouts["high variance state and action"],
      "low",
      event_numbers=event_numbers,
      weight_objective=True
    )
    assert bench == 0, "bench was {}".format(bench)
    bench = benchmark.benchmark_variable(
      mock_target_rollouts["high variance state and action"],
      "high",
      event_numbers=event_numbers,
      weight_objective=True
    )
    assert bench == 0, "bench was {}".format(bench)

    bench_low = benchmark.benchmark_variable(
      mock_target_rollouts["high variance state and action synthetic"],
      "low",
      event_numbers=event_numbers,
      weight_objective=True
    )
    assert abs(bench_low-13.3575579534) < 0.000000001, "bench was {}".format(bench_low)

    bench_high = benchmark.benchmark_variable(
      mock_target_rollouts["high variance state and action synthetic"],
      "high",
      event_numbers=event_numbers,
      weight_objective=True
    )
    assert abs(bench_high-14.1326158343) < 0.000000001, "bench was {}".format(bench_high)
    assert bench_high - bench_low < 1, "bench of the high variance variable {} was too close to low variance variable {}".format(bench_high, bench_low)

def test_bootstrap_sampling():
    pass # todo, it would be good to confirm this is working.
