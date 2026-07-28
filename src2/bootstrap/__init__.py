"""
Bootstrap significance-testing infrastructure -- a distinct concern from
vector_calc / iteration / evaluation, since it needs its own synthetic-
world generation machinery rather than just running against the one real
price history. Split by phase:

- data_gen.py:     generates synthetic price/volume worlds (VectorizedBootstrapEngine,
                    forked from src/statioinary_bootstrap.py) and characterizes them
                    by market regime (bull/bear/choppy trend classification).
- simulation.py:   runs the full vector_calc -> iteration backtest on each
                    synthetic world and collects per-world metrics
                    (evaluation.calculate_metrics) -- pure execution, no
                    data generation or significance testing of its own.
- significance.py: paired significance testing across those per-world
                    metrics (scorer vs. benchmark), using
                    evaluation.significance_test's paired-test picker.

This complements evaluation.significance_test.ic_significance, which only
tests date-to-date noise WITHIN the single real history -- this package
resamples the history itself, a different and larger source of
uncertainty.
"""
from . import data_gen
from . import simulation
from . import significance

__all__ = ["data_gen", "simulation", "significance"]
