"""
Piece 3 of 3 (vector_calc / iteration / evaluation): diagnostics on
df_ranked and on the executed NAV, split by nature:

- metric_calc.py       -- computes a number/table: df_ranked-level (forward return, IC,
                           contribution, turnover) and NAV-level (CAGR, Sharpe/Sortino,
                           drawdown/Calmar -- merged in from src/metrics.py)
- significance_test.py -- is that number statistically real, not noise: single-history
                           (ic_significance) and paired cross-series tests (merged in
                           from src/stat_test.py, used by src2/bootstrap/significance.py)
- plot.py               -- visualizes metric_calc.py's output (and NAV curves generally)

This package has no src/ dependency -- see each file's docstring for what
it was copied from and why (small enough to fork rather than import).
"""
from .metric_calc import (
    add_forward_return,
    information_coefficient,
    contribution_by_ticker,
    turnover_by_date,
    calculate_cagr,
    calculate_volatility_ratios,
    calculate_drawdown_and_calmar,
    calculate_metrics,
)
from .significance_test import (
    ic_significance,
    analyze_metric_difference,
    test_paired_series_parametric,
    test_paired_series_non_parametric,
    automated_paired_test,
    test_paired_series_parametric_one_sided,
    test_paired_series_non_parametric_one_sided,
    automated_paired_test_one_sided,
)
from .plot import (
    plot_rebased_performance,
    plot_ic_over_time,
    plot_contribution_by_ticker,
)

__all__ = [
    "add_forward_return",
    "information_coefficient",
    "contribution_by_ticker",
    "turnover_by_date",
    "calculate_cagr",
    "calculate_volatility_ratios",
    "calculate_drawdown_and_calmar",
    "calculate_metrics",
    "ic_significance",
    "analyze_metric_difference",
    "test_paired_series_parametric",
    "test_paired_series_non_parametric",
    "automated_paired_test",
    "test_paired_series_parametric_one_sided",
    "test_paired_series_non_parametric_one_sided",
    "automated_paired_test_one_sided",
    "plot_rebased_performance",
    "plot_ic_over_time",
    "plot_contribution_by_ticker",
]
