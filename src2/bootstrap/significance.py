"""
Paired significance testing for scorer_comparison_bootstrap.ipynb.

bootstrap_metrics (from compare.run_one_simulation, one row per
sim_id x scorer) pairs every scorer with every benchmark on the SAME
sim_id -- same synthetic world, so a scorer-vs-benchmark comparison is a
paired-samples problem, not an independent-samples one. Reuses
evaluation.significance_test's paired t-test / Wilcoxon signed-rank /
normality-gated picker, and adds a win-rate + percentile CI on the
per-world differences, which needs no distributional assumption at all and
matches the "ask the same question across worlds" spirit of the bootstrap.
"""
import numpy as np
import pandas as pd

from src2.evaluation import significance_test


def _diff_series(bootstrap_metrics: pd.DataFrame, scorer: str, bench: str, metric: str) -> pd.Series:
    wide = bootstrap_metrics.pivot(index="sim_id", columns="scorer", values=metric)
    missing = {scorer, bench} - set(wide.columns)
    if missing:
        raise ValueError(f"scorer(s) not found in bootstrap_metrics['scorer']: {missing}")
    return (wide[scorer] - wide[bench]).dropna()


def compare_scorer_vs_benchmark(
    bootstrap_metrics: pd.DataFrame,
    scorer: str,
    bench: str,
    metric: str = "CAGR",
) -> dict:
    """
    One scorer vs. one benchmark, one metric, across all sim_ids. Returns
    win-rate/CI (assumption-free) plus the automated paired-test p-value
    (t-test if the differences pass a Shapiro normality check, Wilcoxon
    otherwise -- see evaluation.significance_test.automated_paired_test).
    """
    wide = bootstrap_metrics.pivot(index="sim_id", columns="scorer", values=metric)
    diff = _diff_series(bootstrap_metrics, scorer, bench, metric)

    ci_lo, ci_hi = np.percentile(diff, [2.5, 97.5])
    test_result = significance_test.automated_paired_test(wide[scorer], wide[bench])

    return {
        "scorer": scorer,
        "bench": bench,
        "metric": metric,
        "n_sims": len(diff),
        "mean_diff": diff.mean(),
        "median_diff": diff.median(),
        "win_rate": (diff > 0).mean(),
        "ci_2.5": ci_lo,
        "ci_97.5": ci_hi,
        "test_name": test_result["test_name"],
        "statistic": test_result["statistic"],
        "p_value": test_result["p_value"],
        "significant_at_5pct": test_result["significant_at_5pct"],
    }


def compare_scorer_vs_benchmark_one_sided(
    bootstrap_metrics: pd.DataFrame,
    scorer: str,
    bench: str,
    metric: str = "CAGR",
    alternative: str = "greater",
) -> dict:
    """
    One-sided version of compare_scorer_vs_benchmark: tests H1: scorer >
    bench (alternative="greater") or H1: scorer < bench (alternative="less"),
    instead of just "different from". Use this when the question is
    specifically "did the scorer beat the benchmark", which is what most
    scorer-vs-SPY/QQQ comparisons actually are -- a one-sided test has more
    power for that directional question at the same alpha than the
    two-sided compare_scorer_vs_benchmark.
    """
    wide = bootstrap_metrics.pivot(index="sim_id", columns="scorer", values=metric)
    diff = _diff_series(bootstrap_metrics, scorer, bench, metric)

    ci_lo, ci_hi = np.percentile(diff, [2.5, 97.5])
    test_result = significance_test.automated_paired_test_one_sided(wide[scorer], wide[bench], alternative=alternative)

    return {
        "scorer": scorer,
        "bench": bench,
        "metric": metric,
        "alternative": alternative,
        "n_sims": len(diff),
        "mean_diff": diff.mean(),
        "median_diff": diff.median(),
        "win_rate": (diff > 0).mean(),
        "ci_2.5": ci_lo,
        "ci_97.5": ci_hi,
        "test_name": test_result["test_name"],
        "statistic": test_result["statistic"],
        "p_value": test_result["p_value"],
        "significant_at_5pct": test_result["significant_at_5pct"],
    }


def compare_all_one_sided(
    bootstrap_metrics: pd.DataFrame,
    scorers: list[str],
    benches: list[str],
    metrics: list[str] = ("CAGR", "Sharpe_Ratio", "Sortino_Ratio", "Max_Drawdown", "Calmar_Ratio"),
    alternative: str = "greater",
) -> pd.DataFrame:
    """One-sided version of compare_all."""
    rows = [
        compare_scorer_vs_benchmark_one_sided(bootstrap_metrics, scorer, bench, metric, alternative=alternative)
        for scorer in scorers
        for bench in benches
        for metric in metrics
    ]
    return pd.DataFrame(rows)


def compare_all(
    bootstrap_metrics: pd.DataFrame,
    scorers: list[str],
    benches: list[str],
    metrics: list[str] = ("CAGR", "Sharpe_Ratio", "Sortino_Ratio", "Max_Drawdown", "Calmar_Ratio"),
) -> pd.DataFrame:
    """Every scorer x every benchmark x every metric, one row each."""
    rows = [
        compare_scorer_vs_benchmark(bootstrap_metrics, scorer, bench, metric)
        for scorer in scorers
        for bench in benches
        for metric in metrics
    ]
    return pd.DataFrame(rows)
