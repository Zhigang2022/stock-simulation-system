"""
Inferential tests: is a computed metric a real effect, or within noise
given how little data backs it.

ic_significance tests date-to-date noise WITHIN the single real history
passed into metric_calc -- see src2/bootstrap/significance.py for the
complementary, larger source of uncertainty (resampling the history itself
across many synthetic worlds, not just testing across dates within one
history). The paired-test functions below (merged in from src/stat_test.py,
copied not imported -- same reasoning as the rest of src2's src/ forks) are
what bootstrap/significance.py's automated_paired_test(_one_sided) build on
for that cross-world comparison.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats


def ic_significance(df_ic: pd.DataFrame) -> dict:
    """
    One-sample t-test of whether the mean IC across rebalance dates
    (metric_calc.information_coefficient's output) is distinguishable from
    zero -- i.e. is the average IC a real effect, or within noise given
    how few rebalance dates (and how much date-to-date IC variance) back
    it.

    Returns {mean_ic, std_ic, n_dates, t_stat, p_value}. p_value < 0.05 is
    the conventional (not sacred) cutoff for "probably not just noise."
    """
    ic_values = df_ic["ic"].dropna()
    t_stat, p_value = stats.ttest_1samp(ic_values, popmean=0.0)
    return {
        "mean_ic": ic_values.mean(),
        "std_ic": ic_values.std(),
        "n_dates": len(ic_values),
        "t_stat": t_stat,
        "p_value": p_value,
    }


def analyze_metric_difference(list_a, list_b, label_a="Long-Term (IS)", label_b="Mid-Term (OOS)"):
    """
    Performs a Welch's T-test to determine if the means of two simulation
    results are statistically significantly different. Prints a formatted
    report and returns {t_statistic, p_value, significant}.
    """
    arr_a = np.array(list_a)
    arr_b = np.array(list_b)

    mean_a, mean_b = np.mean(arr_a), np.mean(arr_b)
    std_a, std_b = np.std(arr_a, ddof=1), np.std(arr_b, ddof=1)

    # equal_var=False handles different volatilities between time horizons
    t_stat, p_value = stats.ttest_ind(arr_a, arr_b, equal_var=False)

    print("=" * 50)
    print("        STATISTICAL SIGNIFICANCE REPORT        ")
    print("=" * 50)
    print(f"{label_a:<20} | Mean: {mean_a:6.2f}% | StdDev: {std_a:6.2f}%")
    print(f"{label_b:<20} | Mean: {mean_b:6.2f}% | StdDev: {std_b:6.2f}%")
    print("-" * 50)
    print(f"Absolute Difference: {abs(mean_a - mean_b):.2f}%")
    print(f"T-Statistic:        {t_stat:.4f}")
    print(f"P-Value:            {p_value:.4e}")
    print("-" * 50)

    alpha = 0.05
    if p_value < alpha:
        print("Conclusion: STATISTICALLY SIGNIFICANT")
        print(f"-> The difference is real. The strategy performed significantly\n"
              f"   differently in the {label_b} period than the {label_a} period.")
    else:
        print("Conclusion: NOT STATISTICALLY SIGNIFICANT")
        print(f"-> The gap could easily be a byproduct of random sampling noise.\n"
              f"   We cannot confidently say the performance changed.")
    print("=" * 50)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha
    }


def test_paired_series_parametric(series1: pd.Series, series2: pd.Series) -> dict:
    """
    Paired Student's t-test on two aligned series.

    H0: The mean difference between the two paired series is zero.
    H1: The mean difference is not zero.
    """
    df_clean = pd.concat([series1, series2], axis=1).dropna()
    t_stat, p_value = stats.ttest_rel(df_clean.iloc[:, 0], df_clean.iloc[:, 1])

    return {
        "test_name": "Paired t-test",
        "statistic": t_stat,
        "p_value": p_value,
        "significant_at_5pct": p_value < 0.05
    }


def test_paired_series_non_parametric(series1: pd.Series, series2: pd.Series) -> dict:
    """
    Wilcoxon Signed-Rank test on two aligned series -- more robust than the
    paired t-test when differences are fat-tailed/skewed/non-normal (common
    for financial return series), since it ranks absolute differences
    instead of assuming normality.

    H0: The median of the differences between the pairs is zero.
    H1: The median of the differences is not zero.
    """
    df_clean = pd.concat([series1, series2], axis=1).dropna()
    statistic, p_value = stats.wilcoxon(df_clean.iloc[:, 0], df_clean.iloc[:, 1])

    return {
        "test_name": "Wilcoxon Signed-Rank Test",
        "statistic": statistic,
        "p_value": p_value,
        "significant_at_5pct": p_value < 0.05
    }


def automated_paired_test(series1: pd.Series, series2: pd.Series):
    """
    Shapiro-Wilk-gated picker: paired t-test if the differences look
    normal, Wilcoxon signed-rank otherwise.
    """
    df_clean = pd.concat([series1, series2], axis=1).dropna()
    differences = df_clean.iloc[:, 0] - df_clean.iloc[:, 1]

    _, normality_p = stats.shapiro(differences)

    if normality_p > 0.05:
        return test_paired_series_parametric(series1, series2)
    else:
        return test_paired_series_non_parametric(series1, series2)


def test_paired_series_parametric_one_sided(
    series1: pd.Series, series2: pd.Series, alternative: str = "greater"
) -> dict:
    """
    One-sided paired Student's t-test on two aligned series. Use when the
    question is directional ("did series1 beat series2") rather than just
    "did it differ" -- more power at the same alpha, at the cost of being
    unable to detect the opposite direction.

    H0: mean(series1 - series2) <= 0 (if alternative="greater")
    H1: mean(series1 - series2) > 0
    """
    if alternative not in ("greater", "less"):
        raise ValueError(f"alternative must be 'greater' or 'less', got {alternative!r}")

    df_clean = pd.concat([series1, series2], axis=1).dropna()
    t_stat, p_value = stats.ttest_rel(df_clean.iloc[:, 0], df_clean.iloc[:, 1], alternative=alternative)

    return {
        "test_name": f"Paired t-test (one-sided, {alternative})",
        "statistic": t_stat,
        "p_value": p_value,
        "significant_at_5pct": p_value < 0.05,
    }


def test_paired_series_non_parametric_one_sided(
    series1: pd.Series, series2: pd.Series, alternative: str = "greater"
) -> dict:
    """
    One-sided Wilcoxon signed-rank test on two aligned series.

    H0: median(series1 - series2) <= 0 (if alternative="greater")
    H1: median(series1 - series2) > 0
    """
    if alternative not in ("greater", "less"):
        raise ValueError(f"alternative must be 'greater' or 'less', got {alternative!r}")

    df_clean = pd.concat([series1, series2], axis=1).dropna()
    statistic, p_value = stats.wilcoxon(df_clean.iloc[:, 0], df_clean.iloc[:, 1], alternative=alternative)

    return {
        "test_name": f"Wilcoxon Signed-Rank Test (one-sided, {alternative})",
        "statistic": statistic,
        "p_value": p_value,
        "significant_at_5pct": p_value < 0.05,
    }


def automated_paired_test_one_sided(series1: pd.Series, series2: pd.Series, alternative: str = "greater"):
    """Shapiro-Wilk-gated picker, one-sided variant of automated_paired_test."""
    df_clean = pd.concat([series1, series2], axis=1).dropna()
    differences = df_clean.iloc[:, 0] - df_clean.iloc[:, 1]

    _, normality_p = stats.shapiro(differences)

    if normality_p > 0.05:
        return test_paired_series_parametric_one_sided(series1, series2, alternative=alternative)
    else:
        return test_paired_series_non_parametric_one_sided(series1, series2, alternative=alternative)
