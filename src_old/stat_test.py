import numpy as np
import scipy.stats as stats

def analyze_metric_difference(list_a, list_b, label_a="Long-Term (IS)", label_b="Mid-Term (OOS)"):
    """
    Performs a Welch's T-test to determine if the means of two simulation 
    results are statistically significantly different.
    """
    # Convert inputs to numpy arrays just in case
    arr_a = np.array(list_a)
    arr_b = np.array(list_b)
    
    # Calculate basic descriptive stats
    mean_a, mean_b = np.mean(arr_a), np.mean(arr_b)
    std_a, std_b = np.std(arr_a, ddof=1), np.std(arr_b, ddof=1)
    
    # Run Welch's T-test (equal_var=False handles different volatilities between time horizons)
    t_stat, p_value = stats.ttest_ind(arr_a, arr_b, equal_var=False)
    
    # Format and print the report
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
    
    # Interpret the P-value (using the standard 5% significance threshold)
    alpha = 0.05
    if p_value < alpha:
        print("Conclusion: STATISTICALLY SIGNIFICANT")
        print(f"-> The difference is real. The strategy performed significantly\n"
              f"   differently in the {label_b} period than the {label_a} period.")
    else:
        print("Conclusion: NOT STATISTICALLY SIGNIFICANT")
        print(f"-> The $2.5% gap could easily be a byproduct of random sampling noise.\n"
              f"   We cannot confidently say the performance changed.")
    print("=" * 50)
    
    # Return results as a dictionary for further programmatic use if needed
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha
    }


import scipy.stats as stats
import pandas as pd

def test_paired_series_parametric(series1: pd.Series, series2: pd.Series) -> dict:
    """
    Perform a Paired Student's t-test on two aligned series.
    
    H0: The mean difference between the two paired series is zero (they are the same).
    H1: The mean difference is not zero (they are significantly different).
    """
    # Drop any missing values to ensure perfect alignment
    df_clean = pd.concat([series1, series2], axis=1).dropna()
    
    # Run the paired t-test
    t_stat, p_value = stats.ttest_rel(df_clean.iloc[:, 0], df_clean.iloc[:, 1])
    
    return {
        "test_name": "Paired t-test",
        "statistic": t_stat,
        "p_value": p_value,
        "significant_at_5pct": p_value < 0.05
    }

'''
If your data represents financial asset returns, trading strategy metrics, or options pricing data,
 it will likely exhibit fat tails, skewness, or non-normal characteristics. 
 In this scenario, the Wilcoxon Signed-Rank Test is much more robust. 
 It ranks the absolute differences between the pairs and doesn't assume a normal distribution
'''

def test_paired_series_non_parametric(series1: pd.Series, series2: pd.Series) -> dict:
    """
    Perform a Wilcoxon Signed-Rank test on two aligned series.
    
    H0: The median of the differences between the pairs is zero.
    H1: The median of the differences is not zero.
    """
    df_clean = pd.concat([series1, series2], axis=1).dropna()
    
    # Run the Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(df_clean.iloc[:, 0], df_clean.iloc[:, 1])
    
    return {
        "test_name": "Wilcoxon Signed-Rank Test",
        "statistic": statistic,
        "p_value": p_value,
        "significant_at_5pct": p_value < 0.05
    }

'''
You can explicitly test which method to use by checking if the differences between the series 
are normally distributed using the Shapiro-Wilk test:
'''
def automated_paired_test(series1: pd.Series, series2: pd.Series):
    df_clean = pd.concat([series1, series2], axis=1).dropna()
    differences = df_clean.iloc[:, 0] - df_clean.iloc[:, 1]

    # Test for normality
    _, normality_p = stats.shapiro(differences)

    if normality_p > 0.05:
        # Differences are normal -> Use Paired t-test
        return test_paired_series_parametric(series1, series2)
    else:
        # Differences are not normal -> Use Wilcoxon
        return test_paired_series_non_parametric(series1, series2)


'''
One-sided variants: same pairing/statistics as above, but testing a
directional H1 (series1 > series2, or series1 < series2) instead of
"different from zero". Use these when the question is specifically
"did the scorer beat the benchmark", not just "did it differ from it" --
a one-sided test has more power for that directional question at the
same alpha, at the cost of being unable to detect the opposite direction.
`alternative` is passed straight through to scipy, so it must be one of
scipy's own labels: "greater" (series1 > series2) or "less" (series1 < series2).
'''

def test_paired_series_parametric_one_sided(
    series1: pd.Series, series2: pd.Series, alternative: str = "greater"
) -> dict:
    """
    One-sided paired Student's t-test on two aligned series.

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
    df_clean = pd.concat([series1, series2], axis=1).dropna()
    differences = df_clean.iloc[:, 0] - df_clean.iloc[:, 1]

    # Test for normality
    _, normality_p = stats.shapiro(differences)

    if normality_p > 0.05:
        # Differences are normal -> Use one-sided paired t-test
        return test_paired_series_parametric_one_sided(series1, series2, alternative=alternative)
    else:
        # Differences are not normal -> Use one-sided Wilcoxon
        return test_paired_series_non_parametric_one_sided(series1, series2, alternative=alternative)