"""
Vectorized (matrix) momentum/drift scorer. See scorer_mean_reversion.py's
docstring for the shared contract (price_df in, dict of panels out, one key
named "score") and how to add a new scorer file alongside this one.
"""
import numpy as np
import pandas as pd


def rolling_geometric_drift_score(
    price_df: pd.DataFrame,
    window: int,
    min_periods: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Matrix equivalent of src/scorers/moment_scorers.GeometricDriftScorer.
    score = annualized_log_return - 0.5 * vol**2 over a trailing `window`.
    """
    log_price = np.log(price_df.where(price_df > 0))
    log_returns = log_price.diff()

    vol = log_returns.rolling(window, min_periods=min_periods).std(ddof=1) * np.sqrt(252)
    total_log_return = log_price - log_price.shift(window)

    years = window / 252
    annualized_log_return = total_log_return / years
    score = annualized_log_return - 0.5 * vol ** 2

    n_bars = price_df.rolling(window, min_periods=min_periods).count()
    score = score.where(n_bars >= min_periods)

    return {
        "score": score,
        "volatility": vol,
        "raw_log_return": total_log_return,
    }
