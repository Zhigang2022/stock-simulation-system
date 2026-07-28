"""
Vectorized (matrix) versions of the scorer math in src/scorers/. Each
function takes the WHOLE [date x ticker] price panel and returns a dict of
WHOLE [date x ticker] metric panels -- no per-ticker Python loop, no
per-date loop. Every metric at row `date` only ever depends on data at or
before `date` (via .rolling()/.shift()), which is what keeps this leak-safe
without needing a CalendarIterator-style runtime slice.
"""
import numpy as np
import pandas as pd


def rolling_mean_reversion_score(
    price_df: pd.DataFrame,
    window: int,
    min_periods: int = 20,
    clip_z: float | None = 3.0,
) -> dict[str, pd.DataFrame]:
    """
    Matrix equivalent of src/scorers/mean_reversion_scorers.MeanReversionScorer.
    score = -z_score, z_score = (price - rolling_mean) / rolling_std.
    """
    window_mean = price_df.rolling(window, min_periods=min_periods).mean()
    window_std = price_df.rolling(window, min_periods=min_periods).std(ddof=1)

    raw_z = (price_df - window_mean) / window_std
    raw_z = raw_z.where(window_std != 0, 0.0)

    z_score = raw_z.clip(-clip_z, clip_z) if clip_z is not None else raw_z
    score = -z_score

    return {
        "score": score,
        "z_score": z_score,
        "raw_z_score": raw_z,
        "window_mean": window_mean,
        "window_std": window_std,
    }


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


def rolling_composite_score(
    price_df: pd.DataFrame,
    momentum_window: int = 252,
    reversion_window: int = 20,
    momentum_weight: float = 0.5,
    min_periods: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Blends momentum (rolling_geometric_drift_score) and mean-reversion
    (rolling_mean_reversion_score) into one score: momentum_weight * their
    cross-sectional PERCENTILE RANKS, blended per date. Ranks (not raw
    values) are blended because the two scores live on completely
    different scales (log-return-ish vs a z-score) -- averaging raw values
    would let whichever happens to have larger magnitude dominate.

    momentum_weight=1.0 reduces to pure momentum, 0.0 to pure reversion.
    """
    mom = rolling_geometric_drift_score(price_df, momentum_window, min_periods)
    rev = rolling_mean_reversion_score(price_df, reversion_window, min_periods)

    mom_rank = mom["score"].rank(axis=1, pct=True)
    rev_rank = rev["score"].rank(axis=1, pct=True)
    composite = momentum_weight * mom_rank + (1.0 - momentum_weight) * rev_rank

    return {
        "score": composite,
        "momentum_score": mom["score"],
        "reversion_score": rev["score"],
    }
