"""
Vectorized (matrix) momentum-family scorers. See scorer_mean_reversion.py's
docstring for the shared contract (price_df in, dict of panels out, one key
named "score") and how to add a new scorer file alongside this one.

The three regression-based scorers below (rolling_risk_adjusted_score,
rolling_linear_regression_score, rolling_weighted_regression_score) use a
fixed-window OLS/WLS against a LOCAL time index (0..window-1), which is the
same for every window position. That lets each be computed via rolling sums
instead of a per-date/per-ticker regression loop:

    slope = (w*Sxy - Sx*Sy) / (w*Sxx - Sx^2)
    intercept = (Sy - slope*Sx) / w
    SS_res = Syy - intercept*Sy - slope*Sxy
    SS_tot = Syy - Sy^2/w
    r_squared = 1 - SS_res/SS_tot

Sx/Sxx are constants for a given window size. Sxy needs a global-index trick
(see rolling_linear_regression_score) since the local x resets every window.
All three require a FULL window (min_periods defaults to window) -- a
regression on a partial window is not comparable across dates.
"""
import numpy as np
import pandas as pd


def rolling_geometric_drift_score(
    price_df: pd.DataFrame,
    window: int,
    min_periods: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Matrix equivalent of src_old/scorers/moment_scorers.GeometricDriftScorer.
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


def rolling_risk_adjusted_score(
    price_df: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Matrix equivalent of src_old/scorers/moment_scorers.SimpleRiskAdjustedScorer.
    score = raw cumulative simple return over `window` / annualized daily vol.
    """
    min_periods = min_periods or window

    start_price = price_df.shift(window)
    raw_return = price_df / start_price - 1.0

    daily_returns = price_df.pct_change()
    vol = daily_returns.rolling(window, min_periods=min_periods).std(ddof=1) * np.sqrt(252)

    score = (raw_return / vol).where(vol != 0)

    return {
        "score": score,
        "raw_return": raw_return,
        "volatility": vol,
    }


def rolling_linear_regression_score(
    price_df: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Matrix equivalent of src_old/scorers/moment_scorers.TraditionalLinearRegressionScorer.
    Equal-weighted OLS of log price on a local time index over each trailing
    `window`. score = slope * r_squared.
    """
    min_periods = min_periods or window
    n = len(price_df)

    log_price = np.log(price_df.where(price_df > 0))
    g = pd.Series(np.arange(n, dtype=float), index=price_df.index)

    w = float(window)
    sum_x = w * (w - 1) / 2
    sum_x2 = (w - 1) * w * (2 * w - 1) / 6
    denom = w * sum_x2 - sum_x ** 2

    sum_y = log_price.rolling(window, min_periods=min_periods).sum()
    sum_y2 = (log_price ** 2).rolling(window, min_periods=min_periods).sum()

    gy = log_price.mul(g, axis=0)
    sum_gy = gy.rolling(window, min_periods=min_periods).sum()
    offset = g - (w - 1)
    sum_xy = sum_gy.sub(sum_y.mul(offset, axis=0))

    slope = (w * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / w

    ss_tot = sum_y2 - sum_y ** 2 / w
    ss_res = sum_y2 - intercept * sum_y - slope * sum_xy
    r_squared = (1 - ss_res / ss_tot).where(ss_tot > 0, 0.0)

    score = slope * r_squared

    return {
        "score": score,
        "slope": slope,
        "r_squared": r_squared,
    }


def rolling_weighted_regression_score(
    price_df: pd.DataFrame,
    window: int,
    decay_factor: float = 0.98,
    min_periods: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Matrix equivalent of src_old/scorers/moment_scorers.WeightedLinearRegressionScorer.
    WLS of log price on a local time index over each trailing `window`, with
    exponential weight decay_factor**(window-1-i) favoring recent days
    (i=window-1 is the most recent day in the window, weight 1.0).
    score = slope * r_squared (weighted).

    Implementation note: a fixed-window weighted sum S_k = sum over the
    window of decay^(k-m)*y_m can't use a plain rolling sum (the weight
    resets every window), but it can be built from an unbounded recursive
    EWMA-style running sum U_k = y_k + decay*U_{k-1} via the identity
    S_k = U_k - decay**window * U_{k-window}, since U telescopes to exactly
    the windowed weighted sum. That gives closed-form rolling Sy/Sxy/Syy
    without a per-window regression loop.
    """
    min_periods = min_periods or window
    n = len(price_df)
    w = window

    log_price = np.log(price_df.where(price_df > 0))
    y = log_price.fillna(0.0)
    y2 = y ** 2
    gy = y.mul(np.arange(n, dtype=float), axis=0)

    def _running_weighted_sum(values: pd.DataFrame) -> pd.DataFrame:
        # U_k = sum_{m=0}^{k} decay**(k-m) * values_m, via cumulative rescale.
        decay_pow = decay_factor ** np.arange(n, dtype=float)
        rescaled = values.div(decay_pow, axis=0)
        cum = rescaled.cumsum()
        return cum.mul(decay_pow, axis=0)

    def _windowed(values: pd.DataFrame) -> pd.DataFrame:
        u = _running_weighted_sum(values)
        u_lag = u.shift(w) * (decay_factor ** w)
        return u - u_lag.fillna(0.0)

    local_x = np.arange(w, dtype=float)  # 0 (oldest) .. w-1 (most recent)
    weights = decay_factor ** (w - 1 - local_x)
    sum_w = weights.sum()
    sum_wx = (weights * local_x).sum()
    sum_wx2 = (weights * local_x ** 2).sum()
    denom = sum_w * sum_wx2 - sum_wx ** 2

    sum_wy = _windowed(y)
    sum_wy2 = _windowed(y2)

    # Local x resets every window -- same global-index offset trick as the
    # unweighted regression, but weighted: sum(w_i * x_i * y_i)
    #   = sum(w_i * (g_i - offset) * y_i) = sum(w_i * g_i * y_i) - offset*sum(w_i*y_i)
    sum_w_gy = _windowed(gy)
    offset = pd.Series(np.arange(n, dtype=float), index=price_df.index) - (w - 1)
    sum_wxy = sum_w_gy.sub(sum_wy.mul(offset, axis=0))

    slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
    intercept = (sum_wy - slope * sum_wx) / sum_w

    ss_tot = sum_wy2 - sum_wy ** 2 / sum_w
    ss_res = sum_wy2 - intercept * sum_wy - slope * sum_wxy
    r_squared = (1 - ss_res / ss_tot).where(ss_tot > 0, 0.0)

    score = slope * r_squared

    n_bars = price_df.rolling(window, min_periods=min_periods).count()
    score = score.where(n_bars >= min_periods)
    slope = slope.where(n_bars >= min_periods)
    r_squared = r_squared.where(n_bars >= min_periods)

    return {
        "score": score,
        "slope": slope,
        "r_squared": r_squared,
    }
