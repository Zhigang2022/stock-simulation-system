"""
Descriptive metrics on df_ranked (vector_calc's output) + the price panel --
no trade-log parsing needed, unlike analysis.build_enhanced_ledger_from_balances
which requires re-deriving positions from backtest.log text.

These are approximations of the true executed NAV (they ignore fees, the
trade delay, and partial fills that TransactionExecutor applies) -- use
them to understand the SCORER's behavior, not as a substitute for the
actual g_state.export_nav_dataframe() result from iteration.run_daily_iteration.

Each function here just COMPUTES a number/table -- whether that number is
statistically distinguishable from noise is significance_test.py's job, not
this file's (see e.g. information_coefficient here vs. ic_significance
there).

calculate_cagr / calculate_volatility_ratios / calculate_drawdown_and_calmar
/ calculate_metrics are merged in from src/metrics.py (copied, not
imported -- same "small enough to fork, not worth a src/ coupling"
reasoning as iteration/'s calendar_snapshot.py and trade_implement.py):
NAV-level performance stats (CAGR, Sharpe/Sortino, max drawdown, Calmar),
complementary to the df_ranked-level diagnostics above.
"""
import numpy as np
import pandas as pd
from scipy import stats


def add_forward_return(df_ranked: pd.DataFrame, price_df: pd.DataFrame, rebalance_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """
    Adds a `forward_return` column: each ticker's simple return from this
    rebalance date to the NEXT rebalance date -- i.e. the return actually
    earned by holding the weight assigned at this date until the next
    rebalance. The last rebalance date has no forward period (NaN).
    """
    rebalance_dates = sorted(rebalance_dates)
    next_date = {d: rebalance_dates[i + 1] for i, d in enumerate(rebalance_dates[:-1])}

    df = df_ranked.copy()
    df["next_date"] = df["date"].map(next_date)

    start_price = price_df.stack(future_stack=True).rename("start_price")
    start_price.index.names = ["date", "ticker"]
    end_price = price_df.stack(future_stack=True).rename("end_price")
    end_price.index.names = ["next_date", "ticker"]

    df = df.join(start_price, on=["date", "ticker"])
    df = df.join(end_price, on=["next_date", "ticker"])
    df["forward_return"] = df["end_price"] / df["start_price"] - 1.0
    return df.drop(columns=["start_price", "end_price"])


def information_coefficient(df_ranked_fwd: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """
    Per rebalance date, the Spearman rank correlation between that date's
    score and the forward_return actually realized by each ticker -- i.e.
    "did ranking tickers by this score actually predict which ones would
    do better." Positive/high = scorer has real predictive power that
    period; near 0 = no edge; negative = scorer is backwards.

    Returns one row per date: `ic` and `n_tickers` (sample size backing it,
    since IC on 3 tickers means much less than IC on 30). Whether the mean
    of this series is a real effect is significance_test.ic_significance's
    job, not this function's.
    """
    def _ic(group: pd.DataFrame) -> pd.Series:
        valid = group[[score_col, "forward_return"]].dropna()
        if len(valid) < 3:
            return pd.Series({"ic": np.nan, "n_tickers": len(valid)})
        ic, _ = stats.spearmanr(valid[score_col], valid["forward_return"])
        return pd.Series({"ic": ic, "n_tickers": len(valid)})

    return df_ranked_fwd.groupby("date").apply(_ic).reset_index()


def contribution_by_ticker(df_ranked_fwd: pd.DataFrame) -> pd.DataFrame:
    """
    Per ticker, total contribution to portfolio return across the whole
    backtest: sum over all rebalance periods of (target_weight *
    forward_return). Sorted descending -- the top rows are what actually
    drove performance, the bottom rows are what hurt it. This is the
    direct answer to "why was this good/bad."
    """
    df = df_ranked_fwd.copy()
    df["contribution"] = df["target_weight"] * df["forward_return"]

    summary = df.groupby("ticker").agg(
        total_contribution=("contribution", "sum"),
        n_periods_held=("target_weight", lambda s: (s > 0).sum()),
        avg_weight_when_held=("target_weight", lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
    ).sort_values("total_contribution", ascending=False)

    return summary


def turnover_by_date(df_ranked: pd.DataFrame) -> pd.Series:
    """
    Per rebalance date, sum of |weight_t - weight_{t-1}| across all
    tickers -- a proxy for how much trading (and fee drag) each rebalance
    implies. 0 = no change from last period, 2.0 = a complete portfolio
    swap (sold everything, bought all new names).
    """
    wide = df_ranked.pivot(index="date", columns="ticker", values="target_weight").fillna(0.0)
    return wide.diff().abs().sum(axis=1).rename("turnover")


def calculate_cagr(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the Compound Annual Growth Rate (CAGR) for each ticker column
    in a DataFrame, determining the time horizon directly from the DatetimeIndex.

    CAGR = (Ending_Value / Beginning_Value) ** (1 / Number_of_Years) - 1
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df_sorted = df.sort_index()

    start_date = df_sorted.index[0]
    end_date = df_sorted.index[-1]

    # Using 365.25 days per year to accurately account for leap years
    num_years = (end_date - start_date).days / 365.25

    if num_years <= 0:
        raise ValueError("The time horizon must be greater than zero days to calculate CAGR.")

    # ffill/bfill ensures robustness if tickers have misaligned missing data at the edges
    beginning_values = df_sorted.bfill().iloc[0]
    ending_values = df_sorted.ffill().iloc[-1]

    cagr_series = (ending_values / beginning_values) ** (1 / num_years) - 1
    cagr_series.name = 'CAGR'
    return cagr_series


def calculate_volatility_ratios(df: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Calculate the Annualized Sharpe Ratio and Sortino Ratio for each ticker
    column. The annualization factor is derived from the median time delta
    of the DatetimeIndex, so it works for daily/weekly/monthly/irregular data.
    """
    if df.empty or len(df) <= 1:
        raise ValueError("Insufficient data to calculate volatility ratios.")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df_sorted = df.sort_index()

    time_deltas = df_sorted.index.to_series().diff().dropna()
    median_delta_days = time_deltas.dt.total_seconds() / (24 * 3600)

    if median_delta_days.median() <= 0:
        raise ValueError("The time steps between index rows must be positive.")

    ann_factor = 365.25 / median_delta_days.median()

    returns = df_sorted.pct_change().dropna(how='all')

    period_rf = (1 + risk_free_rate) ** (1 / ann_factor) - 1

    excess_returns = returns.sub(period_rf, axis=0)
    mean_excess = excess_returns.mean()

    total_vol = returns.std()

    # Downside deviation: isolate negative returns, treat positive returns as 0
    downside_returns = returns.clip(upper=0)
    downside_vol = downside_returns.std()

    ann_excess_return = mean_excess * ann_factor
    ann_total_vol = total_vol * np.sqrt(ann_factor)
    ann_downside_vol = downside_vol * np.sqrt(ann_factor)

    sharpe = np.where(ann_total_vol > 0, ann_excess_return / ann_total_vol, np.nan)
    sortino = np.where(ann_downside_vol > 0, ann_excess_return / ann_downside_vol, np.nan)

    return pd.DataFrame({
        'Sharpe_Ratio': sharpe,
        'Sortino_Ratio': sortino
    }, index=df.columns)


def calculate_drawdown_and_calmar(df: pd.DataFrame, cagr_series: pd.Series) -> pd.DataFrame:
    """
    Calculate the Maximum Drawdown (positive decimal) and Calmar Ratio
    (CAGR / Max Drawdown) for each ticker column.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df_sorted = df.sort_index()

    running_peaks = df_sorted.cummax()
    drawdown_series = (df_sorted - running_peaks) / running_peaks
    # Invert to express as a positive percentage/decimal (industry convention)
    max_drawdown = -drawdown_series.min()

    calmar = np.where(max_drawdown > 0, cagr_series / max_drawdown, np.nan)

    return pd.DataFrame({
        'Max_Drawdown': max_drawdown,
        'Calmar_Ratio': calmar
    }, index=df.columns)


def calculate_metrics(df_nav: pd.DataFrame) -> pd.DataFrame:
    """CAGR + Sharpe/Sortino + Max Drawdown/Calmar, combined into one table."""
    series_cagr = calculate_cagr(df_nav)
    df_vol = calculate_volatility_ratios(df_nav)
    df_dd = calculate_drawdown_and_calmar(df_nav, series_cagr)
    return pd.concat([series_cagr, df_vol, df_dd], axis=1)
