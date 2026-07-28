"""
Diagnostics for "why is this scorer good or bad," computed directly off
df_ranked (alpha_engine's output) + the price panel -- no trade-log
parsing needed, unlike analysis.build_enhanced_ledger_from_balances which
requires re-deriving positions from backtest.log text.

These are approximations of the true executed NAV (they ignore fees, the
trade delay, and partial fills that TransactionExecutor applies) -- use
them to understand the SCORER's behavior, not as a substitute for the
actual g_state.export_nav_dataframe() result from run_vectorized.
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
    since IC on 3 tickers means much less than IC on 30).
    """
    def _ic(group: pd.DataFrame) -> pd.Series:
        valid = group[[score_col, "forward_return"]].dropna()
        if len(valid) < 3:
            return pd.Series({"ic": np.nan, "n_tickers": len(valid)})
        ic, _ = stats.spearmanr(valid[score_col], valid["forward_return"])
        return pd.Series({"ic": ic, "n_tickers": len(valid)})

    return df_ranked_fwd.groupby("date").apply(_ic).reset_index()


def ic_significance(df_ic: pd.DataFrame) -> dict:
    """
    One-sample t-test of whether the mean IC across rebalance dates is
    distinguishable from zero -- i.e. is the average IC reported by
    information_coefficient() a real effect, or within noise given how
    few rebalance dates (and how much date-to-date IC variance) back it.

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
