"""
MACD daily-trend filter. See filter_null.py for the shared filter_xxx.py
contract (df_metrics in, boolean Series out).
"""
import pandas as pd


def make_macd_trend_filter(
    price_df: pd.DataFrame,
    htf_resample: str = "D",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
):
    """
    Eligible (True) rows are those on/after the most recently fully-CLOSED
    higher-timeframe (default daily) bar whose MACD histogram rose vs. the
    bar before it -- i.e. "the daily MACD is trending up." Leak-safe: the
    higher-timeframe trend series is shifted by one bar before being
    broadcast down onto price_df's native frequency, so a same-day/
    still-forming higher-timeframe bar is never used, only the last
    genuinely closed one (resampling the FULL precomputed price_df down to
    a higher timeframe -- unlike resampling a live-truncated snapshot, see
    calendar_snapshot.py -- would otherwise leak same-day future bars into
    the current day's bucket).

    Returns a filter_fn(df_metrics) -> pd.Series[bool] closure, matching
    every other filter_xxx.py in this package (see filter_liquidity.py).
    """
    price_htf = price_df.resample(htf_resample).last().dropna(how="all")
    ema_fast = price_htf.ewm(span=fast, adjust=False).mean()
    ema_slow = price_htf.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    daily_score = macd_line - signal_line

    trending_up = (daily_score.diff() > 0).shift(1)  # last CLOSED htf bar only -- leak-safe
    trending_up_ltf = trending_up.reindex(price_df.index, method="ffill")

    def macd_trend_filter(df_metrics: pd.DataFrame) -> pd.Series:
        trending_up_long = trending_up_ltf.stack(future_stack=True).rename("trending_up")
        trending_up_long.index.names = ["date", "ticker"]

        keyed = df_metrics.set_index(["date", "ticker"])
        merged = keyed.join(trending_up_long)["trending_up"]

        eligible = merged.fillna(False).astype(bool)  # no closed daily bar yet -> not eligible
        return pd.Series(eligible.values, index=df_metrics.index)

    return macd_trend_filter
