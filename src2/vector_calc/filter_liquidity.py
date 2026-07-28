"""
Naive liquidity filter. See filter_null.py for the shared filter_xxx.py
contract (df_metrics in, boolean Series out).
"""
import pandas as pd


def make_liquidity_filter(
    price_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    min_dollar_volume: float = 1_000_000.0,
    volume_window: int = 20,
):
    """
    Vectorized equivalent of src/filters/liquidfilter.LiquidityComplianceFilter.
    Same rule: drop a ticker on a given date if its trailing `volume_window`-bar
    average dollar volume (price * volume, rolling mean) is below
    `min_dollar_volume`. Tickers absent from price_df/volume_df pass through
    untouched, matching LiquidityComplianceFilter's "unknown data -> don't
    screen it" convention.

    Returns a filter_fn(df_metrics) -> pd.Series[bool] closure, ready to
    pass as apply_filter_and_rank(..., filter_fn=make_liquidity_filter(...)).
    The rolling mean is computed once up front over the whole panel (not
    per-row), which is why this is a factory rather than a plain function.
    """
    dollar_volume = price_df * volume_df
    addv = dollar_volume.rolling(volume_window, min_periods=1).mean()
    known_tickers = set(price_df.columns) & set(volume_df.columns)

    def liquidity_filter(df_metrics: pd.DataFrame) -> pd.Series:
        addv_long = addv.stack(future_stack=True).rename("addv")
        addv_long.index.names = ["date", "ticker"]

        keyed = df_metrics.set_index(["date", "ticker"])
        merged_addv = keyed.join(addv_long)["addv"]

        eligible = merged_addv >= min_dollar_volume  # NaN comparisons -> False (drop)
        is_known = merged_addv.index.get_level_values("ticker").isin(known_tickers)
        eligible = eligible.where(is_known, True)  # unknown ticker -> pass through

        return pd.Series(eligible.values, index=df_metrics.index)

    return liquidity_filter
