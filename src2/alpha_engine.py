"""
Vectorized alpha layer: price matrix -> metrics -> filter -> rank -> target
weight, computed once for the whole backtest instead of inside a day-by-day
loop. This replaces Selector+Scorer+Filter+BudgetAllocator's ranking step
(src/selectors/, src/scorers/, src/budget_allocator.py) for the case where
none of those stages need portfolio state -- they only need the price panel
and a list of rebalance dates.

Output is plain, inspectable DataFrames at every stage -- no AuditRecord /
StrategyTelemetry indirection needed to see what happened.

The sequential part of the pipeline (CalendarIterator's day-by-day walk,
TransactionExecutor, GlobalState) is UNCHANGED and still needed downstream:
turning a target_weight table into actual trades depends on path-dependent
state (available cash, trade delay, existing holdings) that cannot be
vectorized away. This module only replaces the "what should we target"
decision, not "how do we get there."
"""
import pandas as pd


def build_metrics_table(
    price_df: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    score_fn,
    **score_fn_kwargs,
) -> pd.DataFrame:
    """
    Runs a vectorized scorer (see vectorized_scorers.py) across the whole
    price panel, then narrows to rebalance dates only and reshapes into a
    long/tidy DataFrame: one row per (date, ticker), one column per metric.
    This IS the audit trail for the scoring stage -- nothing further needs
    to be extracted from it.
    """
    metric_panels = score_fn(price_df, **score_fn_kwargs)

    frames = []
    for metric_name, panel in metric_panels.items():
        sliced = panel.loc[panel.index.isin(rebalance_dates)]
        long = sliced.stack(future_stack=True).rename(metric_name)
        frames.append(long)

    df_metrics = pd.concat(frames, axis=1)
    df_metrics.index.names = ["date", "ticker"]
    return df_metrics.reset_index()


def null_filter(df_metrics: pd.DataFrame) -> pd.Series:
    """
    Dummy filter: nothing is filtered out. Vectorized equivalent of
    src/filters/base_filter.NullFilter -- a template/baseline, and the
    default when no real filter is wired in.

    Every filter function follows this contract: take df_metrics, return a
    boolean Series aligned to df_metrics.index (same length as the input,
    one entry per row), True = eligible/kept, False = filtered out. A real
    filter (liquidity, absolute momentum, etc.) reads whatever columns it
    needs from df_metrics and returns the same shape -- apply_filter_and_rank
    doesn't care how the mask was computed.
    """
    return pd.Series(True, index=df_metrics.index)


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


def apply_filter_and_rank(
    df_metrics: pd.DataFrame,
    score_col: str = "score",
    filter_fn=null_filter,
    top_percent: float = 0.10,
    allocation_type: str = "equal",
) -> pd.DataFrame:
    """
    Cross-sectional filter + rank + target weight, applied per date. Adds
    four columns to df_metrics: `filtered_out` (bool, from filter_fn),
    `rank` (1 = best score that date, among ELIGIBLE rows only; NaN if
    filtered out), and `target_weight` (0.0 if filtered out or not in the
    top tier).

    This is the vectorized equivalent of a SignalFilter + budget_allocator's
    quantile-cut, chained: filter first, then rank/select only among
    survivors -- but keeps rank explicit instead of discarding it (see
    scorers.md / selectors.md notes for why the original loses this).
    """
    df = df_metrics.copy()

    eligible = filter_fn(df)
    assert len(eligible) == len(df), "filter_fn must return one entry per input row"
    df["filtered_out"] = ~pd.Series(eligible, index=df.index).astype(bool).values

    score_for_rank = df[score_col].where(~df["filtered_out"])
    df["rank"] = score_for_rank.groupby(df["date"]).rank(ascending=False, method="first")

    def _target_weights(group: pd.DataFrame) -> pd.Series:
        weights = pd.Series(0.0, index=group.index)
        eligible_group = group.dropna(subset=["rank"])
        if eligible_group.empty:
            return weights

        n_keep = max(1, int(round(len(eligible_group) * top_percent)))
        cutoff_rank = eligible_group["rank"].nsmallest(n_keep).max()
        kept = group["rank"] <= cutoff_rank  # NaN rank (filtered out) -> False

        if allocation_type == "equal":
            weights[kept] = 1.0 / kept.sum() if kept.sum() > 0 else 0.0
        elif allocation_type == "score_weighted":
            total = group.loc[kept, score_col].sum()
            if total > 0:
                weights[kept] = group.loc[kept, score_col] / total
        else:
            raise ValueError(f"Unknown allocation_type: {allocation_type}")

        return weights

    df["target_weight"] = df.groupby("date", group_keys=False).apply(_target_weights)
    return df


def build_target_weight_table(
    price_df: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    score_fn,
    filter_fn=null_filter,
    top_percent: float = 0.10,
    allocation_type: str = "equal",
    score_col: str = "score",
    **score_fn_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end vectorized alpha layer. Returns (df_metrics, df_ranked):
      - df_metrics: raw score + diagnostic metrics, one row per date/ticker.
      - df_ranked: df_metrics + filtered_out/rank/target_weight columns.

    Feed df_ranked's (date, ticker, target_weight) directly into the
    existing sequential execution loop in place of a live selector call.
    """
    df_metrics = build_metrics_table(price_df, rebalance_dates, score_fn, **score_fn_kwargs)
    df_ranked = apply_filter_and_rank(
        df_metrics,
        score_col=score_col,
        filter_fn=filter_fn,
        top_percent=top_percent,
        allocation_type=allocation_type,
    )
    return df_metrics, df_ranked
