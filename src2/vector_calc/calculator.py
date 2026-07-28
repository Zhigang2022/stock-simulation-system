"""
Orchestration for the vector_calc package: price matrix -> metrics (any
sibling scorer_*.py) -> filter -> rank -> target weight, computed once for
the whole backtest instead of inside a day-by-day loop. This replaces
Selector+Scorer+Filter+BudgetAllocator's ranking step (src/selectors/,
src/scorers/, src/budget_allocator.py Part A) for the case where none of
those stages need portfolio state -- they only need the price panel and a
list of rebalance dates.

Output is plain, inspectable DataFrames at every stage -- no AuditRecord /
StrategyTelemetry indirection needed to see what happened.

The sequential part of the pipeline (day-by-day walk, TransactionExecutor,
GlobalState, tactical/ad-hoc sleeve) lives in src2/iteration/ and is
UNCHANGED here -- turning a target_weight table into actual trades depends
on path-dependent state (available cash, trade delay, existing holdings)
that cannot be vectorized away. This module only replaces the "what should
we target" decision, not "how do we get there."
"""
import pandas as pd

from .filter_null import null_filter
from .selector_top_percent import rank_and_weight_top_percent


def build_metrics_table(
    price_df: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    score_fn,
    **score_fn_kwargs,
) -> pd.DataFrame:
    """
    Runs a vectorized scorer (any scorer_*.py in this package) across the
    whole price panel, then narrows to rebalance dates only and reshapes into a
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


def apply_filter_and_rank(
    df_metrics: pd.DataFrame,
    score_col: str = "score",
    filter_fn=null_filter,
    selector_fn=rank_and_weight_top_percent,
    top_percent: float = 0.10,
    allocation_type: str = "equal",
) -> pd.DataFrame:
    """
    Cross-sectional filter + rank + target weight, applied per date. Adds
    three columns to df_metrics: `filtered_out` (bool, from filter_fn),
    `rank` and `target_weight` (both from selector_fn).

    This is the vectorized equivalent of a SignalFilter + a Selector +
    budget_allocator's quantile-cut, chained: filter first (any
    filter_xxx.py), then rank/select only among survivors (any
    selector_xxx.py) -- but keeps rank explicit instead of discarding it
    (see scorers.md / selectors.md notes for why the original loses this).
    """
    df = df_metrics.copy()

    eligible = filter_fn(df)
    assert len(eligible) == len(df), "filter_fn must return one entry per input row"
    df["filtered_out"] = ~pd.Series(eligible, index=df.index).astype(bool).values

    return selector_fn(df, score_col=score_col, top_percent=top_percent, allocation_type=allocation_type)


def build_target_weight_table(
    price_df: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    score_fn,
    filter_fn=null_filter,
    selector_fn=rank_and_weight_top_percent,
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
        selector_fn=selector_fn,
        top_percent=top_percent,
        allocation_type=allocation_type,
    )
    return df_metrics, df_ranked
