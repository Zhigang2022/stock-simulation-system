"""
Naive pass-through filter. Every filter_xxx.py in this package follows the
same contract: take df_metrics, return a boolean Series aligned to
df_metrics.index (same length, one entry per row), True = eligible/kept,
False = filtered out. calculator.apply_filter_and_rank doesn't care how the
mask was computed, so a new filter is just a new filter_xxx.py with this
same signature -- nothing else needs to change.
"""
import pandas as pd


def null_filter(df_metrics: pd.DataFrame) -> pd.Series:
    """
    Dummy filter: nothing is filtered out. Vectorized equivalent of
    src/filters/base_filter.NullFilter -- a template/baseline, and the
    default when no real filter is wired in.
    """
    return pd.Series(True, index=df_metrics.index)
