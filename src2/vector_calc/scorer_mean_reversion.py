"""
Vectorized (matrix) mean-reversion scorer. Takes the WHOLE [date x ticker]
price panel and returns a dict of WHOLE [date x ticker] metric panels -- no
per-ticker Python loop, no per-date loop. Every metric at row `date` only
ever depends on data at or before `date` (via .rolling()), which is what
keeps this leak-safe without needing a CalendarIterator-style runtime slice.

To add a new scorer: drop a sibling scorer_xxx.py in this same folder with
the same signature (price_df, **params) -> dict[str, pd.DataFrame], one key
of which must be "score". Nothing else in vector_calc/ needs to change --
calculator.build_metrics_table takes any such function in as a plain
argument (score_fn).
"""
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
