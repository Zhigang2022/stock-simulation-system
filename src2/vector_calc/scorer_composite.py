"""
Vectorized (matrix) composite scorer -- blends the momentum and mean-
reversion scorers below into one score. See scorer_mean_reversion.py's
docstring for the shared per-scorer-file contract; this one is the
exception that imports sibling scorer files directly, since blending them
IS the point of this scorer.
"""
import pandas as pd

from .scorer_momentum import rolling_geometric_drift_score
from .scorer_mean_reversion import rolling_mean_reversion_score


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
