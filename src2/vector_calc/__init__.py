"""
Piece 1 of 3 (vector_calc / iteration / evaluation): price matrix -> score ->
filter -> select/rank -> target weight, computed once for the whole backtest.

- scorer_*.py: one file per scorer (the math only). Add a new scorer by
  adding a new scorer_xxx.py here with signature
  (price_df, **params) -> dict[str, pd.DataFrame], one key of which is
  "score" -- nothing else in this package needs to change.
- filter_*.py: one file per filter (naive so far: null, liquidity). Each
  filter_xxx.py takes df_metrics and returns a boolean Series (True = kept).
- selector_*.py: one file per selection rule (naive so far: top-percent
  rank + equal/score-weighted). Each selector_xxx.py takes df_metrics
  (with `filtered_out` already set by a filter_fn) and returns it with
  `rank`/`target_weight` columns added.
- calculator.py: orchestration -- wires scorer -> filter -> selector into
  df_metrics -> df_ranked, independent of which of each was used.
"""
from .calculator import (
    build_metrics_table,
    apply_filter_and_rank,
    build_target_weight_table,
)
from .scorer_mean_reversion import rolling_mean_reversion_score
from .scorer_momentum import rolling_geometric_drift_score
from .scorer_composite import rolling_composite_score
from .filter_null import null_filter
from .filter_liquidity import make_liquidity_filter
from .selector_top_percent import rank_and_weight_top_percent

__all__ = [
    "build_metrics_table",
    "apply_filter_and_rank",
    "build_target_weight_table",
    "rolling_mean_reversion_score",
    "rolling_geometric_drift_score",
    "rolling_composite_score",
    "null_filter",
    "make_liquidity_filter",
    "rank_and_weight_top_percent",
]
