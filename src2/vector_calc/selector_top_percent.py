"""
Naive cross-sectional top-percent selector: rank eligible tickers by score
(per date) and assign target weight to the top `top_percent` slice, either
equal-weighted or score-weighted. Vectorized equivalent of
src/selectors/cross_selectors.ComprehensiveMultiHorizonStrategy's ranking +
budget_allocator.py Part A's quantile-cut, fused into one step (see
vect_calculator/calculator.py's docstring for why those two stages are
fused here instead of kept separate the way src/ keeps them).

Every selector_xxx.py in this package follows the same contract: take
df_metrics (already carrying a `filtered_out` bool column from a filter_fn)
plus score_col/params, return the same df with `rank` and `target_weight`
columns added. A new selection rule (e.g. "keep top N names instead of top
percent", or "volatility-weighted") is just a new selector_xxx.py with this
same signature -- calculator.apply_filter_and_rank doesn't care which one
ran.
"""
import pandas as pd


def rank_and_weight_top_percent(
    df_metrics: pd.DataFrame,
    score_col: str = "score",
    top_percent: float = 0.10,
    allocation_type: str = "equal",
) -> pd.DataFrame:
    """
    Adds `rank` (1 = best score that date, among ELIGIBLE rows only; NaN if
    filtered out) and `target_weight` (0.0 if filtered out or not in the
    top tier) columns, computed per date. Requires a `filtered_out` bool
    column already present (see filter_null.py / filter_liquidity.py).
    """
    df = df_metrics.copy()

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
