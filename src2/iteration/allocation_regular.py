"""
Regular (CORE sleeve) allocation. Vectorized equivalent of
budget_allocator.py's Part A -- this is where the SIZING decision actually
belongs, not vect_calc/selector_top_percent.py.

vect_calc's selector is precomputed once, stateless, with no g_state -- it
can only answer "which tickers are attractive on this date" (rank +
top_percent cut), plus a naive suggested weight (equal / score_weighted)
that's good enough for fast scorer diagnostics (evaluation.py,
bootstrap_compare.py) where paying for the full stateful loop isn't worth
it. It structurally cannot answer "how many dollars, given what I already
hold and how much cash is free" -- that's live, path-dependent state only
this layer has.

So `weight_fn` here is the real sizing hook: it gets g_state (current
core_positions/core_cash), today's selected rows, and market_prices, and
decides the actual target weights. The default just forwards vect_calc's
precomputed target_weight (today's naive baseline), but a more complex
weight_fn (e.g. only rebalance past a drift threshold, cap turnover, blend
with existing core_positions) can be swapped in without touching vect_calc.
"""
import pandas as pd


def naive_precomputed_weight(g_state, day_rows: pd.DataFrame, market_prices: pd.Series) -> dict[str, float]:
    """Default weight_fn: just forwards vect_calc's precomputed target_weight column."""
    return day_rows.set_index("ticker")["target_weight"].loc[lambda s: s > 0].to_dict()


def apply_regular_allocation(
    g_state,
    df_ranked: pd.DataFrame,
    today: pd.Timestamp,
    market_prices: pd.Series = None,
    weight_fn=naive_precomputed_weight,
) -> dict[str, float]:
    """
    Runs weight_fn(g_state, day_rows, market_prices) against today's rows
    out of df_ranked (vect_calc's selection + naive suggested weight), and
    stages the result on g_state.need_trade for trade_implement to pick up.
    Returns the core_target_weights dict (also useful for the audit ledger).
    """
    day_rows = df_ranked.loc[df_ranked["date"] == today]
    core_target_weights = weight_fn(g_state, day_rows, market_prices)
    g_state.need_trade.set_core(core_target_weights, today)
    return core_target_weights
