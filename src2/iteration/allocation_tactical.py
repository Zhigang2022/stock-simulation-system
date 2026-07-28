"""
Tactical (AD-HOC sleeve) allocation. Runs a live day-by-day strategy (e.g.
a MACD bull/bear-cross scorer, or src/ad_hoc_strategy.py's
AdHocChandelierExit) against a fresh leak-safe snapshot, then decides
POSITION SIZE here -- not in the strategy -- and does its own tactical-cash
budget check before staging orders.

Why sizing lives here: the strategy only knows "MACD crossed up on AAA
today" -- it has no idea how much tactical_cash is free, whether AAA is
already partially held, or how much of the tactical NAV is already at
risk. Only this layer has that (g_state.tactical_cash /
tactical_positions), which is exactly what's needed for anything past
"buy with 100% of whatever's left": capping position size, holding back a
cash safety reserve, or scaling into an existing position rather than
re-buying it.

The budget check here is a local, minimal reimplementation of
budget_allocator.py's Part B, not an import of it: that module's version
ignores the signal's actual quantity_target and hardcodes a 10%-of-NAV
cost estimate, which doesn't match sizing_fn's real number below. It also
pulls in src.global_state's Transient_Signals plumbing for no benefit
here. Re-deriving the ~5 lines of "is there enough tactical_cash" locally
keeps this package free of the src/ dependency.

Unlike allocation_regular, none of this can be precomputed -- it depends on
live GlobalState -- which is why it needs a fresh
calendar_snapshot.get_historical_snapshot every day, not just on rebalance
dates.
"""
import pandas as pd


def naive_position_sizing(
    g_state,
    signal,
    market_prices: pd.Series,
    reserve_cash_pct: float = 0.0,
    max_position_pct: float = 1.0,
) -> float:
    """
    Default sizing rule for an AD_HOC_BUY signal: fill up to
    `max_position_pct` of tactical NAV in this ticker, holding back
    `reserve_cash_pct` of tactical NAV as a safety buffer, and topping up
    rather than re-buying if some of the position is already held.
    Returns an allocation fraction of tactical NAV (the executor's
    AD_HOC_BUY branch reads this off `quantity_target`).
    """
    tactical_nav = g_state.calculate_sleeve_value(
        g_state.tactical_cash, g_state.tactical_positions, market_prices
    )
    if tactical_nav <= 0:
        return 0.0

    current_exposure_pct = (
        g_state.tactical_positions.get(signal.ticker, 0.0) * market_prices[signal.ticker]
    ) / tactical_nav
    room_to_target_pct = max(0.0, max_position_pct - current_exposure_pct)
    deployable_pct = max(0.0, 1.0 - reserve_cash_pct)
    return min(room_to_target_pct, deployable_pct)


def _budget_check(g_state, signals, market_prices: pd.Series):
    """
    Local minimal equivalent of budget_allocator.py's Part B: exits always
    clear immediately (no budget constraint), buys are cleared only if
    tactical_cash actually covers the sized order (unlike
    budget_allocator's hardcoded 10% estimate, this uses the real
    quantity_target set by sizing_fn).
    """
    sell_orders, buy_orders = [], []
    for signal in signals:
        if signal.kind == "AD_HOC_EXIT":
            sell_orders.append(signal)
        elif signal.kind == "AD_HOC_BUY":
            tactical_nav = g_state.calculate_sleeve_value(
                g_state.tactical_cash, g_state.tactical_positions, market_prices
            )
            estimated_cost = tactical_nav * signal.quantity_target
            if g_state.tactical_cash >= estimated_cost:
                buy_orders.append(signal)
    return sell_orders, buy_orders


def apply_tactical_allocation(
    g_state,
    module_c1_adhoc,
    get_snapshot,
    today: pd.Timestamp,
    market_prices: pd.Series,
    sizing_fn=naive_position_sizing,
    reserve_cash_pct: float = 0.0,
    max_position_pct: float = 1.0,
):
    """
    Evaluates module_c1_adhoc for `today` (strategy only decides
    kind/ticker -- BUY vs EXIT -- not size), runs sizing_fn on every
    AD_HOC_BUY signal to set its quantity_target, budget-checks the result,
    and stages the survivors on g_state.need_trade for trade_implement to
    pick up. `get_snapshot` is a callable(today) -> snapshot dict, typically
    calendar_snapshot.get_historical_snapshot bound to world_data_dict.
    Returns (sell_adhoc_orders, buy_adhoc_orders).
    """
    snapshot = get_snapshot(today)
    adhoc_strategy_output = module_c1_adhoc.evaluate_exits(g_state, snapshot)

    for signal in adhoc_strategy_output.signals:
        if signal.kind == "AD_HOC_BUY":
            signal.quantity_target = sizing_fn(
                g_state, signal, market_prices,
                reserve_cash_pct=reserve_cash_pct, max_position_pct=max_position_pct,
            )

    sell_adhoc_orders, buy_adhoc_orders = _budget_check(g_state, adhoc_strategy_output.signals, market_prices)
    g_state.need_trade.set_sell_adhoc(sell_adhoc_orders, today)
    g_state.need_trade.set_buy_adhoc(buy_adhoc_orders, today)
    return sell_adhoc_orders, buy_adhoc_orders
