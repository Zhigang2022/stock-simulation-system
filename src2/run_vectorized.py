"""
Execution loop for the vectorized alpha layer (see alpha_engine.py). Same
role as src/run.py, but the "what should we hold" decision is a precomputed
lookup into df_ranked instead of a live Selector/Scorer/Filter call, and
budget_allocator's Part A (regular-signal -> target weight) is skipped
entirely since df_ranked already IS that result.

TransactionExecutor and GlobalState are untouched and imported straight
from src/ -- this only replaces the alpha-generation side of the loop, not
the stateful execution side. CalendarIterator is still used UPSTREAM (by
the caller, to build rebalance_dates/df_ranked in the first place) but not
inside this function -- rebalance_dates is validated against
world_data_dict['price'].index below instead, to catch a mismatch if the
two were built from different panels.

No tactical/ad-hoc strategy exists yet, so budget_allocator's Part B
(ad-hoc order validation) and the whole ad-hoc branch are omitted rather
than stubbed out. When a tactical strategy is added, this is the place to
bring back: evaluate it inside the loop (it needs live GlobalState, so it
can't be precomputed the way the core signal was), then reintroduce
IntegratedBudgetAllocator's Part B and executor's ad-hoc-exit step -- see
src/run.py for the shape that took.
"""
import pandas as pd

from src import global_state as global_state_module
from src import executor_module as executor_module_module
from src import audit


def run_vectorized(
    df_ranked: pd.DataFrame,
    world_data_dict: dict[str, pd.DataFrame],
    rebalance_dates: list[pd.Timestamp],
    initial_capital: float = 100_000.0,
    core_allocation_pct: float = 1.0,
    trade_delay: int = 1,
    fee_rate: float = 0.001,
    logger=None,
):
    """
    Runs the sequential execution leg against a precomputed target_weight
    table. df_ranked must have columns: date, ticker, target_weight (the
    output of alpha_engine.build_target_weight_table).

    Returns (g_state, audit_ledger), same shape as src.run.run().
    """
    rebalance_dates_set = set(rebalance_dates)
    price_index_set = set(world_data_dict["price"].index)
    missing = rebalance_dates_set - price_index_set
    if missing:
        raise ValueError(
            f"{len(missing)} rebalance date(s) not present in world_data_dict['price'].index "
            f"(e.g. {sorted(missing)[:3]}) -- df_ranked/rebalance_dates was likely computed "
            f"against a different price panel than the one passed in here."
        )

    g_state = global_state_module.GlobalState(
        initial_capital=initial_capital, core_allocation_pct=core_allocation_pct
    )
    executor = executor_module_module.TransactionExecutor(fee_rate=fee_rate, trade_delay=trade_delay)
    audit_ledger: list[audit.AuditRecord] = []

    for today in world_data_dict["price"].index:
        market_prices = world_data_dict["price"].loc[today]

        if today in rebalance_dates_set:
            day_rows = df_ranked.loc[df_ranked["date"] == today]
            core_target_weights = (
                day_rows.set_index("ticker")["target_weight"].loc[lambda s: s > 0].to_dict()
            )
            g_state.need_trade.set_core(core_target_weights, today)

            if logger is not None:
                logger.info(f"=== REBALANCE EVENT: {today.strftime('%Y-%m-%d')} -> {len(core_target_weights)} names ===")

            audit_ledger.append(audit.AuditRecord(
                date=today,
                telemetry=None,
                target_weights=core_target_weights,
            ))

        executor.execute_trades(g_state, today, market_prices)
        g_state.record_daily_snapshot(today, market_prices)

    return g_state, audit_ledger
