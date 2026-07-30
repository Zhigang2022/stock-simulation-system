"""
Piece 2 of 3 (vector_calc / iteration / evaluation): the stateful,
path-dependent execution loop. Same role as src/run.py, but the "what
should we hold" decision for the CORE sleeve starts from a precomputed
lookup into df_ranked (vector_calc.build_target_weight_table's output)
instead of a live Selector/Scorer/Filter call.

This module is now just the orchestrator -- each phase lives in its own
file, in the order they run each day:
  1. allocation_regular.py  -- CORE sleeve: vect_calc selection + WEIGHT sizing (g_state-aware), rebalance days only
  2. allocation_tactical.py -- TACTICAL sleeve: live strategy + WEIGHT/position sizing (g_state-aware) + budget check, every day (optional)
  3. trade_implement.py     -- fills whatever got staged, gated by trade_delay (a pandas frequency string, e.g. "1h"/"1D"/"1ME")

Weight/position sizing deliberately lives in the allocation_*.py files, not
in vect_calc: vect_calc is precomputed and stateless (no g_state), so it
can only produce a naive suggested weight. The allocation layer is the
first point in the pipeline with access to live g_state (current
positions, cash, existing exposure), which is what real sizing logic
needs -- see allocation_regular.py / allocation_tactical.py docstrings.

This package has no src/ dependency:
- states.GlobalState is a fork of src/global_state.py (the genuinely
  stateful core, forked rather than imported -- see src2/states/global_state.py).
- Rebalance dates + the tactical leg's live snapshot use calendar_snapshot.py,
  a local dependency-free replacement for src.calendar_iterator.CalendarIterator.
- allocation_tactical.py's budget check reimplements budget_allocator.py's
  Part B locally instead of importing it.
- trade_implement.py inlines src/executor_module.py's TransactionExecutor.
- RebalanceRecord below replaces src.audit.AuditRecord: this loop never
  populates telemetry/raw-signal fields (those belong to the live
  Selector/Scorer path src/run.py uses, which this loop doesn't have), so
  the only field actually ever written -- target_weights per rebalance
  date -- gets a minimal local record instead of importing the richer,
  mostly-unused dataclass. Charting this history is evaluation/plot.py's
  job, not this record's.

TACTICAL sleeve (optional): pass `adhoc_strategy`, an object with an
`evaluate_exits(g_state, df_metrics_adhoc, today) -> StrategyOutput` method
(see src/ad_hoc_strategy.py's AdHocChandelierExit for the shape), to run a
live, day-by-day tactical strategy (e.g. a MACD bull/bear-cross on a single
ticker) alongside the precomputed core sleeve. `df_metrics_adhoc` is itself
precomputed once, vectorized (e.g. vector_calc.build_metrics_table with
score_fn=vector_calc.macd_cross_score) and passed straight through here --
the strategy only decides kind/ticker (BUY vs EXIT) off that table;
allocation_tactical.py decides size.
"""
from dataclasses import dataclass, field

import pandas as pd

from src2.states import GlobalState

from .allocation_regular import apply_regular_allocation
from .allocation_tactical import apply_tactical_allocation
from .trade_implement import make_executor, implement_trades
from .calendar_snapshot import generate_rebalance_dates


@dataclass
class RebalanceRecord:
    """One rebalance date's core target weights -- see module docstring."""
    date: pd.Timestamp
    target_weights: dict[str, float] = field(default_factory=dict)


def run_daily_iteration(
    df_ranked: pd.DataFrame,
    world_data_dict: dict[str, pd.DataFrame],
    rebalance_dates: list[pd.Timestamp] | None = None,
    interval: str = "ME",
    initial_capital: float = 100_000.0,
    core_allocation_pct: float = 1.0,
    trade_delay: str = "1D",
    fee_rate: float = 0.001,
    tactical_strategy=None,
    df_metrics_adhoc=None,
    tactical_reserve_cash_pct: float = 0.0,
    tactical_max_position_pct: float = 1.0,
    logger=None,
):
    """
    Runs the sequential execution leg against a precomputed selection
    table (CORE sleeve), plus an optional live tactical sleeve (TACTICAL).
    df_ranked must have columns: date, ticker, target_weight (the output of
    vector_calc.build_target_weight_table) -- used as allocation_regular's
    naive baseline weight, not the final word (see allocation_regular.py).

    Returns (g_state, audit_ledger): g_state is a states.GlobalState;
    audit_ledger is a list[RebalanceRecord], one per rebalance date.
    """
    if rebalance_dates is None:
        rebalance_dates = generate_rebalance_dates(world_data_dict["price"], interval=interval)

    rebalance_dates_set = set(rebalance_dates)
    price_index_set = set(world_data_dict["price"].index)
    missing = rebalance_dates_set - price_index_set
    if missing:
        raise ValueError(
            f"{len(missing)} rebalance date(s) not present in world_data_dict['price'].index "
            f"(e.g. {sorted(missing)[:3]}) -- df_ranked/rebalance_dates was likely computed "
            f"against a different price panel than the one passed in here."
        )

    g_state = GlobalState(initial_capital=initial_capital, core_allocation_pct=core_allocation_pct)
    executor = make_executor(fee_rate=fee_rate, trade_delay=trade_delay)
    audit_ledger: list[RebalanceRecord] = []

    for today in world_data_dict["price"].index:
        market_prices = world_data_dict["price"].loc[today]

        # 1. REGULAR (CORE) ALLOCATION -- rebalance days only
        if today in rebalance_dates_set:
            core_target_weights = apply_regular_allocation(g_state, df_ranked, today, market_prices)

            if logger is not None:
                logger.info(f"=== REBALANCE EVENT: {today.strftime('%Y-%m-%d')} -> {len(core_target_weights)} names ===")

            audit_ledger.append(RebalanceRecord(date=today, target_weights=core_target_weights))

        # 2. TACTICAL (AD-HOC) ALLOCATION -- every day, if wired in
        if tactical_strategy is not None:
            apply_tactical_allocation(
                g_state, tactical_strategy, df_metrics_adhoc, today, market_prices,
                reserve_cash_pct=tactical_reserve_cash_pct,
                max_position_pct=tactical_max_position_pct,
            )

        # 3. TRADE IMPLEMENTATION -- fills whatever got staged above, gated by trade_delay
        implement_trades(executor, g_state, today, market_prices)

    return g_state, audit_ledger
