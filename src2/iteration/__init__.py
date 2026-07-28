"""
Piece 2 of 3 (vector_calc / iteration / evaluation): the stateful,
path-dependent day-by-day execution loop, split into phases:

- calendar_snapshot.py   -- rebalance dates + leak-safe daily snapshot (local, no src/ dependency)
- allocation_regular.py  -- CORE sleeve: vect_calc selection + weight sizing (rebalance days only)
- allocation_tactical.py -- TACTICAL sleeve: live ad-hoc strategy + position sizing + local budget check (every day, optional)
- trade_implement.py     -- fills staged orders, gated by trade_delay
- daily_iterator.py       -- orchestrates the phases above; run_daily_iteration is the main entry point

Weight/position sizing lives in the allocation_*.py files (not vect_calc)
because only they have access to live g_state (cash, current positions).

This package has no src/ dependency: GlobalState is forked into
src2/states/ (still the genuinely stateful core, just no longer a src/
import); TransactionExecutor is forked into trade_implement.py;
CalendarIterator and budget_allocator's Part B are reimplemented locally
in calendar_snapshot.py / allocation_tactical.py since they added coupling
without functionality worth importing; src.audit.AuditRecord is replaced
by daily_iterator.py's own minimal RebalanceRecord (this loop never
populated the richer fields anyway). See each file's docstring for the
"fork vs import" reasoning.
"""
from .daily_iterator import run_daily_iteration
from .allocation_regular import apply_regular_allocation
from .allocation_tactical import apply_tactical_allocation
from .trade_implement import make_executor, implement_trades
from .calendar_snapshot import generate_rebalance_dates, get_historical_snapshot

__all__ = [
    "run_daily_iteration",
    "apply_regular_allocation",
    "apply_tactical_allocation",
    "make_executor",
    "implement_trades",
    "generate_rebalance_dates",
    "get_historical_snapshot",
]
