"""
Copied (not imported) from src/global_state.py -- the genuinely stateful
core of the engine (cash, positions, NAV ledger). See global_state.py's
module docstring for why this is a fork rather than a src/ import.
"""
from .global_state import GlobalState, Future_Transaction, Transient_Signals

__all__ = ["GlobalState", "Future_Transaction", "Transient_Signals"]
