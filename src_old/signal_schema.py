# signal_schema.py
from dataclasses import dataclass, field
from typing import Literal, List
import pandas as pd

@dataclass
class SignalPayload:
    """
    Standardized cross-module message token passed from alpha strategy generators 
    (Modules C & C1) into the execution/budget layers (Modules D & D1).
    """
    ticker: str
    kind: Literal["REGULAR_REBALANCE", "AD_HOC_EXIT", "AD_HOC_BUY"]
    score: float = float('nan')           # Used by Regular/Ad-hoc Buys to represent alpha ranking
    quantity_target: float = float('nan') # Used by Exits to denote target liquidation fraction (e.g., -0.50)

class StrategyOutput:
    """
    Container wrapper holding collections of SignalPayload objects generated at 
    any single time loop step. Provides quick formatting to DataFrames for logging.
    """
    def __init__(self, signals: List[SignalPayload] = None):
        self.signals: List[SignalPayload] = signals if signals is not None else []

    def append(self, signal: SignalPayload):
        self.signals.append(signal)

    def to_dataframe(self) -> pd.DataFrame:
        """Helper method to convert signals cleanly for the Event-Driven Logger."""
        if not self.signals:
            return pd.DataFrame(columns=["ticker", "kind", "score", "quantity_target"])
        return pd.DataFrame([s.__dict__ for s in self.signals])
    

@dataclass
class StrategyTelemetry:
    """
    Arbitrary diagnostic data and mathematical factors calculated during a strategy pass.
    Example: metrics = {"R2": {"AAPL": 0.85}, "slope": {"AAPL": 0.015}}
    """
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)