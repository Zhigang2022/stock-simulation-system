from abc import ABC, abstractmethod
from src_old.signal_schema import SignalPayload, StrategyOutput, StrategyTelemetry
import pandas as pd
class BaseSelector(ABC):
    """
    Common interface for all signal-generating strategies (cross-sectional
    rank-based, like ComprehensiveMultiHorizonStrategy, OR absolute/
    self-referential, like SelfSelector below). Any selector must return
    (StrategyOutput, StrategyTelemetry) from a single snapshot.
    """
    @abstractmethod
    def calculate_signals(self, snapshot: dict[str, pd.DataFrame]) -> tuple[StrategyOutput, StrategyTelemetry]:
        raise NotImplementedError
    

