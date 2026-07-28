import pandas as pd
from typing import List
from src_old.signal_schema import StrategyOutput, SignalPayload
from src_old.filters.base_filter import SignalFilter

class LiquidityComplianceFilter(SignalFilter):
    """
    Module D-Gateway: Pure Screening Layer.
    Validates signal eligibility against real-world liquidity data.
    Does NOT calculate portfolio weights.
    """
    def __init__(self, min_dollar_volume: float = 1_000_000.0, volume_window: int = 20):
        self.min_dollar_volume = min_dollar_volume
        self.volume_window = volume_window

    def filter_signals(self, date: pd.Timestamp, strategy_output: StrategyOutput, snapshot: dict[str, pd.DataFrame]) -> StrategyOutput:
        """
        Screens incoming signal streams. Drops assets failing liquidity rules.
        Outputs structural audit feedback.
        """
        prices = snapshot["price"]
        volumes = snapshot["volume"]
        
        sanitized_output = StrategyOutput()
        
        for signal in strategy_output.signals:
            ticker = signal.ticker
            
            # Non-asset instructions or short covers skip liquidity screening
            if ticker not in prices.columns:
                sanitized_output.append(signal)
                continue
                
            # Compute Average Daily Dollar Volume (ADDV)
            ticker_prices = prices[ticker].iloc[-self.volume_window:]
            ticker_volumes = volumes[ticker].iloc[-self.volume_window:]
            mean_dollar_volume = (ticker_prices * ticker_volumes).mean()
            
            # Compliance Screening Check
            if mean_dollar_volume >= self.min_dollar_volume:
                sanitized_output.append(signal)
            else:
                # 1. Print stock being filtered to inform user/logger
                print(f"[{date.strftime('%Y-%m-%d')}] FILTERED: {ticker} dropped. "
                      f"ADDV ${mean_dollar_volume:,.2f} < Minimum ${self.min_dollar_volume:,.2f}")
                
        return sanitized_output
    


# TODO 
'''
3. Dual momentum (Antonacci's full framework, combines both layers)
Step 1 — relative momentum: among a small set of alternatives (say, US stocks vs international stocks vs bonds), pick whichever had the strongest trailing 12-month return.
Step 2 — absolute momentum filter on the winner: check if that winner's own trailing return also beats cash. If yes, hold it. If no — even though it "won" the relative comparison — go to cash instead.
'''
