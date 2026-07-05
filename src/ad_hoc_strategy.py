# adhoc_strategy.py
import pandas as pd
from src.signal_schema import SignalPayload, StrategyOutput

class AdHocChandelierExit:
    def __init__(self, atr_window: int = 20, atr_multiplier: float = 3.0):
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier

    def evaluate_exits(self, global_state, snapshot: dict[str, pd.DataFrame]) -> StrategyOutput:
        """
        Scans active holdings in the global state to trigger emergency daily 
        Chandelier Exits based on volatility trailing bands.
        """
        prices = snapshot["price"]
        output = StrategyOutput()
        
        # Active positions are pulled directly out of live global state tracking memory
        active_tactical_positions = [t for t, shares in global_state.tactical_positions.items() if shares > 0]
        
        if not active_tactical_positions:
            return output
            
        for ticker in active_tactical_positions:
            # Calculate Average True Range (ATR Proxy via daily close volatility)
            recent_prices = prices[ticker].iloc[-self.atr_window:]
            daily_changes = recent_prices.pct_change().dropna()
            volatility_buffer = daily_changes.std() * recent_prices.iloc[-1] * self.atr_multiplier
            
            # Fetch rolling peak state metadata from GlobalState
            highest_peak = global_state.tactical_peaks.get(ticker, recent_prices.iloc[-1])
            stop_loss_threshold = highest_peak - volatility_buffer
            current_price = recent_prices.iloc[-1]
            
            # If price breaks trailing threshold, issue an emergency 100% liquidation signal
            if current_price < stop_loss_threshold:
                output.append(SignalPayload(
                    ticker=ticker,
                    kind="AD_HOC_EXIT",
                    quantity_target=-1.0 # Tells Module D to completely dump all shares
                ))
                
        return output