# strategy_selector.py
import pandas as pd
import numpy as np

class BaseStrategy:
    def calculate_scores(self, snapshot: dict[str, pd.DataFrame]) -> pd.Series:
        raise NotImplementedError


class MomentumStrategy(BaseStrategy):
    def __init__(self, lookback_days: int = 252, top_percent: float = 0.10):
        """
        Args:
            lookback_days: Trading days to calculate momentum (e.g., 252 days ~ 1 Year).
            top_percent: Top fraction of the universe to select (0.10 = Top 10%).
        """
        self.lookback_days = lookback_days
        self.top_percent = top_percent

    def calculate_scores(self, snapshot: dict[str, pd.DataFrame]) -> pd.Series:
        """
        Calculates cross-sectional total return momentum and ranks the universe.
        """
        prices = snapshot["price"]
        
        # If we don't have enough data points yet to look back, return an empty series
        if len(prices) < self.lookback_days:
            return pd.Series(0.0, index=prices.columns)
            
        # 1. Capture current prices and historical prices from X days ago
        current_price = prices.iloc[-1]
        historical_price = prices.iloc[-self.lookback_days]
        
        # 2. Calculate Total Return over the lookback window
        # (Current Price / Historical Price) - 1
        # Handle cases where historical price might be 0 or NaN safely
        momentum_returns = (current_price / historical_price) - 1
        momentum_returns = momentum_returns.fillna(-999.0) # Penalty for missing assets
        
        # 3. Create a binary score: 1.0 for top 10%, 0.0 for others
        # We find the threshold value corresponding to the top N percentile
        cutoff_quantile = 1.0 - self.top_percent
        threshold_value = momentum_returns.quantile(cutoff_quantile)
        
        # Generate binary flags (True/False converted to 1.0/0.0)
        # If there's a tie, pandas handles it gracefully
        scores = (momentum_returns >= threshold_value).astype(float)
        
        return scores,momentum_returns