# filter_optimizer.py
import pandas as pd
import numpy as np

class BaseFilter:
    def generate_weights(self, scores: pd.Series, snapshot: dict[str, pd.DataFrame]) -> pd.Series:
        raise NotImplementedError


class VolumeWeightFilter(BaseFilter):
    def __init__(self, top_percent: float = 0.10, volume_window: int = 20, min_dollar_volume: float = 1_000_000.0, allocation_type: str = "equal"):
        """
        Args:
            top_percent: Top fraction of scores to consider (e.g., 0.10 for top 10%).
            volume_window: Lookback window to calculate average volume.
            min_dollar_volume: Minimum average daily dollar volume (Price * Volume) required.
            allocation_type: "equal" or "score_weighted"
        """
        self.top_percent = top_percent
        self.volume_window = volume_window
        self.min_dollar_volume = min_dollar_volume
        self.allocation_type = allocation_type

    def generate_weights(self, scores: pd.Series, snapshot: dict[str, pd.DataFrame]) -> pd.Series:
        """
        Filters out illiquid stocks from the top-tier momentum names and outputs portfolio weights.
        """
        prices = snapshot["price"]
        volumes = snapshot["volume"]
        
        # Initialize an empty weight vector (default to 0% for all stocks)
        weights = pd.Series(0.0, index=prices.columns)
        
        # 1. Isolate the top N% based on Module C's continuous scores
        cutoff_quantile = 1.0 - self.top_percent
        threshold_score = scores.quantile(cutoff_quantile)
        top_candidates = scores[scores >= threshold_score].index
        
        if len(top_candidates) == 0:
            return weights # Return all zeros if no candidates
            
        # 2. Apply Liquidity Filter to the top candidates
        valid_candidates = []
        for ticker in top_candidates:
            # Calculate daily dollar volume: Price * Volume
            ticker_prices = prices[ticker].iloc[-self.volume_window:]
            ticker_volumes = volumes[ticker].iloc[-self.volume_window:]
            daily_dollar_volume = ticker_prices * ticker_volumes
            
            # Compute Average Daily Dollar Volume (ADDV)
            mean_dollar_volume = daily_dollar_volume.mean()
            
            # If the stock passes the liquidity bar, it stays in the portfolio pool
            if mean_dollar_volume >= self.min_dollar_volume:
                valid_candidates.append(ticker)
                
        if len(valid_candidates) == 0:
            return weights # All candidates were too illiquid! Stay in cash.
            
        # 3. Allocation Optimization Step
        if self.allocation_type == "equal":
            # Split capital completely evenly among valid assets
            allocated_weight = 1.0 / len(valid_candidates)
            for ticker in valid_candidates:
                weights[ticker] = allocated_weight
                
        elif self.allocation_type == "score_weighted":
            # Capital is allocated proportionally to the strength of the R^2 score
            candidate_scores = scores[valid_candidates]
            total_score_sum = candidate_scores.sum()
            
            if total_score_sum > 0:
                for ticker in valid_candidates:
                    weights[ticker] = candidate_scores[ticker] / total_score_sum
                    
        # STRUCTURAL GUARD RAIL TEST: Total portfolio allocation must never exceed 100%
        assert round(weights.sum(), 4) <= 1.0, f"CRITICAL: Portfolio weights sum to {weights.sum()}, exceeding 100%!"
        
        return weights