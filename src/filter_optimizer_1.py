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
        Updated Joint-Filtering Logic: Filters for liquidity first, 
        then optimizes the top tier momentum names from the liquid pool.
        """
        prices = snapshot["price"]
        volumes = snapshot["volume"]
        weights = pd.Series(0.0, index=prices.columns)
        
        # 1. STEP 1: Scan the ENTIRE universe for liquidity first
        liquid_universe = []
        for ticker in prices.columns:
            ticker_prices = prices[ticker].iloc[-self.volume_window:]
            ticker_volumes = volumes[ticker].iloc[-self.volume_window:]
            mean_dollar_volume = (ticker_prices * ticker_volumes).mean()
            
            if mean_dollar_volume >= self.min_dollar_volume:
                liquid_universe.append(ticker)
                
        if len(liquid_universe) == 0:
            return weights # Entire universe is illiquid at this timestamp. Stay in cash.
            
        # 2. STEP 2: Isolate the scores of ONLY the liquid survivors
        liquid_scores = scores[liquid_universe]
        
        # 3. STEP 3: Take the Top N% from the liquid pool
        cutoff_quantile = 1.0 - self.top_percent
        threshold_score = liquid_scores.quantile(cutoff_quantile)
        final_candidates = liquid_scores[liquid_scores >= threshold_score].index
        
        if len(final_candidates) == 0:
            return weights

        # 4. STEP 4: Allocation Optimization
        if self.allocation_type == "equal":
            allocated_weight = 1.0 / len(final_candidates)
            for ticker in final_candidates:
                weights[ticker] = allocated_weight
                
        elif self.allocation_type == "score_weighted":
            candidate_scores = liquid_scores[final_candidates]
            total_score_sum = candidate_scores.sum()
            if total_score_sum > 0:
                for ticker in final_candidates:
                    weights[ticker] = candidate_scores[ticker] / total_score_sum
                    
        return weights