# strategy_selector.py
import pandas as pd
import numpy as np

class BaseStrategy:
    def calculate_scores(self, snapshot: dict[str, pd.DataFrame]) -> pd.Series:
        raise NotImplementedError


class RiskAdjustedMomentum(BaseStrategy):
    def __init__(self, lookback_days: int = 252):
        """
        Args:
            lookback_days: Trading days to calculate trend and volatility (e.g., 252 days).
        """
        self.lookback_days = lookback_days

    def calculate_scores(self, snapshot: dict[str, pd.DataFrame]) -> pd.Series:
        """
        Calculates Risk-Adjusted Momentum (Total Return / Volatility) on a cross-section.
        """
        prices = snapshot["price"]
        
        # Guard rail: Ensure enough historical data exists
        if len(prices) < self.lookback_days:
            return pd.Series(0.0, index=prices.columns)
            
        # Slice the window of interest
        window_prices = prices.iloc[-self.lookback_days:]
        
        # 1. Calculate the Raw Total Return over the window
        current_price = window_prices.iloc[-1]
        historical_price = window_prices.iloc[0]
        raw_returns = (current_price / historical_price) - 1
        
        # 2. Calculate the Volatility (Standard Deviation of Daily Returns)
        # We annualized the standard deviation to keep metrics standard
        daily_returns = window_prices.pct_change()
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        
        # 3. Calculate Risk-Adjusted Momentum (Sharpe Ratio of the trend)
        # Avoid division by zero if volatility is somehow NaN or 0
        risk_adjusted_momentum = raw_returns / annualized_volatility
        risk_adjusted_momentum = risk_adjusted_momentum.replace([np.inf, -np.inf], np.nan)
        
        # 4. Convert to a Cross-Sectional Percentile Rank (0.0 to 1.0)
        # Handles NaNs seamlessly by keeping them neutral at the bottom
        momentum_scores = risk_adjusted_momentum.rank(pct=True, na_option='keep')
        momentum_scores = momentum_scores.fillna(0.0)
        
        return momentum_scores
    
class MomentumStrategy(BaseStrategy):
    def __init__(self, lookback_days: int = 252):
        """
        Args:
            lookback_days: Trading days to calculate momentum (e.g., 252 days ~ 1 Year).
        """
        self.lookback_days = lookback_days

    def calculate_scores(self, snapshot: dict[str, pd.DataFrame]) -> pd.Series:
        """
        Calculates cross-sectional relative momentum scores without throwing away information.
        """
        prices = snapshot["price"]
        
        # Guard rail: If the entire historical matrix is shorter than lookback, return zero scores
        if len(prices) < self.lookback_days:
            return pd.Series(0.0, index=prices.columns)
            
        # 1. Capture point-in-time prices
        current_price = prices.iloc[-1]
        historical_price = prices.iloc[-self.lookback_days]
        
        # 2. Calculate Total Return. 
        # If historical_price is NaN or 0, pandas automatically outputs NaN. No manual penalty.
        raw_returns = (current_price / historical_price) - 1
        
        # 3. Convert to a Cross-Sectional Percentile Rank (0.0 to 1.0)
        # pct=True means the highest return gets 1.0, lowest gets 0.0.
        # na_option='keep' ensures that missing data stays NaN and doesn't skew the rank.
        momentum_scores = raw_returns.rank(pct=True, na_option='keep')
        
        # Fill remaining NaNs with 0.0 *only* so the series is safe to pass down, 
        # but they sit at the absolute bottom of the rank.
        momentum_scores = momentum_scores.fillna(0.0)
        
        return momentum_scores
    

class InformationDiscreteMomentum(BaseStrategy):
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days

    def calculate_scores(self, snapshot: dict[str, pd.DataFrame]) -> pd.Series:
        """
        Calculates Information Discrete Momentum using Linear Regression Slope * R-squared.
        """
        prices = snapshot["price"]
        
        if len(prices) < self.lookback_days:
            return pd.Series(0.0, index=prices.columns)
            
        # 1. Focus on the lookback window and convert to Log Prices 
        # Log prices are mathematically required so the slope represents percentage growth rate
        window_log_prices = np.log(prices.iloc[-self.lookback_days:])
        
        # Create the time independent variable (X axis: 0, 1, 2, ..., 251)
        x = np.arange(self.lookback_days)
        
        scores_dict = {}
        
        # 2. Iterate across the asset cross-section to fit the model
        for ticker in window_log_prices.columns:
            y = window_log_prices[ticker].values
            
            # Skip if the entire column is filled with NaNs (e.g., asset didn't exist yet)
            if np.isnan(y).all() or len(y[~np.isnan(y)]) < self.lookback_days:
                scores_dict[ticker] = np.nan
                continue
                
            # Fit linear regression: y = slope * x + intercept
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calculate R-squared manually for speed
            y_pred = slope * x + intercept
            y_bar = np.mean(y)
            ss_tot = np.sum((y - y_bar) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Core Formula: Slope * R-squared
            # This penalizes any deviation from a straight line
            scores_dict[ticker] = slope * r_squared

        # 3. Format as Series and convert to Cross-Sectional Percentile Rank
        raw_scores = pd.Series(scores_dict)
        momentum_scores = raw_scores.rank(pct=True, na_option='keep')
        momentum_scores = momentum_scores.fillna(0.0)
        
        return momentum_scores
    
class InformationDiscreteWithExhaustion(BaseStrategy):
    def __init__(self, lookback_days: int = 252, vol_window: int = 20, z_threshold: float = 3.0):
        """
        Args:
            lookback_days: Window for fitting the linear regression trend line.
            vol_window: Baseline window to check for normal volume behavior.
            z_threshold: The number of standard deviations that defines an "explosion".
        """
        self.lookback_days = lookback_days
        self.vol_window = vol_window
        self.z_threshold = z_threshold

    def calculate_scores(self, snapshot: dict[str, pd.DataFrame]) -> pd.Series:
        prices = snapshot["price"]
        volumes = snapshot["volume"]
        
        if len(prices) < self.lookback_days:
            return pd.Series(0.0, index=prices.columns)
            
        # 1. Core Model: Calculate Rational R² Momentum Slope
        window_log_prices = np.log(prices.iloc[-self.lookback_days:])
        x = np.arange(self.lookback_days)
        
        raw_momentum_scores = {}
        
        for ticker in window_log_prices.columns:
            y = window_log_prices[ticker].values
            if np.isnan(y).all() or len(y[~np.isnan(y)]) < self.lookback_days:
                raw_momentum_scores[ticker] = np.nan
                continue
                
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            y_bar = np.mean(y)
            ss_tot = np.sum((y - y_bar) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Save the pure mathematical momentum score
            raw_momentum_scores[ticker] = slope * r_squared

        # Convert to Series
        final_scores = pd.Series(raw_momentum_scores)

        # 2. Volume Exhaustion Penalty Layer
        for ticker in prices.columns:
            if ticker not in final_scores.index or pd.isna(final_scores[ticker]):
                continue
                
            # Isolate the volume window (excluding the final day to build an unbiased baseline)
            ticker_vol_history = volumes[ticker].iloc[-self.vol_window:]
            today_vol = ticker_vol_history.iloc[-1]
            baseline_vols = ticker_vol_history.iloc[:-1]
            
            mean_vol = baseline_vols.mean()
            std_vol = baseline_vols.std()
            
            # Calculate the Volume Z-Score safely
            if std_vol > 0:
                vol_z = (today_vol - mean_vol) / std_vol
            else:
                vol_z = 0.0
                
            # If volume explodes, penalize the score.
            # Hard-kill method: set to 0.0 (drops it to the absolute bottom of the rank)
            if vol_z >= self.z_threshold:
                final_scores[ticker] = -999.0 

        # 3. Cross-Sectional Percentile Rank among peers
        momentum_ranks = final_scores.rank(pct=True, na_option='keep')
        momentum_ranks = momentum_ranks.fillna(0.0)
        
        return momentum_ranks