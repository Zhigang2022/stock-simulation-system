# strategy_selector.py
import pandas as pd
import numpy as np
from src.signal_schema import SignalPayload, StrategyOutput, StrategyTelemetry
from abc import ABC, abstractmethod


class MomentumScorer(ABC):
    @abstractmethod
    def compute_score(self, prices: np.ndarray) -> tuple[float, dict[str, float]]:
        """
        Compute the strategy score and return a flexible dictionary of diagnostic metrics.

        Returns
        -------
        score : float
            The final scalar value used for cross-sectional ranking.
        metrics : dict[str, float]
            A dictionary containing any arbitrary calculation parameters for auditing.
        """
        pass

class SimpleRiskAdjustedScorer(MomentumScorer):
    """
    TRADITIONAL BASELINE:
    Calculates raw arithmetic cumulative return divided by annualized daily volatility.
    last element of prices is current day, the first element of the prices is the compare day
    """
    def compute_score(self, prices: np.ndarray) -> tuple[float, dict[str, float]]:
        if len(prices) < 2 or prices[0] <= 0:
            return np.nan, {}
        
        # Cumulative simple return: (P_end / P_start) - 1
        raw_return = (prices[-1] / prices[0]) - 1
        
        # Annualized daily volatility
        daily_returns = np.diff(prices) / prices[:-1]
        vol = np.std(daily_returns) * np.sqrt(252)
        
        if vol == 0 or np.isnan(vol):
            return np.nan, {'volatile': vol}
            
        score = raw_return / vol
        return score, {'volatile':vol}
    

class TraditionalLinearRegressionScorer(MomentumScorer):
    """
    TRADITIONAL BASELINE:
    Performs standard unweighted OLS on log prices.
    Score = slope * R-squared.
    """
    def compute_score(self, prices: np.ndarray) -> tuple[float, dict[str, float]]:
        n = len(prices)
        if n < 3:
            return np.nan, {}
            
        y = np.log(prices)
        x = np.arange(n)
        
        # Equal-weighted Ordinary Least Squares
        slope, intercept = np.polyfit(x, y, 1)
        
        y_pred = slope * x + intercept
        y_bar = np.mean(y)
        ss_tot = np.sum((y - y_bar) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        score = slope * r_squared
        # Return everything you might ever want to debug or analyze later
        metrics_dict = {
            "slope": float(slope),
            "r_squared": float(r_squared),
            "intercept": float(intercept)
        }
        return score, metrics_dict

class GeometricDriftScorer(MomentumScorer):
    """Calculates continuous drift rate adjusted for volatility drag."""
    def compute_score(self, prices: np.ndarray) -> tuple[float, dict[str, float]]:
        if len(prices) < 2 or prices[0] <= 0:
            return np.nan, {}
        
        # Calculate log returns for compounding accuracy
        log_returns = np.diff(np.log(prices))
        vol = np.std(log_returns) * np.sqrt(252)
        
        # Total continuous return over the window
        total_log_return = np.log(prices[-1] / prices[0])

        T = len(prices) / 252  # years
        annualized_log_return = total_log_return / T
        geometric_drift = annualized_log_return - 0.5 * vol**2

        metrics_dict = {
            "volatility": float(vol),
            "raw_log_return": float(total_log_return)
        }
        return geometric_drift, metrics_dict


class WeightedLinearRegressionScorer(MomentumScorer):
    """Performs WLS on log prices to give recent data higher trend weight."""
    def __init__(self, decay_factor: float = 0.98):
        self.decay_factor = decay_factor

    def compute_score(self, prices: np.ndarray) -> tuple[float, dict[str, float]]:
        n = len(prices)
        if n < 3:
            return np.nan, {}
            
        y = np.log(prices)
        x = np.arange(n)
        
        # Create exponential weights favoring recent timestamps
        weights = self.decay_factor ** (n - 1 - x)
        w_matrix = np.diag(weights)
        
        # Design matrix for linear regression
        X = np.vstack([x, np.ones(n)]).T
        
        # Weighted Least Squares analytical solution: (X^T * W * X)^(-1) * X^T * W * y
        try:
            XT_W = X.T @ w_matrix
            beta = np.linalg.inv(XT_W @ X) @ XT_W @ y
            slope = beta[0]
            
            # Weighted R-squared calculation
            y_pred = X @ beta
            y_bar_w = np.sum(weights * y) / np.sum(weights)
            ss_tot = np.sum(weights * (y - y_bar_w) ** 2)
            ss_res = np.sum(weights * (y - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            metrics_dict = {
            "slope": float(slope),
            "r_squared": float(r_squared),
            }

            return slope * r_squared, metrics_dict
        except np.linalg.LinAlgError:
            return np.nan, {}

# ==========================================
# 2. ADVANCED COMPOSITE STRATEGY MODULES
# ==========================================
class ComprehensiveMultiHorizonStrategy:
    """
    Unified execution harness that coordinates multi-horizon (long vs short) 
    cross-sectional ranking using any interchangeable MomentumScorer block.
    """
    def __init__(
        self, 
        scorer: MomentumScorer, 
        long_window: int = 252, 
        short_window: int = 21,
        structural_weight: float = 0.7
    ):
        self.scorer = scorer
        self.long_window = long_window
        self.short_window = short_window
        self.structural_weight = structural_weight
        self.tactical_weight = 1.0 - structural_weight

    def calculate_signals(self, snapshot: dict[str, pd.DataFrame]) -> tuple[StrategyOutput, StrategyTelemetry]:
        prices = snapshot["price"]
        output = StrategyOutput()
        telemetry = StrategyTelemetry()
        
        # We don't hardcode individual metric keys anymore; 
        # we initialize a dynamic multi-level dict structure
        # telemetry.metrics will hold structures like: {"slope_long": {"AAPL": 0.012}, "r_squared_long": {"AAPL": 0.85}}
        
        if len(prices) < self.long_window:
            return output, telemetry
            
        long_snapshot = prices.iloc[-self.long_window:]
        short_snapshot = prices.iloc[-self.short_window:]
        
        long_scores_dict = {}
        short_scores_dict = {}
        
        for ticker in prices.columns:
            # 1. Long Horizon Calculation
            y_long = long_snapshot[ticker].dropna().values
            if len(y_long) == self.long_window:
                l_score, l_metrics = self.scorer.compute_score(y_long)
                long_scores_dict[ticker] = l_score
                
                # Dynamically unpack all metrics provided by the scorer for the long window
                for metric_name, value in l_metrics.items():
                    key = f"{metric_name}_long"
                    if key not in telemetry.metrics:
                        telemetry.metrics[key] = {}
                    telemetry.metrics[key][ticker] = value
            else:
                long_scores_dict[ticker] = np.nan

            # 2. Short Horizon Calculation
            y_short = short_snapshot[ticker].dropna().values
            if len(y_short) == self.short_window:
                s_score, s_metrics = self.scorer.compute_score(y_short)
                short_scores_dict[ticker] = s_score
                
                # Dynamically unpack all metrics provided by the scorer for the short window
                for metric_name, value in s_metrics.items():
                    key = f"{metric_name}_short"
                    if key not in telemetry.metrics:
                        telemetry.metrics[key] = {}
                    telemetry.metrics[key][ticker] = value
            else:
                short_scores_dict[ticker] = np.nan

        # Save the finalized structural composite scores to telemetry for auditing
        telemetry.metrics["raw_score_long"] = {t: float(v) for t, v in long_scores_dict.items()}
        telemetry.metrics["raw_score_short"] = {t: float(v) for t, v in short_scores_dict.items()}

        # 3. Cross-sectional ranking logic
        long_ranks = pd.Series(long_scores_dict).rank(pct=True, na_option='keep')
        short_ranks = pd.Series(short_scores_dict).rank(pct=True, na_option='keep')
        
        composite_scores = (long_ranks * self.structural_weight) + (short_ranks * self.tactical_weight)
        final_percentiles = composite_scores.fillna(0.0).rank(pct=True)

        for ticker, score in final_percentiles.items():
            output.append(SignalPayload(ticker=str(ticker), kind="REGULAR_REBALANCE", score=float(score)))
            
        return output, telemetry

