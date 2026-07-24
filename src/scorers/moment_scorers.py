
import numpy as np
from src.scorers.base_scorer import MomentumScorer


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