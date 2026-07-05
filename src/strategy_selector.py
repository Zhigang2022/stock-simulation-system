# strategy_selector.py
import pandas as pd
import numpy as np
from src.signal_schema import SignalPayload, StrategyOutput, StrategyTelemetry
from abc import ABC, abstractmethod

class BaseStrategy:
    """
    Abstract base class for trading strategies.
    """
    def calculate_signals(self, snapshot: dict[str, pd.DataFrame]) -> tuple[StrategyOutput, dict[str, float] | None]:
        """
        Calculate trading signals based on the provided historical snapshot data.

        Parameters
        ----------
        snapshot : dict[str, pd.DataFrame]
            A dictionary containing historical market data. Typically includes a "price"
            key with a DataFrame where columns represent tickers and rows represent time steps.

        Returns
        -------
        tuple[StrategyOutput, dict[str, float] | None]
            A tuple containing:
            - StrategyOutput: The container holding calculated SignalPayload signals.
            - dict[str, float] | None: Auxiliary metrics computed during execution (e.g., diagnostic values).
        """
        raise NotImplementedError


class RiskAdjustedMomentum(BaseStrategy):
    """
    Risk-Adjusted Momentum strategy.

    This strategy ranks assets based on their cumulative return over a specified lookback
    period, normalized by their annualized daily return volatility over the same period.
    """
    def __init__(self, lookback_days: int = 252):
        """
        Initialize the RiskAdjustedMomentum strategy.

        Parameters
        ----------
        lookback_days : int, default 252
            The number of historical trading days to include in the lookback window.
        """
        self.lookback_days = lookback_days

    def calculate_signals(self, snapshot: dict[str, pd.DataFrame]) -> tuple[StrategyOutput, None]:
        """
        Compute risk-adjusted momentum signals for each asset in the snapshot.

        Calculates the ratio of the historical return to the annualized volatility
        over the lookback window. Scores are then ranked and converted to percentiles.

        Parameters
        ----------
        snapshot : dict[str, pd.DataFrame]
            A dictionary containing historical market data, requiring at least the "price" key.

        Returns
        -------
        tuple[StrategyOutput, None]
            A tuple containing:
            - StrategyOutput: The standardized signals with percentile ranks as scores.
            - None: Placeholder indicating no auxiliary parameters are returned.
        """
        prices = snapshot["price"]
        output = StrategyOutput()
        
        if len(prices) < self.lookback_days:
            return output, None
            
        window_prices = prices.iloc[-self.lookback_days:]
        current_price = window_prices.iloc[-1]
        historical_price = window_prices.iloc[0]
        raw_returns = (current_price / historical_price) - 1
        
        daily_returns = window_prices.pct_change()
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        
        risk_adjusted_momentum = raw_returns / annualized_volatility
        risk_adjusted_momentum = risk_adjusted_momentum.replace([np.inf, -np.inf], np.nan)
        
        momentum_scores = risk_adjusted_momentum.rank(pct=True, na_option='keep')
        momentum_scores = momentum_scores.fillna(0.0)
        
        # Build standard output payload
        for ticker, score in momentum_scores.items():
            output.append(SignalPayload(
                ticker=str(ticker),
                kind="REGULAR_REBALANCE",
                score=float(score)
            ))
            
        return output, None

 
class InformationDiscreteMomentum(BaseStrategy):
    """
    Information Discrete Momentum strategy.

    This strategy measures the consistency and strength of price trends by performing 
    a linear regression on log-prices over a specified lookback window. The score for 
    each asset is computed as the product of the regression slope and the coefficient of 
    determination (R-squared).
    """
    def __init__(self, lookback_days: int = 252):
        """
        Initialize the InformationDiscreteMomentum strategy.

        Parameters
        ----------
        lookback_days : int, default 252
            The number of historical trading days to include in the lookback window.
        """
        self.lookback_days = lookback_days

    def calculate_signals(self, snapshot: dict[str, pd.DataFrame]) -> tuple[StrategyOutput, dict[str, float]]:
        """
        Compute information discrete momentum signals for each asset in the snapshot.

        Performs a linear regression of the natural log of asset prices against time
        over the lookback window. The final score is the regression slope multiplied by
        the regression's R-squared value, which is then ranked and converted to percentiles.

        Parameters
        ----------
        snapshot : dict[str, pd.DataFrame]
            A dictionary containing historical market data, requiring at least the "price" key.

        Returns
        -------
        tuple[StrategyOutput, dict[str, float]]
            A tuple containing:
            - StrategyOutput: The standardized signals with percentile ranks as scores.
            - dict[str, float]: A dictionary mapping each ticker symbol to its computed R-squared value.
        """
        prices = snapshot["price"]
        output = StrategyOutput()
        
        if len(prices) < self.lookback_days:
            return output, {}
            
        window_log_prices = np.log(prices.iloc[-self.lookback_days:])
        x = np.arange(self.lookback_days)
        scores_dict = {}
        r2_dict = {}
        slope_dict={}
        
        for ticker in window_log_prices.columns:
            y = window_log_prices[ticker].values
            if np.isnan(y).all() or len(y[~np.isnan(y)]) < self.lookback_days:
                scores_dict[ticker] = np.nan
                r2_dict[ticker] = 0.0
                slope_dict[ticker]=0.0
                continue
                
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            y_bar = np.mean(y)
            ss_tot = np.sum((y - y_bar) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            scores_dict[ticker] = slope * r_squared
            r2_dict[ticker] = r_squared
            slope_dict[ticker]=slope

        raw_scores = pd.Series(scores_dict)
        momentum_scores = raw_scores.rank(pct=True, na_option='keep')
        momentum_scores = momentum_scores.fillna(0.0)
        
        # Build standard output payload
        for ticker, score in momentum_scores.items():
            output.append(SignalPayload(
                ticker=str(ticker),
                kind="REGULAR_REBALANCE",
                score=float(score)
            ))
            
        strategy_tele=StrategyTelemetry(metrics={'r2':r2_dict,
                                                 'slope':slope_dict})
        return output, strategy_tele


class MomentumScorer(ABC):
    @abstractmethod
    def compute_score(self, y: np.ndarray) -> tuple[float, float]:
        """Returns tuple of (raw_score, auxiliary_telemetry_metric)"""
        pass


class GeometricDriftScorer(MomentumScorer):
    """Calculates continuous drift rate adjusted for volatility drag."""
    def compute_score(self, prices: np.ndarray) -> tuple[float, float]:
        if len(prices) < 2 or prices[0] <= 0:
            return np.nan, np.nan
        
        # Calculate log returns for compounding accuracy
        log_returns = np.diff(np.log(prices))
        vol = np.std(log_returns) * np.sqrt(252)
        
        # Total continuous return over the window
        total_log_return = np.log(prices[-1] / prices[0])
        
        # Volatility drag subtraction: True geometric drift rate
        geometric_drift = total_log_return - (0.5 * (vol ** 2))
        return geometric_drift, vol


class WeightedLinearRegressionScorer(MomentumScorer):
    """Performs WLS on log prices to give recent data higher trend weight."""
    def __init__(self, decay_factor: float = 0.98):
        self.decay_factor = decay_factor

    def compute_score(self, prices: np.ndarray) -> tuple[float, float]:
        n = len(prices)
        if n < 3:
            return np.nan, np.nan
            
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
            
            return slope * r_squared, r_squared
        except np.linalg.LinAlgError:
            return np.nan, np.nan

# ==========================================
# 2. ADVANCED COMPOSITE STRATEGY MODULES
# ==========================================

class ComprehensiveMultiHorizonStrategy:
    """
    Advanced cross-sectional strategy combining Structural (Long-Term) 
    and Tactical (Short-Term) momentum components via pluggable scorers.
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
        
        # Initialize sub-structures in telemetry for clarity and debugging
        telemetry.metrics["long_raw"] = {}
        telemetry.metrics["short_raw"] = {}
        telemetry.metrics["aux_metric"] = {}
        
        # Require enough history for the max lookup window
        if len(prices) < self.long_window:
            return output, telemetry
            
        long_snapshot = prices.iloc[-self.long_window:]
        short_snapshot = prices.iloc[-self.short_window:]
        
        long_raw_dict = {}
        short_raw_dict = {}
        
        for ticker in prices.columns:
            # 1. Compute Structural Momentum (Long-Term)
            y_long = long_snapshot[ticker].dropna().values
            if len(y_long) == self.long_window:
                l_score, l_aux = self.scorer.compute_score(y_long)
                long_raw_dict[ticker] = l_score
                telemetry.metrics["aux_metric"][f"{ticker}_long"] = l_aux
            else:
                long_raw_dict[ticker] = np.nan
                
            # 2. Compute Tactical Momentum (Short-Term)
            y_short = short_snapshot[ticker].dropna().values
            if len(y_short) == self.short_window:
                s_score, s_aux = self.scorer.compute_score(y_short)
                short_raw_dict[ticker] = s_score
                telemetry.metrics["aux_metric"][f"{ticker}_short"] = s_aux
            else:
                short_raw_dict[ticker] = np.nan

        # Convert to series to calculate cross-sectional rank percentiles
        long_series = pd.Series(long_raw_dict)
        short_series = pd.Series(short_raw_dict)
        
        # Store raw parameters in audit trail before percentile adjustments
        for ticker in prices.columns:
            telemetry.metrics["long_raw"][ticker] = float(long_series.get(ticker, np.nan))
            telemetry.metrics["short_raw"][ticker] = float(short_series.get(ticker, np.nan))
            
        long_ranks = long_series.rank(pct=True, na_option='keep')
        short_ranks = short_series.rank(pct=True, na_option='keep')
        
        # 3. Blend standardized cross-sectional scores
        composite_scores = (long_ranks * self.structural_weight) + (short_ranks * self.tactical_weight)
        composite_scores = composite_scores.fillna(0.0)
        
        # Final ranking pass over composite values to normalize signals to standard [0, 1] range
        final_percentiles = composite_scores.rank(pct=True)

        for ticker, score in final_percentiles.items():
            output.append(SignalPayload(
                ticker=str(ticker),
                kind="REGULAR_REBALANCE",
                score=float(score)
            ))
            
        return output, telemetry