# performance_evaluator.py
import pandas as pd
import numpy as np

class PerformanceEvaluator:
    def __init__(self, initial_capital: float = 100000.0, transaction_cost_pct: float = 0.001):
        """
        Args:
            initial_capital: Starting cash for the backtest.
            transaction_cost_pct: Slip and commission fee per trade (e.g., 0.001 = 0.1% per buy/sell).
        """
        self.initial_capital = initial_capital
        self.fee_rate = transaction_cost_pct
        
        # Internal state tracking
        self.current_capital = initial_capital
        self.current_weights = None  # Series of weights from the previous period
        self.nav_history = []        # List of dicts to capture daily or monthly equity curve
        
    def update_portfolio_state(self, current_date: pd.Timestamp, target_weights: pd.Series, price_matrix: pd.DataFrame):
        """
        Executes a periodic rebalance and records the portfolio value.
        
        Args:
            current_date: The active rebalance timestamp T.
            target_weights: The optimized weight vector from Module D.
            price_matrix: The entire historical price matrix (to calculate returns between T and T+1).
        """
        # If this is the absolute first rebalance, we just deploy cash according to target weights
        if self.current_weights is None:
            # Assume 100% of capital pays commission to establish positions
            turnover = target_weights.sum()
            fees = self.current_capital * turnover * self.fee_rate
            self.current_capital -= fees
            
            self.current_weights = target_weights.copy()
            self.nav_history.append({"date": current_date, "nav": self.current_capital})
            return

        # 1. Calculate the holding return from the LAST rebalance date to CURRENT date
        last_date = self.nav_history[-1]["date"]
        
        # Get price returns for all assets across this interval
        period_prices = price_matrix.loc[[last_date, current_date]]
        # Calculate asset returns: (Price_current / Price_last) - 1
        asset_returns = (period_prices.iloc[-1] / period_prices.iloc[0]) - 1
        asset_returns = asset_returns.fillna(0.0) # Handle missing/delisted assets safely
        
        # 2. Update capital based on the performance of our existing weights
        portfolio_return = (self.current_weights * asset_returns).sum()
        self.current_capital *= (1.0 + portfolio_return)
        
        # 3. Calculate Portfolio Turnover & Transaction Fees for the new shift
        # Turnover is the sum of absolute changes in weights
        weight_delta = target_weights - self.current_weights
        turnover = weight_delta.abs().sum()
        
        transaction_fees = self.current_capital * turnover * self.fee_rate
        self.current_capital -= transaction_fees
        
        # 4. Lock in state for the next interval
        self.current_weights = target_weights.copy()
        self.nav_history.append({"date": current_date, "nav": self.current_capital})

    def generate_report(self, forward_months: int) -> dict:
        """
        Computes standard performance metrics and strictly splits In-Sample vs Forward Performance.
        """
        df_nav = pd.DataFrame(self.nav_history).set_index("date")
        if df_nav.empty:
            return {}
            
        # Define the forward-performance boundary
        last_date = df_nav.index[-1]
        cutoff_date = last_date - pd.DateOffset(months=forward_months)
        
        # Split Equity Curves
        is_nav = df_nav.loc[:cutoff_date]["nav"]
        oos_nav = df_nav.loc[cutoff_date:]["nav"]
        
        def calculate_metrics(nav_series):
            if len(nav_series) < 2:
                return {"Total Return": 0.0, "Max Drawdown": 0.0}
            
            total_return = (nav_series.iloc[-1] / nav_series.iloc[0]) - 1
            
            # Max Drawdown calculation
            rolling_max = nav_series.cummax()
            drawdowns = (nav_series - rolling_max) / rolling_max
            max_dd = drawdowns.min()
            
            return {
                "Total Return": total_return,
                "Max Drawdown": max_dd,
                "Final Value": nav_series.iloc[-1]
            }
            
        return {
            "In-Sample Performance": calculate_metrics(is_nav),
            "Forward Performance (OOS)": calculate_metrics(oos_nav)
        }