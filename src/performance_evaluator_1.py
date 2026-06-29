# performance_evaluator.py
import pandas as pd
import numpy as np


class AdvancedEvaluator:
    def __init__(self, initial_capital: float = 100000.0, transaction_cost_pct: float = 0.001, execution_lag: int = 1):
        self.initial_capital = initial_capital
        self.fee_rate = transaction_cost_pct
        self.execution_lag = execution_lag
        
        self.current_capital = initial_capital
        self.active_weights = None
        
        # Track daily equity values for true risk analysis
        self.daily_nav = pd.Series(dtype=float)
        self.rebalance_log = []

    def log_daily_value(self, date: pd.Timestamp, price_matrix: pd.DataFrame, base_rebalance_date: pd.Timestamp):
        """
        Calculates the exact value of the portfolio on a DAILY basis to capture 
        intra-month volatility and real drawdowns.
        """
        if self.active_weights is None:
            self.daily_nav[date] = self.initial_capital
            return

        # Calculate asset returns from the rebalance execution day to today
        initial_prices = price_matrix.loc[base_rebalance_date]
        current_prices = price_matrix.loc[date]
        
        asset_returns = (current_prices / initial_prices) - 1
        asset_returns = asset_returns.fillna(0.0)
        
        # Current value reflects daily price fluctuations of the chosen weights
        portfolio_return = (self.active_weights * asset_returns).sum()
        real_time_capital = self.current_capital * (1.0 + portfolio_return)
        
        self.daily_nav[date] = real_time_capital

    def execute_rebalance(self, rebalance_date: pd.Timestamp, target_weights: pd.Series, price_matrix: pd.DataFrame):
        """
        Updates the structural capital account after accounting for fees and shifts.
        """
        # Find the actual trading date when execution occurs based on lag
        trading_dates = price_matrix.index
        rebalance_idx = trading_dates.get_loc(rebalance_date)
        execution_idx = min(rebalance_idx + self.execution_lag, len(trading_dates) - 1)
        execution_date = trading_dates[execution_idx]

        if self.active_weights is not None:
            # Finalize the capital gain/loss from the old period up to this execution date
            initial_prices = price_matrix.loc[self.rebalance_log[-1]['executed_at']]
            execution_prices = price_matrix.loc[execution_date]
            
            period_returns = (execution_prices / initial_prices) - 1
            portfolio_return = (self.active_weights * period_returns.fillna(0.0)).sum()
            self.current_capital *= (1.0 + portfolio_return)

            # Calculate turnover friction
            weight_delta = target_weights - self.active_weights
            turnover = weight_delta.abs().sum()
            self.current_capital -= (self.current_capital * turnover * self.fee_rate)

        else:
            # First allocation fees
            self.current_capital -= (self.current_capital * target_weights.sum() * self.fee_rate)

        self.active_weights = target_weights.copy()
        self.rebalance_log.append({
            'scheduled_at': rebalance_date,
            'executed_at': execution_date,
            'weights': target_weights
        })

    def generate_advanced_report(self, forward_months: int, risk_free_rate: float = 0.04) -> dict:
        df = pd.DataFrame({"nav": self.daily_nav})
        if df.empty: return {}

        cutoff_date = df.index[-1] - pd.DateOffset(months=forward_months)
        
        is_curve = df.loc[:cutoff_date]["nav"]
        oos_curve = df.loc[cutoff_date:]["nav"]

        def analyze_curve(curve, label):
            daily_returns = curve.pct_change().dropna()
            if daily_returns.empty: return {}

            # Annualized Performance Metrics
            total_return = (curve.iloc[-1] / curve.iloc[0]) - 1
            ann_return = (1 + total_return) ** (252 / len(curve)) - 1
            ann_vol = daily_returns.std() * np.sqrt(252)
            
            # Risk Adjusted Metrics (Sharpe & Sortino)
            excess_return = ann_return - risk_free_rate
            sharpe = excess_return / ann_vol if ann_vol > 0 else 0
            
            downside_returns = daily_returns[daily_returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252)
            sortino = excess_return / downside_vol if downside_vol > 0 else 0
            
            # True Peak-to-Trough Max Drawdown
            rolling_max = curve.cummax()
            max_dd = ((curve - rolling_max) / rolling_max).min()

            return {
                f"{label}_Annualized_Return": round(ann_return * 100, 2),
                f"{label}_Annualized_Vol": round(ann_vol * 100, 2),
                f"{label}_Sharpe_Ratio": round(sharpe, 2),
                f"{label}_Sortino_Ratio": round(sortino, 2),
                f"{label}_Max_Drawdown": round(max_dd * 100, 2),
            }

        return {
            "In-Sample": analyze_curve(is_curve, "IS"),
            "Forward_OOS": analyze_curve(oos_curve, "OOS")
        }