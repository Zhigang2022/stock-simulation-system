# performance_evaluator.py
import pandas as pd
import numpy as np

class PerformanceEvaluator:
    """
    Module E: Clean Analytical Performance Evaluator.
    Consumes historical NAV timelines post-run to construct institutional risk-adjusted reports.
    """
    def __init__(self, annual_rf: float = 0.04):
        self.annual_rf = annual_rf
        self.daily_rf = (1 + annual_rf) ** (1 / 252) - 1

    def _compute_metrics(self, nav_series: pd.Series) -> dict:
        """Helper method to isolate mathematical calculations from reporting wrappers."""
        if len(nav_series) < 2:
            return {"CAGR (%)": 0.0, "Sharpe": 0.0, "Sortino": 0.0, "Max Drawdown (%)": 0.0, "Calmar": 0.0}

        # 1. CAGR Calculation
        years_elapsed = len(nav_series) / 252.0
        cagr = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (1 / years_elapsed) - 1 if nav_series.iloc[0] > 0 else 0
        
        # 2. Daily Returns Analysis
        daily_returns = nav_series.pct_change().dropna()
        excess_returns = daily_returns - self.daily_rf
        
        # 3. Sharpe & Sortino
        total_std = daily_returns.std()
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std(ddof=0)
        
        sharpe = (excess_returns.mean() / total_std) * np.sqrt(252) if total_std > 0 else 0
        sortino = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # 4. Maximum Drawdown
        rolling_max = nav_series.cummax()
        drawdowns = (nav_series - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        
        # 5. Calmar Ratio
        calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0
        
        return {
            "CAGR (%)": round(cagr * 100, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2),
            "Max Drawdown (%)": round(max_dd * 100, 2),
            "Calmar Ratio": round(calmar, 2)
        }

    def generate_comprehensive_report(self, state_df: pd.DataFrame, forward_months: int = 0) -> dict:
        """
        Analyzes the compiled historical state output and breaks performance down by sleeve.
        
        Args:
            state_df: Unified DataFrame returned by GlobalState.export_nav_dataframe()
            forward_months: If > 0, chunks the tail end of the data to validate Out-of-Sample stability.
        """
        report = {}
        
        # If running a validation horizon split, carve up the timeline dataframe
        if forward_months > 0:
            last_date = state_df.index[-1]
            cutoff_date = last_date - pd.DateOffset(months=forward_months)
            
            in_sample_df = state_df.loc[:cutoff_date]
            out_of_sample_df = state_df.loc[cutoff_date:]
            
            report["In_Sample"] = {
                "Core_Sleeve": self._compute_metrics(in_sample_df["Core_NAV"]),
                "Tactical_Sleeve": self._compute_metrics(in_sample_df["Tactical_NAV"]),
                "Total_Combined": self._compute_metrics(in_sample_df["Total_NAV"])
            }
            report["Forward_Validation"] = {
                "Core_Sleeve": self._compute_metrics(out_of_sample_df["Core_NAV"]),
                "Tactical_Sleeve": self._compute_metrics(out_of_sample_df["Tactical_NAV"]),
                "Total_Combined": self._compute_metrics(out_of_sample_df["Total_NAV"])
            }
        else:
            # Standard Full-History Audit
            report["Full_Backtest"] = {
                "Core_Sleeve": self._compute_metrics(state_df["Core_NAV"]),
                "Tactical_Sleeve": self._compute_metrics(state_df["Tactical_NAV"]),
                "Total_Combined": self._compute_metrics(state_df["Total_NAV"])
            }
            
        return report