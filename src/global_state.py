import pandas as pd
import numpy as np
from typing import Dict, Any

class GlobalState:
    """
    The Stateful Core of the backtesting engine. Manages cash balances, 
    active holdings across multiple strategic sleeves, and maintains 
    the master chronological historical NAV ledger.
    """
    def __init__(self, initial_capital: float, core_allocation_pct: float = 0.70):
        """
        Initializes the global state with a strict behavioral firewall dividing 
        capital into virtual sub-accounts (sleeves).
        
        Args:
            initial_capital: Total dollar amount starting capital.
            core_allocation_pct: Fraction of capital reserved for the slow, 
                                 regular rebalancing sleeve (default 70%).
        """
        if not (0.0 <= core_allocation_pct <= 1.0):
            raise ValueError("Allocation percentage must be between 0.0 and 1.0")
            
        # Hard capital partitions (The Behavioral Firewall)
        self.core_cash: float = initial_capital * core_allocation_pct
        self.tactical_cash: float = initial_capital * (1.0 - core_allocation_pct)
        
        # Position Trackers: Dict[ticker, shares_count]
        # Positive shares = Long, Negative shares = Short
        self.core_positions: Dict[str, float] = {}
        self.tactical_positions: Dict[str, float] = {}
        
        # Micro-state properties for exit strategy tracking
        # Dict[ticker, highest_tracked_price_since_entry]
        self.tactical_peaks: Dict[str, float] = {}
        self.tactical_entry_dates: Dict[str, Any] = {}
        
        # Historical Append-Only Timeseries Ledgers (For Module E Evaluation)
        self.nav_history: Dict[Any, float] = {}      # {date: total_combined_nav}
        self.core_nav_history: Dict[Any, float] = {} # {date: core_sleeve_nav}
        self.tactical_nav_history: Dict[Any, float] = {} # {date: tactical_sleeve_nav}

    @property
    def total_cash(self) -> float:
        """Returns aggregate unallocated cash across both accounts."""
        return self.core_cash + self.tactical_cash

    def calculate_sleeve_value(self, cash: float, positions: Dict[str, float], market_prices: pd.Series) -> float:
        """
        Calculates the liquidation value of a specific sleeve portfolio.
        Handles long and short equity valuations correctly.
        """
        equity_value = 0.0
        for ticker, shares in positions.items():
            if shares == 0:
                continue
            if ticker not in market_prices or np.isnan(market_prices[ticker]):
                raise ValueError(f"Missing current market price for asset: {ticker}")
                
            # Long positions add positive value, Short positions subtract/add based on price direction
            # Mark-to-market valuation: Cash + (Shares * Current Price)
            equity_value += shares * market_prices[ticker]
            
        return cash + equity_value

    def record_daily_snapshot(self, date: Any, market_prices: pd.Series) -> float:
        """
        Calculates the exact Mark-to-Market NAV for each individual sleeve 
        and appends it to the historical evaluation ledger.
        
        Must be called precisely at the end of every trading day step.
        """
        # Calculate isolated sub-account values
        core_nav = self.calculate_sleeve_value(self.core_cash, self.core_positions, market_prices)
        tactical_nav = self.calculate_sleeve_value(self.tactical_cash, self.tactical_positions, market_prices)
        combined_nav = core_nav + tactical_nav
        
        # Save records to history
        self.core_nav_history[date] = core_nav
        self.tactical_nav_history[date] = tactical_nav
        self.nav_history[date] = combined_nav
        
        # Dynamically update tactical trailing peaks for open long positions
        for ticker, shares in self.tactical_positions.items():
            if shares > 0 and ticker in market_prices:
                current_price = market_prices[ticker]
                self.tactical_peaks[ticker] = max(self.tactical_peaks.get(ticker, current_price), current_price)
                
        return combined_nav

    def export_nav_dataframe(self) -> pd.DataFrame:
        """
        Compiles the historical snapshot ledgers into a structured, unified 
        Pandas DataFrame ready for the Module E Performance Evaluator.
        """
        df = pd.DataFrame({
            'Core_NAV': pd.Series(self.core_nav_history),
            'Tactical_NAV': pd.Series(self.tactical_nav_history),
            'Total_NAV': pd.Series(self.nav_history)
        })
        df.index.name = 'Date'
        return df