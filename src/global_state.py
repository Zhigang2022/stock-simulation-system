import pandas as pd
import numpy as np
from typing import Dict, Any


from datetime import date
from dataclasses import dataclass
from typing import Optional, Dict
from src.executor_module import ExecutedTrade

@dataclass
class Future_Transaction:
    # Defines the structure and handles default initialization
    core_target_weights: Optional[Dict[str, float]] = None
    core_target_date: Optional[date] = None
    buy_adhoc: Optional[Dict[str, float]] = None
    buy_adhoc_date: Optional[date] = None
    sell_adhoc: Optional[Dict[str, float]] = None
    sell_adhoc_date: Optional[date] = None

    def clear_core(self):
        self.core_target_date=None
        self.core_target_weights=None
    
    def clear_adhoc_buy(self):
        self.buy_adhoc=None
        self.buy_adhoc_date=None

    def clear_adhoc_sell(self):
        self.sell_adhoc=None
        self.sell_adhoc_date=None

    def set_core(self,core_target_weights,core_target_date):
        if len(core_target_weights)!=0:
            self.core_target_weights=core_target_weights
            self.core_target_date=core_target_date

    def set_sell_adhoc(self,sell_adhoc,sell_adhoc_date):
        if len(sell_adhoc)!=0:
            self.sell_adhoc=sell_adhoc
            self.sell_adhoc_date=sell_adhoc_date

    def set_buy_adhoc(self,buy_adhoc,buy_adhoc_date):
        if len(buy_adhoc)!=0:
            self.buy_adhoc=buy_adhoc
            self.buy_adhoc_date=buy_adhoc_date

    def is_trade_core(self,today,delay_days):
        if self.core_target_date is not None:
            if (today-self.core_target_date).days>=delay_days:
                return True
        return False
    
    def is_sell_adhoc(self,today,delay_days):
        if self.sell_adhoc_date is not None:
            if (today-self.sell_adhoc_date).days>=delay_days:
                return True
        
        return False
    
    def is_buy_adhoc(self,today,delay_days):
        if self.buy_adhoc_date is not None:
            if (today-self.buy_adhoc_date).days>=delay_days:
                return True
        return False

@dataclass
class Transient_Signals():
    regular_strategy_output=None
    adhoc_strategy_output=None

    def update_signals(self,regular_strategy_output,adhoc_strategy_output):
        self.regular_strategy_output=regular_strategy_output
        self.adhoc_strategy_output=adhoc_strategy_output

    def clear_signals(self):
        self.regular_strategy_output=None
        self.adhoc_strategy_output=None


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

        # Historical implimented Timeseries Ledgers (For Module E Evaluation)
        self.nav_act_history: Dict[Any, float] = {}      # {date: total_combined_nav}
        self.core_nav_act_history: Dict[Any, float] = {} # {date: core_sleeve_nav}
        self.tactical_nav_act_history: Dict[Any, float] = {} # {date: tactical_sleeve_nav}

        # total transaction fee_paid
        self.fees_paid=0

        # transaction needed
        self.need_trade=Future_Transaction()
        self.signals=Transient_Signals()
        


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

    def commit_executed_trade(self, trade: "ExecutedTrade"):
        """
        The absolute single source of truth for portfolio updates.
        Applies a validated trade transaction ledger entry to state balances.
        """
        if trade.sleeve == "CORE":
            if trade.action == "SELL":
                self.core_cash += trade.net_value  # gross - fee
                self.core_positions[trade.ticker] -= trade.shares
            elif trade.action == "BUY":
                self.core_positions[trade.ticker] = self.core_positions.get(trade.ticker, 0.0) + trade.shares
                self.core_cash -= (trade.gross_value + trade.fee)
            self.fees_paid += trade.fee
                
        elif trade.sleeve == "TACTICAL":
            if trade.action == "SELL":
                self.tactical_cash += trade.net_value
                self.tactical_positions.pop(trade.ticker, None) #sell all
                self.tactical_peaks.pop(trade.ticker, None)
            if trade.action == "BUY":
                # Enforce clean baseline initialization to avoid fractional leakage
                current_shares = self.tactical_positions.get(trade.ticker, 0.0)
                self.tactical_positions[trade.ticker] = current_shares + trade.shares
                self.tactical_cash -= (trade.gross_value + trade.fee)
            self.fees_paid += trade.fee


    def record_daily_snapshot(self, date: Any, market_prices: pd.Series,kind='daily') -> float:
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
        if kind=='act':
            self.core_nav_act_history[date] = core_nav
            self.tactical_nav_act_history[date] = tactical_nav
            self.nav_act_history[date] = combined_nav
        else:
            self.core_nav_history[date] = core_nav
            self.tactical_nav_history[date] = tactical_nav
            self.nav_history[date] = combined_nav

        
        # Dynamically update tactical trailing peaks for open long positions
        for ticker, shares in self.tactical_positions.items():
            if shares > 0 and ticker in market_prices:
                current_price = market_prices[ticker]
                self.tactical_peaks[ticker] = max(self.tactical_peaks.get(ticker, current_price), current_price)
                
        return combined_nav

    def export_nav_dataframe(self,kind='daily') -> pd.DataFrame:
        """
        Compiles the historical snapshot ledgers into a structured, unified 
        Pandas DataFrame ready for the Module E Performance Evaluator.
        """
        if kind=='act':
            df = pd.DataFrame({
            'Core_NAV': pd.Series(self.core_nav_act_history),
            'Tactical_NAV': pd.Series(self.tactical_nav_act_history),
            'Total_NAV': pd.Series(self.nav_act_history)
        })
        else:
            df = pd.DataFrame({
                'Core_NAV': pd.Series(self.core_nav_history),
                'Tactical_NAV': pd.Series(self.tactical_nav_history),
                'Total_NAV': pd.Series(self.nav_history)
            })
            df.index.name = 'Date'
        return df
    
    def display_current_state(self, current_date: Any, market_prices: pd.Series) -> None:
        """
        Micro-level Monitoring Utility.
        Generates a clear, multi-sleeve structural breakdown of active capital,
        current holdings, mark-to-market valuation weights, and active trailing peaks.
        """
        date_str = pd.Timestamp(current_date).strftime('%Y-%m-%d')
        print("=" * 80)
        print(f" SYSTEM PORTFOLIO STATE SNAPSHOT: {date_str}")
        print("=" * 80)

        # -----------------------------------------------------------------
        # SECTION 1: CORE SLEEVE DETAILS (LONG-TERM)
        # -----------------------------------------------------------------
        core_nav = self.calculate_sleeve_value(self.core_cash, self.core_positions, market_prices)
        print(f"\n[CORE SLEEVE] Allocation Ratio: {self.core_cash / (core_nav if core_nav > 0 else 1):.1%} Cash")
        print(f"  - Core Liquid Cash : ${self.core_cash:,.2f}")
        print(f"  - Total Core NAV   : ${core_nav:,.2f}")
        
        core_rows = []
        for ticker, shares in self.core_positions.items():
            if shares != 0:
                price = market_prices.get(ticker, np.nan)
                position_value = shares * price
                weight_in_sleeve = position_value / core_nav if core_nav > 0 else 0.0
                core_rows.append({
                    "Ticker": ticker,
                    "Shares": round(shares, 4),
                    "Price": f"${price:,.2f}" if not np.isnan(price) else "NaN",
                    "Market Value": position_value,
                    "Sleeve Weight": f"{weight_in_sleeve:.2%}"
                })
        
        if core_rows:
            df_core = pd.DataFrame(core_rows).set_index("Ticker")
            # Format dollar values for neat terminal display
            df_core["Market Value"] = df_core["Market Value"].map(lambda x: f"${x:,.2f}")
            print(df_core.to_string())
        else:
            print("  (No active structural long positions currently held in Core)")

        # -----------------------------------------------------------------
        # SECTION 2: TACTICAL SLEEVE DETAILS (SHORT-TERM / AD-HOC)
        # -----------------------------------------------------------------
        tactical_nav = self.calculate_sleeve_value(self.tactical_cash, self.tactical_positions, market_prices)
        print(f"\n[TACTICAL SLEEVE] Allocation Ratio: {self.tactical_cash / (tactical_nav if tactical_nav > 0 else 1):.1%} Cash")
        print(f"  - Tactical Cash     : ${self.tactical_cash:,.2f}")
        print(f"  - Total Tactical NAV: ${tactical_nav:,.2f}")
        
        tactical_rows = []
        for ticker, shares in self.tactical_positions.items():
            if shares != 0:
                price = market_prices.get(ticker, np.nan)
                position_value = shares * price
                weight_in_sleeve = position_value / tactical_nav if tactical_nav > 0 else 0.0
                
                # Extract tracked micro-state for Chandelier exits
                highest_peak = self.tactical_peaks.get(ticker, np.nan)
                current_drawdown = (price - highest_peak) / highest_peak if highest_peak > 0 else 0.0
                
                tactical_rows.append({
                    "Ticker": ticker,
                    "Shares": round(shares, 4),
                    "Price": f"${price:,.2f}" if not np.isnan(price) else "NaN",
                    "Market Value": position_value,
                    "Sleeve Weight": f"{weight_in_sleeve:.2%}",
                    "Highest Peak": f"${highest_peak:,.2f}" if not np.isnan(highest_peak) else "NaN",
                    "Peak Drawdown": f"{current_drawdown:.2%}" if shares > 0 else "N/A (Short)"
                })
                
        if tactical_rows:
            df_tactical = pd.DataFrame(tactical_rows).set_index("Ticker")
            df_tactical["Market Value"] = df_tactical["Market Value"].map(lambda x: f"${x:,.2f}")
            print(df_tactical.to_string())
        else:
            print("  (No active ad-hoc positions or trailing stops active in Tactical)")

        # -----------------------------------------------------------------
        # SECTION 3: COMBINED CONSOLIDATED METRICS
        # -----------------------------------------------------------------
        total_nav = core_nav + tactical_nav
        print("\n" + "-" * 50)
        print(f" CONSOLIDATED PORTFOLIO EQUITY NET VALUE: ${total_nav:,.2f}")
        print(f"  - Core Allocation Weight     : {core_nav / (total_nav if total_nav > 0 else 1):.2%}")
        print(f"  - Tactical Allocation Weight : {tactical_nav / (total_nav if total_nav > 0 else 1):.2%}")
        print("-" * 50 + "\n")


    def log_current_state(self, current_date: Any, market_prices: pd.Series) -> None:
        """
        Flat Monitoring Utility optimized for structured log output.
        Records cash, positions, prices, dynamic sleeve NAVs, and historical fees.
        """
        date_str = pd.Timestamp(current_date).strftime('%Y-%m-%d')
        
        # 1. Calculate underlying Sleeve Values
        core_nav = self.calculate_sleeve_value(self.core_cash, self.core_positions, market_prices)
        tactical_nav = self.calculate_sleeve_value(self.tactical_cash, self.tactical_positions, market_prices)
        total_nav = core_nav + tactical_nav
        
        # 2. Extract and flatten positions into easily searchable string summaries
        # Format: TICKER(SHARES@PRICE)
        core_pos_list = [
            f"{t}({shares:,.1f}@${market_prices.get(t, 0):,.2f})" 
            for t, shares in self.core_positions.items() if shares != 0
        ]
        tactical_pos_list = [
            f"{t}({shares:,.1f}@${market_prices.get(t, 0):,.2f})" 
            for t, shares in self.tactical_positions.items() if shares != 0
        ]
        
        core_pos_str = ", ".join(core_pos_list) if core_pos_list else "None"
        tactical_pos_str = ", ".join(tactical_pos_list) if tactical_pos_list else "None"

        # 3. Safely fetch cumulative fees paid (defaulting to 0.0 if not yet initialized in state)
        fees_paid = getattr(self, "fees_paid", 0.0)

        return_str=f'''
    --- STATE SNAPSHOT [{date_str}] ---
    Cash SUMMARY | Total: ${self.total_cash:,.2f} | Core : ${self.core_cash:,.2f} | Tactical : ${self.tactical_cash:,.2f}
    Nav SUMMARY | Total: ${total_nav:,.2f} | Core : ${core_nav:,.2f} | Tactical : ${tactical_nav:,.2f}
    CORE POSITIONS | {core_pos_str}
    TACTICAL POSITIONS | {tactical_pos_str}
    FINANCIAL METRICS  | Cumulative Fees Paid: ${fees_paid:,.2f}
        '''
        return return_str
    
