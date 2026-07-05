# executor_module.py
import pandas as pd
from typing import Dict, List
from src.signal_schema import SignalPayload

# TODO: by now, only support ad_hoc buy/sell can't happen at the same day
class TransactionExecutor:
    """
    Module D: Refactored Stateful Transaction Executor.
    Flipped to act as a direct receiver of completed target weights and approved orders.
    """
    def __init__(self, fee_rate: float = 0.001,trade_delay=1):
        self.fee_rate = fee_rate
        self.trade_delay=trade_delay

    def execute_trades(self, 
                       state, 
                       today, 
                       current_prices: pd.Series):
        """
        Executes explicit, validated transactions against isolated global state properties.
        Enforces execution order: Exits handled before Rebalances are established.
        """
        is_core=state.need_trade.is_trade_core(today,self.trade_delay)
        is_sell_adhoc=state.need_trade.is_sell_adhoc(today,self.trade_delay)
        # print('*****')
        # print(f'is_core: {is_core}, is_sell_adhoc: {is_sell_adhoc}')
        
        if not(is_core or is_sell_adhoc):
            return 
            
        # 1. PROCESS HIGH-PRIORITY EXITS FIRST
        if is_sell_adhoc:
            sell_adhoc=state.need_trade.sell_adhoc
            for signal in sell_adhoc:
                if signal.kind == "AD_HOC_EXIT" and signal.ticker in state.tactical_positions:
                    shares = state.tactical_positions[signal.ticker]
                    if shares > 0:
                        proceeds = (shares * current_prices[signal.ticker]) * (1.0 - self.fee_rate)
                        state.tactical_cash += proceeds
                        state.tactical_positions.pop(signal.ticker, None)
                        state.tactical_peaks.pop(signal.ticker, None)

            print(f'sell :{sell_adhoc}')
            state.trade.clear_adhoc_sell()

        # 2. PROCESS OPTIMIZED POSITION NETTING TO MINIMIZE TRANSACTION FEES
        if is_core:
            core_target_weights=state.need_trade.core_target_weights
            # Calculate current total value of the core sleeve BEFORE making any changes
            total_core_value = state.calculate_sleeve_value(state.core_cash, state.core_positions, current_prices)
            
            # Map out target dollar amounts for each asset
            target_dollars = {ticker: total_core_value * weight for ticker, weight in core_target_weights.items()}
            
            # Map out current dollar amounts for each asset
            current_dollars = {ticker: state.core_positions.get(ticker, 0.0) * current_prices[ticker] for ticker in current_prices.index}
            
            # Calculate the net difference vector (Target $ - Current $)
            # Positive delta = Need to BUY more. Negative delta = Need to SELL/TRIM.
            deltas = {}
            all_tickers = set(target_dollars.keys()).union(set(current_dollars.keys()))
            for ticker in all_tickers:
                deltas[ticker] = target_dollars.get(ticker, 0.0) - current_dollars.get(ticker, 0.0)

            # -----------------------------------------------------------------
            # STEP 1: EXECUTE ALL SELLS & TRIMS FIRST (Generates Cash Buffer)
            # -----------------------------------------------------------------
            for ticker, delta in deltas.items():
                if delta < 0:  # We hold too much or need to drop it entirely
                    dollars_to_free_up = abs(delta)
                    current_shares = state.core_positions.get(ticker, 0.0)
                    
                    # Prevent floating-point over-selling errors
                    shares_to_sell = min(dollars_to_free_up / current_prices[ticker], current_shares)
                    
                    if shares_to_sell > 0:
                        gross_proceeds = shares_to_sell * current_prices[ticker]
                        fee = gross_proceeds * self.fee_rate
                        
                        # Update State
                        state.core_cash += (gross_proceeds - fee)
                        state.fees_paid += fee
                        state.core_positions[ticker] -= shares_to_sell

            # -----------------------------------------------------------------
            # STEP 2: EXECUTE ALL BUYS & EXPANSIONS SECOND (Deploys Cash)
            # -----------------------------------------------------------------
            # Create a small safety haircut fraction across our buy orders to ensure commissions don't overdraw cash
            buy_orders = {ticker: delta for ticker, delta in deltas.items() if delta > 0}
            total_requested_buys = sum(buy_orders.values())
            
            if total_requested_buys > 0:
                # Scale the buys proportionally to fit exactly inside our available cash minus transaction costs
                available_buy_pool = state.core_cash * (1.0 - self.fee_rate)
                scaling_factor = min(available_buy_pool / total_requested_buys, 1.0)
                
                total_fee=0
                for ticker, delta in buy_orders.items():
                    allocated_cash = delta * scaling_factor
                    fee = allocated_cash * self.fee_rate
                    
                    shares_to_buy = allocated_cash / current_prices[ticker]
                    
                    # Update State
                    state.core_positions[ticker] = state.core_positions.get(ticker, 0.0) + shares_to_buy
                    state.core_cash -= (allocated_cash + fee)
                    state.fees_paid += fee
                    total_fee+=fee
            print(f'{today} rebalance to {core_target_weights}, paid fee: {total_fee}')
            state.need_trade.clear_core()
            state.record_daily_snapshot(today,current_prices,kind='act')

        # # 3. PROCESS SPECULATIVE AD-HOC BUY ENTRIES LAST
        # if is_adhod:
        #     for signal in approved_ad_hoc:
        #         if signal.kind == "AD_HOC_BUY":
        #             # Allocating a predefined tactical cash buffer slot to position entry
        #             tactical_allocation = state.tactical_cash * 0.20 
        #             state.tactical_positions[signal.ticker] = tactical_allocation / current_prices[signal.ticker]
        #             state.tactical_cash -= tactical_allocation