# executor_module.py
import pandas as pd
from dataclasses import dataclass
from src_old import logging
logger=logging.setup_logger(name="backtest_logger", log_file="backtest.log")


@dataclass(frozen=True)
class ExecutedTrade:
    """Immutable single asset transaction record."""
    date: pd.Timestamp
    ticker: str
    action: str        # "BUY" or "SELL"
    sleeve: str        # "CORE" or "TACTICAL"
    shares: float
    price: float
    gross_value: float
    fee: float
    net_value: float


class TransactionExecutor:
    """
    Module D: Stateful Transaction Executor.
    Evaluates net weight targets and applies transactional mutations atomically.
    """
    def __init__(self, fee_rate: float = 0.001, trade_delay: int = 1):
        self.fee_rate = fee_rate
        self.trade_delay = trade_delay

    def execute_trades(self, state, today: pd.Timestamp, current_prices: pd.Series):
        """
        Evaluates and runs explicit adjustments against the global state.
        Enforces execution priority: Exits/Sells executed before Buys/Deployments.
        """
        is_core = state.need_trade.is_trade_core(today, self.trade_delay)
        is_sell_adhoc = state.need_trade.is_sell_adhoc(today, self.trade_delay)
        is_buy_adhoc = state.need_trade.is_buy_adhoc(today, self.trade_delay)

        if not (is_core or is_sell_adhoc or is_buy_adhoc):
            return

        # -----------------------------------------------------------------
        # STEP 1: PROCESS HIGH-PRIORITY AD-HOC EXITS
        # -----------------------------------------------------------------
        if is_sell_adhoc:
            sell_adhoc = state.need_trade.sell_adhoc
            for signal in sell_adhoc:
                if signal.kind == "AD_HOC_EXIT" and signal.ticker in state.tactical_positions:
                    shares = state.tactical_positions[signal.ticker]
                    if shares > 0:
                        price = current_prices[signal.ticker]
                        gross = shares * price
                        fee = gross * self.fee_rate
                        
                        trade_rec = ExecutedTrade(
                            date=today, ticker=signal.ticker, action="SELL", sleeve="TACTICAL",
                            shares=shares, price=price, gross_value=gross, fee=fee, net_value=gross - fee
                        )
                        # State is updated immediately following calculation loop
                        logger.info(f'    {today}: ADHOC_SELL {signal.ticker} ... {round(shares,2)}@{round(price,2)}, fee: {fee}')
                        state.commit_executed_trade(trade_rec)

            state.need_trade.clear_adhoc_sell()

        # -----------------------------------------------------------------
        # STEP 2: PROCESS AD-HOC BUYS (Deploys Tactical Sleeve Cash)
        # -----------------------------------------------------------------
        if is_buy_adhoc:
            buy_adhoc = state.need_trade.buy_adhoc
            for signal in buy_adhoc:
                if signal.kind != "AD_HOC_BUY":
                    continue
                price = current_prices[signal.ticker]
                # allocation_pct is stashed on quantity_target by the caller
                # (budget_allocator's Part B), e.g. 0.10 for "10% of tactical NAV"
                allocation_pct = signal.quantity_target
                tactical_nav = state.calculate_sleeve_value(
                    state.tactical_cash, state.tactical_positions, current_prices
                )
                target_dollars = min(tactical_nav * allocation_pct, state.tactical_cash)
                fee = target_dollars * self.fee_rate
                shares_to_buy = (target_dollars - fee) / price

                if shares_to_buy > 0:
                    gross = shares_to_buy * price
                    trade_rec = ExecutedTrade(
                        date=today, ticker=signal.ticker, action="BUY", sleeve="TACTICAL",
                        shares=shares_to_buy, price=price, gross_value=gross, fee=fee, net_value=gross + fee
                    )
                    logger.info(f'    {today}: ADHOC_BUY {signal.ticker} ... {round(shares_to_buy,2)}@{round(price,2)}, fee: {fee}')
                    state.commit_executed_trade(trade_rec)

            state.need_trade.clear_adhoc_buy()

        # -----------------------------------------------------------------
        # STEP 3: PROCESS CORE POSITION REBALANCING (NETTING METHOD)
        # -----------------------------------------------------------------
        if is_core:
            core_target_weights = state.need_trade.core_target_weights
            total_core_value = state.calculate_sleeve_value(state.core_cash, state.core_positions, current_prices)

            # Map target and current value vectors
            target_dollars = {t: total_core_value * w for t, w in core_target_weights.items()}
            current_dollars = {t: state.core_positions.get(t, 0.0) * current_prices[t] for t in current_prices.index}

            deltas = {}
            all_tickers = set(target_dollars.keys()).union(set(current_dollars.keys()))
            for ticker in all_tickers:
                deltas[ticker] = target_dollars.get(ticker, 0.0) - current_dollars.get(ticker, 0.0)

            # Use a threshold relative to portfolio value, not a fixed tiny dollar amount
            MIN_TRADE_DOLLARS = max(1.0, total_core_value * 1e-4)

            # --- SUB-STEP A: SELLS & TRIMS FIRST (Generates Cash Buffer) ---
            for ticker, delta in deltas.items():
                if delta < -MIN_TRADE_DOLLARS:
                    dollars_to_free_up = abs(delta)
                    current_shares = state.core_positions.get(ticker, 0.0)
                    price = current_prices[ticker]

                    # Account for fee up front: to NET dollars_to_free_up after fee,
                    # we need to sell slightly more gross.
                    gross_needed = dollars_to_free_up / (1.0 - self.fee_rate)
                    shares_to_sell = min(gross_needed / price, current_shares)

                    if shares_to_sell > 0:
                        gross = shares_to_sell * price
                        fee = gross * self.fee_rate
                        net = gross - fee

                        sell_trade = ExecutedTrade(
                            date=today, ticker=ticker, action="SELL", sleeve="CORE",
                            shares=shares_to_sell, price=price, gross_value=gross, fee=fee, net_value=net
                        )
                        logger.info(f'    {today}:CORE_SELL {ticker} ... {round(shares_to_sell,2)}@{round(price,2)}, fee: {round(fee,2)}')
                        state.commit_executed_trade(sell_trade)

            # --- SUB-STEP B: BUYS & ALLOCATIONS SECOND (Deploys Generated Cash) ---
            buy_orders = {t: d for t, d in deltas.items() if d > MIN_TRADE_DOLLARS}
            total_requested_buys = sum(buy_orders.values())

            if total_requested_buys > 0:
                # Use actual available cash (existing cash + what selling just raised),
                # haircut for fees on the buy side too.
                available_buy_pool = state.core_cash * (1.0 - self.fee_rate)
                scaling_factor = min(available_buy_pool / total_requested_buys, 1.0)

                for ticker, delta in buy_orders.items():
                    allocated_cash = delta * scaling_factor
                    price = current_prices[ticker]
                    fee = allocated_cash * self.fee_rate
                    shares_to_buy = (allocated_cash - fee) / price  # fee comes OUT of allocated cash, not on top

                    if shares_to_buy > 0:
                        gross = shares_to_buy * price
                        buy_trade = ExecutedTrade(
                            date=today, ticker=ticker, action="BUY", sleeve="CORE",
                            shares=shares_to_buy, price=price, gross_value=gross, fee=fee, net_value=gross + fee
                        )
                        logger.info(f'    {today}:CORE_BUY {ticker} ... {round(shares_to_buy,2)}@{round(price,2)}, fee: {round(fee,2)}')
                        state.commit_executed_trade(buy_trade)

            # Finalize block execution parameters
            state.need_trade.clear_core()
            state.record_daily_snapshot(today, current_prices, kind='act')