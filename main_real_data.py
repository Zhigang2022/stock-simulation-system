import numpy as np
np.random.seed(42)
import pandas as pd
from tqdm import tqdm

from src import data_broker
from src import statioinary_bootstrap
from src import global_state
from src import calendar_iterator
from src import strategy_selector, ad_hoc_strategy
from src import executor_module, performance_evaluator
from src import compliance_filters, budget_allocator
from src import audit
from src import signal_schema
from src import metrics
from src import logging


logger=logging.setup_logger(name="backtest_logger", log_file="backtest.log")


tickers='ALGN,AAPL,PANW,AMD,ZS,MDLZ,XEL,VRSN,INSM,ON,WDAY,ENPH,DLTR,ADI,COST,ABNB,CEG,MSFT,GOOGL,CDNS,PYPL,FAST,JD,MTCH,SNPS,LULU,CMCSA,ADBE,ADSK'.split(',')
broker = data_broker.DataBroker(tickers=tickers, start_date="2022-01-01", end_date="2026-04-01")
universe_data = broker.fetch_universe_data()


risk_adj_moment=strategy_selector.ComprehensiveMultiHorizonStrategy(
    scorer=strategy_selector.SimpleRiskAdjustedScorer(),
    long_window=252,
    short_window=21,
    structural_weight=0.6
)

linear_strategy = strategy_selector.ComprehensiveMultiHorizonStrategy(
    scorer=strategy_selector.TraditionalLinearRegressionScorer(),
    long_window=252,
    short_window=21,
    structural_weight=0.6
)

# Configuration A: Advanced Risk-Adjusted Blended Momentum
advanced_risk_strat = strategy_selector.ComprehensiveMultiHorizonStrategy(
    scorer=strategy_selector.GeometricDriftScorer(),
    long_window=252,
    short_window=21,
    structural_weight=0.6
)

# Configuration B: Advanced Information Discrete Momentum (Time-Weighted)
advanced_info_strat = strategy_selector.ComprehensiveMultiHorizonStrategy(
    scorer=strategy_selector.WeightedLinearRegressionScorer(decay_factor=0.97),
    long_window=126,
    short_window=10,
    structural_weight=0.7
)

portfolio_filter=compliance_filters.LiquidityComplianceFilter()


world_data_dict= universe_data
# module_c_regular=risk_adj_moment 
module_c_regular=linear_strategy
module_c1_adhoc=None #adhoc_exit

#def run(module_c_regular):
allocator=budget_allocator.IntegratedBudgetAllocator(top_percent=.10, allocation_type='equal')
g_state=global_state.GlobalState(initial_capital=100_000,core_allocation_pct=1.0)
executor=executor_module.TransactionExecutor(trade_delay=1)
calendar = calendar_iterator.CalendarIterator(world_data_dict, interval="ME")
rebalance_dates = calendar.generate_rebalance_dates()



# Initialize a formal ledger
audit_ledger: list[audit.AuditRecord] = []
logger.info("Initializing Backtest Engine...")

for i, today in enumerate(world_data_dict['price'].index):
    snapshot = calendar.get_historical_snapshot(today)
    market_prices = snapshot['price'].loc[today]
    
    # Initialize containers for this specific time-step's audit trail
    current_telemetry = signal_schema.StrategyTelemetry()
    g_state.signals.clear_signals()
    
    # 1. Strategy Calculation Pass
    regular_strategy_output,adhoc_strategy_output=None, None
    raw_regular_signals,filtered_regular_signals=[],[]
    if today in rebalance_dates and (module_c_regular is not None):
        # Update your strategy to return the StrategyTelemetry object instead of loose info dicts
        regular_strategy_output, current_telemetry = module_c_regular.calculate_signals(snapshot)
        raw_regular_signals = list(regular_strategy_output.signals)
        regular_strategy_output = portfolio_filter.filter_signals(today, regular_strategy_output, snapshot)
        filtered_regular_signals=list(regular_strategy_output.signals)

        if regular_strategy_output.signals:
            logger.info(f"=== REBALANCE EVENT TRIGGERED: {today.strftime('%Y-%m-%d')} ===")

    raw_adhoc_signals,filtered_adhoc_signals=[],[]
    if module_c1_adhoc is not None:
        adhoc_strategy_output = module_c1_adhoc.evaluate_exits(g_state, snapshot)
        raw_adhoc_signals= list(adhoc_strategy_output.signals)
        adhoc_strategy_output = portfolio_filter.filter_signals(today, adhoc_strategy_output, snapshot)
        filtered_adhoc_signals=list(adhoc_strategy_output.signals)
    g_state.signals.update_signals(regular_strategy_output,adhoc_strategy_output)
    

    # 2. Allocation Pass
    core_target_weights, sell_adhoc_orders, buy_adhoc_orders = allocator.allocate_capital(
         g_state, market_prices
    ) 
    
    
    g_state.need_trade.set_core(core_target_weights, today)
    g_state.need_trade.set_sell_adhoc(sell_adhoc_orders, today)
    g_state.need_trade.set_buy_adhoc(buy_adhoc_orders, today)
    
    # 3. Execution & Portfolio State Capture
    # (Capture the current trades before or during execution context if needed)
    executor.execute_trades(g_state, today, market_prices)
    g_state.record_daily_snapshot(today, market_prices)
    
    # 4. Commit to Audit Ledger only on actionable days to save memory, 
    # or every day depending on your debugging requirements
    if today in rebalance_dates and len(getattr(current_telemetry,'metrics',{}))!=0:
        
        audit_ledger.append(audit.AuditRecord(
            date=today,
            telemetry=current_telemetry,
            raw_regular_signals=raw_regular_signals,
            filtered_regular_signals=filtered_regular_signals,
            raw_adhoc_signals=raw_adhoc_signals,
            filtered_adhoc_signals=filtered_adhoc_signals,
            target_weights=core_target_weights,
            executed_trades=[] # Populate from executor if your executor keeps track
        ))

logger.info("Backtest execution completed successfully.")