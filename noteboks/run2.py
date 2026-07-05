from src import data_broker
from src import statioinary_bootstrap
from src import global_state

from src import calendar_iterator
from src import strategy_selector, ad_hoc_strategy

from src import executor_module, performance_evaluator
from src import compliance_filters, budget_allocator

from src import signal_schema



def run(world_data_dict,module_c_regular,module_c1_adhoc,portfolio_filter,logger):
    allocator=budget_allocator.IntegratedBudgetAllocator(top_percent=.10, allocation_type='equal')
    g_state=global_state.GlobalState(initial_capital=100_000,core_allocation_pct=1.0)
    executor=executor_module.TransactionExecutor()
    calendar = calendar_iterator.CalendarIterator(world_data_dict, interval="ME")
    rebalance_dates = calendar.generate_rebalance_dates()
    
    logger.info("Initializing Backtest Engine...")
    logger.info(f"Initial Capital: {g_state.total_cash} | Core Allocation Pct: {g_state.core_cash/g_state.total_cash}")

    # only consider about the ME rebalance
    for i,today in enumerate(rebalance_dates): #world_data_dict['price'].index):
        snapshot=calendar.get_historical_snapshot(today)
        market_prices=snapshot['price'].loc[today]
         
        # 1. Collect arbitrary signal outputs from your strategy classes
        regular_output,adhoc_output=signal_schema.StrategyOutput(),signal_schema.StrategyOutput()
        
        if today in rebalance_dates and not (module_c_regular is None):
            logger.info(f"=== REBALANCE EVENT TRIGGERED: {today.strftime('%Y-%m-%d')} ===")
            
            regular_output=module_c_regular.calculate_signals(snapshot)
            # if regular_output.signals:
            #     break
            logger.info(f"Generated {len(regular_output.signals)} raw regular signals.")
            regular_output=portfolio_filter.filter_signals(today,regular_output,snapshot)
            logger.info(f"Post-filter regular signals remaining: {len(regular_output.signals)}")
            
    
        if not module_c1_adhoc is None:
            adhoc_output=module_c1_adhoc.evaluate_exits(g_state, snapshot)
            logger.info(f"[{today.strftime('%Y-%m-%d')}] Ad-hoc module generated {len(adhoc_output.signals)} raw exit signals.")
            adhoc_output=portfolio_filter.filter_signals(today,adhoc_output,snapshot)
            logger.info(f"[{today.strftime('%Y-%m-%d')}] Post-filter ad-hoc signals remaining: {len(adhoc_output.signals)}")
    
    
        
        regular_weights, approved_ad_hoc=allocator.allocate_capital(regular_output.signals,adhoc_output.signals,g_state,market_prices) 
        # Log allocation outputs if action happened
        if regular_weights:
            logger.info(f"[{today.strftime('%Y-%m-%d')}] Allocation targets finalized: {regular_weights}")
        if approved_ad_hoc:
            logger.info(f"[{today.strftime('%Y-%m-%d')}] Ad-hoc orders approved: {[s.ticker for s in approved_ad_hoc]}")
        
        executor.execute_trades(g_state,regular_weights,approved_ad_hoc,market_prices)
        g_state.record_daily_snapshot(today,market_prices)
        
        logger.info(g_state.log_current_state(today,market_prices))
        logger.info('\n---\n')
    logger.info("Backtest execution completed successfully.")    
    return g_state


def get_bootstrap_words(data_for_boot,num_simulations,expected_block_size=21):
    # Anchor prices to seed the start of our synthetic worlds
    anchor_prices = data_for_boot['price'].iloc[0].to_dict()

    engine = statioinary_bootstrap.VectorizedBootstrapEngine(data_for_boot, expected_block_size=21)
    # Instantly generates a 3D NumPy Tensor: Shape (1000, Num_Days, 4)
    synthetic_data_cube = engine.generate_all_worlds(start_prices=anchor_prices)
    # Unpack tensors from the engine output
    prices_3d = synthetic_data_cube['price_tensor']
    volume_3d = synthetic_data_cube['volume_tensor']
    return engine,prices_3d,volume_3d


import random

def get_random_sample(original_list: list, sample_size: int) -> list:
    """
    Returns a sample of strictly unique elements from the original list,
    ensuring the returned list length equals sample_size.
    """
    # 1. Deduplicate the original list using a set, then convert back to a list
    unique_base_list = list(set(original_list))
    
    # 2. Check the guard rail against the *unique* count, not the original count
    if sample_size > len(unique_base_list):
        raise ValueError(
            f"Sample size ({sample_size}) cannot be larger than the number of "
            f"unique elements in the original list ({len(unique_base_list)})."
        )
        
    # 3. random.sample guarantees unique selection and exact length
    return random.sample(unique_base_list, sample_size)

def load_total_tickers():
    return 'ATVI,SNPS,CPRT,SBUX,SPLK,AMZN,AVGO,NFLX,KLAC,ABNB,JD,ILMN,INTU,FTNT,AMAT,ENPH,KHC,CSGP,AMD,DOCU,ANSS,ROST,MNST,TMUS,TRI,PTON,MDB,ADSK,MU,CDNS,LRCX,BKNG,CEG,NVDA,META,PEP,EXC,FAST,VRSN,INSM,ZM,COST,CTSH,ON,TEAM,MAR,NTES,INTC,PANW,VRTX,FI,PDD,AZN,MRVL,CTAS,GOOG,XEL,CSX,TXN,GOOGL,ORLY,VRSK,ASML,ZS,DDOG,TTD,ALGN,TSLA,AMGN,ADP,SMCI,NXPI,CSCO,BIDU,MELI,MTCH,ADBE,HON,MDLZ,EBAY,CRWD,LULU,PYPL,WDAY,REGN,DXCM,MCHP,AAPL,GILD,CHTR,SWKS,CMCSA,OKTA,LCID,SIRI,WBA,KDP,IDXX,XLNX,CDW,QCOM,AEP,PCAR,DLTR,ISRG,EA,MRNA,BIIB,PAYX,MSFT,SGEN,ADI'.split(',')