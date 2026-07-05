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

from src import signal_schema

from src import logging 

logger=logging.setup_logger(name="backtest_logger", log_file="backtest.log")

tickers=['ENPH',
 'HON',
 'ORLY',
 'XEL',
 'GILD',
 'MCHP',
 'MRVL',
 'CSGP',
 'VRTX',
 'ROST',
 'AMD',
 'SMCI',
 'PAYX',
 'AMGN',
 'MSFT',
 'SPLK',
 'SPLK',
 'TSLA',
 'TTD',
 'ADSK']
broker = data_broker.DataBroker(tickers=tickers, start_date="2022-01-01", end_date="2026-04-01")
universe_data = broker.fetch_universe_data()

data_for_boot=universe_data
NUM_SIMULATIONS = 10


# strategies
risk_adj_moment=strategy_selector.RiskAdjustedMomentum()
info_moment=strategy_selector.InformationDiscreteMomentum()
adhoc_exit=ad_hoc_strategy.AdHocChandelierExit()

portfolio_filter = compliance_filters.LiquidityComplianceFilter()
# Anchor prices to seed the start of our synthetic worlds
anchor_prices = data_for_boot['price'].iloc[0].to_dict()
engine = statioinary_bootstrap.VectorizedBootstrapEngine(data_for_boot, expected_block_size=21)
# Instantly generates a 3D NumPy Tensor: Shape (1000, Num_Days, 4)
synthetic_data_cube = engine.generate_all_worlds(
    num_simulations=NUM_SIMULATIONS, 
    start_prices=anchor_prices
)

# Unpack tensors from the engine output
prices_3d = synthetic_data_cube['price_tensor']
volume_3d = synthetic_data_cube['volume_tensor']

# Run the Macro Orchestrator validation pass
for sim_id in tqdm(range(NUM_SIMULATIONS)):
    # Pack the 3D slices into clean 2D DataFrames with the ORIGINAL calendar dates index
    world_data_dict = {
        'price': pd.DataFrame(prices_3d[sim_id, :, :], index=engine.dates_index, columns=engine.tickers),
        'volume': pd.DataFrame(volume_3d[sim_id, :, :], index=engine.dates_index, columns=engine.tickers)
    }
    break