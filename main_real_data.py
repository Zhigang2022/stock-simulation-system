from src import logging
from src import ticker_loader
from src import data_broker
from src import run

import pickle

from src.selectors import cross_selectors

class RealDataBacktester:
    """
    A class to run backtests on real historical data using multiple momentum/regression strategies.
    """
    def __init__(self, tickers=None, start_date="2022-01-01", end_date="2026-04-01", log_file="backtest_real.log"):
        self.logger = logging.setup_logger(name="backtest_logger", log_file=log_file)

        if tickers is None:
            # Default "动量亏本组合" tickers
            self.tickers = 'ALGN,AAPL,PANW,AMD,ZS,MDLZ,XEL,VRSN,INSM,ON,WDAY,ENPH,DLTR,ADI,COST,ABNB,CEG,MSFT,GOOGL,CDNS,PYPL,FAST,JD,MTCH,SNPS,LULU,CMCSA,ADBE,ADSK'.split(',')
        elif isinstance(tickers, str):
            self.tickers = tickers.split(',')
        else:
            self.tickers = tickers
            
        self.start_date = start_date
        self.end_date = end_date
        
        self.broker = None
        self.universe_data = None
        self.selector_type=None
        self.portfolio_filter = None
        self.strategies = {}
        self.states = {}



    def set_filter(self,filter):
        self.portfolio_filter =filter

    def fetch_data(self):
        """Fetches the universe data and sets up compliance filters."""
        self.broker = data_broker.DataBroker(
            tickers=self.tickers,
            start_date=self.start_date,
            end_date=self.end_date
        )
        universe_data = self.broker.fetch_universe_data()
        self.universe_data = ticker_loader.fix_data(universe_data)

        

    def fetch_bench_data(self,bench_tickers=['SPY','QQQ']):
        self.bench_tickers=bench_tickers
        self.bench_broker = data_broker.DataBroker(tickers=self.bench_tickers, start_date=self.start_date, end_date=self.end_date)
        self.bench_universe_data = self.bench_broker.fetch_universe_data()


    def set_strategy_selector(self,selector_type):
        self.selector_type=selector_type

    def build_strategies(self,scorers_dict,**kwargs):
        """Initializes the strategies to be evaluated."""
        if self.selector_type is None:
            print('need to set selector_type first')

        strategies={}
        for name in scorers_dict:
            print(f'  **create strategy for {name}')
            strategy=  self.selector_type(
            scorer=scorers_dict[name],
            **kwargs
        )
            strategies[name]=strategy

        self.strategies = strategies
        return self.strategies
        



    def run_backtest(self, allocation_type=None):
        """Runs the backtest loop across all loaded strategies."""
        if self.universe_data is None:
            self.fetch_data()
        if not self.strategies:
            print('error: no strategies')

        if self.portfolio_filter is None:
            #pportfolio_filter will be define with the total input price data for pre-calculation
            print('error: no portfolio_filter')

        self.states = {}
        self.ledger_dict={}
        for strat_name, strategy in self.strategies.items():
            self.logger.info(f'******* {strat_name} *******')
            kwargs = {}
            if allocation_type is not None:
                kwargs['allocation_type'] = allocation_type
            
            state,audit_ledger = run.run(
                self.logger,
                self.universe_data,
                self.portfolio_filter,
                strategy,
                **kwargs
            )
            self.states[strat_name] = state
            self.ledger_dict[strat_name] = audit_ledger
        return self.states

    def save_states(self, filepath='data/states.pickle'):
        """Pickles the resulting backtest states into a file."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb+') as f:
            pickle.dump(self.states, f)


from src import metrics
def evaluate(g_state,effective_date='2023-01-01'):
    df_nav=g_state.export_nav_dataframe()
    df_nav_effect=df_nav.loc[df_nav.index>=effective_date]
    print(metrics.calculate_metrics(df_nav_effect))
    return df_nav_effect

def print_state(g_state,today,market_prices):    
    g_state.display_current_state(today,market_prices)


if __name__ == "__main__":
    # Example configurations/options from original script:
    # totak_tickers = ticker_loader.load_total_tickers()
    # tickers = totak_tickers
    # tickers = ticker_loader.get_random_sample(totak_tickers, 30)
    # ### section ETFs, most are beta, not good for
    # tickers = ['SPY','QQQ','QQQE','IWM','XSW','XLY','XLV','XLU','XLRE','XLP','XLK','XLI','XLF','XLE','XLC','XLB','LABU','XBI']
    
    tickers='NEE,SO,DUK,CEG,AEP,SRE,D,ETR,XEL,VST,EXC,ED,PEG,WEC,PCG,DTE,AEE,ATO,CNP,EIX,AWK,ES,FE,CMS,LNT,EVRG,AES,NI,NRG,PNW'.split(',')
    from src.scorers import mean_reversion_scorers
    reverse_scorers={
    'mean_reverse':mean_reversion_scorers.MeanReversionScorer(),
    'bollinger_reversion':mean_reversion_scorers.BollingerReversionScorer(),
    }

    from src.filters import base_filter
    filter=base_filter.NullFilter()

    # Run with default '动量亏本组合'
    backtester = RealDataBacktester(tickers=tickers, start_date="2022-01-01", end_date="2026-04-01", log_file="backtest_real.log")

    backtester.set_filter(filter)
    backtester.build_strategies(reverse_scorers)
    states = backtester.run_backtest()
    
    # to save states:
    # backtester.save_states('data/states.pickle')


