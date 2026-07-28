"""
Real-market-data acquisition -- yfinance calls (data_broker.py) and static
universe/ticker lists (ticker_loader.py). Forked from src/, not imported
(same "small enough to fork, not worth a src/ coupling" reasoning as the
rest of src2). Not to be confused with src2/bootstrap/data_gen.py, which
generates SYNTHETIC data by resampling whatever this package fetched.
"""
from .data_broker import DataBroker
from .ticker_loader import (
    get_random_sample,
    load_total_tickers_nasdaq_2022,
    load_xlu_tickers,
    load_xlp_tickers,
    load_xlre_tickers,
    load_overall_market,
    fix_data,
)

__all__ = [
    "DataBroker",
    "get_random_sample",
    "load_total_tickers_nasdaq_2022",
    "load_xlu_tickers",
    "load_xlp_tickers",
    "load_xlre_tickers",
    "load_overall_market",
    "fix_data",
]
