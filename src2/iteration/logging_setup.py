"""
Console + file logger for trade_implement.py's "backtest_logger" (and
daily_iterator.py's optional rebalance-event `logger`). Forked from
src_old/logging.py's setup_logger -- same name/format, no src_old
dependency, since it's just a few lines and this package forks rather
than imports across that boundary (see __init__.py's docstring).
"""
import logging
import sys


def setup_logger(name: str = "backtest_logger", log_file: str = "backtest.log", console: bool = False) -> logging.Logger:
    """
    Sets up a file logger (plus console, unless console=False), overwriting
    log_file on each call. Pass the returned logger as run_daily_iteration's
    `logger` kwarg to also log CORE/TACTICAL rebalance events into the same
    file as trade_implement.py's BUY/SELL fills (which already log under
    this same "backtest_logger" name).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w") #model="a" append
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
