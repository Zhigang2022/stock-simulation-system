import logging
import sys
import datetime
import re
import io
import pandas as pd

def get_log_filename(base_name='backtest_real'):
    # Get current date and time
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") # Format: 20260707_134700
    log_filename = f"{base_name}_{timestamp}.log"
    return log_filename

def setup_logger(name="backtest_logger", log_file="backtest.log"):
    """Sets up a dual-destination logger (Console + File)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 🔥 Reset handlers to avoid duplication / stale config
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(formatter)

    # File Handler
    f_handler = logging.FileHandler(log_file, mode='w')
    f_handler.setFormatter(formatter)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger
# # Initialize the logger
# logger = setup_logger()



def parse_backtest_log_by_strategy(log_input):
    # --- trade line pattern (your existing one) ---
    trade_pattern = re.compile(
        r"^.*?:\s*(\d{4}-\d{2}-\d{2}).*?(CORE_BUY|CORE_SELL)\s+(\w+)\s+\.\.\.\s+([\d.]+)\s*@\s*([\d.]+),\s*fee:\s*([\d.]+)"
    )

    # --- section header pattern ---
    section_pattern = re.compile(r"\*{7}\s*(.*?)\s*\*{7}")

    # --- read input ---
    if isinstance(log_input, str) and "\n" in log_input:
        lines = io.StringIO(log_input).readlines()
    elif isinstance(log_input, str):
        with open(log_input, "r") as f:
            lines = f.readlines()
    else:
        lines = log_input.readlines()

    results = {}
    current_strat = None
    current_rows = []

    def flush_section():
        """Convert collected rows into DataFrame and store."""
        if current_strat and current_rows:
            df = pd.DataFrame(current_rows)
            results[current_strat] = df

    # --- iterate through lines ---
    for line in lines:
        # Check if new section starts
        section_match = section_pattern.search(line)
        if section_match:
            # save previous section
            flush_section()

            # start new section
            current_strat = section_match.group(1)
            print(f"Parsing strategy: {current_strat}")
            current_rows = []
            continue

        # Parse trade lines
        match = trade_pattern.search(line)
        if match:
            trade_date, tx_type, ticker, shares, price, fee = match.groups()
            current_rows.append(
                {
                    "date": pd.to_datetime(trade_date),
                    "type": tx_type,
                    "ticker": ticker,
                    "shares": float(shares),
                    "price": float(price),
                    "fee": float(fee),
                }
            )

    # flush last section
    flush_section()

    return results