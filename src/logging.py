import logging
import sys

def setup_logger(name="backtest_logger", log_file="backtest.log"):
    """Sets up a dual-destination logger (Console + File)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s]: %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)
        
        # File Handler
        f_handler = logging.FileHandler(log_file, mode='w')  # 'w' overwrites each run
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
        
    return logger

# # Initialize the logger
# logger = setup_logger()