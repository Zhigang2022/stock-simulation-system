import numpy as np
import pandas as pd

def get_hurst_exponent(price_series, min_lag=10, max_lag=100):
    """
    Calculates the Hurst Exponent (H) of a stock price series using 
    Rescaled Range (R/S) analysis.
    
    Parameters:
    -----------
    price_series : array-like
        The historical price data (e.g., Close prices).
    min_lag : int
        The minimum sub-window size (default 10).
    max_lag : int
        The maximum sub-window size (default 100). Must be less than len(price_series).
        
    Returns:
    --------
    float : The Hurst Exponent (H) bounded theoretically between 0 and 1.
    """
    # 1. Convert prices to Log Returns to ensure stationarity
    prices = np.asarray(price_series)
    log_returns = np.diff(np.log(prices))
    N = len(log_returns)
    
    # Ensure we have enough data points
    if N < max_lag:
        max_lag = N // 2
        if max_lag < min_lag:
            raise ValueError("Price series is too short to calculate Hurst Exponent.")

    # 2. Create an array of sub-window sizes (tau / lags)
    # We use a logarithmic scale or step spacing to evaluate different time scales
    lags = np.unique(np.logspace(np.log10(min_lag), np.log10(max_lag), num=20, dtype=int))
    
    rs_values = []
    valid_lags = []
    
    # 3. Calculate R/S for each sub-window size (lag)
    for lag in lags:
        # Number of non-overlapping sub-windows we can fit
        num_windows = N // lag
        if num_windows == 0:
            continue
            
        rs_sub_windows = []
        
        for i in range(num_windows):
            # Extract the specific chunk of log returns
            start_idx = i * lag
            end_idx = start_idx + lag
            chunk = log_returns[start_idx:end_idx]
            
            # Mean center the data
            mean_centered = chunk - np.mean(chunk)
            
            # Cumulative deviations from the mean
            cum_deviations = np.cumsum(mean_centered)
            
            # Calculate Range (R)
            R = np.max(cum_deviations) - np.min(cum_deviations)
            
            # Calculate Standard Deviation (S)
            S = np.std(chunk)
            
            # Avoid division by zero if volatility is flat
            if S > 0:
                rs_sub_windows.append(R / S)
        
        # Average R/S across all chunks of this specific lag size
        if rs_sub_windows:
            rs_values.append(np.mean(rs_sub_windows))
            valid_lags.append(lag)
            
    # 4. Linear regression on log-log scale to find the slope (Hurst Exponent)
    poly = np.polyfit(np.log(valid_lags), np.log(rs_values), 1)
    
    return poly[0] # The slope of the line is H


import numpy as np

def get_dfa_hurst(price_series, min_lag=10, max_lag=100):
    """
    Calculates the Hurst Exponent using Detrended Fluctuation Analysis (DFA).
    
    Parameters:
    -----------
    price_series : array-like
        The historical price data (e.g., Close prices).
    min_lag : int
        The minimum sub-window size (default 10).
    max_lag : int
        The maximum sub-window size (default 100).
        
    Returns:
    --------
    float : The DFA-calculated Hurst Exponent (H).
            H < 0.5: Mean-reverting
            H = 0.5: Random Walk
            H > 0.5: Trending
    """
    prices = np.asarray(price_series)
    # 1. Convert to log returns and center around the mean
    log_returns = np.diff(np.log(prices))
    mean_normalized = log_returns - np.mean(log_returns)
    
    # 2. Integrate the time series (Cumulative Sum)
    Y = np.cumsum(mean_normalized)
    N = len(Y)
    
    # Generate scale sizes (lags) evenly spaced on a log scale
    lags = np.unique(np.logspace(np.log10(min_lag), np.log10(max_lag), num=20, dtype=int))
    
    fluctuations = []
    valid_lags = []
    
    # 3. Calculate fluctuations for each window size
    for lag in lags:
        num_windows = N // lag
        if num_windows == 0:
            continue
            
        window_variances = []
        
        for i in range(num_windows):
            start_idx = i * lag
            end_idx = start_idx + lag
            
            # Slice the integrated segment
            y_segment = Y[start_idx:end_idx]
            x_segment = np.arange(lag)
            
            # Detrending step: Fit a local linear trend line (y = mx + c)
            poly_coeffs = np.polyfit(x_segment, y_segment, 1)
            trend = np.polyval(poly_coeffs, x_segment)
            
            # Calculate the variance of the residuals (distance from the trend line)
            rms = np.mean((y_segment - trend) ** 2)
            window_variances.append(rms)
            
        if window_variances:
            # Fluctuation F(n) is the root-mean-square of all window variances
            F_n = np.sqrt(np.mean(window_variances))
            fluctuations.append(F_n)
            valid_lags.append(lag)
            
    # 4. Fit a line to the log-log plot
    poly = np.polyfit(np.log(valid_lags), np.log(fluctuations), 1)
    
    # The slope is alpha, which directly maps to the Hurst Exponent for returns
    return poly[0]