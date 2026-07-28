import pandas as pd
import numpy as np

def calculate_cagr(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the Compound Annual Growth Rate (CAGR) for each ticker column 
    in a DataFrame, determining the time horizon directly from the DatetimeIndex.

    The CAGR is calculated using the formula:
    CAGR = (Ending_Value / Beginning_Value) ** (1 / Number_of_Years) - 1

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame where the index consists of pandas Timestamps (sorted or unsorted) 
        and the columns represent different stock tickers containing historical prices.

    Returns
    -------
    pd.Series
        A series containing the CAGR for each stock ticker, indexed by the ticker name.

    Raises
    ------
    ValueError
        If the DataFrame index is empty or cannot be treated as datetime objects.
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty.")
    
    # Ensure the index is a sorted DatetimeIndex to capture exact start and end points
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    df_sorted = df.sort_index()

    # Calculate the precise time delta in years
    start_date = df_sorted.index[0]
    end_date = df_sorted.index[-1]
    
    # Using 365.25 days per year to accurately account for leap years
    num_years = (end_date - start_date).days / 365.25
    
    if num_years <= 0:
        raise ValueError("The time horizon must be greater than zero days to calculate CAGR.")

    # Extract the first and last non-null rows for the actual values
    # (ffill/bfill ensures robustness if tickers have misaligned missing data at the edges)
    beginning_values = df_sorted.bfill().iloc[0]
    ending_values = df_sorted.ffill().iloc[-1]
    
    # Vectorized CAGR calculation across all columns
    cagr_series = (ending_values / beginning_values) ** (1 / num_years) - 1
    cagr_series.name='CAGR'
    return cagr_series


def calculate_volatility_ratios(df: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Calculate the Annualized Sharpe Ratio and Sortino Ratio for each ticker column.
    
    The annualization factor is extracted directly from the median time delta 
    of the DatetimeIndex, making it completely agnostic to the underlying data 
    frequency (daily, weekly, monthly, or irregular).

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame of historical stock prices with a DatetimeIndex.
    risk_free_rate : float, default 0.0
        The annualized risk-free rate (e.g., 0.04 for 4%).

    Returns
    -------
    pd.DataFrame
        A DataFrame with 'Sharpe_Ratio' and 'Sortino_Ratio' as columns, 
        indexed by ticker names.
    """
    if df.empty or len(df) <= 1:
        raise ValueError("Insufficient data to calculate volatility ratios.")
        
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    df_sorted = df.sort_index()
    
    # 1. Calculate the dynamic annualization factor from the index
    # Find the median time gap between consecutive observations
    time_deltas = df_sorted.index.to_series().diff().dropna()
    median_delta_days = time_deltas.dt.total_seconds() / (24 * 3600)  # Convert to days float
    
    if median_delta_days.median() <= 0:
        raise ValueError("The time steps between index rows must be positive.")
        
    # Annualization factor = Total days in a year / days per observation step
    ann_factor = 365.25 / median_delta_days.median()
    
    # 2. Calculate returns per period
    returns = df_sorted.pct_change().dropna(how='all')
    
    # 3. Convert annualized risk-free rate to the period rate matching the index frequency
    period_rf = (1 + risk_free_rate) ** (1 / ann_factor) - 1
    
    # 4. Calculate excess returns per period
    excess_returns = returns.sub(period_rf, axis=0)
    mean_excess = excess_returns.mean()
    
    # 5. Volatility calculations (Standard vs Downside)
    total_vol = returns.std()
    
    # Downside deviation: isolate negative returns, treat positive returns as 0
    downside_returns = returns.clip(upper=0)
    downside_vol = downside_returns.std()
    
    # 6. Annualize the performance statistics using our dynamically derived factor
    ann_excess_return = mean_excess * ann_factor
    ann_total_vol = total_vol * np.sqrt(ann_factor)
    ann_downside_vol = downside_vol * np.sqrt(ann_factor)
    
    # 7. Compute final ratios
    sharpe = np.where(ann_total_vol > 0, ann_excess_return / ann_total_vol, np.nan)
    sortino = np.where(ann_downside_vol > 0, ann_excess_return / ann_downside_vol, np.nan)
    
    return pd.DataFrame({
        'Sharpe_Ratio': sharpe,
        'Sortino_Ratio': sortino
    }, index=df.columns)


def calculate_drawdown_and_calmar(df: pd.DataFrame, cagr_series: pd.Series) -> pd.DataFrame:
    """
    Calculate the Maximum Drawdown and Calmar Ratio for each ticker column.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame of historical stock prices with a DatetimeIndex.
    cagr_series : pd.Series
        The pre-calculated CAGR series for the exact same asset universe.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing 'Max_Drawdown' (as a positive decimal) and 
        'Calmar_Ratio' columns, indexed by ticker names.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    df_sorted = df.sort_index()
    
    # 1. Track the running peak price for each column up to each point in time
    running_peaks = df_sorted.cummax()
    
    # 2. Calculate drawdowns as a percentage drop from that running peak
    drawdown_series = (df_sorted - running_peaks) / running_peaks
    
    # 3. Maximum Drawdown is the minimum (most negative) value found
    # We invert it to express it as a positive percentage/decimal value for industry standard convention
    max_drawdown = -drawdown_series.min()
    
    # 4. Compute Calmar Ratio (CAGR / Max Drawdown)
    # Handle edge case where max drawdown is 0 to avoid zero-division errors
    calmar = np.where(max_drawdown > 0, cagr_series / max_drawdown, np.nan)
    
    return pd.DataFrame({
        'Max_Drawdown': max_drawdown,
        'Calmar_Ratio': calmar
    }, index=df.columns)


def calculate_metrics(df_nav):
    series_cagr=calculate_cagr(df_nav)
    df_vol=calculate_volatility_ratios(df_nav)
    df_dd=calculate_drawdown_and_calmar(df_nav,series_cagr)
    return pd.concat([series_cagr,df_vol,df_dd],axis=1)