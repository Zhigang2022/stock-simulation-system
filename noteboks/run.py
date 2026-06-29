import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import performance_evaluator_1
import numpy as np

def run(universe_data,rebalance_dates, calendar,strategy,portfolio_filter):

    evaluator = performance_evaluator_1.AdvancedEvaluator(initial_capital=100000.0, execution_lag=1)


    active_rebalance_date = rebalance_dates[0]
    # Loop through every single daily date index in your data matrix
    for today in universe_data["price"].index:
        
        # 1. Update and track the portfolio's actual value today
        evaluator.log_daily_value(today, universe_data["price"], active_rebalance_date)
        
        # 2. Check if today is a scheduled rebalance day
        if today in rebalance_dates:
            snapshot = calendar.get_historical_snapshot(today)
            scores = strategy.calculate_scores(snapshot)

            target_weights = portfolio_filter.generate_weights(scores, snapshot )
            # print(target_weights)
            
            # Fire rebalance execution (which safely occurs at T + execution_lag)
            evaluator.execute_rebalance(today, target_weights, universe_data["price"])
            active_rebalance_date = today
    return evaluator



def get_df_nav(evaluator,forward_months=6):
    evaluator.daily_nav.name='nav'
    evaluator.daily_nav.index=evaluator.daily_nav.index.rename('date')
    df_nav=pd.DataFrame(evaluator.daily_nav)

    report = evaluator.generate_advanced_report(forward_months=forward_months)
    return df_nav,report



def plot_nav_charts(df_nav):
    # Set style for a clean, professional look
    sns.set_theme(style="whitegrid")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Plot 1: Net Asset Value ---
    ax1.plot(df_nav.index, df_nav['nav'], color='#1f77b4', lw=2, label='Strategy NAV')
    
    # Mark the Out-of-Sample (Forward Performance) Boundary
    cutoff_date = df_nav.index[-1] - pd.DateOffset(months=6) # matching your N_months
    ax1.axvline(x=cutoff_date, color='#d62728', linestyle='--', lw=2, 
                label=f'last 6 month ({cutoff_date.strftime("%Y-%m-%d")})')
                #label=f'Forward Window Start ({cutoff_date.strftime("%Y-%m-%d")})')
    
    # Visual shading for the Forward Performance Window
    ax1.axvspan(cutoff_date, df_nav.index[-1], color='#d62728', alpha=0.05)
    
    ax1.set_title('Advanced Portfolio Performance & Risk Analysis', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    
    # --- Plot 2: Underwater Drawdown ---
    rolling_max = df_nav['nav'].cummax()
    drawdown = (df_nav['nav'] - rolling_max) / rolling_max
    
    ax2.fill_between(drawdown.index, drawdown * 100, 0, color='#e377c2', alpha=0.3)
    ax2.plot(drawdown.index, drawdown * 100, color='#d62728', lw=1)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    
    plt.tight_layout()
    plt.show()


def display_report(report):
    metrics_data = {
        "Metric Profile": ["Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown"],
        "In-Sample (Historical)": [f"{report['In-Sample']['IS_Annualized_Return']}%", 
                                    f"{report['In-Sample']['IS_Annualized_Vol']}%", 
                                    report['In-Sample']['IS_Sharpe_Ratio'], 
                                    report['In-Sample']['IS_Sortino_Ratio'], 
                                    f"{report['In-Sample']['IS_Max_Drawdown']}%"],
        "Forward Performance (OOS)": [f"{report['Forward_OOS']['OOS_Annualized_Return']}%", 
                                       f"{report['Forward_OOS']['OOS_Annualized_Vol']}%", 
                                       report['Forward_OOS']['OOS_Sharpe_Ratio'], 
                                       report['Forward_OOS']['OOS_Sortino_Ratio'], 
                                       f"{report['Forward_OOS']['OOS_Max_Drawdown']}%"]
    }
    
    df_metrics = pd.DataFrame(metrics_data).set_index("Metric Profile")
    display(df_metrics.style.set_properties(**{'text-align': 'center'})
                            .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}]))
    

import pandas as pd

class SimulationCollector:
    def __init__(self):
        """Initializes empty lists for all metrics."""
        self.metrics = {
            'is_return': [], 'oos_return': [],
            'is_vol': [],    'oos_vol': [],
            'is_sharpe': [], 'oos_sharpe': [],
            'is_sortino': [], 'oos_sortino': [],
            'is_drawdown': [], 'oos_drawdown': []
        }
        
    def collect(self, sim_dict):
        """
        Extracts metrics directly from the simulation's nested dictionary output.
        """
        # Extract In-Sample (IS) metrics
        is_data = sim_dict['In-Sample']
        self.metrics['is_return'].append(is_data['IS_Annualized_Return'])
        self.metrics['is_vol'].append(is_data['IS_Annualized_Vol'])
        self.metrics['is_sharpe'].append(is_data['IS_Sharpe_Ratio'])
        self.metrics['is_sortino'].append(is_data['IS_Sortino_Ratio'])
        self.metrics['is_drawdown'].append(is_data['IS_Max_Drawdown'])
        
        # Extract Out-of-Sample (OOS) metrics
        oos_data = sim_dict['Forward_OOS']
        self.metrics['oos_return'].append(oos_data['OOS_Annualized_Return'])
        self.metrics['oos_vol'].append(oos_data['OOS_Annualized_Vol'])
        self.metrics['oos_sharpe'].append(oos_data['OOS_Sharpe_Ratio'])
        self.metrics['oos_sortino'].append(oos_data['OOS_Sortino_Ratio'])
        self.metrics['oos_drawdown'].append(oos_data['OOS_Max_Drawdown'])

    def to_dataframe(self):
        """Converts the collected lists into a final consolidated DataFrame."""
        return pd.DataFrame(self.metrics)
    

import numpy as np
import scipy.stats as stats

def analyze_metric_difference(list_a, list_b, label_a="Long-Term (IS)", label_b="Mid-Term (OOS)"):
    """
    Performs a Welch's T-test to determine if the means of two simulation 
    results are statistically significantly different.
    """
    # Convert inputs to numpy arrays just in case
    arr_a = np.array(list_a)
    arr_b = np.array(list_b)
    
    # Calculate basic descriptive stats
    mean_a, mean_b = np.mean(arr_a), np.mean(arr_b)
    std_a, std_b = np.std(arr_a, ddof=1), np.std(arr_b, ddof=1)
    
    # Run Welch's T-test (equal_var=False handles different volatilities between time horizons)
    t_stat, p_value = stats.ttest_ind(arr_a, arr_b, equal_var=False)
    
    # Format and print the report
    print("=" * 50)
    print("        STATISTICAL SIGNIFICANCE REPORT        ")
    print("=" * 50)
    print(f"{label_a:<20} | Mean: {mean_a:6.2f}% | StdDev: {std_a:6.2f}%")
    print(f"{label_b:<20} | Mean: {mean_b:6.2f}% | StdDev: {std_b:6.2f}%")
    print("-" * 50)
    print(f"Absolute Difference: {abs(mean_a - mean_b):.2f}%")
    print(f"T-Statistic:        {t_stat:.4f}")
    print(f"P-Value:            {p_value:.4e}")
    print("-" * 50)
    
    # Interpret the P-value (using the standard 5% significance threshold)
    alpha = 0.05
    if p_value < alpha:
        print("Conclusion: STATISTICALLY SIGNIFICANT")
        print(f"-> The difference is real. The strategy performed significantly\n"
              f"   differently in the {label_b} period than the {label_a} period.")
    else:
        print("Conclusion: NOT STATISTICALLY SIGNIFICANT")
        print(f"-> The $2.5% gap could easily be a byproduct of random sampling noise.\n"
              f"   We cannot confidently say the performance changed.")
    print("=" * 50)
    
    # Return results as a dictionary for further programmatic use if needed
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha
    }




def compute_performance_metrics(dollar_return: pd.DataFrame, annual_rf: float = 0.04) -> pd.DataFrame:
    """
    Computes a comprehensive suite of financial performance metrics for multiple 
    strategies based on their absolute dollar NAV curves.

    Metrics include: Annualized CAGR, Annualized Sharpe Ratio, Annualized Sortino Ratio,
    Maximum Drawdown (MDD), and the Calmar Ratio.

    Parameters:
    -----------
    dollar_return : pd.DataFrame
        DataFrame where columns are strategy names/tickers and the index is dates.
        Values must be absolute dollar amounts (NAV).
    annual_rf : float, default 0.04
        The annualized risk-free rate (e.g., 0.04 for 4%).

    Returns:
    --------
    pd.DataFrame
        A performance matrix sorted by Sharpe Ratio in descending order.
    """
    # 1. Calculate Daily Returns for Sharpe and Sortino
    df_returns = dollar_return.pct_change().dropna()
    
    # De-annualize risk-free rate to a daily baseline
    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    
    metrics_summary = {}

    for strategy in dollar_return.columns:
        nav_series = dollar_return[strategy].dropna()
        return_series = df_returns[strategy]
        
        if len(nav_series) < 2:
            continue
            
        # --- A. Calculate CAGR ---
        starting_value = nav_series.iloc[0]
        ending_value = nav_series.iloc[-1]
        
        # Use trading days to determine years elapsed to keep timeline consistent
        years_elapsed = len(nav_series) / 252.0
        cagr = (ending_value / starting_value) ** (1 / years_elapsed) - 1 if starting_value > 0 else 0
        
        # --- B. Calculate Annualized Sharpe Ratio ---
        excess_returns = return_series - daily_rf
        mean_excess = excess_returns.mean()
        std_dev = return_series.std()
        
        sharpe = (mean_excess / std_dev) * np.sqrt(252) if std_dev > 0 else 0
        
        # --- C. Calculate Annualized Sortino Ratio ---
        # Isolate downside returns (only returns below 0)
        downside_returns = return_series[return_series < 0]
        downside_std = downside_returns.std(ddof=0) # Population std for downside deviation
        
        sortino = (mean_excess / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # --- D. Calculate Maximum Drawdown ---
        # Track the expanding historical peak value of the portfolio
        rolling_max = nav_series.cummax()
        # Calculate drawdown percentage from peak to current value
        drawdowns = (nav_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() # The largest negative number represents the max drop
        
        # --- E. Calculate Calmar Ratio ---
        # Avoid division-by-zero if Max Drawdown is perfectly 0%
        calmar = (cagr / abs(max_drawdown)) if max_drawdown < 0 else 0
        
        # Save results to dictionary
        metrics_summary[strategy] = {
            'CAGR (%)': cagr * 100,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown (%)': max_drawdown * 100,
            'Calmar Ratio': calmar
        }
        
    # Convert dictionary payload to a beautiful presentation DataFrame
    df_metrics = pd.DataFrame(metrics_summary).transpose()
    
    # Sort by your primary diagnostic metric (Sharpe)
    df_metrics = df_metrics.sort_values(by='Sharpe Ratio', ascending=False)
    
    return df_metrics


def calculate_cagr(df_navs: pd.DataFrame, use_trading_days: bool = False) -> pd.Series:
    """
    Calculates the Compound Annual Growth Rate (CAGR) for multiple investment 
    strategies based on their absolute dollar Net Asset Value (NAV) curves.

    The formula applied is:
        CAGR = (Ending_Value / Starting_Value) ** (1 / Years_Elapsed) - 1

    Parameters:
    -----------
    df_navs : pd.DataFrame
        A DataFrame where columns represent different trading strategies/portfolios
        and rows represent time steps containing absolute dollar values (NAV).
        The index should ideally be a pandas DatetimeIndex.
    use_trading_days : bool, default False
        If True, calculates years elapsed assuming 252 trading days per year 
        (Total Rows / 252). If False, uses the exact calendar days elapsed 
        between the first and last date timestamp divided by 365.25.

    Returns:
    --------
    pd.Series
        A pandas Series containing the annualized compounding rate (as a decimal)
        for each strategy column in the input DataFrame.

    Raises:
    -------
    ValueError
        If the DataFrame index cannot be converted to timestamps when 
        use_trading_days=False, or if a strategy contains a starting NAV <= 0.
        
    Example:
    --------
    >>> df_navs = pd.DataFrame({'Strategy_A': [10000, 12000, 15000]}, 
                               index=pd.to_datetime(['2024-01-01', '2025-01-01', '2026-01-01']))
    >>> calculate_cagr(df_navs)
    Strategy_A    0.224745
    Name: CAGR, dtype: float64
    """
    cagr_results = {}
    
    # Force index copy to ensure we don't accidentally modify the original dataframe
    df = df_navs.copy()
    
    if not use_trading_days:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(
                "DataFrame index must be convertible to Datetime format to use calendar days. "
                "Alternatively, set `use_trading_days=True`."
            ) from e

    for strategy in df.columns:
        # Extract the specific strategy curve and drop missing values
        series = df[strategy].dropna()
        
        if len(series) < 2:
            cagr_results[strategy] = float('nan')
            continue
            
        starting_value = series.iloc[0]
        ending_value = series.iloc[-1]
        
        # Guard rails for invalid financial inputs
        if starting_value <= 0:
            raise ValueError(
                f"Strategy '{strategy}' has a starting NAV of {starting_value}. "
                "Starting NAV must be strictly greater than 0 to compute CAGR."
            )
        if ending_value <= 0:
            cagr_results[strategy] = -1.0  # Total loss of capital
            continue

        # Calculate time elapsed (N)
        if use_trading_days:
            years_elapsed = len(series) / 252.0
        else:
            days_elapsed = (series.index[-1] - series.index[0]).days
            years_elapsed = days_elapsed / 365.25
            
        # Avoid division by zero if time window is essentially empty
        if years_elapsed == 0:
            cagr_results[strategy] = 0.0
            continue
            
        # Compute CAGR
        cagr = (ending_value / starting_value) ** (1 / years_elapsed) - 1
        cagr_results[strategy] = cagr

    return pd.Series(cagr_results, name='CAGR')