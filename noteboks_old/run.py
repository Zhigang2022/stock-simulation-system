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



