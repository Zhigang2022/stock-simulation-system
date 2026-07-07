import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import re


def plot_rebased_performance(df_price):
    """
    Plot rebased performance (starting at 100) for multiple tickers.

    Parameters
    ----------
    df_price : pd.DataFrame
        Index: datetime
        Columns: tickers
        Values: price or NAV

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """

    # 1. Ensure datetime index & sort
    df = df_price.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 2. Normalize to 100 at start
    df_rebased = df / df.iloc[0] * 100

    # 3. Create figure
    fig = go.Figure()

    # 4. Add each ticker
    for col in df_rebased.columns:
        fig.add_trace(
            go.Scatter(
                x=df_rebased.index,
                y=df_rebased[col],
                mode="lines",
                name=col,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Date: %{x|%b %d, %Y}<br>"
                    "Value: %{y:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    # 5. Layout
    fig.update_layout(
        title="Rebased Performance (Start = 100)",
        xaxis_title="Date",
        yaxis_title="Index (Base = 100)",
        hovermode="closest",  # 👈 avoids your earlier confusion issue
        template="plotly_white",
        # legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        ### legend on left
        # legend=dict(
        # orientation="v",
        # y=1,
        # yanchor="top",
        # x=-0.02,         # slightly outside plot
        ### xanchor="right"
        # )
        # legend on right
        legend=dict(
            orientation="v",
            y=1,
            yanchor="top",
            x=1.02,          # slightly outside plot
            xanchor="left"
            )
    )
    fig.update_layout(
    
    )

    return fig

def plot_ticker_attribution(df_slice, ticker_symbol="CMCSA"):
    """Filters a dataframe slice for a specific ticker and plots its price,

    buy/sell execution markers, and daily dollar PnL on a dual y-axis.
    """
    # 1. Filter and sort data for the specific ticker
    df_ticker = df_slice[df_slice["ticker"] == ticker_symbol].copy()

    if df_ticker.empty:
        print(f"No data found for ticker: {ticker_symbol}")
        return None

    df_ticker["date"] = pd.to_datetime(df_ticker["date"])
    df_ticker = df_ticker.sort_values("date")

    # 2. Separate rebalance days into buys and sells based on share changes
    buys = df_ticker[df_ticker["is_rebalance_day"] & (df_ticker["shares_change"] > 0)]
    sells = df_ticker[df_ticker["is_rebalance_day"] & (df_ticker["shares_change"] < 0)]

    # 3. Create a dual y-axis subplot canvas
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- Primary Y-Axis: Price & Execution Markers ---
    # Main Price Line
    fig.add_trace(
        go.Scatter(
            x=df_ticker["date"],
            y=df_ticker["current_price"],
            name=f"{ticker_symbol} Price",
            line=dict(color="#1f77b4", width=2),
        ),
        secondary_y=False,
    )

    # Buy Markers (Green Up-Triangles)
    fig.add_trace(
        go.Scatter(
            x=buys["date"],
            y=buys["current_price"],
            mode="markers",
            name="Buy",
            marker=dict(
                symbol="triangle-up",
                size=11,
                color="#2ca02c",
                line=dict(width=1, color="white"),
            ),
            hovertemplate="<b>Buy Trade</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>Change: +%{customdata:,} shares",
            customdata=buys["shares_change"],
        ),
        secondary_y=False,
    )

    # Sell Markers (Red Down-Triangles)
    fig.add_trace(
        go.Scatter(
            x=sells["date"],
            y=sells["current_price"],
            mode="markers",
            name="Sell",
            marker=dict(
                symbol="triangle-down",
                size=11,
                color="#d62728",
                line=dict(width=1, color="white"),
            ),
            hovertemplate="<b>Sell Trade</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>Change: %{customdata:,} shares",
            customdata=sells["shares_change"],
        ),
        secondary_y=False,
    )

    # --- Secondary Y-Axis: Daily PnL ($) ---
    # Daily PnL Bar Chart (Dynamic green/red coloring)
    pnl_colors = df_ticker["pnl_dollar"].apply(
        lambda x: "#2ca02c" if x >= 0 else "#d62728"
    )

    fig.add_trace(
        go.Bar(
            x=df_ticker["date"],
            y=df_ticker["pnl_dollar"],
            name="Daily PnL ($)",
            marker_color=pnl_colors,
            opacity=0.35,
            hovertemplate="Date: %{x}<br>Daily PnL: $%{y:,.2f}",
        ),
        secondary_y=True,
    )

    # 4. Refine Layout & Axis Formatting
    fig.update_layout(
        title=dict(
            text=f"{ticker_symbol} Performance Attribution & Execution History",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis_title="Date",
        hovermode="closest",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )

    fig.update_yaxes(title_text="<b>Stock Price ($)</b>", secondary_y=False)
    fig.update_yaxes(
        title_text="<b>Daily PnL ($)</b>",
        secondary_y=True,
        showgrid=False,  # Keep grid clean by hiding secondary gridlines
    )

    return fig


def parse_backtest_log(log_input):
    # Regex Breakdown:
    # ^.*?:\s* -> Ignores everything up to the first colon space (strips the log runtime prefix)
    # (\d{4}-\d{2}-\d{2}) -> Group 1: Captures the true trade date
    # (CORE_BUY|CORE_SELL) -> Group 2: Action Type
    # (\w+) -> Group 3: Ticker
    # ([\d.]+) -> Group 4: Shares
    # ([\d.]+) -> Group 5: Price
    # ([\d.]+) -> Group 6: Fee
    pattern = r"^.*?:\s*(\d{4}-\d{2}-\d{2}).*?(CORE_BUY|CORE_SELL)\s+(\w+)\s+\.\.\.\s+([\d.]+)\s*@\s*([\d.]+),\s*fee:\s*([\d.]+)"

    parsed_rows = []

    # Handle string buffer or file reading
    if isinstance(log_input, str) and "\n" in log_input:
        file_stream = io.StringIO(log_input)
    else:
        file_stream = open(log_input, "r") if isinstance(log_input, str) else log_input

    with file_stream as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                trade_date, tx_type, ticker, shares, price, fee = match.groups()
                parsed_rows.append(
                    {
                        "date": pd.to_datetime(trade_date),
                        "type": tx_type,
                        "ticker": ticker,
                        "shares": float(shares),
                        "price": float(price),
                        "fee": float(fee),
                    }
                )

    return pd.DataFrame(parsed_rows)



import numpy as np

def generate_balance_df(df_trade_history, daily_resolution=True):
    """Takes a trade history DataFrame (as seen in Screenshot 2026-07-06 at 11.40.43 AM.png)

    and generates a rolling balance (positions held) DataFrame.
    """
    df = df_trade_history.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 1. If 'shares2' doesn't exist or isn't signed, generate it dynamically
    if "shares2" not in df.columns:
        df["shares2"] = np.where(
            df["type"].str.contains("BUY", case=False, na=False),
            df["shares"],
            -df["shares"],
        )

    # 2. Aggregate multiple trades of the same ticker on the same day
    df_daily = (
        df.groupby(["date", "ticker"])["shares2"].sum().reset_index()
    )

    # 3. Pivot wide to put tickers as columns and dates as rows
    df_pivot = df_daily.pivot(
        index="date", columns="ticker", values="shares2"
    ).fillna(0)

    # 4. Compute the cumulative sum across time to get actual balances held
    df_balance = round(df_pivot.cumsum(),3)

    # 5. Optional: Fill in missing calendar days so you have a continuous timeline
    if daily_resolution:
        all_days = pd.date_range(
            start=df_balance.index.min(), end=df_balance.index.max(), freq="D"
        )
        df_balance = df_balance.reindex(all_days).ffill().fillna(0)
        df_balance.index.name = "date"

    return df_balance


def unpivot_balance_df(df_balance):
    """Takes a wide balance DataFrame (dates as index, tickers as columns)

    and unpivots it back to a long format: date, ticker, shares.
    """
    # 1. Bring 'date' out of the index so it can be used as an identifier variable
    df_long = df_balance.reset_index()

    # 2. Melt the dataframe from wide to long
    df_unpivoted = df_long.melt(
        id_vars="date",  # The column that remains fixed
        var_name="ticker",  # What the old column headers (tickers) will be named
        value_name="shares",  # What the cell values (share counts) will be named
    )

    # 3. Sort chronologically and by ticker for clean readability
    df_unpivoted = df_unpivoted.sort_values(by=["date", "ticker"]).reset_index(
        drop=True
    )
    
    return df_unpivoted.loc[df_unpivoted['shares']!=0]



import numpy as np
import pandas as pd


def build_enhanced_ledger_from_balances(df_balance, df_prices):
    """Builds a daily attribution ledger treating df_balance as an inventory

    snapshot (Balance Sheet model).
    """
    # 1. Unpivot prices to create a continuous Date-Ticker backbone
    df_price_long = df_prices.melt(
        ignore_index=False, var_name="ticker", value_name="current_price"
    ).reset_index()
    df_price_long["date"] = pd.to_datetime(df_price_long["date"])

    # 2. Standardize balance dataframe
    df_balance = df_balance.copy()
    df_balance["date"] = pd.to_datetime(df_balance["date"])
    df_balance = df_balance.rename(columns={"shares": "reported_shares"})

    # ------------------------------------------------------------------
    # Build balance-sheet snapshots correctly
    # Missing ticker on a snapshot date = 0 shares
    # Holdings persist until the next snapshot.
    # ------------------------------------------------------------------

    # Every ticker ever seen
    all_tickers = sorted(
        set(df_price_long["ticker"]).union(df_balance["ticker"])
    )

    # Every balance-sheet date
    snapshot_dates = sorted(df_balance["date"].unique())

    # Expand every snapshot to include every ticker
    full_snapshots = (
        pd.MultiIndex.from_product(
            [snapshot_dates, all_tickers],
            names=["date", "ticker"]
        )
        .to_frame(index=False)
    )

    full_snapshots = (
        full_snapshots
        .merge(
            df_balance[["date", "ticker", "reported_shares"]],
            on=["date", "ticker"],
            how="left"
        )
    )

    # Missing on a snapshot = sold
    full_snapshots["reported_shares"] = (
        full_snapshots["reported_shares"].fillna(0)
    )

    # Merge snapshots onto the daily price calendar
    df = (
        df_price_long
        .merge(
            full_snapshots,
            on=["date", "ticker"],
            how="left"
        )
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    # Carry holdings forward until the next balance-sheet snapshot
    df["shares_holding"] = (
        df.groupby("ticker")["reported_shares"]
        .ffill()
        .fillna(0)
        .round(6)
    )

    # Transaction deltas
    df["yesterday_shares"] = (
        df.groupby("ticker")["shares_holding"]
        .shift(1)
        .fillna(0)
    )

    df["shares_change"] = (
        df["shares_holding"] - df["yesterday_shares"]
    )

    df["is_rebalance_day"] = (
        df["shares_change"] != 0
    )




    # 6. Track Entry Price via Rolling Weighted Average Cost (Cost Basis)
    # ------------------------------------------------------------------
    # Calculate cash flow ONLY for additions to the position (buys).
    # If shares_change > 0, we spent (shares_change * current_price).
    # If shares_change <= 0 (selling or holding), new capital deployed is 0.
    df["capital_injected"] = np.where(
        df["shares_change"] > 0,
        df["shares_change"] * df["current_price"],
        0.0
    )

    # We need to calculate the rolling cost basis dynamically.
    # To do this cleanly across the vectorised timeline:
    # Cost Basis Today = (Value of Yesterday's Remaining Shares at Yesterday's Cost Basis + New Deployed Capital) / Current Shares
    
    # Initialize a temporary column for execution
    entry_prices = np.zeros(len(df))
    
    # Because calculating an average cost requires the *previous* day's average cost,
    # we group by ticker and compute the rolling average cost basis.
    for ticker, group in df.groupby("ticker"):
        idx = group.index
        shares = group["shares_holding"].values
        change = group["shares_change"].values
        price = group["current_price"].values
        injected = group["capital_injected"].values
        
        current_basis = 0.0
        
        for i in range(len(idx)):
            if shares[i] == 0:
                current_basis = 0.0
            elif change[i] > 0:
                # Scaled up: Average the old remaining cost with the new execution price
                total_cost = (shares[i] - change[i]) * current_basis + injected[i]
                current_basis = total_cost / shares[i]
            else:
                # Scaled down or just holding: Average cost basis does NOT change
                pass
                
            entry_prices[idx[i]] = current_basis

    df["entry_price"] = entry_prices
    # Replace zeros with NaN when not holding for a cleaner dataframe view
    df.loc[df["shares_holding"] == 0, "entry_price"] = np.nan
    # ------------------------------------------------------------------

    # 7. Calculate Values & NAV Weights
    df["current_value"] = df["shares_holding"] * df["current_price"]
    df["yesterday_price"] = df.groupby("ticker")["current_price"].shift(1)
    df["yesterday_value"] = df["yesterday_shares"] * df["yesterday_price"].fillna(0)

    daily_nav = df.groupby("date")["current_value"].transform("sum")
    df["weight"] = np.where(daily_nav > 0, df["current_value"] / daily_nav, 0)

    # 8. Calculate Returns & PnL
    df["pnl_dollar"] = df["yesterday_shares"] * (
        df["current_price"] - df["yesterday_price"].fillna(df["current_price"])
    )

    df["return_change"] = np.where(
        df["yesterday_value"] > 0, df["pnl_dollar"] / df["yesterday_value"], 0
    )

    df["cum_pnl"] = np.where(
        df["shares_holding"] > 0,
        (df["current_price"] - df["entry_price"]) * df["shares_holding"],
        0,
    )

    # 9. Clean up and return final columns
    final_cols = [
        "date",
        "ticker",
        "shares_holding",
        "current_price",
        "current_value",
        "yesterday_value",
        "return_change",
        "pnl_dollar",
        "entry_price",
        "cum_pnl",
        "weight",
        "shares_change",
        "is_rebalance_day",
    ]

    return df[final_cols].sort_values(["date", "ticker"]).reset_index(drop=True)