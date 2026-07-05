import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_ticker_prices(df: pd.DataFrame, title: str = "Stock Prices Over Time") -> go.Figure:
    """Creates an interactive Plotly line chart from a wide-form DataFrame.

    This function expects a DataFrame where the index consists of dates (or a 
    time series), columns represent ticker names, and the cell values represent 
    prices. It automatically formats the axes and maps each column to a distinct 
    line.

    Args:
        df (pd.DataFrame): A pandas DataFrame with a DatetimeIndex (or similar),
            where each column is a stock ticker and values are prices.
        title (str, optional): The title of the generated line chart. 
            Defaults to "Stock Prices Over Time".

    Returns:
        plotly.graph_objects.Figure: The interactive Plotly figure object.

    Raises:
        ValueError: If the input DataFrame is empty.

    Example:
        >>> import pandas as pd
        >>> dates = pd.date_range(start="2026-01-01", periods=3)
        >>> data = {'AAPL': [150, 152, 151], 'MSFT': [300, 305, 302]}
        >>> df = pd.DataFrame(data, index=dates)
        >>> fig = plot_ticker_prices(df, title="Tech Giants Performance")
        >>> fig.show()
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    # Generate the line plot using wide-form data structure
    fig = px.line(
        df,
        title=title,
        labels={"index": "Date", "value": "Price ($)", "variable": "Ticker"},
    )

    # Apply layout customizations for clean visual presentation
    fig.update_layout(
        title_x=0.5,                  # Center alignment for the title
        hovermode="x unified",        # Consolidate all ticker values into a single hover tooltip
        template="plotly_white",      # Professional white grid theme
        xaxis_title="Date",
        yaxis_title="Price ($)"
    )

    return fig