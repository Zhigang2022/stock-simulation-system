"""
Plots for evaluation/'s own outputs -- the practical reason to keep an
audit-trail-shaped record at all is usually "so I can look at a chart of
it," so this is where that visualization lives, next to the metric_calc.py
/ significance_test.py functions whose output it's plotting.

plot_rebased_performance is adapted from src/analysis.py (same function,
inlined rather than imported -- see iteration/'s calendar_snapshot.py /
trade_implement.py for the same "small enough to fork, not worth a src/
coupling" reasoning applied here).
"""
import pandas as pd
import plotly.graph_objects as go


def plot_rebased_performance(df_nav: pd.DataFrame, start_date=None, end_date=None) -> go.Figure:
    """
    Rebased performance (start = 100) for one or more NAV/price series --
    e.g. g_state.export_nav_dataframe()'s Total_NAV column per scorer, or
    bootstrap/compare.py's build_world_nav output. Index: datetime,
    columns: series name, values: price or NAV.
    """
    df = df_nav.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if start_date is not None:
        df = df.loc[start_date:]
    if end_date is not None:
        df = df.loc[:end_date]

    df_rebased = df / df.iloc[0] * 100

    fig = go.Figure()
    for col in df_rebased.columns:
        fig.add_trace(go.Scatter(
            x=df_rebased.index, y=df_rebased[col], mode="lines", name=col,
            hovertemplate="<b>%{fullData.name}</b><br>Date: %{x|%b %d, %Y}<br>Value: %{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        title="Rebased Performance (Start = 100)",
        xaxis_title="Date", yaxis_title="Index (Base = 100)",
        hovermode="closest", template="plotly_white",
        legend=dict(orientation="v", y=1, yanchor="top", x=1.02, xanchor="left"),
    )
    return fig


def plot_ic_over_time(df_ic: pd.DataFrame) -> go.Figure:
    """
    Bar chart of information_coefficient's per-date IC, with the mean IC
    drawn as a reference line -- the visual companion to
    significance_test.ic_significance's numeric summary of the same series.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_ic["date"], y=df_ic["ic"], name="IC"))
    mean_ic = df_ic["ic"].mean()
    fig.add_hline(y=mean_ic, line_dash="dash", annotation_text=f"mean IC = {mean_ic:.3f}")
    fig.update_layout(
        title="Information Coefficient by Rebalance Date",
        xaxis_title="Date", yaxis_title="IC (Spearman)",
        template="plotly_white",
    )
    return fig


def plot_contribution_by_ticker(df_contribution: pd.DataFrame, top_n: int | None = None) -> go.Figure:
    """
    Horizontal bar chart of contribution_by_ticker's total_contribution per
    ticker, sorted (already sorted descending coming in) -- direct visual
    of "what drove performance." top_n limits to the N largest-magnitude
    contributors when the ticker universe is large.
    """
    df = df_contribution.copy()
    if top_n is not None:
        df = pd.concat([df.head(top_n // 2 + top_n % 2), df.tail(top_n // 2)]).drop_duplicates()

    fig = go.Figure(go.Bar(
        x=df["total_contribution"], y=df.index, orientation="h",
        marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in df["total_contribution"]],
    ))
    fig.update_layout(
        title="Contribution to Return by Ticker",
        xaxis_title="Total Contribution", yaxis_title="Ticker",
        template="plotly_white",
    )
    return fig
