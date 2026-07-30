"""
TACTICAL sleeve (AD-HOC) strategies: MACD bull/bear-cross on a single
ticker (MacdCrossStrategy), and the same cross gated by a higher-timeframe
MACD trend filter (MacdCrossDailyTrendFilterStrategy). Both implement the
module_c1_adhoc interface
daily_iterator.py expects -- evaluate_exits(g_state, df_metrics_adhoc, today)
-> StrategyOutput -- see src_old/ad_hoc_strategy.py's AdHocChandelierExit for
the shape this mirrors.

MACD itself is NOT computed here: df_metrics_adhoc is precomputed once,
vectorized, via vector_calc.build_metrics_table(score_fn=vector_calc.macd_cross_score)
-- this file only reads that table's score/macd_line/signal_line columns
and makes the BUY/EXIT decision. It's leak-safe by construction (score's
ewm is causal, so row t only reflects price data through t), so no live
snapshot slicing is needed here.

MacdCrossDailyTrendFilterStrategy additionally reads a boolean trend
column (default "macd_trend_up") off the same precomputed table -- that
column comes from vector_calc.make_macd_trend_filter, a standard
filter_xxx.py (df_metrics in, boolean Series out), merged onto
df_metrics_adhoc by the caller before the backtest starts. No trend math
lives here either; this file only reads columns and decides BUY/EXIT.

Neither strategy decides size: sizing is allocation_tactical.py's job
(sizing_fn / naive_position_sizing), which is the only layer with access
to live g_state (tactical_cash, existing exposure).
"""
import pandas as pd
from dataclasses import dataclass
from typing import List, Literal


@dataclass
class SignalPayload:
    """
    Standardized message from a tactical strategy (module_c1_adhoc) into
    allocation_tactical.py's sizing/budget-check layer.
    """
    ticker: str
    kind: Literal["AD_HOC_EXIT", "AD_HOC_BUY"]
    quantity_target: float = float("nan")  # sizing_fn fills this in for AD_HOC_BUY


class StrategyOutput:
    """Container for the SignalPayload(s) a strategy emits on a given day."""
    def __init__(self, signals: List[SignalPayload] | None = None):
        self.signals: List[SignalPayload] = signals if signals is not None else []

    def append(self, signal: SignalPayload):
        self.signals.append(signal)


class MacdCrossStrategy:
    """MACD bull/bear-cross on a single ticker, read off a precomputed metrics table."""

    def __init__(self, ticker: str, fast: int = 12, slow: int = 26, signal: int = 9):
        self.ticker = ticker
        # kept only as metadata documenting what df_metrics_adhoc should have been built with
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def evaluate_exits(self, g_state, df_metrics_adhoc: pd.DataFrame, today: pd.Timestamp) -> StrategyOutput:
        output = StrategyOutput()

        rows = df_metrics_adhoc.loc[
            (df_metrics_adhoc["ticker"] == self.ticker) & (df_metrics_adhoc["date"] <= today)
        ].sort_values("date")
        if len(rows) < 2:
            return output

        prev_score, curr_score = rows["score"].iloc[-2], rows["score"].iloc[-1]
        bull_cross = prev_score <= 0 and curr_score > 0
        bear_cross = prev_score > 0 and curr_score <= 0

        holding = g_state.tactical_positions.get(self.ticker, 0.0) > 0

        if bear_cross and holding:
            output.append(SignalPayload(ticker=self.ticker, kind="AD_HOC_EXIT"))
        elif bull_cross and not holding:
            output.append(SignalPayload(ticker=self.ticker, kind="AD_HOC_BUY"))

        return output


class MacdCrossDailyTrendFilterStrategy:
    """
    Same hourly bull/bear-cross entry/exit as MacdCrossStrategy, but BUY
    signals additionally require df_metrics_adhoc's `trend_col` (default
    "macd_trend_up", from vector_calc.make_macd_trend_filter) to be True
    for today's row. EXIT signals are never gated -- flattening a position
    should never be blocked by a trend filter.
    """

    def __init__(
        self, ticker: str, fast: int = 12, slow: int = 26, signal: int = 9,
        trend_col: str = "macd_trend_up",
    ):
        self.ticker = ticker
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.trend_col = trend_col

    def evaluate_exits(self, g_state, df_metrics_adhoc: pd.DataFrame, today: pd.Timestamp) -> StrategyOutput:
        output = StrategyOutput()

        rows = df_metrics_adhoc.loc[
            (df_metrics_adhoc["ticker"] == self.ticker) & (df_metrics_adhoc["date"] <= today)
        ].sort_values("date")
        if len(rows) < 2:
            return output

        prev_score, curr_score = rows["score"].iloc[-2], rows["score"].iloc[-1]
        bull_cross = prev_score <= 0 and curr_score > 0
        bear_cross = prev_score > 0 and curr_score <= 0
        trend_ok = bool(rows[self.trend_col].iloc[-1])

        holding = g_state.tactical_positions.get(self.ticker, 0.0) > 0

        if bear_cross and holding:
            output.append(SignalPayload(ticker=self.ticker, kind="AD_HOC_EXIT"))
        elif bull_cross and not holding and trend_ok:
            output.append(SignalPayload(ticker=self.ticker, kind="AD_HOC_BUY"))

        return output
