import pandas as pd
from src_old.signal_schema import StrategyOutput, SignalPayload


class AbsoluteMomentumFilter:
    """
    Module D1-Gateway: Pure Screening Layer.
    Validates signal eligibility against the asset's own trend (time-series
    momentum), independent of its relative rank against peers.
    Does NOT calculate portfolio weights, and does NOT rank/select signals.

    PERFORMANCE NOTE: the trend test (moving average / trailing return) is
    computed ONCE, for the entire price panel, at construction time --
    producing a (date x ticker) boolean pass panel and a matching diagnostic
    panel. filter_signals() then does pure O(1) lookups by (date, ticker)
    instead of recomputing rolling windows on every call. This assumes the
    underlying price panel is fixed across repeated calls/backtest steps
    (e.g. a block-bootstrap over evaluation dates on one fixed historical
    price series). If you are instead bootstrapping/simulating alternate
    PRICE PATHS (return-shuffling Monte Carlo), the panel must be rebuilt
    per sample -- construct a fresh AbsoluteMomentumFilter for each price
    path rather than reusing one precomputed panel.

    Two interchangeable rule modes:
      - "moving_average": require price > trailing N-day/month moving average
      - "absolute_return": require trailing lookback return > a benchmark
        return (e.g. cash / T-bill proxy, or 0.0 for a simple "positive
        trend" test) -- this is the Antonacci-style absolute momentum rule.

    Design intent: run this BEFORE selecting the top-N% by score, so that
    ranking happens only within an already trend-confirmed universe. This
    preserves the "escape to cash" property of dual momentum -- if most of
    the universe fails the absolute trend check (e.g. broad market topping),
    the eligible pool shrinks or empties out, naturally reducing exposure
    instead of just picking the least-bad relative winner.
    """

    def __init__(
        self,
        df_price: pd.DataFrame,        # full price panel: index=dates, columns=tickers (fixed, historical)
        mode: str = "moving_average",
        ma_window: int = 200,          # trading days, used if mode == "moving_average"
        lookback_window: int = 252,    # trading days (~12mo), used if mode == "absolute_return"
        cash_return: float = 0.0,      # benchmark hurdle for absolute_return mode
        benchmark_ticker: str | None = None,  # optional: e.g. "QQQ" to gate the WHOLE universe
    ):
        assert mode in ("moving_average", "absolute_return")
        self.mode = mode
        self.ma_window = ma_window
        self.lookback_window = lookback_window
        self.cash_return = cash_return
        self.benchmark_ticker = benchmark_ticker

        self.pass_df, self.diag_df = self._precompute(df_price)

    # ------------------------------------------------------------------ #
    # One-time precomputation over the full panel
    # ------------------------------------------------------------------ #
    def _precompute(self, df_price: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.mode == "moving_average":
            window = self.ma_window
            ma = df_price.rolling(window=window, min_periods=window).mean()
            diag_df = df_price / ma - 1.0
            pass_df = df_price > ma
        else:  # absolute_return
            window = self.lookback_window
            trailing_return = df_price.pct_change(periods=window)
            diag_df = trailing_return - self.cash_return
            pass_df = trailing_return > self.cash_return

        # Rows with insufficient history are NaN -> treat as fail-safe False
        pass_df = pass_df.fillna(False)
        return pass_df, diag_df

    # ------------------------------------------------------------------ #
    # O(1) lookup helpers
    # ------------------------------------------------------------------ #
    def _lookup(self, date: pd.Timestamp, ticker: str) -> tuple[bool, float]:
        if ticker not in self.pass_df.columns or date not in self.pass_df.index:
            return False, float("nan")
        passes = bool(self.pass_df.at[date, ticker])
        diagnostic = self.diag_df.at[date, ticker]
        return passes, diagnostic

    def filter_signals(
        self,
        date: pd.Timestamp,
        strategy_output: StrategyOutput,
        snapshot: dict[str, pd.DataFrame] = None,  # kept for interface parity; unused (precomputed)
    ) -> StrategyOutput:
        """
        Screens incoming signal streams via precomputed lookup. Drops assets
        failing the absolute (time-series) momentum rule. Exit/liquidation
        signals always pass through untouched -- this filter only gates
        NEW/HELD exposure.
        """
        sanitized_output = StrategyOutput()

        # Optional global gate: benchmark itself failing the trend test
        # blocks all new/held exposure this step (regime filter).
        benchmark_blocks_all = False
        if self.benchmark_ticker is not None:
            bench_passes, bench_diag = self._lookup(date, self.benchmark_ticker)
            if not bench_passes:
                benchmark_blocks_all = True
                print(f"[{date.strftime('%Y-%m-%d')}] REGIME FILTER: "
                      f"{self.benchmark_ticker} failed absolute trend test "
                      f"(diagnostic={bench_diag:.4f}). Blocking new/held exposure this step.")

        for signal in strategy_output.signals:
            ticker = signal.ticker

            # Exits are never blocked
            if signal.kind == "AD_HOC_EXIT":
                sanitized_output.append(signal)
                continue

            # Non-asset instructions (not in the precomputed panel) skip this screen
            if ticker not in self.pass_df.columns:
                sanitized_output.append(signal)
                continue

            if benchmark_blocks_all:
                print(f"[{date.strftime('%Y-%m-%d')}] FILTERED: {ticker} dropped. "
                      f"Regime filter ({self.benchmark_ticker}) is in a non-trending state.")
                continue

            passes, diagnostic = self._lookup(date, ticker)

            if passes:
                sanitized_output.append(signal)
            else:
                print(f"[{date.strftime('%Y-%m-%d')}] FILTERED: {ticker} dropped. "
                      f"Absolute momentum check failed (mode={self.mode}, "
                      f"diagnostic={diagnostic:.4f}).")

        return sanitized_output