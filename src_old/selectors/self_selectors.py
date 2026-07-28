import pandas as pd
import numpy as np
from src_old.signal_schema import SignalPayload, StrategyOutput, StrategyTelemetry
from src_old.scorers.moment_scorers import MomentumScorer
from src_old.selectors.base_selector import BaseSelector


class SelfSelector(BaseSelector):
    """
    Absolute / self-referential selector: the direct fix for the
    forced-ranking problem above. Each ticker's eligibility is judged
    PURELY against its own fixed threshold -- never relative to how its
    peers happen to score today.

      - buy_threshold:  a ticker only generates a REGULAR_REBALANCE buy
        signal if its own score clears this bar. If NO ticker in the
        universe clears it, NO buy signals are produced that step --
        the strategy legitimately sits in cash/holds rather than being
        forced to crown a "least bad" candidate. This is the direct
        analog of the escape-to-cash property discussed for absolute
        momentum, applied to the mean-reversion side.

      - exit_threshold: a ticker generates an AD_HOC_EXIT signal
        (full liquidation by default) if its own score falls to or below
        this bar -- again, independent of relative rank. A ticker is
        never exited just because peers currently look relatively
        better; it's only exited because its OWN absolute signal says
        the setup is no longer valid.

    This directly addresses the "buy at highs / sell at lows" failure
    mode: under ComprehensiveMultiHorizonStrategy, a correlated universe
    that moves together forces a buy/sell decision every step regardless
    of whether ANYTHING is actually attractive/unattractive in absolute
    terms. SelfSelector can produce zero signals on a given date, which is
    the correct behavior when the whole group is simultaneously overbought
    or oversold together.
    """

    def __init__(
        self,
        scorer: MomentumScorer,
        window: int = 60,
        min_periods: int = 20,
        buy_threshold: float = 1.0,
        exit_threshold: float = -1.0,
        exit_quantity_target: float = -1.0,  # full liquidation by default
    ):
        self.scorer = scorer
        self.window = window
        self.min_periods = min_periods
        self.buy_threshold = buy_threshold
        self.exit_threshold = exit_threshold
        self.exit_quantity_target = exit_quantity_target

    def calculate_signals(self, snapshot: dict[str, pd.DataFrame]) -> tuple[StrategyOutput, StrategyTelemetry]:
        prices = snapshot["price"]
        output = StrategyOutput()
        telemetry = StrategyTelemetry()

        if len(prices) < self.min_periods:
            return output, telemetry

        window_snapshot = prices.iloc[-self.window:] if len(prices) >= self.window else prices

        raw_scores = {}
        for ticker in prices.columns:
            y = window_snapshot[ticker].dropna().values
            if len(y) < self.min_periods:
                raw_scores[ticker] = float("nan")
                continue

            score, metrics = self.scorer.compute_score(y)
            raw_scores[ticker] = score

            for metric_name, value in metrics.items():
                telemetry.metrics.setdefault(metric_name, {})[ticker] = value

        telemetry.metrics["raw_score"] = {t: float(v) for t, v in raw_scores.items()}

        n_buys = 0
        n_exits = 0
        for ticker, score in raw_scores.items():
            if np.isnan(score):
                continue

            if score >= self.buy_threshold:
                output.append(SignalPayload(ticker=ticker, kind="REGULAR_REBALANCE", score=float(score)))
                n_buys += 1
            elif score <= self.exit_threshold:
                output.append(SignalPayload(
                    ticker=ticker, kind="AD_HOC_EXIT", quantity_target=self.exit_quantity_target
                ))
                n_exits += 1
            # else: score is in the "neutral zone" between thresholds --
            # no signal at all. Existing positions are simply held as-is,
            # new entries are not initiated. This neutral zone is exactly
            # what a rank-based selector cannot express.

        telemetry.metrics["selector_summary"] = {
            "n_qualified_buys": float(n_buys),
            "n_forced_exits": float(n_exits),
            "n_universe": float(len(raw_scores)),
        }

        return output, telemetry