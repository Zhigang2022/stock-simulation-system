# strategy_selector.py
import pandas as pd
import numpy as np
from src_old.signal_schema import SignalPayload, StrategyOutput, StrategyTelemetry
from src_old.scorers.moment_scorers import MomentumScorer


from src_old.selectors.base_selector import BaseSelector

class ComprehensiveMultiHorizonStrategy(BaseSelector):
    """
    Unified execution harness that coordinates multi-horizon (long vs short)
    cross-sectional ranking using any interchangeable MomentumScorer block.

    IMPORTANT LIMITATION (confirmed by the XLU/XLP/XLRE test): because this
    strategy converts every ticker's score into a CROSS-SECTIONAL PERCENTILE
    RANK before selection, it is structurally forced to always produce a
    "top" and a "bottom" of the ranking, even when:
      - the ENTIRE universe is simultaneously overbought (nothing is
        actually attractive by absolute standards, but something still
        gets ranked #1 and bought), or
      - the ENTIRE universe is simultaneously oversold (nothing deserves
        to be sold, but something still gets ranked last and exited).
    This is especially dangerous on a small, highly correlated universe
    (e.g. a single GICS sector like Utilities), where names mostly move
    together on one common factor (rate expectations) rather than on
    independent, idiosyncratic drivers -- the cross-sectional rank ends up
    reflecting noise in the relative ordering of a correlated group, not a
    repeatable stock-specific edge. See SelfSelector for a fix.
    """
    def __init__(
        self,
        scorer: MomentumScorer,
        long_window: int = 252,
        short_window: int | None = 21,
        structural_weight: float = 0.7
    ):
        self.scorer = scorer
        self.long_window = long_window
        self.short_window = short_window
        self.structural_weight = structural_weight
        self.tactical_weight = 0.0 if short_window is None else 1.0 - structural_weight

    def calculate_signals(self, snapshot: dict[str, pd.DataFrame]) -> tuple[StrategyOutput, StrategyTelemetry]:
        prices = snapshot["price"]
        output = StrategyOutput()
        telemetry = StrategyTelemetry()

        if len(prices) < self.long_window:
            return output, telemetry

        long_snapshot = prices.iloc[-self.long_window:]

        long_scores_dict = {}
        short_scores_dict = {}

        for ticker in prices.columns:
            y_long = long_snapshot[ticker].dropna().values
            if len(y_long) == self.long_window:
                l_score, l_metrics = self.scorer.compute_score(y_long)
                long_scores_dict[ticker] = l_score
                for metric_name, value in l_metrics.items():
                    key = f"{metric_name}_long"
                    telemetry.metrics.setdefault(key, {})[ticker] = value
            else:
                long_scores_dict[ticker] = np.nan

            if self.short_window is not None:
                short_snapshot = prices.iloc[-self.short_window:]
                y_short = short_snapshot[ticker].dropna().values
                if len(y_short) == self.short_window:
                    s_score, s_metrics = self.scorer.compute_score(y_short)
                    short_scores_dict[ticker] = s_score
                    for metric_name, value in s_metrics.items():
                        key = f"{metric_name}_short"
                        telemetry.metrics.setdefault(key, {})[ticker] = value
                else:
                    short_scores_dict[ticker] = np.nan

        telemetry.metrics["raw_score_long"] = {t: float(v) for t, v in long_scores_dict.items()}
        if self.short_window is not None:
            telemetry.metrics["raw_score_short"] = {t: float(v) for t, v in short_scores_dict.items()}

        long_ranks = pd.Series(long_scores_dict).rank(pct=True, na_option='keep').fillna(0.0)

        if self.short_window is not None:
            short_ranks = pd.Series(short_scores_dict).rank(pct=True, na_option='keep').fillna(0.0)
            composite_scores = (long_ranks * self.structural_weight) + (short_ranks * self.tactical_weight)
        else:
            composite_scores = long_ranks * self.structural_weight

        final_percentiles = composite_scores.fillna(0.0).rank(pct=True)

        for ticker, score in final_percentiles.items():
            output.append(SignalPayload(ticker=str(ticker), kind="REGULAR_REBALANCE", score=float(score)))

        return output, telemetry

class SingleHorizonStrategy(ComprehensiveMultiHorizonStrategy):
    """
    Single-horizon special case of ComprehensiveMultiHorizonStrategy.

    Implemented via proper subclassing + super().__init__ (the earlier
    `self = ComprehensiveMultiHorizonStrategy(...)` pattern does NOT work
    -- reassigning the local `self` name inside __init__ has no effect on
    the actual instance being constructed; it just discards a temporary
    object when __init__ returns, leaving the real instance with no
    attributes set at all).

    Also avoids `short_window=0`, which is NOT a safe way to "disable" the
    short leg: pandas/numpy interpret `iloc[-0:]` as `iloc[0:]` (since
    -0 == 0), i.e. the ENTIRE price history, not an empty slice. That
    causes every ticker's short-horizon length check to fail, producing
    NaN short scores -- which then poison the composite via
    `NaN * tactical_weight == NaN` (not 0), even when tactical_weight is
    exactly 0.0.

    Instead, this sets short_window == long_window (so the short leg
    computes on the same, fully-populated data as the long leg -- never
    NaN) and relies on structural_weight=1.0 / tactical_weight=0.0 to
    mathematically zero out its contribution to the composite score. The
    short leg is computed but always contributes 0 -- correct, at the
    cost of one redundant (harmless) computation. See
    SingleHorizonStrategyExplicit for a version that skips the redundant
    computation entirely.
    """

    def __init__(self, scorer: MomentumScorer, window: int = 60):
        super().__init__(
            scorer=scorer,
            long_window=window,
            short_window=window,      # NOT 0 -- avoids the -0 slicing bug
            structural_weight=1.0,    # tactical_weight becomes 0.0 automatically
        )