import numpy as np
import pandas as pd
from src.signal_schema import StrategyOutput, SignalPayload
from src.filters.base_filter import SignalFilter

class TrendAccelerationDiagnostic(SignalFilter):
    """
    Module D2-Diagnostic: Pure signal-quality layer.
    Does NOT filter or drop signals -- produces a per-ticker "is this trend
    losing momentum" diagnostic that can be layered on top of the absolute
    momentum filter (or fed into scoring/weighting) to catch the specific
    failure mode discussed earlier: a name that is still technically above
    its trend line (1st derivative still positive) but decelerating (2nd
    derivative turning negative) -- i.e. a fading trend that a binary
    level-based filter won't catch until it's already too late.

    Definitions (all computed on a smoothed price series to avoid noise):
      smoothed(t)      = rolling mean of price, window = smooth_window
      slope(t)         = smoothed(t) - smoothed(t - slope_window)   [1st derivative]
      acceleration(t)  = slope(t) - slope(t - accel_window)          [2nd derivative]

    A name is flagged "fading" when:
      slope > 0   (still in an uptrend by the 1st-derivative test)
      AND acceleration < 0   (the uptrend is decelerating)

    This is deliberately NOT the same computation as the absolute momentum
    filter's own MA/return test -- it operates on rate of CHANGE, not level,
    so it adds information rather than duplicating the D1 filter (directly
    relevant to the "redundant signal -> concentration" problem seen with
    the advanced strategies).

    All panels are precomputed once at construction time (same performance
    pattern as AbsoluteMomentumFilter) -- lookups afterward are O(1).
    """

    def __init__(
        self,
        df_price: pd.DataFrame,     # full price panel: index=dates, columns=tickers
        smooth_window: int = 10,    # light smoothing to denoise before differencing
        slope_window: int = 20,     # lookback for 1st derivative
        accel_window: int = 20,     # lookback for 2nd derivative (applied to the slope series)
        normalize: bool = True,     # express slope/accel as % of price rather than raw price units
    ):
        self.smooth_window = smooth_window
        self.slope_window = slope_window
        self.accel_window = accel_window
        self.normalize = normalize

        (
            self.smoothed_df,
            self.slope_df,
            self.accel_df,
            self.fading_df,
        ) = self._precompute(df_price)

    def _precompute(self, df_price: pd.DataFrame):
        smoothed = df_price.rolling(window=self.smooth_window, min_periods=self.smooth_window).mean()

        slope = smoothed.diff(self.slope_window)
        accel = slope.diff(self.accel_window)

        if self.normalize:
            # express as % change relative to the smoothed price level at that point,
            # so slope/accel are comparable across tickers with different price scales
            slope = slope / smoothed.shift(self.slope_window)
            accel = accel / smoothed.shift(self.slope_window + self.accel_window)

        fading = (slope > 0) & (accel < 0)
        fading = fading.fillna(False)

        return smoothed, slope, accel, fading

    # ------------------------------------------------------------------ #
    # O(1) lookups
    # ------------------------------------------------------------------ #
    def is_fading(self, date: pd.Timestamp, ticker: str) -> bool:
        if ticker not in self.fading_df.columns or date not in self.fading_df.index:
            return False
        return bool(self.fading_df.at[date, ticker])

    def get_diagnostics(self, date: pd.Timestamp, ticker: str) -> dict:
        """Returns raw slope/acceleration values for logging or score adjustment."""
        if ticker not in self.slope_df.columns or date not in self.slope_df.index:
            return {"slope": float("nan"), "acceleration": float("nan"), "fading": False}
        return {
            "slope": self.slope_df.at[date, ticker],
            "acceleration": self.accel_df.at[date, ticker],
            "fading": bool(self.fading_df.at[date, ticker]),
        }

    def fading_tickers(self, date: pd.Timestamp) -> list[str]:
        """Convenience: all tickers flagged as fading on a given date."""
        if date not in self.fading_df.index:
            return []
        row = self.fading_df.loc[date]
        return row[row].index.tolist()

    # ------------------------------------------------------------------ #
    # Optional: soft score adjustment, rather than hard filtering
    # ------------------------------------------------------------------ #
    def apply_score_penalty(
        self,
        date: pd.Timestamp,
        ticker: str,
        score: float,
        max_penalty: float = 0.5,
        scale: float = 0.05,
    ) -> float:
        """
        Magnitude-scaled soft penalty, replacing the earlier flat-0.5 cut.

        Rationale: a binary "fading -> -50%" haircut throws away the one
        thing this diagnostic actually computes -- HOW MUCH the trend is
        decelerating. A name that just barely tipped into deceleration and
        one that's decelerating hard were being penalized identically,
        which made this behave like a second hard filter wearing a soft
        filter's clothes.

        Instead, the penalty scales continuously with |acceleration|:
          - acceleration near 0 (barely fading)   -> penalty near 0
          - acceleration strongly negative        -> penalty approaches max_penalty
        via tanh, which saturates smoothly rather than growing unbounded
        (so one extreme outlier ticker can't blow the score negative or
        past max_penalty).

        `scale` sets how much |acceleration| it takes to reach roughly
        63% of max_penalty (tanh(1) ~ 0.76 at magnitude == scale). Since
        acceleration is normalized (% terms, when normalize=True at
        construction), scale=0.05 means ~5% deceleration over the
        accel_window gets you most of the way to max_penalty. Tune this
        against the actual distribution of accel_df for your universe
        (see calibrate_scale() below) rather than guessing.

        Only fading names (slope > 0 AND acceleration < 0, per is_fading)
        are penalized at all -- a name with positive acceleration, or one
        already outright declining (slope < 0, which the hard absolute
        filter should catch separately), is untouched here.
        """
        if not self.is_fading(date, ticker):
            return score

        acceleration = self.accel_df.at[date, ticker]
        if pd.isna(acceleration):
            return score

        magnitude = abs(acceleration)
        penalty_frac = max_penalty * np.tanh(magnitude / scale)
        return score * (1.0 - penalty_frac)

    def calibrate_scale(self, quantile: float = 0.75) -> float:
        """
        Helper to pick a reasonable `scale` for apply_score_penalty from
        the actual data, instead of guessing a fixed constant. Returns the
        given quantile of |acceleration| across all fading observations in
        the precomputed panel -- i.e. "scale" defaults to roughly the
        deceleration magnitude of a fairly severe (but not extreme) case,
        so typical fading cases land in the low-to-mid part of the penalty
        curve rather than immediately saturating.
        """
        fading_accel = self.accel_df.where(self.fading_df).abs()
        values = fading_accel.values.flatten()
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return 0.05  # fallback default
        return float(np.quantile(values, quantile))

    # ------------------------------------------------------------------ #
    # Standard interface: filter_signals (matches LiquidityComplianceFilter /
    # AbsoluteMomentumFilter signature so it can be chained in the same
    # pipeline). Main algorithm here is the SOFT score penalty, not a hard
    # drop -- deliberately, to avoid stacking a second binary gate on top
    # of the absolute momentum filter (the redundancy/concentration issue
    # discussed earlier). Exits always pass through untouched.
    # ------------------------------------------------------------------ #
    def filter_signals(
        self,
        date: pd.Timestamp,
        strategy_output: StrategyOutput,
        snapshot: dict[str, pd.DataFrame] = None,  # unused; kept for interface parity
        max_penalty: float = 0.5,
        scale: float = 0.05,
    ) -> StrategyOutput:
        """
        Applies a magnitude-scaled soft score penalty to signals whose
        ticker is currently flagged as "fading" (still trending per 1st
        derivative, but decelerating per 2nd derivative). No signals are
        dropped here -- scores are shrunk proportionally to how severe the
        deceleration is, so mildly-fading names barely change while
        sharply-decelerating names get pushed down harder. Nothing is
        binarily excluded.
        """
        sanitized_output = StrategyOutput()

        for signal in strategy_output.signals:
            ticker = signal.ticker

            # Exits are never touched
            if signal.kind == "AD_HOC_EXIT":
                sanitized_output.append(signal)
                continue

            # Non-asset instructions or tickers outside the precomputed panel pass through
            if ticker not in self.fading_df.columns:
                sanitized_output.append(signal)
                continue

            fading = self.is_fading(date, ticker)
            if fading and not np.isnan(signal.score):
                adjusted_score = self.apply_score_penalty(
                    date, ticker, signal.score, max_penalty=max_penalty, scale=scale
                )
                print(f"[{date.strftime('%Y-%m-%d')}] FADING: {ticker} score adjusted "
                      f"{signal.score:.4f} -> {adjusted_score:.4f} "
                      f"(accel={self.accel_df.at[date, ticker]:.4f}).")
                signal = SignalPayload(
                    ticker=signal.ticker,
                    kind=signal.kind,
                    score=adjusted_score,
                    quantity_target=signal.quantity_target,
                )

            sanitized_output.append(signal)

        return sanitized_output