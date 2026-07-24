from abc import ABC, abstractmethod
import pandas as pd
from src.signal_schema import StrategyOutput, SignalPayload


class SignalFilter(ABC):
    """
    Common interface for all "Module D-style" pure screening layers
    (liquidity, absolute momentum, regime gates, custom compliance rules,
    etc.). A SignalFilter only ever ADDS, DROPS, or ADJUSTS signals it is
    given -- it never computes portfolio weights, and it never invents new
    tickers/signals that weren't already present in strategy_output.

    Implementations should follow these conventions so they can be chained
    interchangeably in a pipeline:

      1. filter_signals(date, strategy_output, snapshot) -> StrategyOutput
         is the ONLY required entry point. Everything else (precomputation,
         thresholds, internal state) is implementation detail.

      2. AD_HOC_EXIT signals should generally pass through untouched unless
         a filter has an explicit, deliberate reason to also gate exits
         (rare -- most filters should never block a position from being
         reduced/closed).

      3. Tickers absent from the relevant snapshot data (e.g. non-asset
         instructions, or data not covered by this filter) should pass
         through untouched rather than being dropped by default -- a
         filter should only act on the data it actually knows how to
         evaluate.

      4. Filters may DROP signals (hard filters, like liquidity or
         absolute momentum) or ADJUST signals in place (soft filters, like
         a score penalty) -- both are valid uses of this interface. A
         filter should not do both silently in a way that's hard to audit;
         prefer clear logging either way (see log()).

      5. Filters should be side-effect-free with respect to
         strategy_output -- always build and return a NEW StrategyOutput
         rather than mutating the input in place, so upstream callers can
         safely reuse the original if needed.
    """

    #: If False, suppress the per-ticker audit print in log(). Useful when
    #: chaining many filters and you only want warnings/summary output.
    verbose: bool = True

    @abstractmethod
    def filter_signals(
        self,
        date: pd.Timestamp,
        strategy_output: StrategyOutput,
        snapshot: dict[str, pd.DataFrame],
    ) -> StrategyOutput:
        """
        Screens/adjusts incoming signals for a single time step.

        Parameters
        ----------
        date : pd.Timestamp
            The current rebalance/evaluation date.
        strategy_output : StrategyOutput
            Signals produced upstream (by an alpha strategy, or by a prior
            filter in the chain).
        snapshot : dict[str, pd.DataFrame]
            Market data available at `date`, keyed by field name (e.g.
            "price", "volume"). Filters that rely on precomputed panels
            (built at construction time) may ignore this and accept it
            only for interface parity.

        Returns
        -------
        StrategyOutput
            A new, sanitized/adjusted signal set.
        """
        raise NotImplementedError

    def log(self, date: pd.Timestamp, message: str) -> None:
        """Uniform audit-log formatting, so all filters print consistently."""
        if self.verbose:
            print(f"[{date.strftime('%Y-%m-%d')}] {self.__class__.__name__}: {message}")


class NullFilter(SignalFilter):
    """
    Dummy / pass-through filter. Does nothing -- returns strategy_output
    unchanged. Useful as:
      - a default/no-op placeholder in a configurable filter chain
        (e.g. "use AbsoluteMomentumFilter if enabled, else NullFilter")
      - a baseline in A/B testing a pipeline with vs without a given filter
      - a template to copy when starting a brand new filter implementation
    """

    def filter_signals(
        self,
        date: pd.Timestamp,
        strategy_output: StrategyOutput,
        snapshot: dict[str, pd.DataFrame] = None,
    ) -> StrategyOutput:
        # Return a new StrategyOutput wrapping the same signals, rather
        # than the original object, to keep the "always return a new
        # StrategyOutput" convention consistent with every other filter.
        return StrategyOutput(list(strategy_output.signals))