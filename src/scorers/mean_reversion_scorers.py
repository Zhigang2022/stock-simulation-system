import numpy as np
from src.scorers.base_scorer import MomentumScorer


class MeanReversionScorer(MomentumScorer):
    """
    Mean-reversion "opposite" of momentum: scores HIGHER when price is
    BELOW its own mean over the GIVEN input window (oversold, more
    attractive to buy) and LOWER when price is ABOVE that mean
    (overbought). This is a deliberate sign-flip of a plain z-score, so it
    can be used as a drop-in cross-sectional score with the same
    "higher = more attractive" ranking convention as a momentum scorer.

    DATE-AGNOSTIC BY DESIGN, matching the MomentumScorer convention: this
    scorer does NOT own any lookback-window concept -- it computes its
    mean/std from the ENTIRE `prices` array it is given, whatever length
    that happens to be. "How much history to use" (e.g. a 60-day
    short-term window vs a 252-day long-term window, or blending both) is
    the SELECTOR's responsibility, not the scorer's -- the selector slices
    the price series to the desired lookback and hands that exact slice
    to compute_score(). This avoids two independent, easily-inconsistent
    definitions of "window" living in different layers of the system.

    Well suited for low-alpha, beta-dominated, range-bound names (e.g.
    XLU) where price tends to oscillate around a stable trend rather than
    persistently trend the way growth/momentum names do -- betting on
    reversion to the mean is structurally more appropriate there than
    betting on continuation.

    score = -(last_price - mean_of_input) / std_of_input
          = -z_score

    A large positive score means price is well below the mean of the
    given input window (oversold -> attractive long candidate). A large
    negative score means price is well above that mean (overbought ->
    unattractive / candidate for exit or short, depending on how your
    ranking is used).

    Optionally clips the raw z-score before scoring, since mean-reversion
    signals are prone to extreme outliers when a name's volatility regime
    shifts sharply (e.g. a stock in the early stages of a genuine
    structural break, not just a normal oscillation) -- clipping keeps one
    outlier observation from dominating a cross-sectional rank.
    """

    def __init__(
        self,
        min_periods: int = 20,       # minimum bars required in the GIVEN input to compute a score
        clip_z: float | None = 3.0,  # clip |z| at this level before scoring; None = no clipping
    ):
        self.min_periods = min_periods
        self.clip_z = clip_z

    def compute_score(self, prices: np.ndarray) -> tuple[float, dict[str, float]]:
        prices = np.asarray(prices, dtype=float)
        prices = prices[~np.isnan(prices)]

        if len(prices) < self.min_periods:
            return float("nan"), {
                "z_score": float("nan"),
                "window_mean": float("nan"),
                "window_std": float("nan"),
                "last_price": float("nan"),
                "n_bars_used": len(prices),
            }

        window_mean = prices.mean()
        window_std = prices.std(ddof=1)
        last_price = prices[-1]

        if window_std == 0 or np.isnan(window_std):
            z_score = 0.0
        else:
            z_score = (last_price - window_mean) / window_std

        raw_z = z_score
        if self.clip_z is not None:
            z_score = float(np.clip(z_score, -self.clip_z, self.clip_z))

        score = -z_score  # flip sign: below-mean (oversold) -> higher score

        metrics = {
            "z_score": z_score,
            "raw_z_score": raw_z,
            "window_mean": float(window_mean),
            "window_std": float(window_std),
            "last_price": float(last_price),
            "n_bars_used": len(prices),
        }
        return score, metrics


class BollingerReversionScorer(MomentumScorer):
    """
    Variant expressed in Bollinger-Band terms rather than a raw z-score --
    same underlying math, different diagnostic framing (percent-B style).
    Same date-agnostic design as MeanReversionScorer above: uses the
    ENTIRE given `prices` input as its window; the selector owns lookback
    length.

    percent_b = (price - lower_band) / (upper_band - lower_band)
      ~0.5  -> price at the mean of the given input
      ~0.0  -> price at the lower band (oversold)
      ~1.0  -> price at the upper band (overbought)

    score = 0.5 - percent_b, so oversold (percent_b near 0) scores high,
    overbought (percent_b near 1) scores low -- same ranking convention
    as MeanReversionScorer above.
    """

    def __init__(self, num_std: float = 2.0, min_periods: int = 20):
        self.num_std = num_std
        self.min_periods = min_periods

    def compute_score(self, prices: np.ndarray) -> tuple[float, dict[str, float]]:
        prices = np.asarray(prices, dtype=float)
        prices = prices[~np.isnan(prices)]

        if len(prices) < self.min_periods:
            return float("nan"), {
                "percent_b": float("nan"),
                "upper_band": float("nan"),
                "lower_band": float("nan"),
                "window_mean": float("nan"),
                "last_price": float("nan"),
            }

        window_mean = prices.mean()
        window_std = prices.std(ddof=1)
        last_price = prices[-1]

        upper_band = window_mean + self.num_std * window_std
        lower_band = window_mean - self.num_std * window_std
        band_width = upper_band - lower_band

        if band_width == 0 or np.isnan(band_width):
            percent_b = 0.5
        else:
            percent_b = (last_price - lower_band) / band_width

        score = 0.5 - percent_b

        metrics = {
            "percent_b": float(percent_b),
            "upper_band": float(upper_band),
            "lower_band": float(lower_band),
            "window_mean": float(window_mean),
            "last_price": float(last_price),
        }
        return score, metrics