# `src/scorers/` — turn one ticker's price series into a single numeric score

Used inside a selector: a selector owns a scorer (constructor arg) and calls `compute_score()` on a price window it slices itself.

## `base_scorer.py` — `MomentumScorer` (ABC)
Meso-level contract every scorer implements. One abstract method:

```python
compute_score(prices: np.ndarray) -> tuple[float, dict[str, float]]
```

- `score`: scalar used for cross-sectional ranking (by a selector) or absolute-threshold comparison (by `SelfSelector`).
- `metrics`: free-form diagnostics dict, folded into `StrategyTelemetry` by whichever selector calls it.

**Design convention (not enforced by the ABC, but followed by every implementation):** a scorer is **date-agnostic / window-agnostic** — it computes purely off the `prices` array it's handed, whatever length that happens to be. "How much history to use" is entirely the **selector's** responsibility (it slices the series before calling `compute_score`), keeping window-length from being defined in two places at once.

## `moment_scorers.py` — momentum-family scorers
All implement `MomentumScorer.compute_score(prices) -> (score, metrics)`; higher score = more attractive momentum candidate.

- **`SimpleRiskAdjustedScorer`** — traditional baseline. `score = raw_arithmetic_return / annualized_daily_vol`, where `raw_return = prices[-1]/prices[0] - 1` and vol is `std(daily_returns) * sqrt(252)`. Returns `NaN` if fewer than 2 prices, non-positive start price, or zero/NaN vol. Metrics: `volatile` (the vol value).
- **`TraditionalLinearRegressionScorer`** — traditional baseline. Unweighted OLS of `log(prices)` on `arange(n)`. `score = slope * r_squared`. Requires ≥3 points. Metrics: `slope`, `r_squared`, `intercept`.
- **`GeometricDriftScorer`** — `score = annualized_log_return - 0.5 * vol²` (geometric/continuous drift net of volatility drag, the GBM-consistent version of a return estimate). `vol = std(diff(log(prices))) * sqrt(252)`; `T = len(prices)/252` years. Metrics: `volatility`, `raw_log_return`.
- **`WeightedLinearRegressionScorer(decay_factor=0.98)`** — like `TraditionalLinearRegressionScorer` but WLS instead of OLS: exponential weights `decay_factor ** (n-1-x)` favor recent observations. Solved via the analytical WLS normal equations (`(XᵀWX)⁻¹XᵀWy`). `score = slope * r_squared` (weighted R²). Returns `NaN` on `LinAlgError` (singular matrix) or `n < 3`. Metrics: `slope`, `r_squared`.

## `mean_reversion_scorers.py` — mean-reversion-family scorers
Both are the "opposite" of momentum: score HIGHER when price is below its own window mean (oversold, more attractive to buy), LOWER when above (overbought) — a deliberate sign-flip so they drop into the same "higher = more attractive" ranking convention as the momentum scorers above.

**Date-agnostic by design** (see `base_scorer.py` above): neither owns a lookback-window concept — mean/std is computed from the *entire* `prices` array handed in. Whether that's a 60-day or 252-day slice, or a blend of both, is decided by the selector, not the scorer.

Well suited to low-alpha, beta-dominated, range-bound names (e.g. XLU) that tend to oscillate around a stable trend rather than persistently trend — reversion is structurally more appropriate than continuation there.

- **`MeanReversionScorer(min_periods=20, clip_z=3.0)`** — `score = -z_score = -(last_price - window_mean) / window_std`. Optionally clips `|z|` at `clip_z` *before* negating (guards against one outlier from a volatility-regime shift dominating a cross-sectional rank); pass `clip_z=None` to disable. Returns `NaN` (with all-`NaN` metrics) if fewer than `min_periods` valid bars. Metrics: `z_score` (post-clip), `raw_z_score` (pre-clip), `window_mean`, `window_std`, `last_price`, `n_bars_used`.
- **`BollingerReversionScorer(num_std=2.0, min_periods=20)`** — same underlying math, reframed in Bollinger-Band / percent-B terms:
  ```
  percent_b = (price - lower_band) / (upper_band - lower_band)   # ~0 oversold, ~0.5 mean, ~1 overbought
  score = 0.5 - percent_b                                        # oversold -> high score, same convention as MeanReversionScorer
  ```
  `upper_band`/`lower_band` = `window_mean ± num_std * window_std`. If band width is 0/NaN, `percent_b` defaults to `0.5` (neutral). Metrics: `percent_b`, `upper_band`, `lower_band`, `window_mean`, `last_price`.
