# `src2/vectorized_scorers.py` + IC findings

Matrix-form scorers used by [alpha_engine.py](../src2/alpha_engine.py). See [scorers.md](scorers.md) for the original per-ticker class-based versions these mirror (`rolling_mean_reversion_score` == `MeanReversionScorer`, `rolling_geometric_drift_score` == `GeometricDriftScorer`, verified to agree bit-for-bit in [alpha_engine_demo.ipynb](../src2/alpha_engine_demo.ipynb)).

## The three score functions
- **`rolling_mean_reversion_score(price_df, window, min_periods=20, clip_z=3.0)`** — `score = -z_score = -(price - rolling_mean) / rolling_std`. Below its own rolling mean → higher score → more attractive to buy. A short-term snap-back bet.
- **`rolling_geometric_drift_score(price_df, window, min_periods=20)`** — `score = annualized_log_return - 0.5*vol²` over the trailing window. This is a plain trend-following/momentum score — whichever ticker has run up the most (risk-adjusted) over the window ranks highest. No reversion element.
- **`rolling_composite_score(price_df, momentum_window, reversion_window, momentum_weight, min_periods=20)`** — blends the two above via **cross-sectional percentile RANK** (not raw score, since the two live on incompatible scales): `momentum_weight * mom_rank + (1-momentum_weight) * rev_rank`. `momentum_weight=1.0` → pure momentum, `0.0` → pure reversion.

## What IC means
Information Coefficient: per rebalance date, the Spearman rank correlation between the score assigned to each ticker and the return that ticker actually realized before the next rebalance (see `src2/evaluation.py::information_coefficient`, using `add_forward_return`). Answers "did ranking by this score actually predict which names would do better." `0` = no skill, `+1` = perfect rank, negative = backwards. Rule of thumb in equity-factor research: ~0.02–0.05 mean IC = weakly-but-genuinely predictive; 0.1+ is strong for a single factor.

## Findings (from [scorer_comparison.ipynb](../src2/scorer_comparison.ipynb), 29-ticker tech/growth universe, 2022-01 to 2026-04)

Window sweep (mean IC, sorted):
| scorer | mean IC |
|---|---|
| mean_reversion_40d | 0.0246 |
| geometric_drift_21d | 0.0201 |
| mean_reversion_60d | 0.0193 |
| mean_reversion_90d | 0.0188 |
| mean_reversion_10d | 0.0169 |
| geometric_drift_252d | 0.0161 |
| geometric_drift_126d | 0.0106 |
| mean_reversion_120d | 0.0070 |
| mean_reversion_20d | 0.0011 |
| geometric_drift_189d | -0.0114 |
| geometric_drift_63d | -0.0327 |

Composite sweep (momentum_window=252, reversion_window=60, varying momentum_weight):
| blend | mean IC |
|---|---|
| 50/50 | 0.0227 |
| 100% momentum | 0.0161 |
| 25% momentum | 0.0151 |
| 75% momentum | 0.0119 |
| 0% momentum (pure reversion) | 0.0081 |

**Significance check (added after the sweep, via `evaluation.ic_significance` — one-sample t-test of mean IC vs 0):**

| scorer | mean IC | std IC (across dates) | n dates | p-value |
|---|---|---|---|---|
| mean_reversion_60d | 0.019 | 0.202 | 50 | 0.50 |
| mean_reversion_20d | 0.001 | 0.216 | 50 | 0.97 |
| geometric_drift_126d | 0.011 | 0.241 | 44 | 0.77 |
| geometric_drift_252d | 0.016 | 0.262 | 38 | 0.71 |

**Every one of these is statistically indistinguishable from zero.** The date-to-date IC standard deviation (~0.2–0.26) completely swamps the tiny mean, and with only ~40–50 rebalance dates there isn't enough data to distinguish a true 0.02 edge from pure noise even if it existed. This changes the earlier read: the "best" result from the window/composite sweep (`mean_reversion_40d` at 0.0246) is **not** a real finding — it's the top of a noise distribution, not evidence of a working signal. Ranking scorers by raw mean IC without a significance check is misleading; don't do it again without this test attached.

**Caveats, unresolved:**
- Small universe (29 tickers) and short window (~4 years) is a likely structural cause of the low statistical power here (large `std_ic`, small `n_dates`) — a bigger universe and/or longer history would narrow the standard error and could reveal whether a real (if small) edge exists, or confirm there isn't one.
- `src/stat_test.py` (`automated_paired_test`, referenced in the old `general`-branch `6_real_run_dev.ipynb`) has other significance-testing utilities not yet applied here — worth checking whether they add anything `ic_significance`'s plain t-test doesn't.
- `mean_reversion_20d`'s earlier finding (near-zero IC, high turnover, worst CAGR) still stands as the clearest result in this whole exercise — not because its IC is more negative, but because the combination of "no signal" + "high turnover" is enough to explain the bad CAGR on its own, without needing the IC magnitude to be significant.

See [03_tactical_sleeve_gap.md](03_tactical_sleeve_gap.md) for the separate, unrelated open issue on the tactical/ad-hoc sleeve.
