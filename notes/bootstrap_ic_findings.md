# Bootstrap IC robustness check (2026-07-24) — significant result, unresolved caveat

[src2/bootstrap_compare.py](../src2/bootstrap_compare.py) + [scorer_comparison_bootstrap.ipynb](../src2/scorer_comparison_bootstrap.ipynb): reuses `src/statioinary_bootstrap.VectorizedBootstrapEngine` to generate 200 synthetic-but-realistic price/volume histories (same dates/tickers, block-resampled daily returns+volume, `expected_block_size=21`) and recomputes IC on each — testing world-to-world noise, which [evaluation.ic_significance](evaluation_module.md) (date-to-date noise within the ONE real history) can't see.

## Result
| scorer | mean IC (single real history, from [scorers_vectorized.md](scorers_vectorized.md)) | mean IC across 200 bootstrap worlds | p-value (bootstrap) |
|---|---|---|---|
| mean_reversion_40d | 0.0246 (real-history sweep best) | **-0.0138** | 2.3e-9 |
| mean_reversion_60d | 0.0193 | **-0.0180** | 7.4e-15 |
| geometric_drift_252d | 0.0161 | **+0.0623** | 2.1e-55 |

All three are now highly statistically significant with 200 independent world-samples vs. ~40-50 dates. But the direction changed for mean-reversion (small-positive-but-insignificant on the real history → significantly **negative** across worlds — i.e. actively harmful, not just noise) and momentum's magnitude nearly quadrupled (0.016 → 0.062).

## Unresolved: is the momentum jump real, or a bootstrap artifact?
Two competing explanations, not yet distinguished:
1. **Real effect, revealed by variance reduction** — our one historical path was a below-average draw for momentum by chance; averaging across many resampled worlds is exactly what should recover the true small-but-real edge a single noisy history can't statistically confirm.
2. **Bootstrap methodology artifact** — jump probability is `1/expected_block_size = 1/21`, so a 252-day momentum window almost always spans multiple stitched blocks (splices together the end of one real historical stretch with the start of an unrelated one). If jump points aren't independent of where trending stretches fall in the real data, this stitching could manufacture artificial multi-month trend continuity that doesn't exist in real markets — and would inflate momentum's IC specifically, since its long window is likelier to straddle a splice than mean-reversion's 40-60 day windows.

**Not yet done, needed before trusting this result:**
- Sensitivity check: does momentum's bootstrap IC hold up if `expected_block_size` is increased (fewer splices per window) or decreased (more splices)? If IC is sensitive to this parameter, that points toward explanation #2.
- Check IC computed only within intact blocks (no window spanning a jump) vs. across all windows, to isolate whether splice-spanning windows are driving the effect.
- `mean_reversion`'s negative-and-significant result deserves the same scrutiny — it's a bigger practical finding (actively harmful, not neutral) and shouldn't be accepted without the same artifact check.

Until this is resolved, treat the bootstrap IC numbers as "a strong signal something is going on, not yet confirmed as a real market effect."

## Follow-up: portfolio-level CAGR distribution (corroborates the IC direction)

The IC bootstrap tests the whole 29-ticker ranking, but the portfolio only ever trades the top `top_percent` slice (see [evaluation_module.md](evaluation_module.md)'s "known gap"). `bootstrap_compare.bootstrap_cagr_distribution` closes that gap: runs the REAL `run_vectorized` execution loop (fees, trade delay, actual top-25% selection) on each synthetic world instead of just scoring, giving one CAGR per world.

60-world result:
| scorer | mean CAGR | std CAGR | % worlds positive | p-value |
|---|---|---|---|---|
| mean_reversion_40d | 1.4% | 11.9% | 56.7% | 0.35 (not significant) |
| mean_reversion_60d | 1.2% | 11.7% | 51.7% | 0.44 (not significant) |
| geometric_drift_126d | 7.0% | 11.3% | 71.7% | 1.1e-5 |
| geometric_drift_252d | **10.4%** | 10.6% | 83.3% | 2.6e-10 |

Paired world-by-world (`geometric_drift_252d` − `mean_reversion_40d` CAGR, same world each time): mean +9.0%, median +6.8%, 25th percentile still positive (+0.9%) — momentum beats mean-reversion in most individual worlds, not just on average.

**This corroborates the IC bootstrap's direction** (momentum real and positive, mean-reversion near-zero/negative) via a completely independent measurement — full-universe rank correlation vs. actual top-decile portfolio return. Two different methodologies agreeing meaningfully raises confidence this isn't purely a splice-jump artifact (that would have to fool both measurements consistently), though it still doesn't rule it out — the sensitivity checks listed above (vary `expected_block_size`, check windows that don't span a jump) are still the way to close this out properly.

**Practical read:** on this evidence, momentum (`geometric_drift`, longer windows especially 252d) looks like the more promising scorer family to develop further; the mean-reversion scorers tested so far (20d/40d/60d/90d/120d short-to-medium windows) show no real edge on this universe/period and shouldn't be prioritized without a different formulation.

