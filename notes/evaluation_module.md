# `src2/evaluation.py` — diagnostics off `df_ranked` (status: useful, incomplete)

Computes "why is this scorer good/bad" directly from `alpha_engine`'s output, no trade-log parsing needed. See [scorer_comparison.ipynb](../src2/scorer_comparison.ipynb) for it in use, and [scorers_vectorized.md](scorers_vectorized.md) for the concrete results/history below.

## What exists
- **`add_forward_return(df_ranked, price_df, rebalance_dates)`** — each ticker's return from this rebalance date to the next.
- **`information_coefficient(df_ranked_fwd, score_col="score")`** — per date, `scipy.stats.spearmanr(score, forward_return)` across **every ticker in the universe that date** (~29 names here), not just the ones the portfolio actually holds. Full-rank-order correlation.
- **`ic_significance(df_ic)`** — one-sample t-test of mean IC vs 0 across dates. Added after an initial sweep ranked scorers by raw mean IC with no significance check — that ranking turned out to be ranking noise (see scorers_vectorized.md). Always run this alongside IC, never report mean IC alone.
- **`contribution_by_ticker(df_ranked_fwd)`** — `target_weight * forward_return` summed per ticker across the whole backtest; approximates (ignores fees/delay) which names actually drove the result.
- **`turnover_by_date(df_ranked)`** — `sum(|weight_t - weight_{t-1}|)` per date; trading-intensity/fee-drag proxy.

## Known gap, flagged 2026-07-24 (not yet implemented)
**IC tests a different question than the one that actually matters for NAV.** `apply_filter_and_rank` only ever keeps the top `top_percent` (10–25% here, ~3-7 of 29 names) — it never cares whether rank 15 beat rank 16. Full-universe Spearman IC is a *stricter* test: "is the entire 29-name ordering correct," when what the portfolio needs is much narrower: "is the top slice meaningfully better than what got excluded."

This is exactly why the results looked contradictory in this session: IC across 4 scorers was statistically indistinguishable from zero (all p > 0.5), yet the same 4 scorers' realized CAGR spanned -0.5% to +9.1% over the backtest. **These aren't actually contradictory** — they're answering different questions on different slices of the same data (whole-universe rank order vs. top-decile-only realized return) — but it means a null IC result should NOT be read as "this scorer's portfolio construction doesn't work," only as "the full cross-sectional ranking isn't statistically validated by this test."

**What would close the gap:** a top-vs-bottom quantile spread metric that matches how `apply_filter_and_rank` actually trades the signal — e.g. per date, average forward return of `target_weight > 0` tickers minus average forward return of the bottom `top_percent` (or the full excluded set) — plus its own significance test (mean spread vs 0, same t-test pattern as `ic_significance`). This is the standard alternative to IC in factor research specifically because it's portfolio-construction-aware rather than whole-ranking-aware. Not yet built — natural next addition to `evaluation.py` (e.g. `quantile_spread(df_ranked_fwd)` / `quantile_spread_significance(...)`).

## Also unresolved (carried from scorers_vectorized.md)
- Small universe (29 tickers), short backtest (~4 years) → low statistical power on any per-date test; a bigger universe/longer history would narrow standard errors.
- `src/stat_test.py` (`automated_paired_test`) exists in the repo, unused here — worth checking if it adds anything the plain t-test in `ic_significance` doesn't.
