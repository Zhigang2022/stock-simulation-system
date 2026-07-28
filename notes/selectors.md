# `src/selectors/` — Module C: decide which tickers get signals, and what kind

## `base_selector.py` — `BaseSelector` (ABC)
Meso-level contract every selector implements. One abstract method:

```python
calculate_signals(snapshot: dict[str, pd.DataFrame]) -> tuple[StrategyOutput, StrategyTelemetry]
```

- `snapshot` is the `{"price": ..., "volume": ...}` dict handed out by `CalendarIterator` for one rebalance date, sliced to `[:today]` only (look-ahead-bias guard lives upstream, not here).
- Returns a `StrategyOutput` (list of `SignalPayload`) and a `StrategyTelemetry` (free-form metrics dict) — see [signal_schema.py](../src/signal_schema.py).

Two implementation philosophies subclass this, with very different behavior on a correlated universe — `self_selectors.py` vs `cross_selectors.py` below.

## `self_selectors.py` — `SelfSelector`
Absolute / self-referential selector: the fix for cross-sectional forced-ranking (see below). Each ticker is judged purely against its **own fixed threshold**, never relative to peers.

**Mechanics:**
1. Takes a `window` (default 60) / `min_periods` (default 20) slice of price history per ticker, runs it through an injected `MomentumScorer.compute_score()`.
2. Per ticker, compares the raw score against two independent absolute thresholds:
   - `score >= buy_threshold` (default `1.0`) → `REGULAR_REBALANCE` buy signal.
   - `score <= exit_threshold` (default `-1.0`) → `AD_HOC_EXIT` signal, liquidating `exit_quantity_target` (default `-1.0`, i.e. full exit).
   - Otherwise: **no signal** — the ticker sits in a "neutral zone," existing position held as-is, no new entry forced.
3. If a ticker's window has fewer than `min_periods` valid bars, its score is `NaN` and it's skipped entirely (no signal either way).

**Why this exists:** under `ComprehensiveMultiHorizonStrategy` (cross-sectional rank), a buy/sell decision is forced on every rebalance regardless of whether anything is actually attractive/unattractive — because a percentile rank always produces a "top" and a "bottom" even when the entire universe is simultaneously overbought or oversold together. `SelfSelector` can legitimately produce **zero signals** on a date, which is the correct behavior for a correlated universe that moves as one block (e.g. a single GICS sector).

**Telemetry:** `raw_score` per ticker, plus a `selector_summary` block (`n_qualified_buys`, `n_forced_exits`, `n_universe`) for quick audit-trail sanity checks.

## `cross_selectors.py` — `ComprehensiveMultiHorizonStrategy`, `SingleHorizonStrategy`

### `ComprehensiveMultiHorizonStrategy`
Cross-sectional rank-based selector: blends a long-horizon score (`long_window`, default 252 bars) and an optional short-horizon score (`short_window`, default 21 bars) from an injected `MomentumScorer`, each converted to a **percentile rank** across the universe, weighted by `structural_weight` (long) / `tactical_weight = 1 - structural_weight` (short), then re-ranked into a final percentile. Every ticker gets a `REGULAR_REBALANCE` signal with `score = final_percentile` — there is no "no signal" case.

**Known structural limitation** (confirmed via an XLU/XLP/XLRE test): because scores are converted to cross-sectional percentile ranks, the strategy is forced to always produce a "top" and "bottom" of the ranking — even when the entire universe is simultaneously overbought (nothing is actually attractive, but something is still ranked #1 and bought) or oversold (nothing deserves selling, but something is still ranked last). This is especially dangerous on a small, highly correlated universe (e.g. one GICS sector) where names move together on one common factor rather than independent drivers — the rank reflects noise in relative ordering, not a repeatable stock-specific edge. `SelfSelector` above exists specifically to fix this.

If `short_window` is `None`, the tactical leg is skipped entirely (`tactical_weight = 0.0`).

### `SingleHorizonStrategy(scorer, window)`
Subclass, single-horizon special case. Two documented gotchas baked into how it's implemented:
- Uses proper `super().__init__()` subclassing — an earlier `self = ComprehensiveMultiHorizonStrategy(...)` pattern inside `__init__` does **not** work, since reassigning the local `self` name has no effect on the actual instance being constructed (it just discards a throwaway object).
- Does **not** disable the short leg via `short_window=0`: pandas/numpy treat `iloc[-0:]` as `iloc[0:]` (since `-0 == 0`), i.e. the *entire* price history rather than an empty slice — this fails every ticker's short-horizon length check, producing `NaN` short scores that poison the composite (`NaN * tactical_weight == NaN`, not `0`, even when `tactical_weight` is exactly `0.0`). Instead it sets `short_window == long_window` (so the short leg computes on the same fully-populated data, never `NaN`) and relies on `structural_weight=1.0` / `tactical_weight=0.0` to mathematically zero out its contribution — one redundant but harmless computation.
