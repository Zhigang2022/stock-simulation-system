# stock_simulation_system

## Active codebase: `src2/` only

`src2/` is the current, actively-developed backtest engine. `src_old/`
and `noteboks_old/` are frozen legacy (renamed from `src/`/`noteboks/`
2026-07-28) — do not add new code there, do not extend them. The only
legitimate reason to import from `src_old` is a deliberate cross-check
against the original implementation (see `notebooks/dev_workflow.ipynb`
for the one example of this). Everything else works out of `src2/` and
`notebooks/`.

`src2` has **zero dependency on `src_old`** internally — every file it
would otherwise have needed (`GlobalState`, `TransactionExecutor`,
`CalendarIterator`, `DataBroker`, `VectorizedBootstrapEngine`,
`metrics.py`, `stat_test.py`) was forked in (copied, not imported) rather
than imported across the boundary. See
[notes/04_src2_migration_status.md](notes/04_src2_migration_status.md)
for the full migration history, bugs fixed, and open items.

## `src2/` package order and structure

Packages are listed in pipeline order — each one builds on the ones
above it. Within each package, files are listed in **running order**
(the entry point, the thing you actually call from outside the package,
is listed last, since it orchestrates everything above it).

### `states/` — the stateful core (used by everything downstream)
1. **`global_state.py`** — `GlobalState`/`Future_Transaction`/`Transient_Signals`. Holds cash, positions (core + tactical sleeves), and the NAV history ledger. No dependencies of its own. Every `iteration/` and `bootstrap/` call operates on an instance of this.

### `data_ingest/` — real market data acquisition
1. **`ticker_loader.py`** — static universe/ticker lists + sampling helper.
2. **`data_broker.py`** — `DataBroker` (yfinance wrapper). Takes a ticker list (often from `ticker_loader.py`) and returns `{"price": df, "volume": df}` — the `world_data_dict` shape everything downstream expects.

### `vector_calc/` — score → filter → select → weight (stateless, precomputed once, no `states` dependency)
1. **`scorer_mean_reversion.py`** / **`scorer_momentum.py`** — base scorers, pure math on the whole price panel.
2. **`scorer_composite.py`** — blends the two above into one score.
3. **`filter_null.py`** / **`filter_liquidity.py`** — eligibility masks (bool per row).
4. **`selector_top_percent.py`** — ranks by score among eligible rows, cuts to top-percent, assigns a **naive** suggested weight (equal or score-weighted). This weight is a fast baseline only — see `iteration/allocation_regular.py`.
5. **`calculator.py`** — entry point. `build_target_weight_table()` wires scorer → `build_metrics_table` → `apply_filter_and_rank` (filter_fn + selector_fn) → `(df_metrics, df_ranked)`.

To add a new scorer/filter/selector: add a new `scorer_xxx.py` / `filter_xxx.py` / `selector_xxx.py` with the matching signature — nothing else in this package needs to change.

### `iteration/` — day-by-day execution loop (stateful, uses `states.GlobalState`)
1. **`calendar_snapshot.py`** — `generate_rebalance_dates()` / `get_historical_snapshot()`.
2. **`allocation_regular.py`** — CORE sleeve. Looks up `vector_calc`'s `df_ranked` for today, applies a `weight_fn` (**real weight sizing lives here**, `g_state`-aware — not in `vector_calc`, which has no access to live cash/positions). Rebalance days only.
3. **`allocation_tactical.py`** — TACTICAL sleeve. A live strategy decides kind/ticker only (e.g. MACD bull/bear cross); **position sizing happens here** via `sizing_fn`, using live cash/exposure — not in the strategy. Runs its own local budget check. Every day, if wired in.
4. **`trade_implement.py`** — `TransactionExecutor`. Fills whatever steps 2–3 staged, gated by `trade_delay`. Priority: ad-hoc exits → ad-hoc buys → core rebalance.
5. **`daily_iterator.py`** — **main entry point**: `run_daily_iteration()`. Orchestrates 1→2→3→4 once per day for the whole backtest.

**Key design rule** (came up explicitly this session, worth remembering): weight/position sizing belongs in `iteration/allocation_*.py`, never in `vector_calc/`. `vector_calc` is stateless and precomputed by design — it structurally cannot know current cash or existing positions, which is exactly what real sizing logic (safety cash reserves, position caps, drift thresholds) needs.

### `evaluation/` — diagnostics (stateless, reads outputs of the above)
1. **`metric_calc.py`** — computes numbers/tables: `df_ranked`-level (forward return, IC, contribution, turnover) and NAV-level (CAGR, Sharpe/Sortino, drawdown/Calmar).
2. **`significance_test.py`** — is a `metric_calc.py` number real or noise (`ic_significance`), or is series A different from series B (paired t-test/Wilcoxon picker — also used by `bootstrap/significance.py`).
3. **`plot.py`** — visualizes `metric_calc.py`'s output. Called last.

### `bootstrap/` — significance testing across synthetic worlds (uses `vector_calc` + `iteration` + `evaluation`)
1. **`data_gen.py`** — `VectorizedBootstrapEngine` generates synthetic price/volume worlds from real data; classifies each world's market regime (bull/bear/choppy).
2. **`simulation.py`** — `run_one_simulation()`/`build_world_nav()`: runs the full `vector_calc` → `iteration` pipeline on one synthetic world, collects `evaluation.calculate_metrics`. Typically via `ProcessPoolExecutor`, one call per world.
3. **`significance.py`** — paired significance testing (scorer vs. benchmark) across all per-world metrics `simulation.py` collected. Runs last.

## Notebooks

`notebooks/` holds the current, verified-working demo notebooks:
`dev_workflow.ipynb`, `scorer_comparison.ipynb`, `scorer_comparison_bootstrap.ipynb`.
All three run end-to-end against real fetched data with zero errors (verified via
`jupyter nbconvert --execute`). Put any new demo/dev notebook here, not inside `src2/`.

## Known open item

A QQQ MACD bull/bear-cross tactical strategy (the original motivation for
several of the design decisions above) has not actually been implemented
yet. `iteration/allocation_tactical.py` is ready for it — see
[notes/04_src2_migration_status.md](notes/04_src2_migration_status.md)
"Not done / open".
