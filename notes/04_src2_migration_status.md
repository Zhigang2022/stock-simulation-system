# src2 migration status (2026-07-28)

**Start here in new sessions: work out of `src2/` and `notebooks/` only.**
`src_old/` and `noteboks_old/` are frozen legacy references (renamed from
`src/` and `noteboks/` this session) — don't extend them, don't add new
code there. They exist only so old notebooks/scripts that still name them
explicitly don't silently rot; new work belongs in `src2/`.

## Why this migration happened

`src` (now `src_old`) was a live day-by-day loop architecture
(`CalendarIterator` → `Selector`/`Scorer` → `Filter` → `BudgetAllocator` →
`TransactionExecutor` → `GlobalState`). `src2` is a from-scratch rebuild:
score/filter/select computed **once, vectorized, up front** (no
CalendarIterator-style runtime slice needed for the core sleeve), then a
much thinner stateful execution loop consumes the precomputed result.
Over this session `src2` went from two flat files (`alpha_engine.py`,
`run_vectorized.py`) to five packages, and every `src_old` dependency was
either forked (copied, not imported) or eliminated.

## Current src2/ package map

```
src2/
├── vector_calc/     # score -> filter -> select/rank -> target_weight (STATELESS, precomputed)
│   ├── scorer_mean_reversion.py / scorer_momentum.py / scorer_composite.py
│   ├── filter_null.py / filter_liquidity.py
│   ├── selector_top_percent.py   # rank + top-percent cut + naive suggested weight
│   └── calculator.py             # orchestration: build_metrics_table / apply_filter_and_rank / build_target_weight_table
├── states/
│   └── global_state.py           # GlobalState/Future_Transaction/Transient_Signals -- forked from src_old, the one genuinely stateful core
├── iteration/        # the day-by-day execution loop (STATEFUL)
│   ├── calendar_snapshot.py      # rebalance dates + leak-safe snapshot (local fork of CalendarIterator)
│   ├── allocation_regular.py     # CORE sleeve: real weight_calc lives HERE (g_state-aware), not in vector_calc
│   ├── allocation_tactical.py    # TACTICAL sleeve: live ad-hoc strategy + position sizing (g_state-aware) + local budget check
│   ├── trade_implement.py        # TransactionExecutor, forked from executor_module.py
│   └── daily_iterator.py         # orchestrates the above; run_daily_iteration is the entry point
├── evaluation/
│   ├── metric_calc.py            # df_ranked diagnostics (IC, contribution, turnover) + NAV metrics (merged from src_old/metrics.py)
│   ├── significance_test.py      # ic_significance + paired tests (merged from src_old/stat_test.py)
│   └── plot.py                   # plot_rebased_performance / plot_ic_over_time / plot_contribution_by_ticker
├── bootstrap/
│   ├── data_gen.py                # VectorizedBootstrapEngine (forked from statioinary_bootstrap.py) + regime classification
│   ├── simulation.py              # run_one_simulation / build_world_nav (runs vector_calc -> iteration per synthetic world)
│   └── significance.py            # paired significance across bootstrap worlds (scorer vs benchmark)
└── data_ingest/
    ├── data_broker.py             # DataBroker (yfinance), forked from src_old
    └── ticker_loader.py           # static ticker lists, forked from src_old
```

**Key design decision, in case it comes up again**: weight/position
sizing lives in `iteration/allocation_regular.py` /
`allocation_tactical.py`, NOT in `vector_calc/selector_top_percent.py`.
`vector_calc` is stateless/precomputed and only ever produces a naive
suggested weight (still useful as `evaluation`'s fast baseline). Real
sizing needs live `g_state` (current cash/positions/exposure), which only
the `iteration` layer has — that's where a real strategy's position
sizing (e.g. safety cash reserve, position caps) should be added.

**Zero `src` imports remain anywhere in `src2`** (verified via grep sweep,
several times over the session, most recently after the `bootstrap/`
reorg). Earlier in the session `bootstrap/compare.py`'s `run_one_simulation`
took a `bootstrap_result` dict built externally by `src.statioinary_bootstrap`
— that boundary is now closed too: `bootstrap.data_gen.VectorizedBootstrapEngine`
generates it from inside `src2`.

## What's been verified end-to-end this session

All three via actual execution (`jupyter nbconvert --execute`), not just
import checks, against real yfinance-fetched data:

1. **`notebooks/dev_workflow.ipynb`** (moved from `src2/alpha_engine_demo.ipynb`)
   — vectorized mean-reversion score matches the original per-ticker
   `MeanReversionScorer` class exactly (`-1.221381` both sides), and
   `iteration.run_daily_iteration` produces a sane NAV/metrics table. This
   notebook is the one deliberate exception with a live `src_old` import
   (`src_old.scorers.mean_reversion_scorers.MeanReversionScorer`) — kept on
   purpose as the cross-check target, per explicit instruction.
2. **`notebooks/scorer_comparison.ipynb`** — 4 scorer configs, full
   summary/IC/contribution/turnover/window-sweep sections, zero errors.
   No `src_old` references at all.
3. **`notebooks/scorer_comparison_bootstrap.ipynb`** — full bootstrap
   pipeline (synthetic worlds → `ProcessPoolExecutor` simulation → regime
   classification → two-sided + one-sided significance → conditional
   per-regime comparison), zero errors. No `src_old` references at all.
   One bug fixed along the way: the original notebook's cell using
   `stat_test.automated_paired_test_one_sided` would have `NameError`'d
   (`stat_test` was never imported there) — now correctly calls
   `evaluation.automated_paired_test_one_sided`.

## Bugs found and fixed this session (not just renames)

- **`.gitignore`**'s `data*` pattern (line 3) was matching by path
  component anywhere in the tree, not just a top-level `data/` dir —
  silently excluding `src2/data_ingest/` and `src2/bootstrap/data_gen.py`
  from git. Narrowed to `/data/`.
- `src_old`'s own internal files still self-referenced the old `src.`
  package name after the `src` → `src_old` rename (13 files: `run.py`,
  `global_state.py`, `executor_module.py`, `audit.py`,
  `ad_hoc_strategy.py`, all of `filters/`/`selectors/`,
  `scorers/mean_reversion_scorers.py`/`moment_scorers.py`) — fixed
  (`from src.` → `from src_old.`) so the legacy package is importable
  again, needed for `dev_workflow.ipynb`'s cross-check cell.
- `budget_allocator.py`'s Part B (tactical cash-check) ignored the
  signal's real `quantity_target` and hardcoded a 10%-of-NAV cost
  estimate — the local reimplementation in
  `iteration/allocation_tactical.py`'s `_budget_check` checks against the
  actual sized amount instead.
- `src_old/executor_module.py` had a dead `AD_HOC_BUY` stub (a bare
  comment, no code) — the tactical buy-execution branch was implemented
  (both in `src_old` and forked properly into
  `iteration/trade_implement.py`).

## Not done / open

- The original ask that kicked off this whole session — a QQQ MACD
  bull/bear-cross tactical strategy with `delay_days=0` — was never
  actually implemented. `iteration/allocation_tactical.py` is now fully
  ready for it (strategy decides kind/ticker only, allocation layer sizes
  the position), but no MACD strategy class exists yet in `src2`.
- `noteboks_old/*.ipynb` (the even-older exploration notebooks, pre-dating
  `src`/`src2` split) were not touched or migrated — they're legacy, not
  in scope.
- `src2/`'s three demo notebooks now live in top-level `notebooks/`; if
  more get added, put them there too, not back inside `src2/`.
