# System Overview — how the backtester works end to end

This is the top-level (Macro) map. This project is deliberately built using a **Macro / Meso / Micro** layering — see [01_macro_meso_micro.md](01_macro_meso_micro.md) for that framework and how it maps onto this codebase. In short: this file is the Macro layer (the orchestration graph), the ABCs/`signal_schema.py` are the Meso layer (module & data contracts), and the `select_*.md` files are the Micro layer (one file's concrete implementation each).

## Entry point
[noteboks/6_real_run.ipynb](../noteboks/6_real_run.ipynb) (and equivalently `main_real_data.py`'s `__main__` block) is where a run is configured and launched.

## The five layers (a signal pipeline)

```
DataBroker  →  Selector (+ Scorer)  →  Filter(s)  →  BudgetAllocator  →  TransactionExecutor  →  GlobalState
 (Module B)      (Module C)             (Module D)     (Module D1)          (Module D2)          (state store)
```

1. **`DataBroker`** ([src/data_broker.py](../src/data_broker.py)) — pulls historical OHLCV from `yfinance`, aligns everything into two `[date x ticker]` matrices: `price` and `volume`. Missing tickers become empty series (not an error) so the shape stays consistent.

2. **`CalendarIterator`** ([src/calendar_iterator.py](../src/calendar_iterator.py)) — drives the day-by-day loop. Decides monthly (or other interval) rebalance dates from real trading days, and for each `today` hands out a `snapshot` (a `{"price": ..., "volume": ...}` dict sliced to `[:today]` only). This is the look-ahead-bias guard — it asserts the snapshot never contains data past `today`.

3. **Selector + Scorer (Module C)** — on rebalance days, the selector scores every ticker (via a pluggable `MomentumScorer`) and turns scores into a `StrategyOutput` (a list of `SignalPayload`: ticker, kind, score/quantity). Two selector philosophies exist — see `self_selectors.md` vs `cross_selectors.md` for why they behave very differently on a correlated universe.

4. **Filter(s) (Module D)** — a chain of `SignalFilter` implementations screens/adjusts the raw signals: hard-drop (liquidity, absolute momentum) or soft-adjust (trend-acceleration score penalty). Currently the notebook run uses `NullFilter` (no-op) — real filters exist but aren't wired in yet.

5. **`IntegratedBudgetAllocator` (Module D1)** ([src/budget_allocator.py](../src/budget_allocator.py)) — takes the filtered signals, keeps only the top decile by score (`top_percent=0.10`), turns that into target portfolio weights (`equal` or `score_weighted`), and separately validates ad-hoc tactical buy/sell orders against available tactical cash.

6. **`TransactionExecutor` (Module D2)** ([src/executor_module.py](../src/executor_module.py)) — converts target weights into actual trades: ad-hoc exits first, then core rebalancing (sells before buys, to raise cash first), applies a flat fee rate, and commits every trade to `GlobalState`. There's a 1-day trade delay (`trade_delay=1`) between a signal being generated and being executed — meant to simulate realistic order lag.

7. **`GlobalState`** ([src/global_state.py](../src/global_state.py)) — the single source of truth. Splits capital into two "sleeves": **Core** (slow, regular monthly rebalancing) and **Tactical** (fast, ad-hoc buy/sell). State is mutated only through `commit_executed_trade()`. Records daily NAV snapshots (both the "intended" mark and the "act"/executed mark) for later performance evaluation.

## Orchestration
[src/run.py](../src/run.py) is the actual day-by-day loop that wires steps 2–7 together for **one strategy**. [main_real_data.py](../main_real_data.py)'s `RealDataBacktester` class is the outer harness that runs `src.run.run()` once per named strategy (e.g. `{'mean_reverse': ..., 'bollinger_reversion': ...}`), collecting a `GlobalState` and an audit ledger per strategy.

## Audit trail
Every rebalance-day step optionally appends an `AuditRecord` ([src/audit.py](../src/audit.py)) capturing: date, all scorer telemetry (`StrategyTelemetry.metrics` — raw scores, z-scores, R², etc.), raw vs filtered signals, and final target weights. `audit.analyze_audit_trail()` flattens the whole ledger into one long DataFrame (date, ticker, all metric columns, final_weight) for post-hoc analysis in the notebook — this is the "why did we buy/sell X" trail.

## Message contract between layers
[src/signal_schema.py](../src/signal_schema.py) defines the shared vocabulary every layer speaks:
- `SignalPayload(ticker, kind, score, quantity_target)` — `kind` is one of `REGULAR_REBALANCE` (core sleeve), `AD_HOC_EXIT`, `AD_HOC_BUY` (tactical sleeve).
- `StrategyOutput` — a list wrapper of `SignalPayload`s for one time-step.
- `StrategyTelemetry` — free-form diagnostics dict for the audit trail.

## Pluggability (why the "restructure")
Every stage — Selector, Scorer, Filter — has an ABC base class (`base_selector.py`, `base_scorer.py`, `base_filter.py`) so implementations are swappable without touching the run loop: e.g. swap `SelfSelector` for `ComprehensiveMultiHorizonStrategy`, or `MeanReversionScorer` for `GeometricDriftScorer`, purely via constructor arguments passed into `RealDataBacktester`. This is what the `main` branch's restructure (splitting the old flat `strategy_selector.py`/`compliance_filters.py` into `src/selectors/`, `src/scorers/`, `src/filters/` packages) was for.

## Known issues found while documenting (not yet fixed unless noted)
- [src/executor_module.py:67](../src/executor_module.py:67) — **fixed**: was referencing an undefined `ticker` instead of `signal.ticker` in a log line (would `NameError` whenever an ad-hoc exit actually executes).
- [src/filters/liquidfilter.py:4](../src/filters/liquidfilter.py:4) — **not fixed yet**: `from filters.base_filter import SignalFilter` is missing the `src.` prefix used everywhere else; importing this module currently raises `ModuleNotFoundError`. Not currently wired into the live pipeline, which is why it hasn't surfaced.

## Framework & source notes
- [01_macro_meso_micro.md](01_macro_meso_micro.md) — the layering method itself, and a map of which folders in `notes/量化股市/` (the Obsidian export) are actually about this trading system vs. an unrelated side project that happens to share the same macro/meso/micro vocabulary.

## Per-file deep dives
- [selectors.md](selectors.md) — `base_selector.py`, `self_selectors.py`, `cross_selectors.py`
- [scorers.md](scorers.md) — `base_scorer.py`, `moment_scorers.py`, `mean_reversion_scorers.py`
- [select_base_filter.md](select_base_filter.md), [select_abosolution_momentum_filter.md](select_abosolution_momentum_filter.md), [select_trend_acceleration_filter.md](select_trend_acceleration_filter.md), [select_liquidfilter.md](select_liquidfilter.md)
