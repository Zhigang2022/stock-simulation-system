# Open issue: tactical/ad-hoc sleeve is not implemented (2026-07-24)

`GlobalState` ([src/global_state.py](../src/global_state.py)) already hard-partitions capital into two sleeves (`core_cash`/`core_positions` vs `tactical_cash`/`tactical_positions`, never mixed) — the data model is ready. What's missing is everything that would actually populate the tactical sleeve.

## What's missing, concretely
1. **No tactical strategy exists.** `src/run.py`'s `module_c1_adhoc` parameter is always `None` in current usage; no `evaluate_exits(state, snapshot) -> StrategyOutput` implementation lives anywhere in `main`. (The old `general` branch referenced `ad_hoc_strategy.AdHocChandelierExit` in `noteboks/6_real_run_dev.ipynb`, but that module isn't present even there.)
2. **Sizing is undecided.** Even once a strategy signals "buy this ticker tactically," nothing decides *how much* — `budget_allocator.py`'s Part B currently hardcodes a fixed 10% of tactical NAV per ad-hoc buy ([budget_allocator.py:89](../src/budget_allocator.py:89), flagged `# TODO` in the source itself) rather than sizing off signal strength/confidence.
3. **`TransactionExecutor` doesn't implement the buy side at all.** [executor_module.py](../src/executor_module.py) is 149 lines; it ends right after a `### ADHOC buy` comment with no code following it. Only `AD_HOC_EXIT` (sell/exit) is implemented ([executor_module.py:52-70](../src/executor_module.py:52)). This is a real missing feature, not an integration question — it blocks tactical buys regardless of whether the alpha layer is the old live-loop version or the new [src2/](../src2/) vectorized one.

## Why it can't be precomputed like the core sleeve
Unlike the core sleeve (now vectorized — see [src2/alpha_engine.py](../src2/alpha_engine.py)), a tactical strategy inherently needs live portfolio state: entry price and running high-water-mark since entry (`GlobalState.tactical_peaks`) only exist once a trade has actually executed. There's no way to turn "when do I exit" into a precomputed table independent of realized fills — it has to stay in the sequential execution loop.

## What implementing it requires (in order)
1. A tactical strategy class with `evaluate_exits(state, snapshot) -> StrategyOutput`, called every day inside the loop (not just rebalance days).
2. Its output passed through the existing `SignalFilter` chain.
3. `IntegratedBudgetAllocator`'s Part B (ad-hoc order validation against `tactical_cash`) wired back in — currently unused by [src2/run_vectorized.py](../src2/run_vectorized.py) since nothing produces ad-hoc signals yet.
4. A real position-sizing rule for tactical buys (replacing the hardcoded 10%).
5. `TransactionExecutor`'s missing `AD_HOC_BUY` execution path, mirroring the existing `CORE_BUY` logic (compute shares from allocated cash/price, fee, `ExecutedTrade`, `state.commit_executed_trade`).

Deferred until there's a concrete tactical idea to implement against (trailing-stop exit, fixed take-profit/stop-loss, or something else) — implementing the executor's buy path speculatively, without a real strategy driving it, isn't worth doing yet.
