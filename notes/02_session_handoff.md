# Session handoff (2026-07-23)

State of things after this session, for picking up in a fresh conversation without the messy context.

## Environment
- Project was moved/renamed (`Project Z` → `stock_simulation_system`). `.venv` was stale (baked-in old path) — recreated in place via `uv venv` + reinstall from a frozen package list. Verified working.
- [requirements.txt](../requirements.txt) created from that freeze (117 pinned packages) so the env is reproducible going forward: `uv pip install -r requirements.txt` into a fresh `.venv`.
- `.git` was never broken — repo location-independence is automatic; remote is `https://github.com/Zhigang2022/Project-Z.git`.

## Git branches
Two local branches, diverged after a common ancestor:
- `main` (current) — has the "restructure": `src/filters/`, `src/scorers/`, `src/selectors/` packages. This is the active/newer architecture.
- `general` — older flat structure (`src/strategy_selector.py`, `src/compliance_filters.py`, since deleted on `main`). Tracks `origin/general`.
- Common ancestor: commit `652abc5 "July 5th"`. A `main`↔`general` merge was discussed but **not attempted** — likely to hit real conflicts given the restructure. Revisit with a diff/merge-preview before actually merging.

## Bugs found & status
- **Fixed**: [src/executor_module.py:67](../src/executor_module.py:67) — referenced undefined `ticker` instead of `signal.ticker` in a log line (would `NameError` on any executed ad-hoc exit).
- **Not fixed** (flagged, not currently used in the live pipeline so low urgency): [src/filters/liquidfilter.py:4](../src/filters/liquidfilter.py:4) — `from filters.base_filter import SignalFilter` is missing the `src.` prefix every sibling file uses; importing this module currently raises `ModuleNotFoundError`.

## Notes written this session (in `notes/`)
- [00_overview.md](00_overview.md) — Macro-level pipeline map: `DataBroker → CalendarIterator → Selector(+Scorer) → Filter(s) → BudgetAllocator → TransactionExecutor → GlobalState`. Start here.
- [01_macro_meso_micro.md](01_macro_meso_micro.md) — the Macro/Meso/Micro framework this project follows, and a map of which parts of the Obsidian export (`notes/量化股市/`) are actually about this trading system vs. an unrelated side project ("task_plan", a codebase-navigator LLM tool) that happens to reuse the same layering vocabulary.
- 10 `select_*.md` files — Micro-level deep dives, one per file, covering every selector/scorer/filter implementation (`self_selectors`, `cross_selectors`, `moment_scorers`, `mean_reversion_scorers`, `abosolution_momentum_filter`, `trend_acceleration_filter`, `liquidfilter`, plus the three `base_*` ABCs).

## Open TODOs (not started / explicitly deferred)
1. Decide whether/how to merge `general` into `main` (or vice versa) — no diff/conflict preview has been run yet.

## Resolved since last session (2026-07-23)
- `liquidfilter.py` import bug fixed: [src/filters/liquidfilter.py:4](../src/filters/liquidfilter.py:4) now imports `from src.filters.base_filter import SignalFilter`; verified it loads under `.venv`.
- `notes/量化股市/` was moved out of the project by the user — not relevant here. The TODO to mine it for original macro/meso/micro definitions is dropped; [01_macro_meso_micro.md](01_macro_meso_micro.md) updated accordingly.
- Dropped the TODO to write Micro-level notes for non-selector modules (`global_state.py`, `run.py`, `budget_allocator.py`, etc.) — not doing that.
