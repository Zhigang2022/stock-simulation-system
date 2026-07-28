# The Macro / Meso / Micro framework applied to this project

This project was deliberately developed using a 3-layer thinking method: **Macro (orchestration graph) → Meso (modules & data interfaces) → Micro (concrete function implementations)**. The idea: it's easy to get stuck at Meso/Micro level (staring at one function or one file) without ever building a Macro-level mental model of the whole system — this framework exists to force that top-down anchor first.

## The three layers, as applied here

### Macro — the orchestration graph
The end-to-end pipeline and its control flow: what stages exist, in what order, and how control/data passes between them. This is what [00_overview.md](00_overview.md) documents:
```
DataBroker → CalendarIterator → Selector(+Scorer) → Filter(s) → BudgetAllocator → TransactionExecutor → GlobalState
```
Macro-level questions: "what are the stages," "what's the lifecycle of one backtest run," "why does the pipeline look like this." Not concerned with any single function's internals.

### Meso — modules & data interfaces
The concrete module boundaries and the data contracts that make the Macro graph real:
- The ABC base classes (`BaseSelector`, `MomentumScorer`, `SignalFilter`) — these define *what* a module must expose, decoupling stages from each other.
- `src/signal_schema.py` (`SignalPayload`, `StrategyOutput`, `StrategyTelemetry`) — the shared vocabulary every stage communicates through.
- `src/audit.py` (`AuditRecord`) — the data contract for the audit trail.

Meso-level questions: "what does a Selector need to return so the Filter can consume it," "what's the schema of a signal." This is the layer that makes stages swappable without touching the run loop — the reason for the `main`-branch "restructure" from a flat `strategy_selector.py` into `src/selectors/`, `src/scorers/`, `src/filters/` packages.

### Micro — concrete function implementations
The actual math/logic inside one selector, one scorer, one filter — e.g. how `MeanReversionScorer.compute_score()` computes a z-score, or how `TrendAccelerationDiagnostic` computes slope/acceleration. This is what the `select_*.md` notes in this folder document one file at a time.

## Why this matters practically
When adding a new strategy idea, work top-down: first place it on the Macro graph (which stage does this belong to — a new Selector? a new Filter?), then check it fits the Meso contract (does it return a `StrategyOutput`/consume a `snapshot` the same way siblings do), and only then write the Micro-level math. This avoids ending up with one-off code that doesn't compose with the rest of the pipeline.

## Note on the source material
The Obsidian export (`notes/量化股市/`) that this framework was originally drafted alongside has since been moved out of this project by the user (2026-07-23) as not quite relevant here. It mixed notes on this trading system with an unrelated side project ("task_plan"); that distinction no longer matters since the export isn't in this repo anymore.
