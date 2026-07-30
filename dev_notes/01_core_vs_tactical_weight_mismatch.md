# CORE (regular) vs TACTICAL (ad-hoc) sleeve: how selection actually works, and a real bug found building the QQQ MACD strategy (2026-07-28)

## How CORE selection works, end to end

`vector_calc`'s `rank_and_weight_top_percent` ([src2/vector_calc/selector_top_percent.py](../src2/vector_calc/selector_top_percent.py))
computes, once per rebalance date, for **every eligible ticker**:
- `rank`: 1, 2, 3, … N, best score first (NaN if filtered out). This is an
  audit/diagnostic column, not itself the selection test.
- `target_weight`: derived from `rank` — `cutoff_rank` is the rank of the
  `top_percent`-th name; any row with `rank <= cutoff_rank` gets a nonzero
  weight (equal or score-weighted), everything else gets `0.0`.

Inside the loop, `allocation_regular.naive_precomputed_weight`
([src2/iteration/allocation_regular.py](../src2/iteration/allocation_regular.py))
never looks at `rank` — it only reads `target_weight`, filters to
`> 0`, and hands that `{ticker: weight}` dict to the executor. So: `rank`
is kept for inspection, `target_weight` (derived from a `rank <= cutoff`
test, not `rank > 0`) is the only thing execution actually consumes.

This whole table is precomputable because it only needs the price panel —
no live cash/positions — so it's built once, before `run_daily_iteration`
even starts.

## How TACTICAL is different in kind, not just in code path

TACTICAL strategies (e.g. `MacdCrossStrategy`,
[src2/iteration/adhoc_macd_strategy.py](../src2/iteration/adhoc_macd_strategy.py))
aren't a selector at all — there's no ranked universe, no `score`/`rank`/
`target_weight` table. It's a live yes/no decision re-evaluated every
single day inside the loop, because it needs `g_state.tactical_positions`
(am I already holding this?) to decide BUY vs EXIT — state the precomputed
`vector_calc` stage structurally cannot see.

The real axis isn't "selector vs. strategy" — it's **precomputable vs.
path-dependent**:

| | CORE (regular) | TACTICAL (ad-hoc) |
|---|---|---|
| Decided | Once, up front, in `vector_calc` | Live, every day, in the loop |
| Needs live state? | No | Yes (checks current position) |
| Frequency | Rebalance dates only | Every day |
| Output | `target_weight` table | One `BUY`/`EXIT` signal or nothing |
| Sizing | Precomputed (naive), unless overridden | Always live (`allocation_tactical.py`'s `sizing_fn`) |

## Real bug found: CORE's `naive_precomputed_weight` cannot represent "flatten to zero" for a single-ticker strategy

First attempt at MACD wrongly modeled it as a CORE-sleeve scorer (score =
MACD histogram, filter = `score > 0`, daily rebalance). This surfaced a
genuine bug, not just a design mismatch:

`naive_precomputed_weight` does `.loc[lambda s: s > 0]` before returning
the target dict — rows with weight 0 are dropped entirely, not included as
`{"TICKER": 0.0}`. `Future_Transaction.set_core`
([src2/states/global_state.py](../src2/states/global_state.py)) then does:
```python
if len(core_target_weights) != 0:
    self.core_target_weights = core_target_weights
    self.core_target_date = core_target_date
```
An **empty** dict is treated as "no signal today, keep whatever target was
last set" — not "sell everything." So a single-ticker on/off strategy
routed through CORE would buy once on the first bull cross and then
**never sell**, silently degenerating into buy-and-hold. Confirmed
empirically: an 8-year QQQ backtest logged exactly 2 fills total instead
of the ~250 expected from real MACD crosses.

This is invisible for CORE's actual use case (sparse multi-name
rebalances) because a dropped name is implicitly zero *relative to
survivors* — the weights of the names that stay in are what change, so
`core_target_weights` is never empty as long as anything is still held.
It only breaks down for a single-ticker, fully-in/fully-out signal, which
is exactly what a tactical strategy is for. This is *why* MACD belongs in
`allocation_tactical.py`, not just a stylistic preference — routing it
through CORE is actively broken today, not merely "off-pattern."

## What to change later

Worth revisiting once there's a second single-ticker/on-off CORE use case
(not just MACD) to justify it:
- Either make `naive_precomputed_weight` keep zero-weight rows by default
  (a `full_precomputed_weight` variant already exists in git history —
  see the reverted commit around 2026-07-28 — but was pulled back out
  since MACD moved to TACTICAL instead), or
- Make `Future_Transaction.set_core` distinguish "no signal, keep prior
  target" from "explicit target of zero everything" (e.g. `None` vs `{}`
  vs a nonempty dict), so an empty *selection* can still mean "flatten."

Until then: **CORE sleeve is not safe to use for any strategy that needs
to represent "hold 0% of everything" as a real state change** — only use
it for cross-sectional multi-name rebalances where the selected set never
goes fully empty.
