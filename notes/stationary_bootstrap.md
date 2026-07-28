# Stationary bootstrap — notes

Implementation: [`src/statioinary_bootstrap.py`](../src/statioinary_bootstrap.py), class `VectorizedBootstrapEngine`. Used from [`src2/bootstrap_compare.py`](../src2/bootstrap_compare.py) and [`scorer_comparison_bootstrap.ipynb`](../src2/scorer_comparison_bootstrap.ipynb) to check whether a scorer's edge survives across many alternate histories, not just the one real price path we have. See [`bootstrap_ic_findings.md`](bootstrap_ic_findings.md) for results.

## 1. The problem it solves

We have exactly **one** realized history of prices for our universe (2022-01 → 2026-04, ~1000 trading days). Any backtest metric computed on it — CAGR, Sharpe, mean IC — is a single draw from whatever the "true" data-generating process is. Two ways that number can mislead:

- **Small-sample noise.** `evaluation.ic_significance` in `scorer_comparison.ipynb` already showed that date-to-date IC standard deviation (~0.2) swamps the tiny mean IC (~0.02) with only ~40-50 rebalance dates — not enough samples to trust the sign, let alone the magnitude.
- **Single-path luck.** Even a "significant" result could just be an artifact of *this specific* history — this specific sequence of rallies, crashes, and sector rotations. Would the scorer still look good if 2022's drawdown had landed in month 3 instead of month 1, or if the AI rally in the sample had been milder?

The bootstrap answers: **generate many synthetic-but-realistic alternate histories, rerun the whole pipeline on each, and look at the distribution of outcomes** — not just its one realized value. If a scorer's win rate over the benchmark or its mean IC is only good in the one real history and evaporates across resampled histories, it was never real.

## 2. Why a *stationary* bootstrap, not IID or plain block bootstrap

**Naive fix — resample daily returns IID (with replacement):** wrong, because it destroys the autocorrelation structure of markets. Volatility clusters (calm/turbulent regimes persist for weeks), returns exhibit momentum/reversion over short horizons, and volume trends persist. Shuffling every day independently erases all of that and creates a return series that "looks like" the real one only in its 1-day marginal distribution — useless for testing a scorer whose entire premise is that multi-day patterns (60-day mean reversion, 252-day drift) are informative.

**Better — fixed-length block bootstrap:** resample contiguous *blocks* of, say, 21 days at a time, preserving within-block autocorrelation. Problem: a fixed block length is itself an arbitrary, discontinuous choice — the resampled series has artificial "seams" every exactly-21 days where a regime discontinuity is guaranteed, and the result isn't strictly stationary (statistics depend on where in a block you are).

**Stationary bootstrap (Politis & Romano, 1994):** instead of fixed-length blocks, use **random block lengths drawn from a geometric distribution**. Concretely: walk forward through time either (a) continuing sequentially from the previous resampled day (i.e. `t → t+1` in the original series), or (b) with a small fixed probability `p` at every step, "jump" to a uniformly random new starting point. This produces variable-length runs of contiguous original-history segments, spliced together at random points. Key properties that make this "the" standard tool here:

- **Expected block length = `1/p`.** Setting `p = jump_probability = 1/expected_block_size` gives you direct control over how much local structure survives (this codebase defaults to `expected_block_size=21`, ≈ one trading month, matching monthly rebalancing).
- **No fixed seams.** Block lengths are random (geometric), so there's no single periodicity artifact — statistically, the resampled process is genuinely stationary, and its asymptotic properties (motivating the name) match the original process better than fixed-block schemes when the true DGP is weakly stationary.
- **Same-day cross-sectional coherence, if applied correctly.** Because the jump/continue decision is made once per day and applied to *all tickers and both features (return, volume) simultaneously* (see §3), the resampled world preserves that day's cross-sectional structure — correlated moves across the 29-ticker universe on the same synthetic day stay correlated, rather than each ticker being resampled independently and destroying co-movement. This matters a lot for testing a cross-sectional ranking scorer, whose whole job is to compare tickers *against each other on the same date*.

**What it does *not* fix:** it resamples the *realized* return distribution — it can't invent volatility regimes, correlation structures, or tail events that never happened in the source history. It's better understood as "how much does path-order/timing luck matter, given the return distribution we've already observed" rather than "what if the world had been fundamentally different."

## 3. Code implementation — key points

Reading straight through `VectorizedBootstrapEngine.generate_all_worlds`:

### a. Setup (`__init__`)
- Converts prices → daily returns via `pct_change().dropna()` — this trims the first day. **Consequence downstream:** `dates_index`/`tickers` must be read off the *engine* (`engine.dates_index`), not the original `universe_data`, because the tensor is one day shorter — this bit the bootstrap notebook once (see git history / earlier session note).
- Volume is re-indexed to match the trimmed returns index — return and volume series must be perfectly aligned since they get shuffled together (see step 2 below).
- `jump_probability = 1 / expected_block_size` — the geometric-distribution parameter from §2.

### b. Step 1 — generate the randomized timeline (the actual "stationary bootstrap" mechanism)
```python
jump_trials = np.random.rand(num_simulations, num_days) < jump_probability
sim_indices[:, 0] = np.random.randint(0, num_days, size=num_simulations)
for t in range(1, num_days):
    random_jumps      = np.random.randint(0, num_days, size=num_simulations)
    sequential_steps  = (sim_indices[:, t-1] + 1) % num_days
    sim_indices[:, t] = np.where(jump_trials[:, t], random_jumps, sequential_steps)
```
- This is *only* generating **integer indices into the original history**, one path per simulation — nothing about prices yet. `sim_indices[sim, t]` says "on synthetic day `t` of world `sim`, use the real data from original day `sim_indices[sim, t]`."
- `jump_trials[:, t] == True` → jump to a fresh uniform-random day (block ends, new block starts). Otherwise → continue sequentially from yesterday's chosen day (`+1`, wrapped via `% num_days` so it never runs off the end of history) — this is what keeps a block's *internal* return sequence identical to a contiguous stretch of real history, preserving autocorrelation within a block.
- Fully vectorized across `num_simulations` worlds at once via the `(num_simulations, num_days)` shape — no per-simulation Python loop, only a per-*day* loop (necessary because each day's index depends on the previous day's, i.e. it's an inherently sequential Markov-chain-style construction). This is the "Vectorized" in the class name: vectorized across worlds, sequential across time.

### c. Step 2 — resample returns and volume *together*, per ticker
```python
features_matrix = np.hstack([returns_df.values, volume_df.values])   # (num_days, 2*num_tickers)
shuffled_features_3d = features_matrix[sim_indices]                   # (num_sims, num_days, 2*num_tickers)
```
- Advanced NumPy indexing (`features_matrix[sim_indices]`) applies the *same* per-day index chosen in step 1 to every column at once — i.e. every ticker's return **and** volume on a given synthetic day all come from the *same* original day. This is what preserves the cross-sectional (same-day, cross-ticker) correlation structure discussed in §2, and also keeps each ticker's own return/volume relationship intact (e.g. high-volume days stay paired with their actual return, not some other day's).
- Concatenating returns and volume into one matrix before indexing is purely a vectorization trick — it lets one fancy-index operation resample both feature types identically instead of two separate (and potentially inconsistent) indexing calls.

### d. Step 3 — reconstruct price paths from resampled returns
```python
price_factors = 1.0 + shuffled_returns_3d
price_factors = concat([ones_layer, price_factors], axis=1)   # prepend a day-0 factor of 1.0
synthetic_prices_3d = start_arr * np.cumprod(price_factors, axis=1)
synthetic_prices_3d = synthetic_prices_3d[:, 1:, :]             # drop the day-0 layer
```
- Since the bootstrap operates on **returns** (not prices directly), the synthetic price path must be rebuilt by compounding: `price[t] = start_price * ∏(1 + r[1..t])`. `np.cumprod` along the time axis does this for every world and ticker simultaneously.
- `start_arr` defaults to a flat 100.0 for every ticker/world (`start_prices=None`) — meaning by default all synthetic worlds start every ticker at the same nominal 100, discarding the real starting price level entirely. Only relative moves (returns) are preserved; absolute price level is not calibrated to reality unless `start_prices` is explicitly passed. This is fine for return/metric-based comparisons (CAGR, Sharpe, IC) which are scale-invariant, but would matter if something downstream cared about absolute price (e.g. dollar-based position sizing against a minimum share price).
- The `ones_layer`/prepend-then-drop dance is just to make `cumprod`'s day-0 factor exactly 1.0 (i.e. `price[0] = start_price` exactly) without a separate initial-value special case.

### e. Output
```python
return {'price_tensor': (num_sims, num_days, num_tickers), 'volume_tensor': (same shape)}
```
Consumed by `bootstrap_compare.build_synthetic_universe`/`run_one_simulation`, which slice one `[sim_id]` world at a time into a `{'price', 'volume'}` DataFrame pair shaped exactly like real `universe_data`, so the existing `alpha_engine` → `run_vectorized` pipeline runs on it unmodified — the whole point of doing this at the tensor level rather than reimplementing the backtest loop per synthetic world.

## 4. Practical gotchas (learned from using it in this repo)

- **Off-by-one day:** `dates_index`/`tickers`/`rebalance_dates` must be sourced from `engine.dates_index`/`engine.tickers`, filtered against the *engine's* index — not the raw `universe_data` — because `pct_change().dropna()` trims day 1.
- **Benchmarks need bootstrapping too.** An earlier version of `scorer_comparison_bootstrap.ipynb` bootstrapped only the 29-ticker universe and left SPY/QQQ as fixed real data across all synthetic worlds — that's an apples-to-oranges comparison (scorer performance varies by world, benchmark doesn't). Fix: include `'SPY', 'QQQ'` in the ticker list passed into `VectorizedBootstrapEngine` so they get resampled through the identical synthetic-index mechanism as everything else, then split them back out per-world in `run_one_simulation` (`bench_tickers` param) rather than reading a fixed `bench_price` DataFrame.
- **Paired, not independent, comparisons.** Because scorer and benchmark share the same `sim_id` (same synthetic world), comparing them across simulations is a paired-samples problem — use a paired test (paired t-test / Wilcoxon signed-rank on the differences), not an unpaired one. See [`src/stat_test.py`](../src/stat_test.py) and [`src2/bootstrap_significance.py`](../src2/bootstrap_significance.py).
- **Cost:** the per-world cost is dominated by the *downstream* sequential day-by-day execution loop (`run_vectorized`), not the bootstrap generation itself (which is a couple of vectorized NumPy ops). With `NUM_SIMULATIONS` in the hundreds × several scorer configs, this is naturally embarrassingly parallel across `sim_id` — see the `ProcessPoolExecutor` usage in the notebook.

## 5. Regime classification — conditioning results on what kind of world it was

Pooling all `NUM_SIMULATIONS` worlds into one win-rate/mean-diff number (as in [`bootstrap_significance.py`](../src2/bootstrap_significance.py)) answers "does this scorer beat the benchmark on average, across all kinds of histories" — but averages away the far more useful question "in *which* kinds of histories." A mean-reversion scorer that only wins in choppy/sideways markets and a momentum scorer that only wins in trending markets can both average out to a mediocre pooled win rate, while each is actually a strong conditional edge. Conditioning on the realized regime per world recovers that.

### Why not just bucket on total return

The obvious first idea — bucket each world by its benchmark's net return (up/down/flat) — conflates two different things: **net direction** and **path smoothness**. A world that rallied +40% then fully round-tripped back to flat nets ~0% return, same bucket as a world that just sat dead flat the whole time — but those are very different regimes for a scorer to operate in (the first is intensely trendy, just not net-directional by the end; the second has no exploitable trend at all). Classifying only by net return would mislabel the first as "flat" alongside the second.

### The two-axis approach used here (`bootstrap_compare.compute_trend_features` / `classify_world_regime` / `label_regimes`)

1. **Fit `log(price) ~ t`** (OLS, `np.polyfit`) on the benchmark's (SPY or QQQ — pick whichever you consider closer to "the market" for your purposes) price series for that world, over the same post-warmup window the scorer metrics were computed on.
2. **`slope`** — sign gives direction, independent of how well the path fits a line.
3. **`r2`** — how much of the log-price path's variance the straight-line trend explains. High R² (default threshold 0.7) = smooth, persistent trend in one direction. Low R² = the price wandered/oscillated without a persistent direction, regardless of where it net landed — this is the "choppy/consolidation" bucket, and it catches the round-trip case above (low R² even though a naive return-bucket might have called it "flat").
4. **`total_return`** — still computed and kept, used only for the *direction* label (bull/flat/bear), and only among worlds that already passed the R² trending threshold — bucketed by **tercile across the worlds actually being compared**, not a fixed cutoff, so "what counts as a bull world" adapts to the actual spread of outcomes this bootstrap produced rather than an arbitrary hardcoded number.
5. Final label: `bull_trend` / `bear_trend` (high R², slope's sign) or `choppy` (low R², either sign).

### Why this has to be two passes, not computed inside `run_one_simulation`

`run_one_simulation` (the `ProcessPoolExecutor` worker) computes each world in isolation — it has no visibility into the other 999 worlds, so it cannot compute a population-relative tercile by itself. The regime pipeline is deliberately split:

- **Pass 1 — per world, embarrassingly parallel-able (but cheap enough to just loop):** `classify_world_regime` extracts the benchmark's raw resampled price column directly from `price_tensor[sim_id]` and runs `compute_trend_features` on it. This needs **no backtest at all** — it's a plain OLS fit on a price series, not the scorer pipeline — so unlike `run_one_simulation` it doesn't need the process pool; a sequential Python loop over `NUM_SIMULATIONS` is fast enough on its own.
- **Pass 2 — across all worlds at once:** `label_regimes` takes the full collection of pass-1 rows and applies the population tercile cutoffs, producing the final `regime` label per `sim_id`.
- **Join:** merge `label_regimes`'s output onto `bootstrap_metrics` by `sim_id`, then group/rerun the paired significance tests (`bootstrap_significance.compare_all`) **within each `regime` value** instead of pooled.

**Consistency gotcha:** `classify_world_regime` must be called with the same `warmup_days` used for `run_one_simulation`/`build_world_nav` — otherwise a world's regime label could be based on a stretch of price history (e.g. the warmup period) that the scorer metrics never actually traded through, decoupling the label from what's being explained.
