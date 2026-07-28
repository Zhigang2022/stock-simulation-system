"""
Uses the existing stationary-bootstrap engine (src/statioinary_bootstrap.py)
to test whether a scorer's IC is a real effect or an artifact of the one
historical price path we happen to have. Generates many synthetic-but-
realistic price/volume histories (same dates/tickers, block-resampled
daily returns+volume, preserving autocorrelation structure via the
"stationary bootstrap" jump/continue mechanism) and computes the same
score_fn -> IC -> mean_IC on each, giving a DISTRIBUTION of mean_IC across
independent worlds.

This complements evaluation.ic_significance, which only tests date-to-date
noise WITHIN the single real history: a strategy could look consistent
across dates in our one history just by luck of which history we got.
Bootstrapping resamples the history itself, a different and larger source
of uncertainty.

Dates/tickers are unchanged by the bootstrap (only the return/volume
VALUES are block-resampled) so the same rebalance_dates list computed
against the real data applies directly to every synthetic world -- no
need to recompute per simulation.
"""
import logging
import numpy as np
import pandas as pd

from src import metrics
from src2 import alpha_engine, evaluation, run_vectorized

# executor_module's "backtest_logger" logs one CORE_BUY/CORE_SELL line per
# trade at INFO to both console and backtest.log. Each ProcessPoolExecutor
# worker re-imports it fresh (its own log file, its own inherited stdout),
# so across hundreds of simulations this floods notebook output. Raise the
# level here so it's silenced in every worker at import time.
logging.getLogger("backtest_logger").setLevel(logging.WARNING)


def build_synthetic_universe(
    bootstrap_result: dict,
    sim_id: int,
    dates_index: pd.DatetimeIndex,
    tickers,
) -> dict[str, pd.DataFrame]:
    """One synthetic {'price', 'volume'} universe_data dict from a bootstrap draw."""
    price = pd.DataFrame(bootstrap_result["price_tensor"][sim_id], index=dates_index, columns=tickers)
    volume = pd.DataFrame(bootstrap_result["volume_tensor"][sim_id], index=dates_index, columns=tickers)
    return {"price": price, "volume": volume}


def build_world_nav(
    sim_id,
    price_slice,
    volume_slice,
    dates_index: pd.DatetimeIndex,
    tickers,
    rebalance_dates: list[pd.Timestamp],
    scorer_configs: dict,
    bench_tickers: list[str],
    top_percent: float = 0.25,
    allocation_type: str = "equal",
    warmup_days: int = 0,
) -> pd.DataFrame:
    """
    One bootstrap world's df_nav_all (benchmark + every scorer's NAV column,
    same shape as scorer_comparison.ipynb's df_nav_all) -- the part of
    run_one_simulation that runs the actual backtest, split out so it can be
    called standalone for a single world (e.g. to eyeball
    analysis.plot_rebased_performance(build_world_nav(sim_id=0, ...)))
    without paying for metrics.calculate_metrics or a process pool. Default
    warmup_days=0 (unlike run_one_simulation's 252) since a visual peek at
    one world doesn't need the metrics-bias trimming -- pass warmup_days
    explicitly if you want the plot to match what run_one_simulation scored.
    """
    world_data = {
        "price": pd.DataFrame(price_slice, index=dates_index, columns=tickers),
        "volume": pd.DataFrame(volume_slice, index=dates_index, columns=tickers),
    }
    participate_price = world_data["price"].drop(bench_tickers,axis=1)

    bench_aligned=world_data['price'][bench_tickers]
    df_nav_list = [bench_aligned]
    for name, cfg in scorer_configs.items():
        cfg = dict(cfg)
        score_fn = cfg.pop("score_fn")
        _, df_ranked = alpha_engine.build_target_weight_table(
            participate_price, rebalance_dates, score_fn=score_fn,
            top_percent=top_percent, allocation_type=allocation_type, **cfg,
        )
        g_state, _ = run_vectorized.run_vectorized(df_ranked, world_data, rebalance_dates)
        df_nav = g_state.export_nav_dataframe()[["Total_NAV"]].rename(columns={"Total_NAV": name})
        df_nav_list.append(df_nav)

    df_nav_all = pd.concat(df_nav_list, axis=1).dropna(how="all").ffill()

    if warmup_days > 0:
        df_nav_all = df_nav_all.iloc[warmup_days:]

    return df_nav_all


def compute_trend_features(price_series: pd.Series, warmup_days: int = 0) -> dict:
    """
    Trend-vs-chop features for one price series, via OLS of log(price) on
    time (day index). See notes/stationary_bootstrap.md "Regime
    classification" for why log-price/R² rather than raw return:

    - total_return: net direction/magnitude over the window.
    - slope: OLS slope of log(price) ~ t -- sign = direction, independent
      of R².
    - r2: how much of the path's variance the straight-line trend explains.
      High r2 = smooth, persistent trend (up OR down). Low r2 = price
      wandered/round-tripped without a persistent direction -- chop/
      consolidation/mean-reversion regime -- REGARDLESS of net total_return
      (a world that rallied 40% then gave it all back nets ~0% return but
      is NOT the same regime as a world that just sat flat the whole time).

    warmup_days: leading observations dropped before fitting, same idea as
    build_world_nav/run_one_simulation's warmup_days -- pass the SAME value
    used there so a world's regime label is assessed over the identical
    window the scorer metrics were actually computed on, not a stretch of
    history (e.g. the scorer's own lookback warmup) the metrics never saw.
    """
    price_series = pd.Series(price_series)
    if warmup_days > 0:
        price_series = price_series.iloc[warmup_days:]

    price = np.asarray(price_series, dtype=float)
    if len(price) < 2 or (price <= 0).any():
        raise ValueError("price_series must have >=2 strictly positive observations")

    log_price = np.log(price)
    t = np.arange(len(log_price))

    slope, intercept = np.polyfit(t, log_price, 1)
    fitted = slope * t + intercept
    ss_res = np.sum((log_price - fitted) ** 2)
    ss_tot = np.sum((log_price - log_price.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "total_return": price[-1] / price[0] - 1.0,
        "slope": slope,
        "r2": r2,
    }


def classify_world_regime(
    sim_id,
    price_slice,
    dates_index: pd.DatetimeIndex,
    tickers,
    regime_ticker: str = "SPY",
    warmup_days: int = 0,
) -> dict:
    """
    compute_trend_features on one bootstrap world's benchmark price column
    (no backtest needed -- just the raw resampled price series). Cheap
    enough to run sequentially for every sim_id in a plain for-loop rather
    than through the ProcessPoolExecutor used for run_one_simulation.

    Use the SAME warmup_days you passed to run_one_simulation so the regime
    is assessed over the identical window the scorer metrics were computed
    on, before merging label_regimes' output onto bootstrap_metrics by
    sim_id -- otherwise a world could be labeled by a stretch of price
    history the metrics never actually saw. (Trimming itself happens inside
    compute_trend_features, which takes the same warmup_days parameter.)
    """
    price_df = pd.DataFrame(price_slice, index=dates_index, columns=tickers)
    bench_series = price_df[regime_ticker]

    features = compute_trend_features(bench_series, warmup_days=warmup_days)
    features["sim_id"] = sim_id
    features["regime_ticker"] = regime_ticker
    return features


def label_regimes(trend_features: pd.DataFrame, r2_threshold: float = 0.7) -> pd.DataFrame:
    """
    Second pass: bucket a DataFrame of per-world compute_trend_features rows
    (columns: sim_id, total_return, slope, r2, ...) into regime labels.
    Direction (bull/flat/bear) is by TERCILE of total_return across the
    worlds actually passed in -- relative to this bootstrap's own outcome
    distribution, not an arbitrary fixed cutoff -- so it stays meaningful
    regardless of NUM_SIMULATIONS or which universe/dates this is run on.
    Trendiness is r2 >= r2_threshold (default 0.7, i.e. the linear trend
    explains >=70% of the log-price path's variance).
    """
    df = trend_features.copy()
    df["direction"] = pd.qcut(df["total_return"], q=3, labels=["bear", "flat", "bull"], duplicates="drop")
    df["trending"] = df["r2"] >= r2_threshold

    def _label(row):
        if not row["trending"]:
            return "choppy"
        return "bull_trend" if row["slope"] > 0 else "bear_trend"

    df["regime"] = df.apply(_label, axis=1)
    return df


def sample_sim_ids_per_regime(
    world_regimes: pd.DataFrame,
    n_per_regime: int = 1,
    extreme: bool = True,
    random_state=None,
) -> dict:
    """
    One (or n_per_regime) representative sim_id per regime, from
    label_regimes' output -- for picking a peek_sim_id per regime to feed
    into build_world_nav/analysis.plot_rebased_performance without writing
    the world_regimes[world_regimes['regime'] == ...]['sim_id'] filter by
    hand each time.

    extreme=True (default): pick the sim_id(s) furthest from the tercile
    boundary within each regime -- the most bull-ish of the bull_trend
    worlds, most bear-ish of the bear_trend worlds, lowest-r2 (most
    directionless) of the choppy worlds -- so the peek is a clean example of
    the regime rather than a borderline case near a bucket edge.
    extreme=False: uniform random sample instead (uses random_state).

    Returns {regime_label: [sim_id, ...]}.
    """
    result = {}
    for regime, group in world_regimes.groupby("regime"):
        if extreme:
            if regime == "bull_trend":
                picked = group.nlargest(n_per_regime, "total_return")
            elif regime == "bear_trend":
                picked = group.nsmallest(n_per_regime, "total_return")
            else:  # choppy -- most directionless, i.e. lowest r2
                picked = group.nsmallest(n_per_regime, "r2")
        else:
            picked = group.sample(n=min(n_per_regime, len(group)), random_state=random_state)
        result[regime] = picked["sim_id"].tolist()
    return result


def run_one_simulation(
    sim_id,
    price_slice,
    volume_slice,
    dates_index: pd.DatetimeIndex,
    tickers,
    rebalance_dates: list[pd.Timestamp],
    scorer_configs: dict,
    bench_tickers: list[str],
    top_percent: float = 0.25,
    allocation_type: str = "equal",
    warmup_days: int = 252,
) -> pd.DataFrame:
    """
    One bootstrap world, start to finish: build_world_nav + metrics.calculate_metrics
    on the result. Designed to be called from a ProcessPoolExecutor with one
    call per sim_id -- everything it touches (price/volume slices,
    scorer_configs, bench_tickers) is passed in rather than captured from a
    notebook global, so it's picklable/importable as a top-level function.

    warmup_days: number of leading NAV days dropped before computing metrics.
    The longest-window scorer (e.g. geometric_drift_252d needs 252 days of
    trailing price history) produces NaN/degenerate scores until its rolling
    window fills up, so early rebalances in that window hold no position
    (or a partially-informed one) -- including that period in CAGR/Sharpe/etc.
    would bias every scorer's world toward however much of that no-signal
    "wait period" it happened to sit in. Default 252 covers every scorer
    config currently in this repo; raise it if you add a scorer with a
    longer lookback window. Since this is a bootstrap draw of a long (1063+
    day) synthetic history, trimming more than strictly necessary costs
    only a bit of sample size, not validity -- err on the side of cutting
    too much rather than too little.
    """
    df_nav_all = build_world_nav(
        sim_id, price_slice, volume_slice, dates_index, tickers, rebalance_dates,
        scorer_configs, bench_tickers, top_percent=top_percent,
        allocation_type=allocation_type, warmup_days=warmup_days,
    )

    df_sim_metrics = metrics.calculate_metrics(df_nav_all)
    df_sim_metrics.index.name = "scorer"
    df_sim_metrics = df_sim_metrics.reset_index()
    df_sim_metrics.insert(0, "sim_id", sim_id)
    return df_sim_metrics


