"""
Runs the actual backtest (vector_calc -> iteration) on synthetic worlds
built by data_gen.py, and collects per-world metrics (evaluation.calculate_metrics)
-- the execution step, not data generation or significance testing.
"""
import logging

import pandas as pd

from src2 import vector_calc, evaluation, iteration

# iteration/trade_implement.py's "backtest_logger" logs one CORE_BUY/CORE_SELL
# line per trade at INFO. Each ProcessPoolExecutor worker re-imports it fresh,
# so across hundreds of simulations this floods notebook output. Raise the
# level here so it's silenced in every worker at import time.
logging.getLogger("backtest_logger").setLevel(logging.WARNING)


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
    evaluation.plot_rebased_performance(build_world_nav(sim_id=0, ...)))
    without paying for evaluation.calculate_metrics or a process pool. Default
    warmup_days=0 (unlike run_one_simulation's 252) since a visual peek at
    one world doesn't need the metrics-bias trimming -- pass warmup_days
    explicitly if you want the plot to match what run_one_simulation scored.
    """
    world_data = {
        "price": pd.DataFrame(price_slice, index=dates_index, columns=tickers),
        "volume": pd.DataFrame(volume_slice, index=dates_index, columns=tickers),
    }
    participate_price = world_data["price"].drop(bench_tickers, axis=1)

    bench_aligned = world_data['price'][bench_tickers]
    df_nav_list = [bench_aligned]
    for name, cfg in scorer_configs.items():
        cfg = dict(cfg)
        score_fn = cfg.pop("score_fn")
        _, df_ranked = vector_calc.build_target_weight_table(
            participate_price, rebalance_dates, score_fn=score_fn,
            top_percent=top_percent, allocation_type=allocation_type, **cfg,
        )
        g_state, _ = iteration.run_daily_iteration(df_ranked, world_data, rebalance_dates)
        df_nav = g_state.export_nav_dataframe()[["Total_NAV"]].rename(columns={"Total_NAV": name})
        df_nav_list.append(df_nav)

    df_nav_all = pd.concat(df_nav_list, axis=1).dropna(how="all").ffill()

    if warmup_days > 0:
        df_nav_all = df_nav_all.iloc[warmup_days:]

    return df_nav_all


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
    One bootstrap world, start to finish: build_world_nav + evaluation.calculate_metrics
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

    df_sim_metrics = evaluation.calculate_metrics(df_nav_all)
    df_sim_metrics.index.name = "scorer"
    df_sim_metrics = df_sim_metrics.reset_index()
    df_sim_metrics.insert(0, "sim_id", sim_id)
    return df_sim_metrics
