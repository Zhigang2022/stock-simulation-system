"""
Generates and characterizes SYNTHETIC price/volume worlds -- the "data"
side of bootstrap significance testing, separate from actually running a
backtest on that data (simulation.py) or testing the results (significance.py).

VectorizedBootstrapEngine is forked from src/statioinary_bootstrap.py (not
imported): block-resamples real daily returns+volume via a "stationary
bootstrap" jump/continue mechanism, preserving autocorrelation structure,
to produce many synthetic-but-realistic histories over the SAME
dates/tickers as the real data. Dates/tickers are unchanged (only the
return/volume VALUES are resampled), so a single rebalance_dates list
computed against the real data applies directly to every synthetic world.

This is a bootstrap-specific statistical technique, not a general data
concern (unlike src2/data_ingest/, which acquires real market data) --
nobody outside significance testing needs stationary-bootstrap resampling,
which is why it's forked here rather than into data_ingest/.
"""
import numpy as np
import pandas as pd


class VectorizedBootstrapEngine:
    def __init__(self, historical_data_dict: dict[str, pd.DataFrame], expected_block_size=21):
        """
        historical_data_dict: dict like {'price': pd.DataFrame, 'volume': pd.DataFrame}
        expected_block_size: Average trend length in days before a timeline jump
        """
        self.prices_df = historical_data_dict['price']
        self.volume_df = historical_data_dict['volume']

        self.returns_df = self.prices_df.pct_change().dropna()
        assert (self.prices_df.shape[0] - self.returns_df.shape[0]) < 30

        self.volume_df = self.volume_df.loc[self.returns_df.index]

        self.tickers = self.returns_df.columns
        self.dates_index = self.returns_df.index
        self.num_days, self.num_tickers = self.returns_df.shape

        self.jump_probability = 1.0 / expected_block_size

    def generate_all_worlds(self, num_simulations: int = 1000, start_prices=None):
        """
        Instantly generates all synthetic paths.
        Returns {'price_tensor': ndarray, 'volume_tensor': ndarray}, both
        shaped (num_simulations, num_days, num_tickers).
        """
        # 1. Generate the randomized timeline indices (Stationary Bootstrap)
        jump_trials = np.random.rand(num_simulations, self.num_days) < self.jump_probability
        sim_indices = np.zeros((num_simulations, self.num_days), dtype=int)
        sim_indices[:, 0] = np.random.randint(0, self.num_days, size=num_simulations)

        for t in range(1, self.num_days):
            random_jumps = np.random.randint(0, self.num_days, size=num_simulations)
            sequential_steps = (sim_indices[:, t - 1] + 1) % self.num_days
            sim_indices[:, t] = np.where(jump_trials[:, t], random_jumps, sequential_steps)

        # 2. Combine Returns and Volume horizontally to shuffle them together
        features_matrix = np.hstack([self.returns_df.values, self.volume_df.values])
        shuffled_features_3d = features_matrix[sim_indices]

        shuffled_returns_3d = shuffled_features_3d[:, :, :self.num_tickers]
        shuffled_volume_3d = shuffled_features_3d[:, :, self.num_tickers:]

        # 3. Reconstruct the price paths from returns
        if start_prices is None:
            start_arr = np.ones((num_simulations, 1, self.num_tickers)) * 100.0
        else:
            base = np.array([start_prices[t] for t in self.tickers])
            start_arr = np.tile(base, (num_simulations, 1, 1))

        price_factors = 1.0 + shuffled_returns_3d
        ones_layer = np.ones((num_simulations, 1, self.num_tickers))
        price_factors = np.concatenate([ones_layer, price_factors], axis=1)

        synthetic_prices_3d = start_arr * np.cumprod(price_factors, axis=1)
        synthetic_prices_3d = synthetic_prices_3d[:, 1:, :]  # Drop day 0 layer

        return {
            'price_tensor': synthetic_prices_3d,
            'volume_tensor': shuffled_volume_3d
        }


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


def compute_trend_features(price_series: pd.Series, warmup_days: int = 0) -> dict:
    """
    Trend-vs-chop features for one price series, via OLS of log(price) on
    time (day index). See notes/stationary_bootstrap.md "Regime
    classification" for why log-price/R² rather than raw return:

    - total_return: net direction/magnitude over the window.
    - slope: OLS slope of log(price) ~ t -- sign = direction, independent of R².
    - r2: how much of the path's variance the straight-line trend explains.
      High r2 = smooth, persistent trend (up OR down). Low r2 = price
      wandered/round-tripped without a persistent direction -- chop/
      consolidation/mean-reversion regime -- REGARDLESS of net total_return
      (a world that rallied 40% then gave it all back nets ~0% return but
      is NOT the same regime as a world that just sat flat the whole time).

    warmup_days: leading observations dropped before fitting -- pass the
    SAME value used in simulation.run_one_simulation so a world's regime
    label is assessed over the identical window the scorer metrics were
    actually computed on, not a stretch of history the metrics never saw.
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
    than through the ProcessPoolExecutor used for simulation.run_one_simulation.

    Use the SAME warmup_days you passed to run_one_simulation so the regime
    is assessed over the identical window the scorer metrics were computed
    on, before merging label_regimes' output onto bootstrap_metrics by
    sim_id -- otherwise a world could be labeled by a stretch of price
    history the metrics never actually saw.
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
    into simulation.build_world_nav/evaluation.plot_rebased_performance
    without writing the world_regimes[world_regimes['regime'] == ...]['sim_id']
    filter by hand each time.

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
