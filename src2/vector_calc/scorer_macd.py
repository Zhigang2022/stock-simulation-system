"""
Vectorized MACD scorer -- DIAGNOSTIC ONLY. Follows the scorer_mean_reversion.py
contract (price_df in, dict of panels out, one key named "score") so it can
be fed to vector_calc.build_metrics_table for an inspectable
date x ticker table of macd_line/signal_line/score values.

Not used for execution: the actual MACD trading strategy lives in
src2/iteration/strategy_adhoc_macd.py (MacdCrossStrategy /
MacdMultiTimeframeStrategy), run through the TACTICAL sleeve, because it
needs live g_state (already holding it? in-progress bar?) that this
stateless matrix scorer structurally cannot see -- see
dev_notes/01_core_vs_tactical_weight_mismatch.md for why routing MACD
through the CORE sleeve (this scorer + build_target_weight_table) is
broken for a single-ticker on/off signal. Use this file only to eyeball
MACD values over time, e.g.:

    df_metrics = vector_calc.build_metrics_table(
        price_df, price_df.index.tolist(),
        score_fn=vector_calc.macd_cross_score, fast=12, slow=26, signal=9,
    )
"""
import pandas as pd


def macd_cross_score(
    price_df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.DataFrame]:
    """score = MACD histogram (macd_line - signal_line): >0 bullish, <=0 bearish."""
    ema_fast = price_df.ewm(span=fast, adjust=False).mean()
    ema_slow = price_df.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    return {
        "score": macd_line - signal_line,
        "macd_line": macd_line,
        "signal_line": signal_line,
    }
