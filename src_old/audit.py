from dataclasses import dataclass,field
from src_old.signal_schema import StrategyTelemetry
import pandas as pd

@dataclass
class AuditRecord:
    """
    A single point-in-time snapshot containing all telemetry, 
    pre/post filtered signals, and resulting allocations.
    """
    date: pd.Timestamp
    telemetry: StrategyTelemetry

    raw_regular_signals: list = field(default_factory=list)
    filtered_regular_signals: list = field(default_factory=list)
    raw_adhoc_signals: list = field(default_factory=list)
    filtered_adhoc_signals: list = field(default_factory=list)

    target_weights: dict[str, float] = field(default_factory=dict)
    executed_trades: list = field(default_factory=list)

def analyze_audit_trail(ledger: list[AuditRecord], metric_keys: list[str] = None):
    """
    Flattens complex mathematical telemetry down into analytical DataFrames 
    to easily cross-reference 'Why did we buy X?'
    """
    if metric_keys is None:
        # Dynamically discover all metric keys present in the ledger's telemetry
        discovered_keys = set()
        for record in ledger:
            if hasattr(record.telemetry, "metrics") and record.telemetry.metrics:
                discovered_keys.update(record.telemetry.metrics.keys())
        metric_keys = sorted(list(discovered_keys))

    rows = []
    for record in ledger:
        date = record.date
        
        # Extract the metrics for each specified key
        extracted_metrics = {}
        for key in metric_keys:
            extracted_metrics[key] = record.telemetry.metrics.get(key, {}) if hasattr(record.telemetry, "metrics") else {}
            
        # Tickers are the union of keys across the extracted metrics and target weights for this record
        tickers = set(record.target_weights.keys()) if hasattr(record, "target_weights") else set()
        for m_dict in extracted_metrics.values():
            tickers.update(m_dict.keys())
            
        for ticker in sorted(tickers):
            row = {
                "date": date,
                "ticker": ticker,
            }
            for key in metric_keys:
                row[key] = extracted_metrics[key].get(ticker, None)
            row["final_weight"] = record.target_weights.get(ticker, 0.0) if hasattr(record, "target_weights") else 0.0
            rows.append(row)
            
    df_analysis = pd.DataFrame(rows)
    return df_analysis


def recompute_score(
    price_panel: pd.DataFrame,
    ticker: str,
    date: pd.Timestamp,
    scorer,
    window: int,
    min_periods: int = 20,
) -> tuple[float, dict[str, float]]:
    """
    Independently recomputes one ticker's score on one date straight from
    the raw price panel, slicing the window the same way a selector does
    (last `window` bars up to and including `date`). Lets you check a
    live run's number against a from-scratch calculation without needing
    to have stored the raw price array anywhere.
    """
    history = price_panel.loc[:date, ticker].dropna()
    if len(history) < min_periods:
        return float("nan"), {}
    window_snapshot = history.iloc[-window:] if len(history) >= window else history
    return scorer.compute_score(window_snapshot.values)


def verify_score(
    df_ledger: pd.DataFrame,
    price_panel: pd.DataFrame,
    ticker: str,
    date: pd.Timestamp,
    scorer,
    window: int,
    min_periods: int = 20,
    score_col: str = "raw_score",
    atol: float = 1e-6,
) -> dict:
    """
    Recomputes a ticker's score from raw prices and compares it against
    the value already recorded in df_ledger (the output of
    analyze_audit_trail). Returns a dict with both values, their diff,
    and a pass/fail `match` flag -- a quick way to spot-check a selector
    or scorer without re-running the whole backtest.
    """
    date = pd.Timestamp(date)
    recomputed_score, recomputed_metrics = recompute_score(
        price_panel, ticker, date, scorer, window, min_periods
    )

    row = df_ledger[(df_ledger["ticker"] == ticker) & (df_ledger["date"] == date)]
    if row.empty:
        return {
            "ticker": ticker,
            "date": date,
            "recomputed_score": recomputed_score,
            "ledger_score": None,
            "diff": None,
            "match": False,
            "reason": "no matching row in ledger",
        }

    ledger_score = row.iloc[0].get(score_col)
    both_nan = pd.isna(recomputed_score) and pd.isna(ledger_score)
    diff = None if pd.isna(recomputed_score) or pd.isna(ledger_score) else recomputed_score - ledger_score
    match = both_nan or (diff is not None and abs(diff) <= atol)

    return {
        "ticker": ticker,
        "date": date,
        "recomputed_score": recomputed_score,
        "ledger_score": ledger_score,
        "diff": diff,
        "match": match,
        "recomputed_metrics": recomputed_metrics,
    }