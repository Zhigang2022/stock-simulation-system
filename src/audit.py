from dataclasses import dataclass,field
from src.signal_schema import StrategyTelemetry
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