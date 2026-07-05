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
    raw_signals: list = field(default_factory=list)
    filtered_signals: list = field(default_factory=list)
    target_weights: dict[str, float] = field(default_factory=dict)
    executed_trades: list = field(default_factory=list)

def analyze_audit_trail(ledger: list[AuditRecord]):
    """
    Flattens complex mathematical telemetry down into analytical DataFrames 
    to easily cross-reference 'Why did we buy X?'
    """
    rows = []
    for record in ledger:
        date = record.date
        # Extract specific telemetry keys (e.g., your R2 or Slope from Information Discrete)
        r2_metrics = record.telemetry.metrics.get("r2", {})
        slope_metrics = record.telemetry.metrics.get("slope", {})
        
        for ticker in r2_metrics.keys():
            rows.append({
                "date": date,
                "ticker": ticker,
                "r_squared": r2_metrics.get(ticker, None),
                "slope": slope_metrics.get(ticker, None),
                "final_weight": record.target_weights.get(ticker, 0.0)
            })
            
    df_analysis = pd.DataFrame(rows)
    return df_analysis