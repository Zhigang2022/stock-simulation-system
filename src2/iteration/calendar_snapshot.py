"""
Local, dependency-free replacement for src.calendar_iterator.CalendarIterator
-- iteration/ only ever needed two things off it (rebalance-date generation,
and a leak-safe truncated snapshot for the tactical leg), so inlining them
here removes a src/ dependency that added no real coupling value.
"""
import pandas as pd


def generate_rebalance_dates(price_df: pd.DataFrame, interval: str = "ME") -> list[pd.Timestamp]:
    """
    Finds the actual trading days that match the desired interval. Using
    the price matrix index ensures we pick real trading days, avoiding
    weekends and market holidays.
    """
    all_dates = price_df.index
    return pd.Series(all_dates, index=all_dates).resample(interval).last().dropna().tolist()


def get_historical_snapshot(world_data_dict: dict[str, pd.DataFrame], current_date: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """
    Slices every panel in world_data_dict up to (and including) current_date.
    Includes a strict structural check to prevent future data leaks.
    """
    price_matrix = world_data_dict["price"]
    assert current_date <= price_matrix.index[-1], f"Requested date {current_date} is out of bounds."

    snapshot = {key: panel.loc[:current_date] for key, panel in world_data_dict.items()}

    price_snapshot = snapshot["price"]
    if not price_snapshot.empty:
        last_available_date = price_snapshot.index[-1]
        assert last_available_date <= current_date, (
            f"CRITICAL BUG: Future data leaked! Snapshot last date ({last_available_date}) "
            f"is ahead of current backtest time ({current_date})."
        )

    return snapshot
