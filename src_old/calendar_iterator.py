# calendar_iterator.py
import pandas as pd

class CalendarIterator:
    def __init__(self, universe_data: dict[str, pd.DataFrame], interval: str = "ME"):
        """
        Args:
            universe_data: Dict containing 'price' and 'volume' DataFrames from DataBroker.
            interval: Pandas offset alias. 'ME' = Month End, 'W' = Weekly, etc.
        """
        self.price_matrix = universe_data["price"]
        self.volume_matrix = universe_data["volume"]
        self.interval = interval

    def generate_rebalance_dates(self) -> list[pd.Timestamp]:
        """
        Finds the actual trading days that match the desired interval.
        Using the price matrix index ensures we pick real trading days, 
        avoiding weekends and market holidays.
        """
        # Group the existing trading dates by the interval and take the last day of each period
        all_dates = self.price_matrix.index
        rebalance_dates = pd.Series(all_dates, index=all_dates).resample(self.interval).last().dropna().tolist()
        return rebalance_dates

    def get_historical_snapshot(self, current_date: pd.Timestamp) -> dict[str, pd.DataFrame]:
        """
        Slices the universe matrices up to (and including) current_date.
        Includes a strict structural check to prevent future data leaks.
        """
        # Strict Structural Boundary Check
        assert current_date <= self.price_matrix.index[-1], f"Requested date {current_date} is out of bounds."
        
        # Slice matrices up to current_date
        price_snapshot = self.price_matrix.loc[:current_date]
        volume_snapshot = self.volume_matrix.loc[:current_date]
        
        # STRUCTURAL CHECK: Absolute Guard against Look-Ahead Bias
        if not price_snapshot.empty:
            last_available_date = price_snapshot.index[-1]
            assert last_available_date <= current_date, (
                f"CRITICAL BUG: Future data leaked! Snapshot last date ({last_available_date}) "
                f"is ahead of current backtest time ({current_date})."
            )

        return {
            "price": price_snapshot,
            "volume": volume_snapshot
        }