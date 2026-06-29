# data_broker.py
import pandas as pd
import yfinance as yf

class DataBroker:
    def __init__(self, tickers: list[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = None

    def fetch_universe_data(self) -> dict[str, pd.DataFrame]:
        """
        Fetches historical data from yfinance and pivots into aligned matrices.
        """
        # Fetch data for all tickers at once
        # yfinance returns a multi-index column DataFrame if multiple tickers
        df = yf.download(
            tickers=self.tickers, 
            start=self.start_date, 
            end=self.end_date, 
            group_by='ticker'
        )
        self.raw_data = df
        
        price_dict = {}
        volume_dict = {}
        
        # Extract and align Adjusted Close and Volume
        for ticker in self.tickers:
            if ticker in df.columns.levels[0]:
                price_dict[ticker] = df[ticker]['Close']
                volume_dict[ticker] = df[ticker]['Volume']
            else:
                # Handle cases where ticker data is completely missing
                price_dict[ticker] = pd.Series(dtype='float64')
                volume_dict[ticker] = pd.Series(dtype='float64')
                
        # Build the [Time x Asset] Matrices
        price_matrix = pd.DataFrame(price_dict)
        volume_matrix = pd.DataFrame(volume_dict)
        
        # Forward fill and backward fill minor asset mismatches/holidays
        price_matrix = price_matrix.ffill().bfill()
        volume_matrix = volume_matrix.fillna(0) # No trading volume on missing days
        
        return {
            "price": price_matrix,
            "volume": volume_matrix
        }

    def split_train_test(self, universe_data: dict[str, pd.DataFrame], forward_months: int) -> tuple[dict, dict]:
        """
        Splits the price and volume matrices into In-Sample and Out-of-Sample (Forward Window).
        """
        price_df = universe_data["price"]
        volume_df = universe_data["volume"]
        
        # Find the cut-off date based on the last row's timestamp minus N months
        last_date = price_df.index[-1]
        cutoff_date = last_date - pd.DateOffset(months=forward_months)
        
        # Slice matrices
        is_data = {
            "price": price_df.loc[:cutoff_date],
            "volume": volume_df.loc[:cutoff_date]
        }
        
        oos_data = {
            "price": price_df.loc[cutoff_date:],
            "volume": volume_df.loc[cutoff_date:]
        }
        
        return is_data, oos_data
    
    def split_train_test_by_count(self,universe_data: dict[str, pd.DataFrame],last_ratio_as_test=.25):
        # 2. Simple Two-Way Split based on row count
        total_rows = len(universe_data)
        split_idx = int(total_rows * (1-last_ratio_as_test))

        is_data,oos_data={},{}
        for key in universe_data:
            is_data[key]=universe_data[key].iloc[:split_idx] 
            oos_data[key]=universe_data[key].iloc[split_idx:] 

        return is_data,oos_data