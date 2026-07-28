import numpy as np
import pandas as pd

class VectorizedBootstrapEngine:
    def __init__(self, historical_data_dict:dict[str, pd.DataFrame], expected_block_size=21):
        """
        historical_data_dict: dict like {'price': pd.DataFrame, 'volume': pd.DataFrame}
        expected_block_size: Average trend length in days before a timeline jump

        # volume used normalized or raw
        # here is using the raw way.
        """
        self.prices_df = historical_data_dict['price']
        self.volume_df = historical_data_dict['volume']
        
        # Calculate returns for the price component
        self.returns_df = self.prices_df.pct_change().dropna()
        assert (self.prices_df.shape[0]-self.returns_df.shape[0])<30

        # Align volume to match the same dates as the returns
        self.volume_df = self.volume_df.loc[self.returns_df.index]
        
        self.tickers = self.returns_df.columns
        self.dates_index = self.returns_df.index
        self.num_days, self.num_tickers = self.returns_df.shape
        
        self.jump_probability = 1.0 / expected_block_size

    def generate_all_worlds(self, num_simulations:int=1000, start_prices=None):
        """
        Instantly generates all synthetic paths.
        Returns a 3D dictionary structure matching your input types.
        """
        # 1. Step 1: Generate the randomized timeline indices (Stationary Bootstrap)
        jump_trials = np.random.rand(num_simulations, self.num_days) < self.jump_probability
        sim_indices = np.zeros((num_simulations, self.num_days), dtype=int)
        sim_indices[:, 0] = np.random.randint(0, self.num_days, size=num_simulations)
        
        for t in range(1, self.num_days):
            random_jumps = np.random.randint(0, self.num_days, size=num_simulations)
            sequential_steps = (sim_indices[:, t-1] + 1) % self.num_days
            sim_indices[:, t] = np.where(jump_trials[:, t], random_jumps, sequential_steps)
            
        # 2. Step 2: Combine Returns and Volume horizontally to shuffle them together
        # If you have 18 tickers, features_matrix will have 36 columns
        features_matrix = np.hstack([self.returns_df.values, self.volume_df.values])
        
        # Shuffle everything across all parallel universes at once
        # Shape: (num_simulations, num_days, num_tickers * 2)
        shuffled_features_3d = features_matrix[sim_indices]
        
        # Split the features back into their respective Return and Volume buckets
        shuffled_returns_3d = shuffled_features_3d[:, :, :self.num_tickers]
        shuffled_volume_3d = shuffled_features_3d[:, :, self.num_tickers:]
        
        # 3. Step 3: Reconstruct the price paths from returns
        if start_prices is None:
            start_arr = np.ones((num_simulations, 1, self.num_tickers)) * 100.0
        else:
            base = np.array([start_prices[t] for t in self.tickers])
            start_arr = np.tile(base, (num_simulations, 1, 1))
            
        price_factors = 1.0 + shuffled_returns_3d
        ones_layer = np.ones((num_simulations, 1, self.num_tickers))
        price_factors = np.concatenate([ones_layer, price_factors], axis=1)
        
        # Compound returns across the time axis (axis=1)
        synthetic_prices_3d = start_arr * np.cumprod(price_factors, axis=1)
        synthetic_prices_3d = synthetic_prices_3d[:, 1:, :] # Drop day 0 layer
        
        return {
            'price_tensor': synthetic_prices_3d,
            'volume_tensor': shuffled_volume_3d
        }