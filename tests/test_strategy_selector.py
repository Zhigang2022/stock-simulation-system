import unittest
import numpy as np
import pandas as pd
from src.strategy_selector import (
    SimpleRiskAdjustedScorer,
    TraditionalLinearRegressionScorer,
    GeometricDriftScorer,
    WeightedLinearRegressionScorer,
    ComprehensiveMultiHorizonStrategy
)

class TestSimpleRiskAdjustedScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = SimpleRiskAdjustedScorer()

    def test_normal_calculation(self):
        # Monotonically increasing prices
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        score, metrics = self.scorer.compute_score(prices)
        
        # Expect a positive score
        self.assertTrue(score > 0)
        self.assertIn('volatile', metrics)
        self.assertTrue(metrics['volatile'] > 0)

    def test_length_less_than_two(self):
        prices = np.array([10.0])
        score, metrics = self.scorer.compute_score(prices)
        self.assertTrue(np.isnan(score))
        self.assertIsInstance(metrics, dict)

    def test_first_price_zero_or_negative(self):
        prices = np.array([0.0, 10.0])
        score, metrics = self.scorer.compute_score(prices)
        self.assertTrue(np.isnan(score))
        self.assertIsInstance(metrics, dict)

    def test_zero_volatility(self):
        # Constant prices
        prices = np.array([10.0, 10.0, 10.0])
        score, metrics = self.scorer.compute_score(prices)
        self.assertTrue(np.isnan(score))


class TestTraditionalLinearRegressionScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = TraditionalLinearRegressionScorer()

    def test_normal_calculation(self):
        # Linear trend in log prices
        # prices = exp(x * slope)
        x = np.arange(5)
        slope = 0.1
        prices = np.exp(x * slope)
        
        score, metrics = self.scorer.compute_score(prices)
        
        self.assertAlmostEqual(metrics['slope'], slope, places=5)
        # R-squared should be close to 1 since we created a perfect line
        self.assertAlmostEqual(metrics['r_squared'], 1.0, places=5)
        self.assertAlmostEqual(score, slope, places=5)

    def test_length_less_than_three(self):
        prices = np.array([10.0, 11.0])
        score, metrics = self.scorer.compute_score(prices)
        self.assertTrue(np.isnan(score))
        self.assertIsInstance(metrics, dict)


class TestGeometricDriftScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = GeometricDriftScorer()

    def test_normal_calculation(self):
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        score, metrics = self.scorer.compute_score(prices)
        
        self.assertTrue(score > 0)
        self.assertIn('volatility', metrics)
        self.assertIn('raw_log_return', metrics)

    def test_length_less_than_two(self):
        prices = np.array([10.0])
        score, metrics = self.scorer.compute_score(prices)
        self.assertTrue(np.isnan(score))
        self.assertIsInstance(metrics, dict)


class TestWeightedLinearRegressionScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = WeightedLinearRegressionScorer(decay_factor=0.9)

    def test_normal_calculation(self):
        x = np.arange(5)
        prices = np.exp(x * 0.05)
        score, metrics = self.scorer.compute_score(prices)
        
        self.assertTrue(score > 0)
        self.assertIn('slope', metrics)
        self.assertIn('r_squared', metrics)

    def test_length_less_than_three(self):
        prices = np.array([10.0, 11.0])
        score, metrics = self.scorer.compute_score(prices)
        self.assertTrue(np.isnan(score))
        self.assertIsInstance(metrics, dict)


class TestComprehensiveMultiHorizonStrategy(unittest.TestCase):
    def setUp(self):
        self.scorer = TraditionalLinearRegressionScorer()
        self.strategy = ComprehensiveMultiHorizonStrategy(
            scorer=self.scorer,
            long_window=10,
            short_window=5,
            structural_weight=0.7
        )

    def test_insufficient_data(self):
        # We need at least long_window (10) rows
        dates = pd.date_range("2023-01-01", periods=9)
        price_df = pd.DataFrame(
            {"AAPL": np.linspace(10, 15, 9), "MSFT": np.linspace(20, 18, 9)},
            index=dates
        )
        snapshot = {"price": price_df}
        
        output, telemetry = self.strategy.calculate_signals(snapshot)
        # Should return empty StrategyOutput signals list
        self.assertEqual(len(output.signals), 0)
        self.assertEqual(len(telemetry.metrics), 0)

    def test_sufficient_data_signal_generation(self):
        dates = pd.date_range("2023-01-01", periods=15)
        # AAPL goes up, MSFT goes down
        price_df = pd.DataFrame(
            {
                "AAPL": np.linspace(10, 20, 15), 
                "MSFT": np.linspace(20, 10, 15)
            },
            index=dates
        )
        snapshot = {"price": price_df}
        
        output, telemetry = self.strategy.calculate_signals(snapshot)
        
        # Verify signal generation structure
        self.assertTrue(len(output.signals) > 0)
        df_signals = output.to_dataframe()
        self.assertIn("ticker", df_signals.columns)
        self.assertIn("score", df_signals.columns)
        
        # Check telemetry metrics
        self.assertIn("slope_long", telemetry.metrics)
        self.assertIn("slope_short", telemetry.metrics)
        self.assertIn("r_squared_long", telemetry.metrics)
        self.assertIn("r_squared_short", telemetry.metrics)
        self.assertIn("raw_score_long", telemetry.metrics)
        self.assertIn("raw_score_short", telemetry.metrics)
        
        # AAPL (trending up) should have higher score than MSFT (trending down)
        aapl_signal = df_signals[df_signals["ticker"] == "AAPL"].iloc[0]
        msft_signal = df_signals[df_signals["ticker"] == "MSFT"].iloc[0]
        self.assertTrue(aapl_signal["score"] > msft_signal["score"])

    def test_crash_edge_case_with_short_window(self):
        # Using a short window < 3 with TraditionalLinearRegressionScorer (which requires >= 3)
        # to trigger the return of np.nan as metrics, checking if it causes AttributeError.
        strategy = ComprehensiveMultiHorizonStrategy(
            scorer=self.scorer,
            long_window=10,
            short_window=2, # Will trigger n < 3 in TraditionalLinearRegressionScorer
            structural_weight=0.7
        )
        dates = pd.date_range("2023-01-01", periods=10)
        price_df = pd.DataFrame(
            {"AAPL": np.linspace(10, 20, 10)},
            index=dates
        )
        snapshot = {"price": price_df}
        
        # This will raise AttributeError if not handled/fixed in strategy_selector.py
        try:
            strategy.calculate_signals(snapshot)
        except AttributeError as e:
            self.fail(f"Strategy crashed with AttributeError: {e}")



if __name__ == "__main__":
    unittest.main()
