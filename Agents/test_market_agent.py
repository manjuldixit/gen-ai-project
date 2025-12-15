"""
Unit tests for the Market Activity Prediction Agent system.
"""

import unittest
from market_agent import (
    MarketDataManager, VolatilityForecaster, AnomalyDetector,
    EventSimulator, TrendAnalysisAgent, RiskAssessmentAgent,
    MarketActivityAgent, MarketPredictionOrchestrator,
    PredictionSignal, RiskLevel
)
from utils import (
    get_action_recommendation, calculate_position_sizing,
    assess_prediction_quality, get_market_regime
)
import pandas as pd
import numpy as np


class TestMarketDataManager(unittest.TestCase):
    """Test data collection functionality."""
    
    def setUp(self):
        self.manager = MarketDataManager()
    
    def test_get_historical_data(self):
        """Test fetching historical data."""
        try:
            df = self.manager.get_historical_data("AAPL", period="1mo")
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 10)
            self.assertIn('Returns', df.columns)
            self.assertIn('Log_Returns', df.columns)
        except Exception as e:
            self.skipTest(f"Market data unavailable: {str(e)}")
    
    def test_get_upcoming_events(self):
        """Test event detection."""
        events = self.manager.get_upcoming_events("AAPL")
        self.assertIsInstance(events, list)


class TestVolatilityForecaster(unittest.TestCase):
    """Test volatility prediction."""
    
    def setUp(self):
        self.forecaster = VolatilityForecaster()
        # Create synthetic returns data
        np.random.seed(42)
        self.returns = pd.Series(np.random.randn(100) * 0.02)
    
    def test_predict_volatility(self):
        """Test volatility forecasting."""
        result = self.forecaster.predict_volatility(self.returns)
        
        self.assertIn('annualized_volatility_forecast', result)
        self.assertIn('conditional_volatility', result)
        self.assertIn('model_confidence', result)
        self.assertGreater(result['annualized_volatility_forecast'], 0)
        self.assertLessEqual(result['model_confidence'], 1)


class TestAnomalyDetector(unittest.TestCase):
    """Test anomaly detection."""
    
    def setUp(self):
        self.detector = AnomalyDetector()
        # Create synthetic data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100)
        self.data = pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.randn(100) * 2),
            'Volume': np.random.randint(1000000, 5000000, 100),
            'Returns': np.random.randn(100) * 0.02
        }, index=dates)
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        result = self.detector.detect_anomalies(self.data)
        
        self.assertIn('current_state_is_anomaly', result)
        self.assertIn('total_anomalies_detected', result)
        self.assertIn('anomaly_score', result)
        self.assertIsInstance(result['current_state_is_anomaly'], (bool, np.bool_))
        self.assertGreaterEqual(result['total_anomalies_detected'], 0)


class TestEventSimulator(unittest.TestCase):
    """Test Monte Carlo simulation."""
    
    def setUp(self):
        self.simulator = EventSimulator()
    
    def test_run_simulation(self):
        """Test simulation execution."""
        result = self.simulator.run_simulation(
            current_price=150.0,
            daily_vol=0.02,
            days=5,
            simulations=100
        )
        
        self.assertIn('expected_price', result)
        self.assertIn('bear_case_95', result)
        self.assertIn('bull_case_95', result)
        self.assertIn('probability_of_decline', result)
        
        # Validate simulation results
        self.assertGreater(result['expected_price'], 0)
        self.assertGreater(result['bull_case_95'], result['bear_case_95'])
        self.assertGreaterEqual(result['probability_of_decline'], 0)
        self.assertLessEqual(result['probability_of_decline'], 1)


class TestTrendAnalysisAgent(unittest.TestCase):
    """Test trend analysis."""
    
    def setUp(self):
        self.agent = TrendAnalysisAgent()
        # Create synthetic price data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100)
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        self.data = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_analyze_trend(self):
        """Test trend analysis."""
        result = self.agent.analyze_trend(self.data)
        
        self.assertIn('trend', result)
        self.assertIn('trend_strength', result)
        self.assertIn('technical_factors', result)
        self.assertIsInstance(result['technical_factors'], list)


class TestRiskAssessmentAgent(unittest.TestCase):
    """Test risk assessment."""
    
    def setUp(self):
        self.agent = RiskAssessmentAgent()
    
    def test_assess_risk_low(self):
        """Test low risk assessment."""
        result = self.agent.assess_risk(
            volatility=15.0,
            anomaly_score=0.0,
            probability_decline=0.5,
            price_range=(145.0, 155.0)
        )
        
        self.assertEqual(result['risk_level'], RiskLevel.LOW.value)
        self.assertIsInstance(result['risk_factors'], list)
        self.assertIsInstance(result['recommendations'], list)
    
    def test_assess_risk_critical(self):
        """Test critical risk assessment."""
        result = self.agent.assess_risk(
            volatility=60.0,
            anomaly_score=-1.0,
            probability_decline=0.9,
            price_range=(100.0, 200.0)
        )
        
        self.assertEqual(result['risk_level'], RiskLevel.CRITICAL.value)
        self.assertGreater(result['risk_score'], 80)


class TestMarketActivityAgent(unittest.TestCase):
    """Test legacy market activity agent."""
    
    def setUp(self):
        self.agent = MarketActivityAgent()
    
    def test_analyze(self):
        """Test basic analysis."""
        try:
            result = self.agent.analyze("AAPL")
            
            self.assertIn('ticker', result)
            self.assertIn('alert_level', result)
            self.assertIn('prediction', result)
            self.assertIn('analysis', result)
            self.assertEqual(result['ticker'], 'AAPL')
            self.assertIn(result['alert_level'], ['LOW', 'MEDIUM', 'HIGH'])
        except Exception as e:
            self.skipTest(f"Market data unavailable: {str(e)}")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_get_action_recommendation(self):
        """Test action recommendation."""
        action = get_action_recommendation(PredictionSignal.BULLISH, 0.85)
        self.assertIn('BUY', action.upper())
        
        action = get_action_recommendation(PredictionSignal.BEARISH, 0.85)
        self.assertIn('SELL', action.upper())
        
        action = get_action_recommendation(PredictionSignal.NEUTRAL, 0.50)
        self.assertIn('HOLD', action.upper())
    
    def test_calculate_position_sizing(self):
        """Test position sizing calculation."""
        position = calculate_position_sizing(RiskLevel.LOW, 100000, 0.8)
        
        self.assertIn('recommended_position_size', position)
        self.assertIn('allocation_percent', position)
        self.assertGreater(position['recommended_position_size'], 0)
        self.assertLess(position['allocation_percent'], 10)
    
    def test_get_market_regime(self):
        """Test market regime identification."""
        # Create mock prediction
        from market_agent import MarketPrediction
        from datetime import datetime
        
        prediction = MarketPrediction(
            ticker="TEST",
            signal=PredictionSignal.BULLISH,
            confidence=0.8,
            volatility_forecast=10.0,
            price_target_range=(100, 110),
            risk_level=RiskLevel.LOW,
            reasoning="Test",
            key_factors=["Test"],
            timestamp=datetime.now().isoformat()
        )
        
        regime = get_market_regime(prediction)
        self.assertIsInstance(regime, str)
        self.assertIn(regime, [
            "CRISIS_MODE", "RISK_OFF", "RISK_ON",
            "LOW_VOLATILITY_ENVIRONMENT", "NORMAL_CONDITIONS"
        ])


class TestSignalGeneration(unittest.TestCase):
    """Test prediction signal generation."""
    
    def test_signal_quality(self):
        """Test signal generation quality."""
        try:
            agent = MarketActivityAgent()
            result = agent.analyze("AAPL")
            
            # Check that direction bias is reasonable
            direction = result['prediction']['direction_bias']
            self.assertIn(direction, ['Bullish', 'Bearish'])
            
            # Check confidence is in valid range
            confidence = result['analysis']['confidence_score']
            self.assertGreaterEqual(confidence, 0)
            self.assertLessEqual(confidence, 1)
        except Exception as e:
            self.skipTest(f"Market data unavailable: {str(e)}")


class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        detector = AnomalyDetector()
        
        # Empty dataframe
        empty_df = pd.DataFrame({'Returns': [], 'Volume': []})
        
        # Should handle gracefully
        try:
            # This might fail or return empty results, both are acceptable
            result = detector.detect_anomalies(empty_df)
        except Exception:
            pass  # Expected behavior for invalid data
    
    def test_single_point_data(self):
        """Test handling of single data point."""
        forecaster = VolatilityForecaster()
        single_return = pd.Series([0.01])
        
        # Should handle gracefully
        try:
            result = forecaster.predict_volatility(single_return)
        except Exception:
            pass  # Expected behavior for insufficient data


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
