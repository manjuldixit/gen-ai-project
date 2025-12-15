"""
Configuration and constants for the Market Activity Prediction Agent.
"""

from enum import Enum
from typing import Dict

# API Configuration
class APIConfig:
    """Configuration for external APIs."""
    GEMINI_API_KEY = None  # Set via environment or explicitly
    YFINANCE_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3

# Model Configuration
class ModelConfig:
    """Configuration for ML models."""
    
    # GARCH model parameters
    GARCH_P = 1
    GARCH_Q = 1
    VOLATILITY_FORECAST_HORIZON = 5  # days
    
    # Anomaly detection
    ANOMALY_CONTAMINATION = 0.05  # 5% expected anomalies
    ANOMALY_RANDOM_STATE = 42
    
    # Monte Carlo simulation
    MC_SIMULATIONS = 1000
    MC_DAYS = 5
    MC_DRIFT = 0  # Neutral drift for short-term
    
    # Technical analysis
    SMA_SHORT = 20  # days
    SMA_LONG = 50   # days
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Event adjustment
    EVENT_VOLATILITY_MULTIPLIER = 1.5

# Thresholds
class ThresholdConfig:
    """Risk and signal thresholds."""
    
    # Volatility thresholds (%)
    VOLATILITY_CRITICAL = 50
    VOLATILITY_HIGH = 35
    VOLATILITY_MEDIUM = 20
    
    # Anomaly thresholds
    ANOMALY_CRITICAL_THRESHOLD = -0.5
    
    # Probability thresholds
    DOWNSIDE_HIGH_PROBABILITY = 0.7
    DOWNSIDE_MODERATE_PROBABILITY = 0.6
    UPSIDE_MODERATE_PROBABILITY = 0.4
    
    # Price range thresholds
    PRICE_RANGE_WIDE_THRESHOLD = 0.2  # 20% range

# Confidence levels
class ConfidenceConfig:
    """Confidence score thresholds."""
    MINIMUM_CONFIDENCE = 0.5
    HIGH_CONFIDENCE = 0.8
    MAX_CONFIDENCE = 0.95

# GenAI Configuration
class GenAIConfig:
    """Configuration for Gemini AI integration."""
    MODEL_NAME = "gemini-2.0-flash"
    TEMPERATURE = 0.7
    MAX_OUTPUT_TOKENS = 1024
    TOP_P = 0.9
    TOP_K = 40

# Market Data Configuration
class MarketDataConfig:
    """Configuration for market data retrieval."""
    DEFAULT_PERIOD = "2y"
    DEFAULT_INTERVAL = "1d"
    MIN_DATA_POINTS = 252  # 1 year of trading days

# Risk Assessment Weights
class RiskWeights:
    """Weights for risk factor aggregation."""
    VOLATILITY_WEIGHT = 0.3
    ANOMALY_WEIGHT = 0.25
    DIRECTIONAL_WEIGHT = 0.25
    RANGE_WEIGHT = 0.2

# Output Templates
class OutputTemplates:
    """Templates for formatted output."""
    
    PREDICTION_REPORT = """
    ╔════════════════════════════════════════════════════════════════════╗
    ║              MARKET ACTIVITY PREDICTION REPORT                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    Ticker: {ticker}
    Timestamp: {timestamp}
    Current Price: ${current_price:.2f}
    
    ─────────────────────────────────────────────────────────────────────
    PREDICTION SIGNAL
    ─────────────────────────────────────────────────────────────────────
    Signal: {signal}
    Confidence: {confidence:.1%}
    Risk Level: {risk_level}
    
    ─────────────────────────────────────────────────────────────────────
    FORECAST METRICS
    ─────────────────────────────────────────────────────────────────────
    Volatility Forecast (Annualized): {volatility:.2f}%
    Price Target Range (5-day): ${price_low:.2f} - ${price_high:.2f}
    
    ─────────────────────────────────────────────────────────────────────
    KEY FACTORS
    ─────────────────────────────────────────────────────────────────────
    {key_factors}
    
    ─────────────────────────────────────────────────────────────────────
    ANALYSIS & REASONING
    ─────────────────────────────────────────────────────────────────────
    {reasoning}
    
    ═════════════════════════════════════════════════════════════════════
    DISCLAIMER: For educational purposes only. Not financial advice.
    Consult a financial advisor before making investment decisions.
    ═════════════════════════════════════════════════════════════════════
    """

# Agent Descriptions
class AgentDescriptions:
    """Descriptions of each agent's role."""
    
    DATA_AGENT = """
    Data Collection Agent: Fetches and preprocesses market data from external sources.
    Responsibilities:
    - Retrieve historical price and volume data
    - Calculate returns and log returns
    - Identify upcoming market events
    """
    
    TREND_AGENT = """
    Trend Analysis Agent: Analyzes market trends and technical patterns.
    Responsibilities:
    - Calculate moving averages (SMA 20/50)
    - Compute RSI (Relative Strength Index)
    - Calculate MACD and signal lines
    - Identify trend direction and strength
    """
    
    VOLATILITY_AGENT = """
    Volatility Forecasting Agent: Predicts market volatility.
    Responsibilities:
    - Model volatility using GARCH(1,1)
    - Forecast conditional volatility
    - Detect volatility regimes
    """
    
    ANOMALY_AGENT = """
    Anomaly Detection Agent: Identifies unusual market behavior.
    Responsibilities:
    - Detect statistical anomalies in returns and volume
    - Flag current market state anomalies
    - Score anomaly severity
    """
    
    RISK_AGENT = """
    Risk Assessment Agent: Comprehensively evaluates market risks.
    Responsibilities:
    - Assess volatility risk
    - Evaluate directional risk
    - Compute overall risk score
    - Provide risk recommendations
    """
    
    GENAI_AGENT = """
    GenAI Reasoning Agent: Provides AI-powered analysis and explanations.
    Responsibilities:
    - Contextualize market data
    - Generate prediction explanations
    - Provide actionable insights
    - Assess decision rationale
    """
    
    ORCHESTRATOR = """
    Market Prediction Orchestrator: Coordinates all agents and synthesizes insights.
    Responsibilities:
    - Orchestrate agent workflows
    - Aggregate signals and factors
    - Generate final predictions
    - Produce comprehensive reports
    """
