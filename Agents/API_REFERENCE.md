# Market Activity Prediction Agent - API Reference

## Core Classes

### MarketPrediction

Structured result containing all prediction information.

```python
@dataclass
class MarketPrediction:
    ticker: str                          # Stock ticker (e.g., "AAPL")
    signal: PredictionSignal            # Prediction signal (enum)
    confidence: float                    # 0.5-1.0 confidence score
    volatility_forecast: float          # Annualized volatility %
    price_target_range: tuple           # (lower, upper) price bounds
    risk_level: RiskLevel               # Risk level (enum)
    reasoning: str                      # AI-generated explanation
    key_factors: List[str]              # Contributing factors
    timestamp: str                      # ISO 8601 timestamp
```

**Attributes**:
- `ticker`: Stock symbol to analyze
- `signal`: One of BULLISH, BEARISH, NEUTRAL, HIGH_VOLATILITY
- `confidence`: Prediction reliability (higher = more reliable)
- `volatility_forecast`: Expected annualized price volatility
- `price_target_range`: Expected price range (5-day horizon)
- `risk_level`: Market risk assessment
- `reasoning`: AI-powered explanation of prediction
- `key_factors`: List of factors influencing prediction
- `timestamp`: When prediction was generated

---

## Agent Classes

### MarketDataManager

Handles market data collection and preprocessing.

```python
class MarketDataManager:
    def get_historical_data(ticker: str, period: str = "2y") -> pd.DataFrame
    def get_upcoming_events(ticker: str) -> List[str]
```

**Methods**:
- `get_historical_data(ticker, period)`: Fetch historical OHLCV data
  - Returns DataFrame with 'Close', 'Volume', 'Returns', 'Log_Returns'
  - `period`: "1mo", "3mo", "6mo", "1y", "2y", "5y"
  
- `get_upcoming_events(ticker)`: Identify upcoming market events
  - Returns list of event descriptions
  - Currently detects earnings releases

**Example**:
```python
manager = MarketDataManager()
df = manager.get_historical_data("AAPL", period="1y")
events = manager.get_upcoming_events("AAPL")
```

---

### VolatilityForecaster

Predicts market volatility using GARCH(1,1) models.

```python
class VolatilityForecaster:
    def predict_volatility(returns: pd.Series, horizon: int = 5) -> Dict
```

**Methods**:
- `predict_volatility(returns, horizon)`: Forecast conditional volatility
  - `returns`: pd.Series of asset returns
  - `horizon`: Days to forecast ahead (default 5)
  - Returns dict with:
    - `annualized_volatility_forecast`: Annualized vol %
    - `conditional_volatility`: List of daily vols
    - `model_confidence`: GARCH model R-squared

**Example**:
```python
forecaster = VolatilityForecaster()
result = forecaster.predict_volatility(returns_series, horizon=5)
print(f"Vol: {result['annualized_volatility_forecast']:.2f}%")
```

---

### AnomalyDetector

Identifies unusual market behavior using statistical methods.

```python
class AnomalyDetector:
    def detect_anomalies(data: pd.DataFrame) -> Dict
```

**Methods**:
- `detect_anomalies(data)`: Detect statistical anomalies
  - `data`: DataFrame with 'Returns' and 'Volume' columns
  - Returns dict with:
    - `current_state_is_anomaly`: Boolean flag
    - `total_anomalies_detected`: Count in period
    - `anomaly_score`: Severity score

**Example**:
```python
detector = AnomalyDetector()
result = detector.detect_anomalies(data)
if result['current_state_is_anomaly']:
    print("Unusual market conditions detected")
```

---

### EventSimulator

Runs Monte Carlo simulations of price paths.

```python
class EventSimulator:
    def run_simulation(current_price: float, daily_vol: float, 
                      days: int = 5, simulations: int = 1000) -> Dict
```

**Methods**:
- `run_simulation()`: Run price simulations
  - `current_price`: Current asset price
  - `daily_vol`: Daily volatility (decimal, not %)
  - `days`: Simulation horizon
  - `simulations`: Number of Monte Carlo paths
  - Returns dict with:
    - `expected_price`: Mean simulated price
    - `bear_case_95`: 5th percentile
    - `bull_case_95`: 95th percentile
    - `probability_of_decline`: % paths downside

**Example**:
```python
simulator = EventSimulator()
result = simulator.run_simulation(150.0, 0.02, days=5)
print(f"Bear case: ${result['bear_case_95']:.2f}")
print(f"Bull case: ${result['bull_case_95']:.2f}")
```

---

### TrendAnalysisAgent

Analyzes technical trends and patterns.

```python
class TrendAnalysisAgent:
    def analyze_trend(df: pd.DataFrame, period: int = 20) -> Dict
```

**Methods**:
- `analyze_trend(df, period)`: Analyze price trends
  - `df`: DataFrame with 'Close' prices
  - `period`: SMA period (default 20)
  - Returns dict with:
    - `trend`: Trend classification (UPTREND, DOWNTREND, etc.)
    - `trend_strength`: 0-1 strength score
    - `sma_20`, `sma_50`: Moving averages
    - `rsi`: Relative Strength Index
    - `macd`, `macd_signal`: MACD indicators
    - `technical_factors`: List of observations

**Example**:
```python
agent = TrendAnalysisAgent()
result = agent.analyze_trend(df)
print(f"Trend: {result['trend']}")
print(f"RSI: {result['rsi']:.1f}")
```

---

### RiskAssessmentAgent

Evaluates multi-dimensional risk factors.

```python
class RiskAssessmentAgent:
    def assess_risk(volatility: float, anomaly_score: float,
                   probability_decline: float, 
                   price_range: tuple) -> Dict
```

**Methods**:
- `assess_risk()`: Comprehensive risk assessment
  - `volatility`: Annualized volatility %
  - `anomaly_score`: Anomaly detector output
  - `probability_decline`: Downside probability
  - `price_range`: (lower, upper) bounds
  - Returns dict with:
    - `risk_level`: LOW/MEDIUM/HIGH/CRITICAL
    - `risk_score`: 0-100
    - `risk_factors`: List of contributing factors
    - `recommendations`: Risk management suggestions

**Example**:
```python
agent = RiskAssessmentAgent()
risk = agent.assess_risk(25.0, 0.2, 0.55, (145, 155))
print(f"Risk Level: {risk['risk_level']}")
for rec in risk['recommendations']:
    print(f"  - {rec}")
```

---

### GenAIAnalysisAgent

AI-powered analysis using Google Gemini.

```python
class GenAIAnalysisAgent:
    def __init__(api_key: Optional[str] = None)
    def analyze_market_context(market_data: Dict) -> str
    def explain_prediction(prediction_data: Dict, 
                          technical_factors: List) -> str
```

**Methods**:
- `__init__(api_key)`: Initialize with optional API key
  - Uses GEMINI_API_KEY environment variable if not provided
  
- `analyze_market_context(data)`: Provide contextual market analysis
  - `market_data`: Dict with market metrics
  - Returns: String with AI analysis
  
- `explain_prediction(data, factors)`: Generate prediction explanation
  - `prediction_data`: Prediction metrics
  - `technical_factors`: Contributing technical factors
  - Returns: String with detailed explanation

**Example**:
```python
agent = GenAIAnalysisAgent(api_key="your-key")
explanation = agent.explain_prediction(prediction_data, factors)
print(explanation)
```

---

### MarketPredictionOrchestrator

Main agent coordinating all sub-agents.

```python
class MarketPredictionOrchestrator:
    def __init__(api_key: Optional[str] = None)
    def predict_market_activity(ticker: str) -> MarketPrediction
    def get_detailed_report(ticker: str) -> Dict
```

**Methods**:
- `__init__(api_key)`: Initialize orchestrator
  - Sets up all sub-agents
  - Configures GenAI if API key provided
  
- `predict_market_activity(ticker)`: Generate complete prediction
  - `ticker`: Stock symbol
  - Returns: MarketPrediction object
  - Coordinates all agents
  
- `get_detailed_report(ticker)`: Get comprehensive analysis report
  - `ticker`: Stock symbol
  - Returns: Dict with summary, prediction, analysis, indicators

**Example**:
```python
orchestrator = MarketPredictionOrchestrator()
prediction = orchestrator.predict_market_activity("NVDA")

print(f"Signal: {prediction.signal.value}")
print(f"Confidence: {prediction.confidence:.1%}")

report = orchestrator.get_detailed_report("NVDA")
import json
print(json.dumps(report, indent=2))
```

---

## Utility Functions

### format_prediction_report

Format prediction into readable report.

```python
def format_prediction_report(prediction: MarketPrediction, 
                            current_price: float) -> str
```

**Parameters**:
- `prediction`: MarketPrediction object
- `current_price`: Current asset price

**Returns**: Formatted string suitable for display/export

**Example**:
```python
from utils import format_prediction_report
report = format_prediction_report(prediction, 150.25)
print(report)
```

---

### get_action_recommendation

Get trading recommendation based on signal.

```python
def get_action_recommendation(signal: PredictionSignal, 
                             confidence: float) -> str
```

**Parameters**:
- `signal`: Prediction signal (enum)
- `confidence`: Confidence score (0-1)

**Returns**: String with action recommendation

**Example**:
```python
action = get_action_recommendation(prediction.signal, prediction.confidence)
print(f"Action: {action}")
```

---

### calculate_position_sizing

Calculate recommended position size.

```python
def calculate_position_sizing(risk_level: RiskLevel,
                             account_size: float,
                             confidence: float) -> Dict
```

**Parameters**:
- `risk_level`: Market risk level (enum)
- `account_size`: Trading account size ($)
- `confidence`: Prediction confidence (0-1)

**Returns**: Dict with position sizing guidance
- `recommended_position_size`: Position $ amount
- `allocation_percent`: % of account
- `stop_loss_distance`: Suggested distance
- `take_profit_distance`: Suggested distance

**Example**:
```python
position = calculate_position_sizing(RiskLevel.MEDIUM, 100000, 0.8)
print(f"Position: ${position['recommended_position_size']:,.2f}")
```

---

### assess_prediction_quality

Evaluate prediction reliability.

```python
def assess_prediction_quality(prediction: MarketPrediction) -> Dict
```

**Parameters**:
- `prediction`: MarketPrediction object

**Returns**: Dict with quality metrics
- `overall_quality_score`: 0-100 score
- `reliability_rating`: EXCELLENT/GOOD/FAIR/POOR/UNRELIABLE
- `strengths`: List of positive factors
- `issues`: List of concerns
- `suitable_for_trading`: Boolean

**Example**:
```python
quality = assess_prediction_quality(prediction)
if quality['suitable_for_trading']:
    # Proceed with trade
    pass
```

---

### compare_predictions

Compare multiple predictions for consensus.

```python
def compare_predictions(predictions: List[MarketPrediction]) -> Dict
```

**Parameters**:
- `predictions`: List of MarketPrediction objects

**Returns**: Dict with consensus analysis
- `total_predictions`: Number analyzed
- `consensus`: Overall consensus (BULLISH/BEARISH/etc)
- `signal_distribution`: Breakdown by signal
- `average_confidence`: Mean confidence
- `consensus_strength`: Strength of consensus

**Example**:
```python
consensus = compare_predictions([pred1, pred2, pred3])
print(f"Consensus: {consensus['consensus']}")
print(f"Strength: {consensus['consensus_strength']:.0%}")
```

---

### get_market_regime

Identify current market regime.

```python
def get_market_regime(prediction: MarketPrediction) -> str
```

**Parameters**:
- `prediction`: MarketPrediction object

**Returns**: String indicating regime
- CRISIS_MODE
- RISK_OFF
- RISK_ON
- LOW_VOLATILITY_ENVIRONMENT
- NORMAL_CONDITIONS

**Example**:
```python
regime = get_market_regime(prediction)
print(f"Market regime: {regime}")
```

---

### generate_trading_alerts

Generate trading alerts.

```python
def generate_trading_alerts(prediction: MarketPrediction) -> List[str]
```

**Parameters**:
- `prediction`: MarketPrediction object

**Returns**: List of alert strings

**Example**:
```python
alerts = generate_trading_alerts(prediction)
for alert in alerts:
    print(alert)
```

---

## Enums

### PredictionSignal

```python
class PredictionSignal(Enum):
    BULLISH = "BULLISH"              # Upside expected
    BEARISH = "BEARISH"              # Downside expected
    NEUTRAL = "NEUTRAL"              # No clear direction
    HIGH_VOLATILITY = "HIGH_VOLATILITY"  # Significant move expected
```

---

### RiskLevel

```python
class RiskLevel(Enum):
    LOW = "LOW"                      # <20% volatility
    MEDIUM = "MEDIUM"                # 20-35% volatility
    HIGH = "HIGH"                    # 35-50% volatility
    CRITICAL = "CRITICAL"            # >50% volatility
```

---

## Configuration Classes

### ThresholdConfig

Adjust thresholds in `config.py`:

```python
VOLATILITY_CRITICAL = 50      # % threshold
VOLATILITY_HIGH = 35
VOLATILITY_MEDIUM = 20

ANOMALY_CRITICAL_THRESHOLD = -0.5
DOWNSIDE_HIGH_PROBABILITY = 0.7
```

---

### ModelConfig

Adjust model parameters in `config.py`:

```python
GARCH_P = 1                    # GARCH order
GARCH_Q = 1
VOLATILITY_FORECAST_HORIZON = 5  # Days

MC_SIMULATIONS = 1000          # Monte Carlo paths
MC_DAYS = 5                    # Simulation days

SMA_SHORT = 20                 # Moving averages
SMA_LONG = 50
RSI_PERIOD = 14
```

---

## Error Handling

All classes handle errors gracefully:

```python
try:
    prediction = orchestrator.predict_market_activity("INVALID")
except Exception as e:
    print(f"Error: {str(e)}")
    # Fallback to basic analysis
```

---

## Performance Considerations

**Typical execution times**:
- Data fetch: 2-5 seconds
- Volatility forecast: 1-2 seconds
- Anomaly detection: 0.5 seconds
- Trend analysis: 0.5 seconds
- Risk assessment: 0.2 seconds
- GenAI analysis: 3-5 seconds
- **Total**: 7-15 seconds per ticker

---

## Thread Safety

Classes are designed for single-threaded use. For multi-threaded applications:

```python
from threading import Lock

class ThreadSafeOrchestrator:
    def __init__(self):
        self.orchestrator = MarketPredictionOrchestrator()
        self.lock = Lock()
    
    def predict(self, ticker):
        with self.lock:
            return self.orchestrator.predict_market_activity(ticker)
```

---

## API Key Management

### Setting API Key

```bash
# Environment variable
export GEMINI_API_KEY="your-api-key"

# Or in code
import os
os.environ['GEMINI_API_KEY'] = 'your-api-key'

# Or directly
orchestrator = MarketPredictionOrchestrator(api_key="your-api-key")
```

### Getting API Key

Visit: https://ai.google.dev/

---

## Best Practices

1. **Check Confidence**: High confidence predictions more reliable
2. **Assess Quality**: Use quality assessment before trading
3. **Respect Risk**: Follow risk level recommendations
4. **Combine Signals**: Don't rely on single prediction
5. **Monitor Updates**: Run periodic re-analysis
6. **Adjust Configuration**: Customize thresholds for your markets
7. **Log Predictions**: Track accuracy over time

---

## Troubleshooting

### API Errors

```python
try:
    orchestrator = MarketPredictionOrchestrator()
except Exception as e:
    print(f"GenAI unavailable: {str(e)}")
    # System works without GenAI features
```

### Data Errors

```python
try:
    prediction = orchestrator.predict_market_activity("BADTICKER")
except ValueError as e:
    print(f"Invalid ticker: {str(e)}")
```

### Performance Issues

```python
# Reduce simulations for faster execution
from market_agent import EventSimulator
EventSimulator.MC_SIMULATIONS = 500  # Default 1000
```

---

**Last Updated**: December 15, 2025
**Version**: 2.0 (Multi-Agent GenAI)
