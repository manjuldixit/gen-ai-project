# Market Activity Prediction Agent – Multi-Agent GenAI Solution

## Overview

This is a sophisticated **multi-agent GenAI system** designed to predict short-term market activity by combining historical market data, technical analysis, statistical modeling, and AI-powered reasoning. The solution is built to meet the 2025 CCIBT GenAI Hackathon judging criteria by being:

- **GenAI-Central**: Uses Google Gemini for intelligent analysis and explainability
- **Agentic**: Multi-agent orchestration with specialized agents for different tasks
- **Explainable**: Provides transparent reasoning for all predictions
- **Risk-Aware**: Comprehensive risk assessment and position sizing guidance

---

## System Architecture

### Multi-Agent Orchestration

The system employs specialized agents working in concert:

```
┌─────────────────────────────────────────────────────────────┐
│         Market Prediction Orchestrator (Main Agent)         │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┬──────────────┬─────────────┐
        ▼            ▼            ▼              ▼             ▼
   ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌─────────────┐ ┌───────────┐
   │Data Coll│ │Trend Ana-│ │Volatil- │ │Anomaly Dete│ │Risk Assess│
   │  Agent  │ │  lysis   │ │ity Agent│ │  ction     │ │  Agent    │
   └─────────┘ └──────────┘ └─────────┘ └─────────────┘ └───────────┘
        │            │            │              │             │
        └────────────┴────────────┴──────────────┴─────────────┘
                     │
        ┌────────────▼─────────────┐
        │  GenAI Reasoning Agent   │
        │   (Gemini Integration)   │
        └──────────────────────────┘
                     │
        ┌────────────▼─────────────┐
        │  Prediction Signal +     │
        │  Confidence + Reasoning  │
        └──────────────────────────┘
```

### Agent Responsibilities

#### 1. **Data Collection Agent**
- Fetches historical price and volume data via yfinance
- Calculates returns and log returns
- Identifies upcoming market events
- Preprocesses data for downstream analysis

#### 2. **Trend Analysis Agent**
- Calculates Simple Moving Averages (SMA-20, SMA-50)
- Computes RSI (Relative Strength Index)
- Calculates MACD (Moving Average Convergence Divergence)
- Identifies trend direction and strength

#### 3. **Volatility Forecasting Agent**
- Uses GARCH(1,1) model for volatility prediction
- Forecasts conditional volatility over 5-day horizon
- Detects volatility regimes
- Adjusts for upcoming events

#### 4. **Anomaly Detection Agent**
- Uses Isolation Forest for outlier detection
- Flags unusual price/volume behavior
- Computes anomaly severity scores
- Contextualizes current market state

#### 5. **Risk Assessment Agent**
- Evaluates multi-dimensional risk factors
- Assigns risk levels (LOW/MEDIUM/HIGH/CRITICAL)
- Provides risk-based trading recommendations
- Calculates position sizing guidance

#### 6. **GenAI Reasoning Agent**
- Contextualizes all quantitative factors
- Generates human-readable explanations
- Provides actionable insights
- Explains prediction rationale transparently

#### 7. **Market Prediction Orchestrator**
- Coordinates all sub-agents
- Aggregates signals and confidence scores
- Synthesizes final market predictions
- Produces comprehensive reports

---

## Key Features

### Prediction Capabilities

1. **Market Signal Generation**
   - BULLISH: Positive price direction expected
   - BEARISH: Negative price direction expected
   - NEUTRAL: No clear directional bias
   - HIGH_VOLATILITY: Significant price movement expected

2. **Confidence Scoring**
   - 0.5-1.0 confidence range
   - Based on signal strength and data consistency
   - Adjusts for anomalies and uncertainty

3. **Risk Assessment**
   - Multi-factor risk evaluation
   - Risk levels: LOW, MEDIUM, HIGH, CRITICAL
   - Risk score: 0-100

4. **Volatility Forecasting**
   - Annualized volatility forecast
   - 5-day price target range (95% confidence)
   - Event-adjusted volatility

### Explainability

- **AI-Powered Reasoning**: Gemini generates human-readable explanations
- **Transparent Factors**: Lists key factors driving predictions
- **Technical Indicators**: Detailed technical analysis breakdown
- **Risk Warnings**: Clear communication of risk factors
- **Trading Recommendations**: Actionable guidance based on signal

### Risk Management

- **Position Sizing**: Calculates appropriate position sizes based on risk
- **Stop-Loss Guidance**: Recommends stop-loss distances
- **Take-Profit Targets**: Suggests profit-taking levels
- **Market Regime Analysis**: Identifies current market conditions
- **Trading Alerts**: Generates actionable alerts for traders

---

## Technical Stack

### Dependencies

```
python >= 3.8
pandas >= 1.3.0
numpy >= 1.21.0
yfinance >= 0.1.70
arch >= 4.15
scikit-learn >= 0.24.0
scipy >= 1.7.0
google-generativeai >= 0.3.0
```

### Core Technologies

- **Data Processing**: pandas, numpy
- **Statistical Modeling**: arch (GARCH models), scipy
- **Machine Learning**: scikit-learn (Isolation Forest)
- **Market Data**: yfinance
- **Generative AI**: Google Gemini 2.0 Flash
- **Financial Calculations**: Custom implementations

---

## Usage Examples

### Basic Usage (Legacy Agent)

```python
from market_agent import MarketActivityAgent

agent = MarketActivityAgent()
result = agent.analyze("NVDA")
print(result)
```

### Multi-Agent System with GenAI

```python
from market_agent import MarketPredictionOrchestrator
import os

# Set up API key
os.environ['GEMINI_API_KEY'] = 'your-api-key'

# Initialize orchestrator
orchestrator = MarketPredictionOrchestrator()

# Get prediction
prediction = orchestrator.predict_market_activity("NVDA")

print(f"Signal: {prediction.signal.value}")
print(f"Confidence: {prediction.confidence:.1%}")
print(f"Risk Level: {prediction.risk_level.value}")
print(f"Reasoning: {prediction.reasoning}")
```

### Comprehensive Analysis

```python
from market_agent import MarketPredictionOrchestrator
from utils import (
    format_prediction_report,
    get_action_recommendation,
    calculate_position_sizing,
    assess_prediction_quality
)

orchestrator = MarketPredictionOrchestrator()

# Get prediction
prediction = orchestrator.predict_market_activity("AAPL")

# Get action recommendation
action = get_action_recommendation(prediction.signal, prediction.confidence)
print(f"Action: {action}")

# Calculate position sizing
position = calculate_position_sizing(prediction.risk_level, 100000, prediction.confidence)
print(f"Position Size: ${position['recommended_position_size']:,.2f}")

# Assess quality
quality = assess_prediction_quality(prediction)
print(f"Quality Rating: {quality['reliability_rating']}")

# Format report
report = format_prediction_report(prediction, 150.25)
print(report)
```

### Comparative Analysis

```python
from market_agent import MarketPredictionOrchestrator
from utils import compare_predictions

orchestrator = MarketPredictionOrchestrator()

# Analyze multiple tickers
tickers = ["NVDA", "AAPL", "MSFT", "GOOGL"]
predictions = [orchestrator.predict_market_activity(t) for t in tickers]

# Get consensus
consensus = compare_predictions(predictions)
print(f"Consensus: {consensus['consensus']}")
print(f"Strength: {consensus['consensus_strength']:.0%}")
```

---

## Output Structure

### MarketPrediction Object

```python
@dataclass
class MarketPrediction:
    ticker: str                          # Stock ticker symbol
    signal: PredictionSignal            # BULLISH/BEARISH/NEUTRAL/HIGH_VOLATILITY
    confidence: float                    # 0.5-1.0 confidence score
    volatility_forecast: float          # Annualized volatility percentage
    price_target_range: tuple           # (lower_bound, upper_bound)
    risk_level: RiskLevel               # LOW/MEDIUM/HIGH/CRITICAL
    reasoning: str                      # AI-generated explanation
    key_factors: List[str]              # Factors driving prediction
    timestamp: str                      # ISO timestamp
```

### Detailed Report Structure

```python
{
    "summary": {
        "ticker": "NVDA",
        "timestamp": "2025-12-15T...",
        "signal": "BULLISH",
        "confidence": 0.82,
        "current_price": 150.25
    },
    "prediction": {
        "volatility_forecast_annual": "22.45%",
        "price_target_range": [145.50, 165.25],
        "risk_level": "MEDIUM"
    },
    "analysis": {
        "reasoning": "AI-generated detailed explanation...",
        "key_factors": ["Technical factors...", "Risk factors...", ...]
    },
    "technical_indicators": {
        "trend": "UPTREND",
        "sma_20": 148.50,
        "sma_50": 145.75,
        "rsi": 65.2,
        "macd_signal": "POSITIVE"
    }
}
```

---

## Configuration

### Key Configuration Files

#### `config.py`

Contains all system configuration:

- **APIConfig**: API keys and timeouts
- **ModelConfig**: ML model parameters
- **ThresholdConfig**: Risk and signal thresholds
- **GenAIConfig**: Gemini AI settings
- **AgentDescriptions**: Role documentation

### Environment Variables

```bash
# Required for GenAI features
export GEMINI_API_KEY="your-api-key"

# Optional: Market data API configuration
export YFINANCE_TIMEOUT=30
```

---

## Risk Assessment Details

### Risk Level Determination

| Factor | LOW | MEDIUM | HIGH | CRITICAL |
|--------|-----|--------|------|----------|
| Volatility | <20% | 20-35% | 35-50% | >50% |
| Anomaly Score | >-0.2 | -0.2 to -0.5 | <-0.5 | Extreme |
| Downside Prob | <35% | 35-65% | 65-70% | >70% |
| Price Range | <10% | 10-15% | 15-20% | >20% |

### Risk Recommendations

- **LOW**: Standard trading conditions, normal leverage acceptable
- **MEDIUM**: Use appropriate stop-losses, monitor positions
- **HIGH**: Reduce leverage, increase monitoring frequency
- **CRITICAL**: Significantly reduce position size, consider hedging

---

## Performance Considerations

### Computational Efficiency

- Data fetching: ~2-5 seconds per ticker
- Volatility modeling (GARCH): ~1-2 seconds
- Anomaly detection: ~0.5 seconds
- GenAI analysis: ~3-5 seconds (API dependent)
- **Total per ticker**: ~7-15 seconds

### Scalability

- System scales to analyze multiple tickers simultaneously
- Batch processing supported for comparative analysis
- Efficient vectorized calculations using numpy/pandas
- API rate limiting handled gracefully

---

## Examples

### Running Examples

```bash
# Run all examples
python examples.py

# Individual examples
python -c "from examples import example_1_legacy_agent; example_1_legacy_agent()"
```

### Example Output

```
================================================================================
  EXAMPLE 1: Legacy Market Analysis Agent
================================================================================

Ticker: NVDA
Current Price: $150.25
Alert Level: MEDIUM
Volatility (Annualized): 22.45%
5-Day Forecast Range: $145.50 - $165.25
Direction Bias: Bullish
Confidence: 0.85

Rationale:
Market is currently in a normal state with moderate volatility. 
Skewed upside potential detected in Monte Carlo simulations...
```

---

## Advanced Topics

### Custom Model Parameters

Modify `config.py` to adjust model behavior:

```python
class ModelConfig:
    GARCH_P = 2  # Increase GARCH order
    GARCH_Q = 2
    ANOMALY_CONTAMINATION = 0.10  # Adjust anomaly detection sensitivity
    MC_SIMULATIONS = 5000  # More simulations for robustness
    MC_DAYS = 10  # Longer forecast horizon
```

### Extending Agents

Add custom agents by subclassing base agent structure:

```python
class CustomAgent:
    """Custom analysis agent."""
    
    def analyze(self, data):
        # Custom analysis logic
        return insights

# Integrate into orchestrator
orchestrator.custom_agent = CustomAgent()
```

### Data Sources

Currently supports yfinance. Can extend to:

- Alternative market data providers (API connectors)
- Real-time data feeds (WebSocket integration)
- Alternative/crypto markets (Crypto exchanges APIs)
- Macro economic data (FRED, Quandl, etc.)

---

## Limitations & Assumptions

1. **Historical Performance**: Past volatility not guaranteed to predict future volatility
2. **Black Swan Events**: Cannot predict unprecedented market shocks
3. **Data Quality**: Results depend on data accuracy from yfinance
4. **Market Hours**: Analysis based on trading day data
5. **Event Impact**: Event volatility adjustment is heuristic
6. **GenAI Disclaimers**: AI reasoning is helpful but not infallible

---

## Disclaimers

**⚠️ IMPORTANT: FOR EDUCATIONAL AND INFORMATIONAL PURPOSES ONLY**

This tool provides AI-generated predictions and analysis for **educational purposes only**. It does not constitute financial advice, investment recommendations, or endorsements.

- **Past Performance**: Not indicative of future results
- **Risk Disclosure**: All investments carry risk, including potential loss
- **Professional Advice**: Consult qualified financial advisors before investing
- **No Guarantees**: Market predictions are probabilistic, not certain
- **Your Responsibility**: Investment decisions are your responsibility

---

## Contributing

Contributions are welcome! Areas for enhancement:

- Additional data sources and macro indicators
- Alternative volatility models (EGARCH, FIGARCH)
- Machine learning-based signal generation
- Real-time market monitoring
- Integration with trading APIs
- Custom risk models
- Portfolio-level analysis

---

## References

### Academic

- GARCH Models: Bollerslev, T. (1986)
- Anomaly Detection: Liu, F. T., et al. (2008)
- Monte Carlo Methods: Glasserman, P. (2004)
- Technical Analysis: Murphy, J. J. (1999)

### Libraries

- [arch-py](https://arch.readthedocs.io/): GARCH modeling
- [scikit-learn](https://scikit-learn.org/): Machine learning
- [yfinance](https://pypi.org/project/yfinance/): Market data
- [Google Generative AI](https://ai.google.dev/): Gemini API

---

## Support

For issues or questions:

1. Check configuration (API keys, dependencies)
2. Review logs and error messages
3. Verify market data availability
4. Test with simple examples first
5. Check GenAI API quota and limits

---

## License

This project is provided as-is for educational purposes. Use at your own risk.

**Last Updated**: December 15, 2025
**Version**: 2.0 (Multi-Agent GenAI Enhanced)
