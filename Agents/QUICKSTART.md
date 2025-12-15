# Market Activity Prediction Agent - Quick Start Guide

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Set Up API Key (Optional but Recommended)

For GenAI-powered analysis, set your Gemini API key:

```bash
export GEMINI_API_KEY="your-google-ai-api-key"
```

Get your API key at: https://ai.google.dev

### Step 3: Run Basic Example

```python
from market_agent import MarketActivityAgent

# Initialize agent
agent = MarketActivityAgent()

# Analyze a ticker
result = agent.analyze("NVDA")

# Print results
import json
print(json.dumps(result, indent=2))
```

### Step 4: Try Advanced Analysis

```python
from market_agent import MarketPredictionOrchestrator

# Initialize orchestrator (uses GEMINI_API_KEY if available)
orchestrator = MarketPredictionOrchestrator()

# Get AI-powered prediction
prediction = orchestrator.predict_market_activity("AAPL")

# View results
print(f"Signal: {prediction.signal.value}")
print(f"Confidence: {prediction.confidence:.1%}")
print(f"Risk: {prediction.risk_level.value}")
print(f"\nReasoning:\n{prediction.reasoning}")
```

---

## 📊 Common Tasks

### Get Trading Recommendation

```python
from market_agent import MarketPredictionOrchestrator
from utils import get_action_recommendation

orchestrator = MarketPredictionOrchestrator()
prediction = orchestrator.predict_market_activity("TSLA")

# Get action
action = get_action_recommendation(prediction.signal, prediction.confidence)
print(f"Action: {action}")
```

### Calculate Position Size

```python
from utils import calculate_position_sizing

position = calculate_position_sizing(
    risk_level=prediction.risk_level,
    account_size=100000,  # $100k account
    confidence=prediction.confidence
)

print(f"Position Size: ${position['recommended_position_size']:,.2f}")
print(f"Stop Loss Distance: {position['stop_loss_distance']:.1%}")
```

### Assess Prediction Quality

```python
from utils import assess_prediction_quality

quality = assess_prediction_quality(prediction)

print(f"Quality Score: {quality['overall_quality_score']}/100")
print(f"Rating: {quality['reliability_rating']}")
print(f"Suitable for Trading: {quality['suitable_for_trading']}")
```

### Compare Multiple Tickers

```python
from market_agent import MarketPredictionOrchestrator
from utils import compare_predictions

orchestrator = MarketPredictionOrchestrator()

tickers = ["NVDA", "AAPL", "MSFT"]
predictions = [orchestrator.predict_market_activity(t) for t in tickers]

# Get consensus
bullish = sum(1 for p in predictions if p.signal.value == "BULLISH")
print(f"Bullish signals: {bullish}/{len(predictions)}")
```

---

## ⚙️ Configuration

All configuration is in `config.py`. Key settings:

```python
# Volatility thresholds
VOLATILITY_CRITICAL = 50    # %
VOLATILITY_HIGH = 35        # %

# GARCH model
GARCH_P = 1
GARCH_Q = 1
VOLATILITY_FORECAST_HORIZON = 5

# Monte Carlo
MC_SIMULATIONS = 1000
MC_DAYS = 5

# Risk weights
VOLATILITY_WEIGHT = 0.3
ANOMALY_WEIGHT = 0.25
DIRECTIONAL_WEIGHT = 0.25
RANGE_WEIGHT = 0.2
```

Modify these values based on your preferences.

---

## 📈 Understanding Outputs

### Prediction Signal

- **BULLISH**: Strong upside expected
- **BEARISH**: Strong downside expected
- **NEUTRAL**: No clear direction
- **HIGH_VOLATILITY**: Significant movement expected

### Risk Levels

| Level | Volatility | Action |
|-------|-----------|--------|
| LOW | <20% | Standard trading |
| MEDIUM | 20-35% | Use stop-losses |
| HIGH | 35-50% | Reduce leverage |
| CRITICAL | >50% | Minimize position |

### Confidence

- 0.5-0.6: Low confidence
- 0.6-0.8: Moderate confidence
- 0.8-0.95: High confidence

---

## 🔧 Troubleshooting

### Issue: "GEMINI_API_KEY not found"

**Solution**: The system works without GenAI features. For full functionality:
```bash
export GEMINI_API_KEY="your-key"
```

### Issue: "No market data available"

**Solution**: Check internet connection. yfinance requires live data:
```python
import yfinance
yf.Ticker("AAPL").history(period="1d")  # Test connection
```

### Issue: "Insufficient data points"

**Solution**: Try a different ticker or longer period:
```python
prediction = orchestrator.predict_market_activity("SPY")
```

---

## 💡 Tips & Best Practices

1. **Combine with Other Analysis**: Use as one input, not sole decision factor
2. **Monitor Risk**: Always respect position sizing recommendations
3. **Check Quality**: Use `assess_prediction_quality()` to validate results
4. **Track Performance**: Keep records of predictions vs actual outcomes
5. **Adjust Thresholds**: Customize `config.py` based on your markets
6. **Use Confidence**: Higher confidence predictions are more reliable
7. **Watch for Anomalies**: High anomaly scores indicate unusual conditions

---

## 📚 Full Examples

See `examples.py` for comprehensive examples:

```bash
python examples.py
```

This demonstrates all features:
- Legacy agent analysis
- Multi-agent prediction
- Quality assessment
- Trading guidance
- Comparative analysis
- Detailed reporting

---

## 🎯 Next Steps

1. ✅ Install dependencies
2. ✅ Set GEMINI_API_KEY
3. ✅ Run basic example
4. ✅ Try advanced analysis
5. ✅ Review `config.py` settings
6. ✅ Explore `utils.py` functions
7. ✅ Integrate into your trading system

---

## 📖 Full Documentation

For detailed information, see `README_MARKET_AGENT.md`

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**. Not financial advice. Consult professionals before investing.

---

**Version**: 2.0 (Multi-Agent GenAI)
**Updated**: December 15, 2025
