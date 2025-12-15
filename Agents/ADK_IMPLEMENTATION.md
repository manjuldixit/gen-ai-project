# Market Prediction Agent - Google ADK Implementation Guide

## Overview

The Market Activity Prediction Agent has been enhanced to leverage **Google ADK (Agent Development Kit)** for superior multi-agent orchestration, cloud-native deployment, and enterprise-grade reliability.

---

## What is Google ADK?

**Google ADK (Agent Development Kit)** is a framework for building, deploying, and managing multi-agent AI systems on Google Cloud. It provides:

- **Built-in Agent Management**: Automatic agent lifecycle management
- **Cloud Integration**: Seamless Vertex AI integration
- **Tool Management**: Structured tool definitions and execution
- **Observability**: Built-in logging, monitoring, and tracing
- **Scalability**: Enterprise-grade deployment capabilities

---

## Architecture Changes

### Before (Direct Gemini API)
```
Application
    ↓
Direct Gemini API Calls
    ↓
GenAI Responses
```

### After (ADK-Based)
```
Application
    ↓
ADK App
    ├─ Root Agent (Market Prediction)
    ├─ Data Agent (Fetch & preprocess)
    ├─ Analysis Agent (Technical analysis)
    └─ Risk Agent (Risk assessment)
        ↓
    Tool Execution Layer
        ├─ fetch_market_data
        ├─ get_volatility_forecast
        ├─ detect_market_anomalies
        ├─ analyze_market_trend
        └─ assess_market_risk
        ↓
    Vertex AI / Local API
        ↓
    Structured Responses
```

---

## Key Components

### 1. ADK Agents

#### Root Agent
```python
root_agent = Agent(
    name="market_prediction_agent",
    model="gemini-3-pro-preview",
    instruction="Market prediction expert with access to all tools",
    tools=[all_tools]
)
```
- Orchestrates all analysis
- Synthesizes final predictions
- Makes final recommendations

#### Data Agent
```python
data_agent = Agent(
    name="market_data_agent",
    model="gemini-3-pro-preview",
    instruction="Financial data analyst",
    tools=[fetch_market_data, detect_market_anomalies]
)
```
- Fetches market data
- Detects anomalies
- Validates data quality

#### Analysis Agent
```python
analysis_agent = Agent(
    name="market_analysis_agent",
    model="gemini-3-pro-preview",
    instruction="Technical analyst",
    tools=[analyze_market_trend, get_volatility_forecast]
)
```
- Analyzes trends
- Forecasts volatility
- Identifies patterns

#### Risk Agent
```python
risk_agent = Agent(
    name="market_risk_agent",
    model="gemini-3-pro-preview",
    instruction="Risk management expert",
    tools=[assess_market_risk]
)
```
- Assesses risks
- Calculates position sizing
- Provides risk recommendations

### 2. ADK Tools

All tools are now defined as callable functions that ADK agents use:

```python
def fetch_market_data(ticker: str, period: str = "2y") -> str:
    """Tool for ADK: Fetch historical market data."""
    # Implementation
    
def get_volatility_forecast(ticker: str) -> str:
    """Tool for ADK: Get volatility forecast."""
    # Implementation
    
def detect_market_anomalies(ticker: str) -> str:
    """Tool for ADK: Detect market anomalies."""
    # Implementation
```

### 3. ADK App

```python
app = App(
    root_agent=root_agent,
    name="market-activity-prediction-agent",
    description="Multi-agent GenAI system for market prediction"
)
```

---

## Installation & Setup

### 1. Install ADK

```bash
pip install google-adk
```

### 2. Google Cloud Authentication

#### Option A: Vertex AI (Recommended)
```bash
# Authenticate with Google Cloud
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

#### Option B: API Key (Local Development)
```bash
export GOOGLE_API_KEY="your-api-key"
```

### 3. Initialize Environment

```python
from adk_config import initialize_adk_environment
initialize_adk_environment()
```

---

## Usage Examples

### Example 1: Basic Market Analysis

```python
from market_agent import app, root_agent

# Option A: Using the ADK app directly
# (Implementation depends on ADK app execution methods)

# Option B: Using legacy orchestrator with ADK
from market_agent import MarketPredictionOrchestrator

orchestrator = MarketPredictionOrchestrator()
prediction = orchestrator.predict_market_activity("NVDA")

print(f"Signal: {prediction.signal.value}")
print(f"Confidence: {prediction.confidence:.1%}")
print(f"Risk: {prediction.risk_level.value}")
```

### Example 2: Access Specific ADK Agents

```python
from market_agent import data_agent, analysis_agent, risk_agent

# Data agent focuses on data acquisition
# Analysis agent performs technical analysis
# Risk agent assesses risks
# Root agent orchestrates all
```

### Example 3: Custom ADK Agent

```python
from google.adk.agents import Agent

# Create custom agent with specific tools
custom_agent = Agent(
    name="custom_market_agent",
    model="gemini-3-pro-preview",
    instruction="Your custom instructions here",
    tools=[your_tools]
)
```

---

## Configuration

### Using ADK Configuration

```python
from adk_config import get_adk_config, ADKAgentConfig, ADKAppConfig

# Get environment-specific config
config = get_adk_config("production")

# Or use specific config classes
agent_config = ADKAgentConfig()
app_config = ADKAppConfig()
```

### Configuration Options

**Agent Configuration** (`adk_config.py`):
- Model selection (gemini-3-pro-preview)
- Agent names and descriptions
- Tool timeouts and retry policies

**App Configuration**:
- Application name and version
- Logging and monitoring settings
- Cloud vs local deployment

**Cloud Configuration**:
- Project ID
- Location (regional settings)
- Vertex AI vs API key

---

## Deployment Options

### Option 1: Local Development

```bash
# Set API key
export GOOGLE_API_KEY="your-key"

# Run the agent
python market_agent.py
```

### Option 2: Google Cloud Vertex AI

```bash
# Authentication
gcloud auth application-default login
gcloud config set project YOUR_PROJECT

# Set environment
export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"

# Deploy app (using ADK deployment)
# Details depend on ADK deployment framework
```

### Option 3: Google Cloud Run

```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "market_agent.py"]
```

---

## Tools Overview

### Tool: fetch_market_data
- **Purpose**: Fetch historical market data
- **Input**: ticker (str), period (str)
- **Output**: JSON with data summary
- **Used by**: Data agent, Root agent

### Tool: get_volatility_forecast
- **Purpose**: Forecast market volatility
- **Input**: ticker (str)
- **Output**: JSON with volatility metrics
- **Used by**: Analysis agent, Root agent

### Tool: detect_market_anomalies
- **Purpose**: Detect unusual market behavior
- **Input**: ticker (str)
- **Output**: JSON with anomaly detection results
- **Used by**: Data agent, Root agent

### Tool: analyze_market_trend
- **Purpose**: Analyze technical trends
- **Input**: ticker (str)
- **Output**: JSON with trend analysis
- **Used by**: Analysis agent, Root agent

### Tool: assess_market_risk
- **Purpose**: Assess market risk
- **Input**: ticker (str)
- **Output**: JSON with risk metrics
- **Used by**: Risk agent, Root agent

---

## Agent Instructions

### Root Agent Instruction
```
You are the Market Activity Prediction Agent, an expert financial analyst 
powered by Google ADK.

Your responsibilities:
1. Analyze market data using the data agent
2. Perform technical analysis using the analysis agent
3. Assess risks using the risk agent
4. Synthesize insights into actionable predictions
5. Explain predictions with clear reasoning

Provide market predictions with:
- Clear signal (BULLISH/BEARISH/NEUTRAL)
- Confidence level
- Risk assessment
- Actionable recommendations

Keep responses professional and data-driven.
```

### Data Agent Instruction
```
You are a financial data analyst. Use tools to fetch and summarize market data.

Focus on:
- Data quality and completeness
- Anomaly detection
- Market regime identification
- Data validation
```

### Analysis Agent Instruction
```
You are a technical analyst. Use tools to analyze market trends and volatility.

Focus on:
- Trend direction and strength
- Volatility patterns
- Support/resistance levels
- Technical indicators
```

### Risk Agent Instruction
```
You are a risk management expert. Use tools to assess market risk and 
provide recommendations.

Focus on:
- Risk quantification
- Position sizing
- Risk mitigation strategies
- Risk-adjusted recommendations
```

---

## Logging & Monitoring

### Enable Logging

```python
import logging
from adk_config import ADKLoggingConfig

logging.basicConfig(
    format=ADKLoggingConfig.LOG_FORMAT,
    level=ADKLoggingConfig.ROOT_LEVEL
)

logger = logging.getLogger(__name__)
logger.info("Market prediction agent started")
```

### Monitor Agent Performance

```python
from adk_config import ADKMonitoringConfig

# Metrics are automatically collected if enabled
if ADKMonitoringConfig.ENABLE_METRICS:
    # View metrics in Cloud Monitoring
    pass

# Tracing is automatically enabled if ENABLE_TRACING is True
if ADKMonitoringConfig.ENABLE_TRACING:
    # View traces in Cloud Trace
    pass
```

---

## Error Handling

### Tool Timeouts
```python
# Configured in ADKToolConfig
FETCH_DATA_TIMEOUT = 30
ANALYSIS_TIMEOUT = 20
RISK_TIMEOUT = 10

# Automatically handled by ADK
```

### Retries
```python
# Configured in ADKToolConfig
FETCH_DATA_RETRIES = 3

# Automatically handled by ADK
```

### Fallback Strategies
```python
# Sequential fallback if parallel fails
FALLBACK_TO_SEQUENTIAL = True

# Agent will retry with sequential execution
```

---

## Performance Considerations

### Parallel Execution
```python
# Enable parallel agent execution
ENABLE_PARALLEL_AGENTS = True
MAX_PARALLEL_AGENTS = 3

# Data agent, Analysis agent, and Risk agent run in parallel
# Root agent waits for results then synthesizes
```

### Caching
```python
# Enable result caching
ENABLE_CACHING = True
CACHE_TTL_SECONDS = 3600  # 1 hour

# Identical requests cached to improve performance
```

### Resource Optimization
- ADK automatically manages agent lifecycles
- Tools are pooled and reused
- Memory is optimized across agents
- Network calls are batched when possible

---

## Migration Guide

### From Direct Gemini API to ADK

#### Before
```python
import google.generativeai as genai
genai.configure(api_key="YOUR_KEY")
model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content(prompt)
```

#### After
```python
from google.adk.agents import Agent

agent = Agent(
    name="market_prediction_agent",
    model="gemini-3-pro-preview",
    instruction="Your system instruction",
    tools=[your_tools]
)

# Agent uses tools and models automatically
```

### Backward Compatibility

The original `MarketActivityAgent` class and `MarketPredictionOrchestrator` remain unchanged and functional. They can coexist with ADK implementation.

---

## Best Practices

### 1. Agent Design
- One agent per specialized task
- Clear, focused instructions
- Appropriate tool assignment
- Timeout configuration

### 2. Tool Design
- Single responsibility principle
- Clear input/output contracts
- Error handling and validation
- Timeout configuration

### 3. Monitoring
- Log important decisions
- Monitor tool performance
- Track prediction accuracy
- Alert on anomalies

### 4. Deployment
- Use Vertex AI for production
- Use API key for development
- Implement proper authentication
- Monitor resource usage

---

## Troubleshooting

### Issue: "Agent not initialized"
**Solution**: Ensure ADK is installed and environment is properly configured.
```bash
pip install google-adk
export GOOGLE_API_KEY="your-key"
```

### Issue: "Tool execution timeout"
**Solution**: Increase timeout in `adk_config.py`:
```python
FETCH_DATA_TIMEOUT = 60  # Increase from 30
```

### Issue: "Agent cannot access tool"
**Solution**: Ensure tool is registered with agent:
```python
agent = Agent(
    name="my_agent",
    tools=[my_tool]  # Must be in this list
)
```

### Issue: "Google Cloud authentication failed"
**Solution**: Authenticate with Google Cloud:
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

---

## Next Steps

1. **Install ADK**: `pip install google-adk`
2. **Set up authentication**: Use Vertex AI or API key
3. **Review examples.py**: See ADK usage examples
4. **Configure adk_config.py**: Adjust for your environment
5. **Test locally**: Run market_agent.py
6. **Deploy to cloud**: Use Cloud Run or App Engine

---

## Resources

- **ADK Documentation**: [Google ADK Docs](https://cloud.google.com/docs/adk)
- **Vertex AI**: [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- **Market Agent Guide**: See README_MARKET_AGENT.md
- **API Reference**: See API_REFERENCE.md

---

## Support

For issues or questions:
1. Check adk_config.py for configuration options
2. Review logging output for error details
3. Check ADK documentation for framework issues
4. Review market_agent.py for implementation details

---

**Version**: 2.0-adk (ADK-Enhanced)
**Last Updated**: December 15, 2025
**Status**: ✅ Ready for Deployment
