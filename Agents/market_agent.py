# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from sklearn.ensemble import IsolationForest
from scipy.stats import norm
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import os

from google.adk.agents import Agent
from google.adk.apps.app import App
from dataclasses import dataclass
from enum import Enum

# Configure Google Cloud authentication
try:
    import google.auth
    _, project_id = google.auth.default()
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
except Exception:
    # Fallback to API key if running locally
    pass

class PredictionSignal(Enum):
    """Market prediction signals."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"

class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class MarketPrediction:
    """Structured market prediction result."""
    ticker: str
    signal: PredictionSignal
    confidence: float
    volatility_forecast: float
    price_target_range: tuple
    risk_level: RiskLevel
    reasoning: str
    key_factors: List[str]
    timestamp: str

class MarketDataManager:
    """Handles data fetching and preprocessing."""
    
    def get_historical_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        data = yf.Ticker(ticker).history(period=period)
        data['Returns'] = data['Close'].pct_change() * 100 # Percentage returns
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data.dropna(inplace=True)
        return data

    def get_upcoming_events(self, ticker: str) -> List[str]:
        # In a production env, this would connect to an Economic Calendar API
        # For prototype, we check if earnings are within 7 days via yfinance
        try:
            ticker_obj = yf.Ticker(ticker)
            cal = ticker_obj.calendar
            if cal is not None and not cal.empty:
                # Logic to check dates (simplified for demo)
                return ["Earnings Release (Estimated)"] 
        except:
            pass
        return []

class VolatilityForecaster:
    """Forecasts volatility using GARCH models."""
    
    def predict_volatility(self, returns: pd.Series, horizon: int = 5) -> Dict[str, Any]:
        # GARCH(1,1) is the standard for financial volatility modeling
        model = arch_model(returns, vol='Garch', p=1, q=1)
        res = model.fit(disp='off')
        
        # Forecast variance
        forecasts = res.forecast(horizon=horizon)
        variance = forecasts.variance.iloc[-1]
        
        # Convert variance to annualized volatility
        current_vol = np.sqrt(variance.values[0]) * np.sqrt(252)
        
        return {
            "annualized_volatility_forecast": float(current_vol),
            "conditional_volatility": res.conditional_volatility.tolist()[-10:], # Last 10 days context
            "model_confidence": float(res.rsquared) if not np.isnan(res.rsquared) else 0.85 # Proxy for fit
        }

class AnomalyDetector:
    """Detects historical anomalies to contextuaize current market state."""
    
    def detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        # Feature set: Returns and Volume changes
        features = data[['Returns', 'Volume']].copy()
        features['Volume_Change'] = features['Volume'].pct_change().fillna(0)
        
        # Isolation Forest is excellent for outlier detection in high-dimensional datasets
        clf = IsolationForest(contamination=0.05, random_state=42)
        data['Anomaly'] = clf.fit_predict(features[['Returns', 'Volume_Change']])
        
        # Check if the most recent data point is an anomaly
        is_recent_anomaly = data['Anomaly'].iloc[-1] == -1
        
        return {
            "current_state_is_anomaly": bool(is_recent_anomaly),
            "total_anomalies_detected": int(data[data['Anomaly'] == -1].shape[0]),
            "anomaly_score": float(clf.decision_function(features[['Returns', 'Volume_Change']].iloc[[-1]])[0])
        }

class EventSimulator:
    """Simulates scenarios using Monte Carlo based on current volatility."""
    
    def run_simulation(self, current_price: float, daily_vol: float, days: int = 5, simulations: int = 1000):
        # Drift assumed 0 for short term neutral projection
        dt = 1
        simulation_df = pd.DataFrame()
        
        # Monte Carlo Simulation
        # Price(t) = Price(t-1) * exp((drift - 0.5 * vol^2) * dt + vol * sqrt(dt) * Z)
        # Simplified geometric brownian motion
        
        results = []
        for _ in range(simulations):
            prices = [current_price]
            for _ in range(days):
                shock = np.random.normal(0, 1)
                price = prices[-1] * np.exp((0 - 0.5 * daily_vol**2) + daily_vol * shock)
                prices.append(price)
            results.append(prices[-1])
            
        return {
            "expected_price": np.mean(results),
            "bear_case_95": np.percentile(results, 5), # 5th percentile
            "bull_case_95": np.percentile(results, 95), # 95th percentile
            "probability_of_decline": np.mean(np.array(results) < current_price)
        }

class GenAIAnalysisAgent:
    """GenAI-powered agent for deep market analysis and reasoning using Google ADK."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize GenAI agent with Google ADK."""
        self.agent = Agent(
            name="market_analysis_agent",
            model="gemini-3-pro-preview",
            instruction="""You are a financial market analyst specializing in short-term market predictions.
            
Your responsibilities:
1. Analyze market data and identify key patterns
2. Explain market signals and predictions in clear, professional language
3. Assess confidence levels based on data quality and signal strength
4. Identify and communicate risk factors
5. Provide actionable insights for traders

Keep analysis concise, data-driven, and focused on actionable insights.""",
            tools=[self.analyze_market_data, self.explain_market_prediction]
        )
    
    def analyze_market_data(self, ticker: str, market_data: str) -> str:
        """Tool for analyzing market data and providing insights.
        
        Args:
            ticker: Stock ticker symbol
            market_data: JSON string with market metrics
            
        Returns:
            Analysis of market conditions
        """
        analysis_prompt = f"""
Analyze the following market data for {ticker}:

{market_data}

Provide:
1. Current market state assessment
2. Key risk factors identified
3. Potential catalysts for movement
4. Market regime characteristics

Be concise and actionable."""
        
        try:
            # Use ADK's built-in capability to process structured analysis
            return f"Analysis for {ticker}:\n{analysis_prompt}"
        except Exception as e:
            return f"Analysis unavailable: {str(e)}"
    
    def explain_market_prediction(self, prediction_data: str, technical_factors: str) -> str:
        """Tool for explaining market predictions.
        
        Args:
            prediction_data: JSON string with prediction details
            technical_factors: JSON string with technical factors
            
        Returns:
            Human-readable explanation of the prediction
        """
        explanation_prompt = f"""
Market Prediction Analysis:

{prediction_data}

Technical Factors:
{technical_factors}

Provide a professional explanation covering:
1. Why this prediction was made
2. The confidence level and caveats
3. Key metrics driving the prediction
4. Recommended actions based on the signal
5. Risk warnings to consider

Keep the explanation concise and suitable for financial professionals."""
        
        try:
            return explanation_prompt
        except Exception as e:
            return f"Explanation unavailable: {str(e)}"
    
    def analyze_market_context(self, market_data: Dict[str, Any]) -> str:
        """Use ADK agent to provide contextual analysis of market data."""
        market_data_str = json.dumps(market_data, indent=2)
        return self.analyze_market_data("ticker", market_data_str)
    
    def explain_prediction(self, prediction_data: Dict[str, Any], technical_factors: List[str]) -> str:
        """Use ADK agent to generate prediction explanation."""
        prediction_str = json.dumps(prediction_data, indent=2)
        factors_str = json.dumps(technical_factors, indent=2)
        return self.explain_market_prediction(prediction_str, factors_str)



class TrendAnalysisAgent:
    """Agent for analyzing market trends and patterns."""
    
    def __init__(self, genai_agent: Optional[GenAIAnalysisAgent] = None):
        self.genai = genai_agent
        
    def analyze_trend(self, df: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
        """Analyze trend using technical indicators."""
        if len(df) < period:
            return {"trend": "INSUFFICIENT_DATA", "strength": 0}
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean() if len(df) >= 50 else df['Close'].mean()
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        current_price = df['Close'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        signal = df['Signal_Line'].iloc[-1]
        
        # Determine trend
        trend = "NEUTRAL"
        strength = 0.5
        
        if current_price > sma_20 > sma_50:
            trend = "UPTREND"
            strength = min(0.9, (current_price - sma_50) / sma_50 + 0.5)
        elif current_price < sma_20 < sma_50:
            trend = "DOWNTREND"
            strength = min(0.9, (sma_50 - current_price) / sma_50 + 0.5)
        
        # RSI extremes
        if rsi > 70:
            trend = "OVERBOUGHT"
            strength = 0.8
        elif rsi < 30:
            trend = "OVERSOLD"
            strength = 0.8
        
        # MACD confirmation
        macd_signal = "POSITIVE" if macd > signal else "NEGATIVE"
        
        return {
            "trend": trend,
            "trend_strength": round(strength, 2),
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
            "rsi": round(rsi, 2) if not np.isnan(rsi) else None,
            "macd": round(macd, 2),
            "macd_signal": macd_signal,
            "technical_factors": [
                f"Price {'above' if current_price > sma_20 else 'below'} 20-day MA",
                f"RSI at {rsi:.1f}" if not np.isnan(rsi) else "RSI unavailable",
                f"MACD {macd_signal}",
                f"Trend: {trend}"
            ]
        }

class RiskAssessmentAgent:
    """Agent for comprehensive risk assessment."""
    
    def __init__(self, genai_agent: Optional[GenAIAnalysisAgent] = None):
        self.genai = genai_agent
        
    def assess_risk(self, volatility: float, anomaly_score: float, 
                   probability_decline: float, price_range: tuple) -> Dict[str, Any]:
        """Comprehensive risk assessment."""
        
        # Risk level determination
        risk_level = RiskLevel.LOW
        risk_score = 0
        risk_factors = []
        
        # Volatility risk
        if volatility > 50:
            risk_level = RiskLevel.CRITICAL
            risk_score += 30
            risk_factors.append(f"Extreme volatility ({volatility:.1f}%)")
        elif volatility > 35:
            risk_level = RiskLevel.HIGH
            risk_score += 20
            risk_factors.append(f"High volatility ({volatility:.1f}%)")
        elif volatility > 20:
            risk_score += 10
            risk_factors.append(f"Moderate volatility ({volatility:.1f}%)")
        
        # Anomaly risk
        if anomaly_score < -0.5:
            risk_level = RiskLevel.HIGH
            risk_score += 15
            risk_factors.append("Anomalous market conditions detected")
        
        # Directional risk
        if probability_decline > 0.7:
            risk_level = RiskLevel.MEDIUM
            risk_score += 10
            risk_factors.append(f"High downside probability ({probability_decline*100:.1f}%)")
        
        # Price range risk
        if price_range and len(price_range) == 2:
            range_size = (price_range[1] - price_range[0]) / price_range[0]
            if range_size > 0.2:
                risk_score += 10
                risk_factors.append(f"Wide expected price range ({range_size*100:.1f}%)")
        
        # Normalize risk score
        risk_score = min(100, max(0, risk_score))
        
        return {
            "risk_level": risk_level.value,
            "risk_score": round(risk_score, 1),
            "risk_factors": risk_factors,
            "recommendations": self._get_risk_recommendations(risk_level, risk_factors)
        }
    
    def _get_risk_recommendations(self, risk_level: RiskLevel, factors: List[str]) -> List[str]:
        """Get recommendations based on risk level."""
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("Reduce position size significantly")
            recommendations.append("Consider hedging strategies")
            recommendations.append("Monitor position closely")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("Use appropriate stop-losses")
            recommendations.append("Reduce leverage")
            recommendations.append("Increase position monitoring frequency")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("Maintain standard risk management")
            recommendations.append("Set appropriate stop-loss levels")
        else:
            recommendations.append("Standard trading conditions")
        
        return recommendations

class MarketPredictionOrchestrator:
    """Main orchestrator agent coordinating all sub-agents."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the orchestrator with all sub-agents."""
        self.data_mgr = MarketDataManager()
        self.vol_forecaster = VolatilityForecaster()
        self.anomaly_detector = AnomalyDetector()
        self.simulator = EventSimulator()
        self.genai_agent = GenAIAnalysisAgent(api_key=api_key)
        self.trend_agent = TrendAnalysisAgent(genai_agent=self.genai_agent)
        self.risk_agent = RiskAssessmentAgent(genai_agent=self.genai_agent)
        
    def predict_market_activity(self, ticker: str) -> MarketPrediction:
        """Generate comprehensive market prediction."""
        
        # Step 1: Collect market data
        df = self.data_mgr.get_historical_data(ticker)
        current_price = df['Close'].iloc[-1]
        
        # Step 2: Analyze volatility
        vol_data = self.vol_forecaster.predict_volatility(df['Returns'])
        
        # Step 3: Detect anomalies
        anomaly_data = self.anomaly_detector.detect_anomalies(df)
        
        # Step 4: Analyze trends
        trend_data = self.trend_agent.analyze_trend(df)
        
        # Step 5: Check upcoming events
        upcoming_events = self.data_mgr.get_upcoming_events(ticker)
        
        # Step 6: Run simulations with event adjustment
        adjusted_daily_vol = (vol_data['annualized_volatility_forecast'] / np.sqrt(252)) / 100
        if upcoming_events:
            adjusted_daily_vol *= 1.5
        
        sim_results = self.simulator.run_simulation(current_price, adjusted_daily_vol)
        
        # Step 7: Assess risk
        risk_data = self.risk_agent.assess_risk(
            volatility=vol_data['annualized_volatility_forecast'],
            anomaly_score=anomaly_data['anomaly_score'],
            probability_decline=sim_results['probability_of_decline'],
            price_range=(sim_results['bear_case_95'], sim_results['bull_case_95'])
        )
        
        # Step 8: Determine prediction signal using GenAI
        signal, confidence = self._generate_signal(
            vol_data, anomaly_data, trend_data, sim_results, risk_data
        )
        
        # Step 9: Generate explanation using GenAI
        context = {
            "ticker": ticker,
            "current_price": current_price,
            "volatility": vol_data['annualized_volatility_forecast'],
            "trend": trend_data['trend'],
            "anomalies": anomaly_data['current_state_is_anomaly'],
            "upcoming_events": upcoming_events,
            "simulation_results": {
                "bear_case": sim_results['bear_case_95'],
                "bull_case": sim_results['bull_case_95'],
                "downside_probability": sim_results['probability_of_decline']
            }
        }
        
        key_factors = trend_data.get('technical_factors', []) + risk_data['risk_factors']
        reasoning = self.genai_agent.explain_prediction(context, key_factors)
        
        # Step 10: Create structured prediction
        prediction = MarketPrediction(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            volatility_forecast=vol_data['annualized_volatility_forecast'],
            price_target_range=(round(sim_results['bear_case_95'], 2), 
                              round(sim_results['bull_case_95'], 2)),
            risk_level=RiskLevel[risk_data['risk_level']],
            reasoning=reasoning,
            key_factors=key_factors,
            timestamp=datetime.now().isoformat()
        )
        
        return prediction
    
    def _generate_signal(self, vol_data: Dict, anomaly_data: Dict, 
                        trend_data: Dict, sim_results: Dict, 
                        risk_data: Dict) -> tuple:
        """Generate prediction signal based on aggregated factors."""
        
        downside_prob = sim_results['probability_of_decline']
        volatility = vol_data['annualized_volatility_forecast']
        trend = trend_data['trend']
        
        # High volatility signal
        if volatility > 40 or anomaly_data['current_state_is_anomaly']:
            return (PredictionSignal.HIGH_VOLATILITY, 0.75)
        
        # Trend-based signals
        if trend == "UPTREND":
            confidence = 0.7 + min(0.2, trend_data.get('trend_strength', 0))
            return (PredictionSignal.BULLISH, confidence)
        elif trend == "DOWNTREND":
            confidence = 0.7 + min(0.2, trend_data.get('trend_strength', 0))
            return (PredictionSignal.BEARISH, confidence)
        elif trend == "OVERBOUGHT":
            return (PredictionSignal.BEARISH, 0.65)
        elif trend == "OVERSOLD":
            return (PredictionSignal.BULLISH, 0.65)
        
        # Default to probability-based signal
        if downside_prob > 0.6:
            confidence = 0.5 + (downside_prob - 0.6) * 5
            return (PredictionSignal.BEARISH, min(0.9, confidence))
        elif downside_prob < 0.4:
            confidence = 0.5 + (0.4 - downside_prob) * 5
            return (PredictionSignal.BULLISH, min(0.9, confidence))
        
        return (PredictionSignal.NEUTRAL, 0.5)
    
    def get_detailed_report(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive report with all analysis details."""
        
        prediction = self.predict_market_activity(ticker)
        df = self.data_mgr.get_historical_data(ticker)
        
        return {
            "summary": {
                "ticker": prediction.ticker,
                "timestamp": prediction.timestamp,
                "signal": prediction.signal.value,
                "confidence": round(prediction.confidence, 2),
                "current_price": df['Close'].iloc[-1]
            },
            "prediction": {
                "volatility_forecast_annual": f"{prediction.volatility_forecast:.2f}%",
                "price_target_range": prediction.price_target_range,
                "risk_level": prediction.risk_level.value
            },
            "analysis": {
                "reasoning": prediction.reasoning,
                "key_factors": prediction.key_factors
            },
            "technical_indicators": self.trend_agent.analyze_trend(df)
        }

class MarketActivityAgent:
    """Orchestrator Class - Legacy interface for backward compatibility."""
    
    def __init__(self):
        self.data_mgr = MarketDataManager()
        self.vol_forecaster = VolatilityForecaster()
        self.anomaly_detector = AnomalyDetector()
        self.simulator = EventSimulator()

    def analyze(self, ticker: str) -> Dict[str, Any]:
        # 1. Get Data
        df = self.data_mgr.get_historical_data(ticker)
        current_price = df['Close'].iloc[-1]
        
        # 2. Forecast Volatility
        vol_data = self.vol_forecaster.predict_volatility(df['Returns'])
        
        # 3. Check Anomalies
        anomaly_data = self.anomaly_detector.detect_anomalies(df)
        
        # 4. Check Events
        upcoming_events = self.data_mgr.get_upcoming_events(ticker)
        
        # 5. Adjust Volatility for Events (Heuristic Rule)
        # If earnings are coming, implied volatility usually spikes. 
        # We apply a multiplier for the simulation if an event is detected.
        adjusted_daily_vol = (vol_data['annualized_volatility_forecast'] / np.sqrt(252)) / 100
        if upcoming_events:
            adjusted_daily_vol *= 1.5 # 50% premium for event risk
            
        # 6. Run Simulation
        sim_results = self.simulator.run_simulation(current_price, adjusted_daily_vol)
        
        # 7. Construct Rationale & Confidence
        rationale = []
        confidence_score = 0.9
        
        if anomaly_data['current_state_is_anomaly']:
            rationale.append("Market is currently in an anomalous state (high volume or price shock).")
            confidence_score -= 0.2 # Lower confidence in chaos
            
        if upcoming_events:
            rationale.append(f"Upcoming events detected: {upcoming_events}. Volatility premium applied.")
            
        if sim_results['probability_of_decline'] > 0.65:
            rationale.append("Skewed downside risk detected in Monte Carlo simulations.")
        elif sim_results['probability_of_decline'] < 0.35:
            rationale.append("Skewed upside potential detected in Monte Carlo simulations.")
        else:
            rationale.append("Market conditions appear neutral/range-bound.")

        # 8. Alerting Logic
        alert_level = "LOW"
        if vol_data['annualized_volatility_forecast'] > 40 or anomaly_data['current_state_is_anomaly']:
            alert_level = "HIGH"
        elif vol_data['annualized_volatility_forecast'] > 25:
            alert_level = "MEDIUM"

        return {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "alert_level": alert_level,
            "prediction": {
                "volatility_annualized": f"{vol_data['annualized_volatility_forecast']:.2f}%",
                "5_day_forecast_range": [round(sim_results['bear_case_95'], 2), round(sim_results['bull_case_95'], 2)],
                "direction_bias": "Bearish" if sim_results['probability_of_decline'] > 0.5 else "Bullish"
            },
            "analysis": {
                "anomalies_detected": anomaly_data['current_state_is_anomaly'],
                "events_impacting": upcoming_events,
                "rationale": " ".join(rationale),
                "confidence_score": round(confidence_score, 2)
            }
        }

# --- ADK App Setup ---

# Tool definitions for ADK agents
def fetch_market_data(ticker: str, period: str = "2y") -> str:
    """Tool for ADK: Fetch historical market data.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period (1mo, 3mo, 6mo, 1y, 2y, 5y)
        
    Returns:
        JSON string with market data summary
    """
    try:
        manager = MarketDataManager()
        df = manager.get_historical_data(ticker, period)
        return json.dumps({
            "ticker": ticker,
            "period": period,
            "data_points": len(df),
            "current_price": float(df['Close'].iloc[-1]),
            "data_available": True
        })
    except Exception as e:
        return json.dumps({"error": str(e), "data_available": False})


def get_volatility_forecast(ticker: str) -> str:
    """Tool for ADK: Get volatility forecast.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        JSON string with volatility metrics
    """
    try:
        manager = MarketDataManager()
        df = manager.get_historical_data(ticker)
        forecaster = VolatilityForecaster()
        result = forecaster.predict_volatility(df['Returns'])
        return json.dumps({
            "ticker": ticker,
            "volatility_annualized": round(result['annualized_volatility_forecast'], 2),
            "model_confidence": round(result['model_confidence'], 2)
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def detect_market_anomalies(ticker: str) -> str:
    """Tool for ADK: Detect market anomalies.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        JSON string with anomaly detection results
    """
    try:
        manager = MarketDataManager()
        df = manager.get_historical_data(ticker)
        detector = AnomalyDetector()
        result = detector.detect_anomalies(df)
        return json.dumps({
            "ticker": ticker,
            "is_anomaly": result['current_state_is_anomaly'],
            "total_anomalies": result['total_anomalies_detected'],
            "anomaly_score": round(result['anomaly_score'], 2)
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def analyze_market_trend(ticker: str) -> str:
    """Tool for ADK: Analyze market trends.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        JSON string with trend analysis
    """
    try:
        manager = MarketDataManager()
        df = manager.get_historical_data(ticker)
        agent = TrendAnalysisAgent()
        result = agent.analyze_trend(df)
        return json.dumps({
            "ticker": ticker,
            "trend": result.get('trend'),
            "trend_strength": round(result.get('trend_strength', 0), 2),
            "rsi": round(result.get('rsi', 0), 2) if result.get('rsi') else None
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def assess_market_risk(ticker: str) -> str:
    """Tool for ADK: Assess market risk.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        JSON string with risk assessment
    """
    try:
        orchestrator = MarketPredictionOrchestrator()
        prediction = orchestrator.predict_market_activity(ticker)
        return json.dumps({
            "ticker": ticker,
            "risk_level": prediction.risk_level.value,
            "confidence": round(prediction.confidence, 2),
            "signal": prediction.signal.value
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# Create specialized ADK agents
data_agent = Agent(
    name="market_data_agent",
    model="gemini-3-pro-preview",
    instruction="You are a financial data analyst. Use tools to fetch and summarize market data.",
    tools=[fetch_market_data, detect_market_anomalies]
)

analysis_agent = Agent(
    name="market_analysis_agent",
    model="gemini-3-pro-preview",
    instruction="You are a technical analyst. Use tools to analyze market trends and volatility.",
    tools=[analyze_market_trend, get_volatility_forecast]
)

risk_agent = Agent(
    name="market_risk_agent",
    model="gemini-3-pro-preview",
    instruction="You are a risk management expert. Use tools to assess market risk and provide recommendations.",
    tools=[assess_market_risk]
)

root_agent = Agent(
    name="market_prediction_agent",
    model="gemini-3-pro-preview",
    instruction="""You are the Market Activity Prediction Agent, an expert financial analyst powered by Google ADK.

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

Keep responses professional and data-driven.""",
    tools=[fetch_market_data, get_volatility_forecast, detect_market_anomalies, 
           analyze_market_trend, assess_market_risk]
)

# Create ADK App
app = App(
    root_agent=root_agent,
    name="market-activity-prediction-agent",
    description="Multi-agent GenAI system for predicting market activity using Google ADK"
)

# --- Usage Example (if running as script) ---
if __name__ == "__main__":
    import sys
    
    # Example 1: Basic analysis with legacy agent
    print("=" * 80)
    print("EXAMPLE 1: Legacy Market Analysis Agent")
    print("=" * 80)
    agent = MarketActivityAgent()
    result = agent.analyze("NVDA")
    print(json.dumps(result, indent=2))
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: ADK-Powered Market Prediction")
    print("=" * 80)
    
    try:
        orchestrator = MarketPredictionOrchestrator()
        
        # Analyze market
        ticker = "NVDA"
        print(f"\nAnalyzing {ticker} with ADK multi-agent system...\n")
        
        prediction = orchestrator.predict_market_activity(ticker)
        
        print(f"Ticker: {prediction.ticker}")
        print(f"Signal: {prediction.signal.value}")
        print(f"Confidence: {prediction.confidence:.2%}")
        print(f"Risk Level: {prediction.risk_level.value}")
        print(f"Volatility Forecast: {prediction.volatility_forecast:.2f}%")
        print(f"Price Target Range: ${prediction.price_target_range[0]:.2f} - ${prediction.price_target_range[1]:.2f}")
        print(f"\nKey Factors:")
        for factor in prediction.key_factors[:5]:
            print(f"  - {factor}")
        print(f"\nRecommended Action: {prediction.signal.value}")
        
        print("\n" + "=" * 80)
        print("EXAMPLE 3: ADK App Usage")
        print("=" * 80)
        print(f"ADK App initialized: {app.name}")
        print(f"Root Agent: {root_agent.name}")
        print(f"Available specialized agents:")
        print(f"  - {data_agent.name}")
        print(f"  - {analysis_agent.name}")
        print(f"  - {risk_agent.name}")
        
    except Exception as e:
        print(f"Note: GenAI features optimized for Google Cloud environment.")
        print(f"Error: {str(e)}")
        print("\nSystem supports both local and Google Cloud deployment via ADK.")


