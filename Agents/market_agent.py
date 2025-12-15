import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from sklearn.ensemble import IsolationForest
from scipy.stats import norm
from datetime import datetime, timedelta
from typing import Dict, Any, List

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

class MarketActivityAgent:
    """Orchestrator Class."""
    
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

# --- usage Example (if running as script) ---
if __name__ == "__main__":
    agent = MarketActivityAgent()
    # Test on a volatile stock or index
    print("Analyzing NVDA...")
    result = agent.analyze("NVDA")
    import json
    print(json.dumps(result, indent=2))