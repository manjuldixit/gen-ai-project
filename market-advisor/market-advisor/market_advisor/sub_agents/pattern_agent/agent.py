from google.adk.agents import LlmAgent
import yfinance as yf
import pandas as pd
import numpy as np

from . import prompt

MODEL = "gemini-2.5-pro"


def fetch_historical_data(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Fetches market data and calculates basic indicators to assist agents in 
    pattern recognition and anomaly detection.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return f"No data found for {ticker}."
        data.index = pd.to_datetime(data.index)
        data = data[~data.index.duplicated(keep='first')] # Drop duplicate index entries
        # Fill any gaps in the critical 'Close' column to prevent calculation break
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data['Close'] = data['Close'].ffill().bfill()
        if data['Close'].isnull().all():
            return f"Error: After cleaning, all 'Close' prices for {ticker} are NaN. Data is unusable."
        # 1. Feature Engineering: Add Technical Indicators for the Pattern Agent
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['Std_Dev'] = data['Close'].rolling(window=20).std()
        
        data = data.dropna(subset=['SMA_20', 'Std_Dev'])
        # 2. Anomaly Detection (Simple Z-Score)
        # Identifies if price is more than 2 standard deviations from the mean
        data['Upper_Band'] = data['SMA_20'] + (data['Std_Dev'] * 2)
        data['Lower_Band'] = data['SMA_20'] - (data['Std_Dev'] * 2)
        data = data.dropna(subset=['SMA_20', 'Std_Dev']) # Drops the first 19 rows
        data['Is_Anomaly'] = ((data['Close'] > data['Upper_Band'])(data['Close'] < data['Lower_Band'])).fillna(False) # Fill any remaining potential NaNs in the boolean result with False.

        # 3. Format the last 5 days for the Agent to process
        recent_data = data.tail(5).to_dict(orient='index')
        
        # Structure the summary for the LLM
        summary = {
            "ticker": ticker,
            "current_price": round(float(data['Close'].iloc[-1]), 2),
            "recent_trend": "Bullish" if data['Close'].iloc[-1] > data['SMA_20'].iloc[-1] else "Bearish",
            "anomalies_detected": data['Is_Anomaly'].tail(5).any(),
            "data_points": recent_data
        }

        return str(summary)

    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"

pattern_agent = LlmAgent(
    name="PatternAnalyst",
    model=MODEL,
    instruction="""Analyze historical price and volume data. 
    Identify technical patterns (Head & Shoulders, etc.) and anomalies. 
    Output: {prediction, confidence_score, rationale}.""",
    tools=[fetch_historical_data]
)
