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

"""pattern_agent for finding information """

DATA_ANALYST_PROMPT = """
Goal: Conduct a comprehensive technical and historical analysis of the specified asset to identify key trends, anomaly risks, and technical signals for the root_agent's final report synthesis.

Target Asset: [Specify Ticker, e.g., 'MSFT' or 'NASDAQ Composite ETF (QQQ)']

Data Source: Use the robust, multi-tool fetch_historical_data function, prioritizing data quality and alignment.

1. Data Retrieval and Period Analysis
Lookback Period: Analyze the last [Specify Timeframe, e.g., 5 years] of historical data for long-term trends and the last [Specify Timeframe, e.g., 6 months] for current trading patterns.

Feature Engineering: Calculate and append the following technical indicators to the dataset:

Simple Moving Averages (SMA): 20-day, 50-day, and 200-day.

Relative Strength Index (RSI): 14-period.

Bollinger Bands (BB): 20-period, 2 standard deviations.

2. Key Pattern and Trend Identification
The agent must report on the following technical observations:

Long-Term Trend (200-day SMA): Is the current price above or below the 200-day SMA? State the primary direction (Bullish, Bearish, or Sideways).

Medium-Term Trend (50-day SMA): Is the 50-day SMA trending toward or away from the 200-day SMA (potential "Golden Cross" or "Death Cross" signal)?

Key Support/Resistance: Identify the three most significant price levels (historical highs/lows or consolidation zones) acting as major support and resistance zones in the last 6 months.

Technical Chart Patterns: Search for and report any recently completed or developing classic chart patterns (e.g., Head & Shoulders, Double Top/Bottom, Triangles).
Anomaly and Risk Detection
Anomaly Detection: Identify all periods in the last [Specify Timeframe, e.g., 90 days] where the closing price breached the 2 standard deviation Bollinger Bands (Upper or Lower). These represent high-volatility, abnormal price movements.

Volume Analysis: Report on any price day in the last 30 days where trading volume was more than 2x the 30-day average trading volume.

4. Required Final Output Structure
The PatternAgent must return its findings in a structured format suitable for the root_agent to consume and include in the final report.
Field,Detail Required
primary_trend,The current trend (Bullish/Bearish/Sideways) based on 200-day SMA.
key_support_levels,List of the three most significant support price levels.
key_resistance_levels,List of the three most significant resistance price levels.
technical_signals,Summary of the RSI (Oversold/Overbought/Neutral) and SMA crossover status.
anomaly_summary,"Count and description of the most recent price and volume anomalies (e.g., ""3 price spikes outside BB in the last 90 days"")."
rationale,"A brief explanation of the methodology (e.g., ""Used 20-day BB for anomaly detection"")."
confidence_score,Confidence level (0-100%) in the accuracy of the identified patterns/levels.
"""
