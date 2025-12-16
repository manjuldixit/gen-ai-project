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

"""Volatility_modeler_agent for finding the market volatility"""

EXECUTION_ANALYST_PROMPT = """

Goal: Provide a quantitative risk assessment and probabilistic price forecast for the target asset.
 The analysis must cover both short-term trading risk and long-term investment volatility.
 Target Asset: [Specify Ticker, e.g., 'MSFT' or 'NASDAQ Composite ETF (QQQ)']
 Dependencies: The agent should request and incorporate the latest price data and the anomaly report from the PatternAgent to contextualize its models.
 1. Core Statistical Metrics CalculationThe agent must calculate the following foundational risk metrics using the last [Specify Lookback, e.g., 252 trading days (1 year)] of data:Annualized Volatility ($\sigma_{annual}$): Calculate the standard deviation of daily returns, annualized.Beta ($\beta$): Calculate the asset's sensitivity relative to a major market index (e.g., S&P 500 or NASDAQ).Value at Risk (VaR): Calculate the 99% 1-Day VaR (the maximum expected loss over one day with 99% confidence).
 2. Multi-Horizon Volatility ModelingThe agent must forecast future risk using both historical and predictive models:Short-Term Forecast (1-Week): Use a time-series model (e.g., GARCH(1,1) or Exponentially Weighted Moving Average, EWMA) to forecast the expected volatility over the next 7 calendar days.Long-Term Forecast (3-Month): Report the historical Rolling 60-day Annualized Volatility and use it as a proxy for the 3-Month forecast, noting its historical nature in the rationale.
 3. Probabilistic Price ForecastingUsing the current price and the Short-Term Volatility Forecast (1-Week), calculate the following price targets based on a Normal Distribution model:1-Week 95% Confidence Interval (CI): Calculate the upper and lower price boundaries where the asset is expected to trade 95% of the time over the next week.$P_{upper} = P_{current} \times e^{Z \sigma \sqrt{T}}$$P_{lower} = P_{current} \times e^{-Z \sigma \sqrt{T}}$Where $\sigma$ is the 1-Week volatility, $T$ is $7/365$, and $Z$ is 1.96.
 4. Required Final Output StructureThe VolatilityAgent must return its findings in a structured format for the root_agent to consume.
 Field,Detail Required
annualized_volatility,1-Year Historical Annualized Volatility (σannual​).
asset_beta,Calculated Beta relative to the designated index.
var_99_1day,Value at Risk (99% confidence) as a percentage or currency amount.
volatility_forecast_1wk,"Projected 1-Week Volatility (e.g., from GARCH model)."
volatility_forecast_3mo,"Projected 3-Month Volatility (e.g., from Rolling 60-day)."
price_ci_95_upper,The upper bound of the 1-Week 95% CI (in currency).
price_ci_95_lower,The lower bound of the 1-Week 95% CI (in currency).
rationale,"Brief explanation of the models used (e.g., ""GARCH(1,1) used for 1-week forecast"")."
confidence_score,Confidence level (0-100%) in the accuracy of the volatility forecasts.
"""
