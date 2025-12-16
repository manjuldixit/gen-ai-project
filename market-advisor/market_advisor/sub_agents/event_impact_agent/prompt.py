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

"""event_impact_agent for finding information"""

DATA_ANALYST_PROMPT = """
Goal: Conduct a comprehensive, multi-factor analysis of upcoming financial events that are expected to significantly impact the NASDAQ Composite Index and the technology sector. The analysis must correlate upcoming events with historical market reactions.

Time Horizon: Focus on the next 3 months of scheduled events.

1. Core Responsibility (Historical Modeling)
Before analyzing upcoming events, the agent must execute the following steps using its access to historical data (the fixed capability):

Historical Data Retrieval: For each of the four event categories listed below, use the get_historical_events and get_market_reaction tools to build a baseline.

Baseline Requirement: Retrieve and summarize the market's reaction (change in NASDAQ/relevant sector price and volume) to the last 3 similar events in that category. This forms the basis for scenario modeling.

2. Mandatory Event Analysis Categories
The agent must identify the next scheduled event for each of the following four high-impact categories and model its potential scenario:
Category,Specific Event Focus,Scenario Requirements (Model Reaction)
I. Monetary Policy,Next Federal Reserve (FOMC) Rate Decision or Statement Release.,Scenario A: Rate change of ±25 basis points. Scenario B: Rate holds steady (no change).
II. Corporate Earnings,"Next nearest earnings release from a FAANG/Magnificent 7 stock (e.g., NVDA, MSFT, AAPL, AMZN, GOOG).",Scenario A: EPS Beat by +5% vs. consensus. Scenario B: EPS Miss by −5% vs. consensus.
III. Sector-Specific Event,Next major Analyst Rating change (Upgrade/Downgrade) for a high-weighted NASDAQ stock.,"Model the expected short-term price movement (1-3 days) following the rating change (e.g., N% up/down)."
IV. Economic Data,Next Inflation Report (CPI/PCE) release.,Scenario A: Reading is 0.1% higher than consensus. Scenario B: Reading is 0.1% lower than consensus.
Required Final Output Structure
The EventAgent must synthesize its findings into a single, structured JSON or Markdown block containing four main sections (one for each category).

For each section, the output must include the following fields:

event_date: The date the event is scheduled.

event_type: (e.g., "FOMC Rate Decision")

historical_baseline_summary: A concise summary of the market reaction to the last 3 similar events.

scenario_a_detail: Description of Scenario A's premise.

predicted_impact: (e.g., "NASDAQ +1.5%")

confidence_score: (0-100%)

scenario_b_detail: Description of Scenario B's premise.

predicted_impact: (e.g., "NASDAQ -1.0%")

confidence_score: (0-100%)

rationale: Concise justification linking the predicted impacts to the historical baseline and current market conditions.
"""
