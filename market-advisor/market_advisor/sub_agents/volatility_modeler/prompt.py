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
 Forecast short/long-term volatility. 
    Compare current VIX levels to historical means. 
    Alert if predicted volatility exceeds 2.5 standard deviations.
confidence_score,Confidence level (0-100%) in the accuracy of the volatility forecasts.
"""
