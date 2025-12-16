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

"""Financial coordinator: provide reasonable investment strategies."""

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from . import prompt
from .sub_agents.pattern_agent import pattern_agent
from .sub_agents.volatility_modeler import volatility_modeler
from .sub_agents.event_impact_agent import event_impact_agent


MODEL = "gemini-2.5-pro"


financial_coordinator = LlmAgent(
    name="financial_coordinator",
    model=MODEL,
    description=(
        "Primary interface for Market Activity Prediction."
    ),
    instruction="""You are a Senior Market Strategist. 
    1. Delegate data requests to specialized agents.
    2. Synthesize their outputs into a final report.
    3. Always include a 'Confidence Score' based on the consensus of your sub-agents.
    4. If the Volatility Modeler predicts a spike, issue a HIGH VOLATILITY ALERT.""",
    output_key="financial_coordinator_output",
    tools=[
        AgentTool(agent=pattern_agent),
        AgentTool(agent=volatility_modeler),
        AgentTool(agent=event_impact_agent)
    ],
)

root_agent = financial_coordinator
