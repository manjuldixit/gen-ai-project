from google.adk.agents import LlmAgent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from typing import List, Dict
from . import prompt


# Mock tool to simulate fetching historical correlations
def fetch_event_correlations(ticker: str, event_type: str) -> Dict:
    """
    Analyzes how a specific ticker reacted to similar events in the past.
    Returns average price change, volatility jump, and occurrence count.
    """
    # In a real app, this would query a SQL database or a financial API
    return {
        "event": event_type,
        "avg_impact_24h": "+1.5%",
        "volatility_increase": "25%",
        "sample_size": 12,
        "historical_reliability": 0.82
    }

event_impact_agent = LlmAgent(
    name="EventImpactAgent",
    model="gemini-2.0-flash",
    instruction=prompt.DATA_ANALYST_PROMPT,
    tools=[fetch_event_correlations]
)