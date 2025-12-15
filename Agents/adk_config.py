# Market Activity Prediction Agent - ADK Configuration

"""
ADK (Agent Development Kit) configuration for the Market Prediction System.

This module provides ADK-specific configuration for deploying the multi-agent
market prediction system on Google Cloud or locally.
"""

import os
from typing import Optional

# ADK Agent Configuration
class ADKAgentConfig:
    """Configuration for ADK agents."""
    
    # Models
    MAIN_MODEL = "gemini-3-pro-preview"
    DATA_MODEL = "gemini-3-pro-preview"
    ANALYSIS_MODEL = "gemini-3-pro-preview"
    RISK_MODEL = "gemini-3-pro-preview"
    
    # Agent names
    ROOT_AGENT_NAME = "market_prediction_agent"
    DATA_AGENT_NAME = "market_data_agent"
    ANALYSIS_AGENT_NAME = "market_analysis_agent"
    RISK_AGENT_NAME = "market_risk_agent"
    
    # Tool settings
    TOOL_TIMEOUT = 30  # seconds
    MAX_TOOL_ITERATIONS = 3


# ADK App Configuration
class ADKAppConfig:
    """Configuration for ADK application."""
    
    APP_NAME = "market-activity-prediction-agent"
    APP_DESCRIPTION = "Multi-agent GenAI system for predicting market activity"
    APP_VERSION = "2.0-adk"
    
    # Deployment
    ENABLE_LOGGING = True
    LOG_LEVEL = "INFO"
    
    # Cloud configuration
    USE_VERTEX_AI = True
    USE_LOCAL_API = not USE_VERTEX_AI
    

# Google Cloud Configuration
class GoogleCloudConfig:
    """Google Cloud specific configuration."""
    
    # Auto-detect or manual configuration
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", None)
    LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    
    # Vertex AI settings
    USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "True").lower() == "true"
    
    # API Key (for local development)
    API_KEY = os.getenv("GOOGLE_API_KEY", None)


# ADK Tool Configuration
class ADKToolConfig:
    """Configuration for ADK tools."""
    
    # Data fetching tools
    FETCH_DATA_TIMEOUT = 30
    FETCH_DATA_RETRIES = 3
    
    # Analysis tools
    ANALYSIS_TIMEOUT = 20
    VOLATILITY_FORECAST_DAYS = 5
    
    # Anomaly detection
    ANOMALY_TIMEOUT = 15
    ANOMALY_CONTAMINATION = 0.05
    
    # Risk assessment
    RISK_TIMEOUT = 10
    RISK_CALCULATION_METHOD = "multi_factor"


# ADK Workflow Configuration
class ADKWorkflowConfig:
    """Configuration for agent workflows."""
    
    # Parallel execution
    ENABLE_PARALLEL_AGENTS = True
    MAX_PARALLEL_AGENTS = 3
    
    # Sequential fallback
    FALLBACK_TO_SEQUENTIAL = True
    SEQUENTIAL_TIMEOUT = 120
    
    # Result caching
    ENABLE_CACHING = True
    CACHE_TTL_SECONDS = 3600


# ADK Logging Configuration
class ADKLoggingConfig:
    """Configuration for ADK logging."""
    
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "adk_market_agent.log"
    LOG_MAX_SIZE = 10_000_000  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # Log levels
    ROOT_LEVEL = "INFO"
    ADK_LEVEL = "INFO"
    AGENT_LEVEL = "DEBUG"


# ADK Monitoring Configuration
class ADKMonitoringConfig:
    """Configuration for monitoring and observability."""
    
    # Metrics
    ENABLE_METRICS = True
    METRICS_INTERVAL = 60  # seconds
    
    # Tracing
    ENABLE_TRACING = True
    TRACE_SAMPLE_RATE = 0.1  # 10%
    
    # Health checks
    ENABLE_HEALTH_CHECKS = True
    HEALTH_CHECK_INTERVAL = 30  # seconds


# Integration Configuration
class ADKIntegrationConfig:
    """Configuration for integrations with other systems."""
    
    # Data sources
    YFINANCE_ENABLED = True
    YFINANCE_TIMEOUT = 30
    
    # Output formats
    JSON_OUTPUT_ENABLED = True
    STRUCTURED_OUTPUT_ENABLED = True
    
    # External APIs
    EXTERNAL_API_TIMEOUT = 30
    EXTERNAL_API_RETRIES = 3


def get_adk_config(environment: str = "production") -> dict:
    """Get ADK configuration based on environment.
    
    Args:
        environment: "production", "staging", or "development"
        
    Returns:
        Dictionary with environment-specific configuration
    """
    
    base_config = {
        "agent_config": ADKAgentConfig.__dict__,
        "app_config": ADKAppConfig.__dict__,
        "cloud_config": GoogleCloudConfig.__dict__,
        "tool_config": ADKToolConfig.__dict__,
        "workflow_config": ADKWorkflowConfig.__dict__,
        "logging_config": ADKLoggingConfig.__dict__,
        "monitoring_config": ADKMonitoringConfig.__dict__,
        "integration_config": ADKIntegrationConfig.__dict__,
    }
    
    # Environment-specific overrides
    if environment == "development":
        base_config["app_config"]["ENABLE_LOGGING"] = True
        base_config["logging_config"]["ROOT_LEVEL"] = "DEBUG"
        base_config["workflow_config"]["ENABLE_PARALLEL_AGENTS"] = False
        base_config["monitoring_config"]["TRACE_SAMPLE_RATE"] = 1.0
        
    elif environment == "staging":
        base_config["monitoring_config"]["ENABLE_METRICS"] = True
        base_config["workflow_config"]["ENABLE_PARALLEL_AGENTS"] = True
        
    elif environment == "production":
        base_config["workflow_config"]["ENABLE_PARALLEL_AGENTS"] = True
        base_config["workflow_config"]["ENABLE_CACHING"] = True
        base_config["monitoring_config"]["ENABLE_METRICS"] = True
    
    return base_config


def initialize_adk_environment():
    """Initialize Google Cloud environment for ADK."""
    
    if GoogleCloudConfig.USE_VERTEXAI:
        # Configure Vertex AI
        os.environ["GOOGLE_CLOUD_LOCATION"] = GoogleCloudConfig.LOCATION
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
        
        if GoogleCloudConfig.PROJECT_ID:
            os.environ["GOOGLE_CLOUD_PROJECT"] = GoogleCloudConfig.PROJECT_ID
    else:
        # Configure local API key
        if GoogleCloudConfig.API_KEY:
            os.environ["GOOGLE_API_KEY"] = GoogleCloudConfig.API_KEY


# Export configuration classes and functions
__all__ = [
    "ADKAgentConfig",
    "ADKAppConfig",
    "GoogleCloudConfig",
    "ADKToolConfig",
    "ADKWorkflowConfig",
    "ADKLoggingConfig",
    "ADKMonitoringConfig",
    "ADKIntegrationConfig",
    "get_adk_config",
    "initialize_adk_environment",
]
