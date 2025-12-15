"""
Utility functions for the Market Activity Prediction Agent system.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from market_agent import MarketPrediction, PredictionSignal, RiskLevel
from config import OutputTemplates, ThresholdConfig, ConfidenceConfig

def format_prediction_report(prediction: MarketPrediction, current_price: float) -> str:
    """Format a prediction into a readable report."""
    
    key_factors_str = "\n".join([f"  • {factor}" for factor in prediction.key_factors[:7]])
    
    report = OutputTemplates.PREDICTION_REPORT.format(
        ticker=prediction.ticker,
        timestamp=prediction.timestamp,
        current_price=current_price,
        signal=prediction.signal.value,
        confidence=prediction.confidence,
        risk_level=prediction.risk_level.value,
        volatility=prediction.volatility_forecast,
        price_low=prediction.price_target_range[0],
        price_high=prediction.price_target_range[1],
        key_factors=key_factors_str,
        reasoning=prediction.reasoning
    )
    
    return report

def get_action_recommendation(signal: PredictionSignal, confidence: float) -> str:
    """Get recommended action based on signal and confidence."""
    
    if confidence < ConfidenceConfig.MINIMUM_CONFIDENCE:
        return "HOLD - Low confidence in prediction, insufficient data for action."
    
    action_map = {
        PredictionSignal.BULLISH: {
            True: "BUY - Strong bullish signal with high confidence",
            False: "ACCUMULATE - Bullish signal with moderate confidence"
        },
        PredictionSignal.BEARISH: {
            True: "SELL/SHORT - Strong bearish signal with high confidence",
            False: "REDUCE - Bearish signal with moderate confidence"
        },
        PredictionSignal.NEUTRAL: {
            True: "HOLD - Neutral with conviction",
            False: "HOLD - Uncertain conditions"
        },
        PredictionSignal.HIGH_VOLATILITY: {
            True: "CAUTION - High volatility expected, consider hedging",
            False: "CAUTION - Volatile conditions, reduce leverage"
        }
    }
    
    is_high_conf = confidence >= ConfidenceConfig.HIGH_CONFIDENCE
    return action_map[signal][is_high_conf]

def calculate_position_sizing(risk_level: RiskLevel, account_size: float, 
                             confidence: float) -> Dict[str, Any]:
    """Calculate recommended position size based on risk level and confidence."""
    
    # Base position sizes by risk level
    base_allocation = {
        RiskLevel.LOW: 0.05,      # 5% of account
        RiskLevel.MEDIUM: 0.03,   # 3% of account
        RiskLevel.HIGH: 0.01,     # 1% of account
        RiskLevel.CRITICAL: 0.005 # 0.5% of account
    }
    
    # Adjust for confidence
    allocation = base_allocation[risk_level] * confidence
    
    position_size = account_size * allocation
    
    return {
        "recommended_position_size": round(position_size, 2),
        "allocation_percent": round(allocation * 100, 2),
        "stop_loss_distance": 0.02 if risk_level == RiskLevel.LOW else 0.03,
        "take_profit_distance": 0.05
    }

def assess_prediction_quality(prediction: MarketPrediction) -> Dict[str, Any]:
    """Assess the overall quality and reliability of a prediction."""
    
    quality_score = 0
    issues = []
    strengths = []
    
    # Confidence assessment
    if prediction.confidence >= 0.8:
        quality_score += 30
        strengths.append("High confidence in prediction")
    elif prediction.confidence >= 0.6:
        quality_score += 20
        strengths.append("Moderate confidence")
    else:
        issues.append("Low confidence - prediction reliability questionable")
    
    # Risk level assessment
    if prediction.risk_level == RiskLevel.LOW:
        quality_score += 25
        strengths.append("Low risk environment")
    elif prediction.risk_level == RiskLevel.MEDIUM:
        quality_score += 15
    elif prediction.risk_level == RiskLevel.HIGH:
        quality_score += 5
        issues.append("High risk - be cautious")
    else:
        issues.append("Critical risk level detected")
    
    # Volatility assessment
    if prediction.volatility_forecast < 20:
        quality_score += 20
        strengths.append("Low volatility favorable for trading")
    elif prediction.volatility_forecast < 35:
        quality_score += 10
    elif prediction.volatility_forecast >= 50:
        issues.append("Extreme volatility - increased uncertainty")
    
    # Signal clarity
    if prediction.signal != PredictionSignal.NEUTRAL:
        quality_score += 15
        strengths.append(f"Clear {prediction.signal.value} signal")
    
    # Price range assessment
    price_move = (prediction.price_target_range[1] - prediction.price_target_range[0]) / \
                 ((prediction.price_target_range[0] + prediction.price_target_range[1]) / 2)
    if price_move < 0.1:
        issues.append("Narrow expected price range - limited upside")
    
    return {
        "overall_quality_score": min(100, quality_score),
        "reliability_rating": get_reliability_rating(quality_score),
        "strengths": strengths,
        "issues": issues,
        "suitable_for_trading": quality_score >= 60
    }

def get_reliability_rating(score: int) -> str:
    """Convert quality score to reliability rating."""
    if score >= 85:
        return "EXCELLENT"
    elif score >= 70:
        return "GOOD"
    elif score >= 55:
        return "FAIR"
    elif score >= 40:
        return "POOR"
    else:
        return "UNRELIABLE"

def compare_predictions(predictions: List[MarketPrediction]) -> Dict[str, Any]:
    """Compare multiple predictions for consensus analysis."""
    
    if not predictions:
        return {"error": "No predictions to compare"}
    
    bullish_count = sum(1 for p in predictions if p.signal == PredictionSignal.BULLISH)
    bearish_count = sum(1 for p in predictions if p.signal == PredictionSignal.BEARISH)
    neutral_count = sum(1 for p in predictions if p.signal == PredictionSignal.NEUTRAL)
    volatile_count = sum(1 for p in predictions if p.signal == PredictionSignal.HIGH_VOLATILITY)
    
    avg_confidence = sum(p.confidence for p in predictions) / len(predictions)
    avg_volatility = sum(p.volatility_forecast for p in predictions) / len(predictions)
    
    consensus = "NO_CONSENSUS"
    if bullish_count >= len(predictions) * 0.6:
        consensus = "BULLISH"
    elif bearish_count >= len(predictions) * 0.6:
        consensus = "BEARISH"
    elif volatile_count >= len(predictions) * 0.5:
        consensus = "HIGH_VOLATILITY"
    
    return {
        "total_predictions": len(predictions),
        "consensus": consensus,
        "signal_distribution": {
            "bullish": bullish_count,
            "bearish": bearish_count,
            "neutral": neutral_count,
            "high_volatility": volatile_count
        },
        "average_confidence": round(avg_confidence, 2),
        "average_volatility_forecast": round(avg_volatility, 2),
        "consensus_strength": round(max(bullish_count, bearish_count, volatile_count) / len(predictions), 2)
    }

def export_prediction_to_json(prediction: MarketPrediction, filepath: str) -> bool:
    """Export prediction to JSON file."""
    try:
        data = {
            "ticker": prediction.ticker,
            "timestamp": prediction.timestamp,
            "signal": prediction.signal.value,
            "confidence": prediction.confidence,
            "volatility_forecast": prediction.volatility_forecast,
            "price_target_range": {
                "lower_bound": prediction.price_target_range[0],
                "upper_bound": prediction.price_target_range[1]
            },
            "risk_level": prediction.risk_level.value,
            "key_factors": prediction.key_factors,
            "reasoning": prediction.reasoning
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error exporting prediction: {str(e)}")
        return False

def get_market_regime(prediction: MarketPrediction) -> str:
    """Identify current market regime based on prediction."""
    
    vol = prediction.volatility_forecast
    signal = prediction.signal
    risk = prediction.risk_level
    
    if vol > 40 or risk == RiskLevel.CRITICAL:
        return "CRISIS_MODE"
    elif vol > 30 and signal in [PredictionSignal.BEARISH, PredictionSignal.HIGH_VOLATILITY]:
        return "RISK_OFF"
    elif signal == PredictionSignal.BULLISH and risk == RiskLevel.LOW:
        return "RISK_ON"
    elif vol < 15:
        return "LOW_VOLATILITY_ENVIRONMENT"
    else:
        return "NORMAL_CONDITIONS"

def generate_trading_alerts(prediction: MarketPrediction) -> List[str]:
    """Generate trading alerts based on prediction."""
    
    alerts = []
    
    # Extreme volatility alert
    if prediction.volatility_forecast > 50:
        alerts.append(f"🚨 EXTREME VOLATILITY: {prediction.volatility_forecast:.1f}% forecast")
    
    # Clear directional signal
    if prediction.signal == PredictionSignal.BULLISH and prediction.confidence > 0.75:
        alerts.append(f"📈 STRONG BUY SIGNAL: {prediction.confidence:.0%} confidence")
    elif prediction.signal == PredictionSignal.BEARISH and prediction.confidence > 0.75:
        alerts.append(f"📉 STRONG SELL SIGNAL: {prediction.confidence:.0%} confidence")
    
    # Risk warnings
    if prediction.risk_level == RiskLevel.CRITICAL:
        alerts.append("⚠️ CRITICAL RISK LEVEL: Reduce position size or hedge")
    elif prediction.risk_level == RiskLevel.HIGH:
        alerts.append("⚠️ HIGH RISK: Use appropriate stop-losses")
    
    # Wide price range warning
    if prediction.price_target_range[1] - prediction.price_target_range[0] > \
       (prediction.price_target_range[0] + prediction.price_target_range[1]) / 2 * 0.2:
        alerts.append("📊 WIDE PRICE RANGE: Expect high volatility in target range")
    
    return alerts
