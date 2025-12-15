"""
Example usage script for the Market Activity Prediction Agent system.
Demonstrates multi-agent analysis with GenAI-powered reasoning.
"""

import json
import sys
from typing import List
from datetime import datetime
from market_agent import MarketPredictionOrchestrator, MarketActivityAgent
from utils import (
    format_prediction_report,
    get_action_recommendation,
    calculate_position_sizing,
    assess_prediction_quality,
    compare_predictions,
    get_market_regime,
    generate_trading_alerts
)
from config import OutputTemplates

def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")

def example_1_legacy_agent():
    """Example 1: Using the legacy MarketActivityAgent."""
    print_section("EXAMPLE 1: Legacy Market Analysis Agent")
    
    agent = MarketActivityAgent()
    result = agent.analyze("NVDA")
    
    print(f"Ticker: {result['ticker']}")
    print(f"Current Price: ${result['current_price']}")
    print(f"Alert Level: {result['alert_level']}")
    print(f"Volatility (Annualized): {result['prediction']['volatility_annualized']}")
    print(f"5-Day Forecast Range: ${result['prediction']['5_day_forecast_range'][0]} - ${result['prediction']['5_day_forecast_range'][1]}")
    print(f"Direction Bias: {result['prediction']['direction_bias']}")
    print(f"Confidence: {result['analysis']['confidence_score']}")
    print(f"\nRationale:\n{result['analysis']['rationale']}")
    
    return result

def example_2_multi_agent_prediction():
    """Example 2: Advanced multi-agent prediction with GenAI."""
    print_section("EXAMPLE 2: Multi-Agent Prediction System with GenAI")
    
    try:
        orchestrator = MarketPredictionOrchestrator()
        ticker = "NVDA"
        
        print(f"Analyzing {ticker} with multi-agent system...\n")
        prediction = orchestrator.predict_market_activity(ticker)
        
        # Display structured prediction
        print(f"Ticker: {prediction.ticker}")
        print(f"Signal: {prediction.signal.value}")
        print(f"Confidence: {prediction.confidence:.1%}")
        print(f"Risk Level: {prediction.risk_level.value}")
        print(f"Volatility Forecast: {prediction.volatility_forecast:.2f}%")
        print(f"Price Target Range: ${prediction.price_target_range[0]:.2f} - ${prediction.price_target_range[1]:.2f}")
        
        print("\n" + "-" * 80)
        print("KEY FACTORS:")
        print("-" * 80)
        for i, factor in enumerate(prediction.key_factors[:8], 1):
            print(f"{i}. {factor}")
        
        print("\n" + "-" * 80)
        print("AI-POWERED REASONING:")
        print("-" * 80)
        print(prediction.reasoning)
        
        return prediction
        
    except Exception as e:
        print(f"⚠️  GenAI features require GEMINI_API_KEY environment variable")
        print(f"Error: {str(e)}")
        return None

def example_3_quality_assessment(prediction):
    """Example 3: Assessment of prediction quality."""
    if not prediction:
        print("Skipping quality assessment - no valid prediction")
        return
    
    print_section("EXAMPLE 3: Prediction Quality Assessment")
    
    quality = assess_prediction_quality(prediction)
    
    print(f"Quality Score: {quality['overall_quality_score']}/100")
    print(f"Reliability Rating: {quality['reliability_rating']}")
    print(f"Suitable for Trading: {'YES' if quality['suitable_for_trading'] else 'NO'}\n")
    
    print("Strengths:")
    for strength in quality['strengths']:
        print(f"  ✓ {strength}")
    
    if quality['issues']:
        print("\nIssues to Consider:")
        for issue in quality['issues']:
            print(f"  ⚠️  {issue}")

def example_4_trading_guidance(prediction):
    """Example 4: Trading guidance and position sizing."""
    if not prediction:
        print("Skipping trading guidance - no valid prediction")
        return
    
    print_section("EXAMPLE 4: Trading Guidance")
    
    # Get action recommendation
    action = get_action_recommendation(prediction.signal, prediction.confidence)
    print(f"Recommended Action:\n  {action}\n")
    
    # Position sizing (assuming $100,000 account)
    account_size = 100000
    position = calculate_position_sizing(prediction.risk_level, account_size, prediction.confidence)
    
    print(f"Position Sizing (${account_size:,} account):")
    print(f"  Recommended Position Size: ${position['recommended_position_size']:,.2f}")
    print(f"  Allocation Percentage: {position['allocation_percent']:.2f}%")
    print(f"  Stop-Loss Distance: {position['stop_loss_distance']:.1%}")
    print(f"  Take-Profit Distance: {position['take_profit_distance']:.1%}\n")
    
    # Market regime
    regime = get_market_regime(prediction)
    print(f"Market Regime: {regime}\n")
    
    # Trading alerts
    alerts = generate_trading_alerts(prediction)
    if alerts:
        print("Trading Alerts:")
        for alert in alerts:
            print(f"  {alert}")

def example_5_comparative_analysis():
    """Example 5: Comparative analysis across multiple tickers."""
    print_section("EXAMPLE 5: Comparative Analysis (Multiple Tickers)")
    
    try:
        orchestrator = MarketPredictionOrchestrator()
        tickers = ["NVDA", "AAPL", "MSFT"]
        
        predictions = []
        print(f"Analyzing {', '.join(tickers)}...\n")
        
        for ticker in tickers:
            try:
                prediction = orchestrator.predict_market_activity(ticker)
                predictions.append(prediction)
                print(f"✓ {ticker}: {prediction.signal.value} ({prediction.confidence:.0%} confidence)")
            except Exception as e:
                print(f"✗ {ticker}: Analysis failed - {str(e)}")
        
        if predictions:
            print("\n" + "-" * 80)
            print("CONSENSUS ANALYSIS:")
            print("-" * 80)
            
            bullish = sum(1 for p in predictions if p.signal.value == "BULLISH")
            bearish = sum(1 for p in predictions if p.signal.value == "BEARISH")
            
            print(f"Bullish Signals: {bullish}/{len(predictions)}")
            print(f"Bearish Signals: {bearish}/{len(predictions)}")
            print(f"Average Confidence: {sum(p.confidence for p in predictions)/len(predictions):.1%}")
            print(f"Average Volatility: {sum(p.volatility_forecast for p in predictions)/len(predictions):.2f}%")
        
    except Exception as e:
        print(f"Error in comparative analysis: {str(e)}")

def example_6_detailed_report(prediction):
    """Example 6: Generate comprehensive detailed report."""
    if not prediction:
        print("Skipping detailed report - no valid prediction")
        return
    
    print_section("EXAMPLE 6: Comprehensive Detailed Report")
    
    try:
        orchestrator = MarketPredictionOrchestrator()
        report = orchestrator.get_detailed_report(prediction.ticker)
        
        print("Summary:")
        for key, value in report['summary'].items():
            print(f"  {key}: {value}")
        
        print("\nPrediction Metrics:")
        for key, value in report['prediction'].items():
            print(f"  {key}: {value}")
        
        print("\nTechnical Indicators:")
        tech = report.get('technical_indicators', {})
        for key, value in tech.items():
            if key not in ['technical_factors']:
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")

def example_7_formatted_report(prediction):
    """Example 7: Display formatted prediction report."""
    if not prediction:
        print("Skipping formatted report - no valid prediction")
        return
    
    # Get current price for report
    from market_agent import MarketDataManager
    data_mgr = MarketDataManager()
    df = data_mgr.get_historical_data(prediction.ticker)
    current_price = df['Close'].iloc[-1]
    
    report = format_prediction_report(prediction, current_price)
    print(report)

def main():
    """Run all examples."""
    print("\n" + "🚀 " * 20)
    print("MARKET ACTIVITY PREDICTION AGENT - COMPREHENSIVE EXAMPLES")
    print("🚀 " * 20)
    
    try:
        # Example 1: Legacy agent (always works)
        legacy_result = example_1_legacy_agent()
        
        # Example 2: Multi-agent system
        prediction = example_2_multi_agent_prediction()
        
        if prediction:
            # Examples 3-7: Only run if prediction was successful
            example_3_quality_assessment(prediction)
            example_4_trading_guidance(prediction)
            example_6_detailed_report(prediction)
            example_7_formatted_report(prediction)
            example_5_comparative_analysis()
        
        print_section("Analysis Complete")
        print("✅ All examples executed successfully!")
        print("\nKey Takeaways:")
        print("1. Multi-agent system coordinates specialized analysis")
        print("2. GenAI provides reasoning and explainability")
        print("3. Comprehensive risk assessment supports decision-making")
        print("4. Structured predictions enable systematic trading strategies")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
