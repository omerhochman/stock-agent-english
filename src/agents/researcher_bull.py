from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint
import json
import ast


@agent_endpoint("researcher_bull", "Bullish researcher, analyzing market data from a bullish perspective and proposing investment theses")
def researcher_bull_agent(state: AgentState):
    """Analyzes signals from a bullish perspective and generates optimistic investment thesis."""
    show_workflow_status("Bullish Researcher")
    show_reasoning = state["metadata"]["show_reasoning"]

    # Fetch messages from analysts
    technical_message = next(
        msg for msg in state["messages"] if msg.name == "technical_analyst_agent")
    fundamentals_message = next(
        msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    sentiment_message = next(
        msg for msg in state["messages"] if msg.name == "sentiment_agent")
    valuation_message = next(
        msg for msg in state["messages"] if msg.name == "valuation_agent")

    try:
        fundamental_signals = json.loads(fundamentals_message.content)
        technical_signals = json.loads(technical_message.content)
        sentiment_signals = json.loads(sentiment_message.content)
        valuation_signals = json.loads(valuation_message.content)
    except Exception as e:
        fundamental_signals = ast.literal_eval(fundamentals_message.content)
        technical_signals = ast.literal_eval(technical_message.content)
        sentiment_signals = ast.literal_eval(sentiment_message.content)
        valuation_signals = ast.literal_eval(valuation_message.content)

    # Get risk management information if available
    risk_message = next(
        (msg for msg in state["messages"] if msg.name == "risk_management_agent"), None)
    risk_data = None
    if risk_message:
        try:
            risk_data = json.loads(risk_message.content)
        except:
            try:
                risk_data = ast.literal_eval(risk_message.content)
            except:
                risk_data = None

    # Get macro analysis information if available
    macro_message = next(
        (msg for msg in state["messages"] if msg.name == "macro_analyst_agent"), None)
    macro_data = None
    if macro_message:
        try:
            macro_data = json.loads(macro_message.content)
        except:
            try:
                macro_data = ast.literal_eval(macro_message.content)
            except:
                macro_data = None

    # Analyze from bullish perspective
    bullish_points = []
    confidence_scores = []

    # Technical Analysis
    tech_bull_score = analyze_bullish_technical(technical_signals)
    bullish_points.append(tech_bull_score["point"])
    confidence_scores.append(tech_bull_score["confidence"])

    # Fundamental Analysis
    fund_bull_score = analyze_bullish_fundamental(fundamental_signals)
    bullish_points.append(fund_bull_score["point"])
    confidence_scores.append(fund_bull_score["confidence"])

    # Sentiment Analysis
    sent_bull_score = analyze_bullish_sentiment(sentiment_signals)
    bullish_points.append(sent_bull_score["point"])
    confidence_scores.append(sent_bull_score["confidence"])

    # Valuation Analysis
    val_bull_score = analyze_bullish_valuation(valuation_signals)
    bullish_points.append(val_bull_score["point"])
    confidence_scores.append(val_bull_score["confidence"])

    # Market Opportunity Factors Analysis
    market_opportunity_adjustment = analyze_market_opportunity_factors(
        fundamental_signals, technical_signals, valuation_signals, risk_data, macro_data)

    # Calculate overall bullish confidence with market opportunity adjustment
    base_confidence = sum(confidence_scores) / len(confidence_scores)
    final_confidence = min(base_confidence + market_opportunity_adjustment, 0.9)  # Cap at 90%

    # Add opportunity adjustment reasoning
    if market_opportunity_adjustment > 0:
        bullish_points.append(f"Market opportunity factor analysis increases bullish confidence: +{market_opportunity_adjustment:.2f}")

    message_content = {
        "perspective": "bullish",
        "confidence": final_confidence,
        "base_confidence": base_confidence,
        "opportunity_adjustment": market_opportunity_adjustment,
        "thesis_points": bullish_points,
        "reasoning": f"Bullish perspective based on comprehensive analysis of technical, fundamental, sentiment and valuation factors, considering overall market opportunity environment",
        "opportunity_factors": {
            "market_risk_score": risk_data.get("risk_score") if risk_data else None,
            "macro_environment": macro_data.get("macro_environment") if macro_data else None,
            "volatility_regime": risk_data.get("risk_metrics", {}).get("volatility_regime") if risk_data else None
        }
    }

    message = HumanMessage(
        content=json.dumps(message_content),
        name="researcher_bull_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "Bullish Researcher")
        # Save reasoning information to metadata for API use
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("Bullish Researcher", "completed")
    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
        "metadata": state["metadata"],
    }


def analyze_bullish_technical(technical_signals: dict) -> dict:
    """Analyze bullish factors in technical indicators"""
    signal = technical_signals.get("signal", "neutral")
    confidence = technical_signals.get("confidence", 0.5)
    
    # Convert confidence format
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bullish":
        return {
            "point": f"Technical indicators show bullish trend, confidence: {confidence:.1%}",
            "confidence": confidence
        }
    elif signal == "neutral":
        # Neutral signal may indicate breakout opportunity
        confidence_adjustment = 0.1 if confidence > 0.7 else 0.05
        return {
            "point": f"Technical indicators neutral, but may indicate upward breakout (base confidence: {confidence:.1%})",
            "confidence": 0.4 + confidence_adjustment
        }
    else:  # bearish
        # Extreme pessimism may be a bottom signal
        if confidence > 0.8:
            return {
                "point": f"Technical indicators extremely bearish, oversold opportunity, possible reversal (confidence: {confidence:.1%})",
                "confidence": 0.55
            }
        elif confidence > 0.6:
            return {
                "point": f"Technical indicators bearish, but may be near support level (confidence: {confidence:.1%})",
                "confidence": 0.35
            }
        else:
            return {
                "point": f"Technical indicators mildly bearish, watch for rebound opportunity (confidence: {confidence:.1%})",
                "confidence": 0.3
            }


def analyze_bullish_fundamental(fundamental_signals: dict) -> dict:
    """Analyze bullish factors in fundamental indicators"""
    signal = fundamental_signals.get("signal", "neutral")
    confidence = fundamental_signals.get("confidence", 0.5)
    reasoning = fundamental_signals.get("reasoning", {})
    
    # Convert confidence format
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bullish":
        return {
            "point": f"Strong fundamentals, multiple positive indicators, confidence: {confidence:.1%}",
            "confidence": confidence
        }
    elif signal == "neutral":
        # Analyze specific growth indicators
        growth_indicators = []
        if reasoning:
            # Check growth indicators
            if "growth" in str(reasoning).lower() and ("positive" in str(reasoning).lower() or "increas" in str(reasoning).lower()):
                growth_indicators.append("Strong growth momentum")
            # Check profitability quality
            if "profitability" in str(reasoning).lower() and ("improv" in str(reasoning).lower() or "strong" in str(reasoning).lower()):
                growth_indicators.append("Enhanced profitability")
            # Check cash flow
            if "cash" in str(reasoning).lower() and ("flow" in str(reasoning).lower() and "positive" in str(reasoning).lower()):
                growth_indicators.append("Improved cash flow")
        
        growth_count = len(growth_indicators)
        confidence_adjustment = growth_count * 0.1
        
        return {
            "point": f"Fundamentals neutral but with {growth_count} positive factors: {', '.join(growth_indicators)}",
            "confidence": 0.4 + confidence_adjustment
        }
    else:  # bearish
        # Even with poor fundamentals, look for turnaround signs
        if confidence > 0.8:
            return {
                "point": f"Fundamentals temporarily difficult, but may be fully reflected, recovery opportunity exists (confidence: {confidence:.1%})",
                "confidence": 0.4
            }
        else:
            return {
                "point": f"Fundamentals need improvement, watch management response measures (confidence: {confidence:.1%})",
                "confidence": 0.3
            }


def analyze_bullish_sentiment(sentiment_signals: dict) -> dict:
    """Analyze bullish factors in market sentiment"""
    signal = sentiment_signals.get("signal", "neutral")
    confidence = sentiment_signals.get("confidence", 0.5)
    
    # Convert confidence format
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bullish":
        return {
            "point": f"Market sentiment turning optimistic, confidence: {confidence:.1%}",
            "confidence": confidence
        }
    elif signal == "neutral":
        return {
            "point": f"Market sentiment neutral, providing buying opportunity for bulls",
            "confidence": 0.4
        }
    else:  # bearish
        # Extreme pessimism may be a bottom signal
        if confidence > 0.8:
            return {
                "point": f"Market sentiment overly pessimistic, may be near bottom reversal point (confidence: {confidence:.1%})",
                "confidence": 0.6
            }
        elif confidence > 0.6:
            return {
                "point": f"Market sentiment bearish, but sentiment recovery opportunity exists (confidence: {confidence:.1%})",
                "confidence": 0.4
            }
        else:
            return {
                "point": f"Market sentiment mildly pessimistic, watch for improvement trend (confidence: {confidence:.1%})",
                "confidence": 0.3
            }


def analyze_bullish_valuation(valuation_signals: dict) -> dict:
    """Analyze bullish factors in valuation"""
    signal = valuation_signals.get("signal", "neutral")
    confidence = valuation_signals.get("confidence", 0.5)
    valuation_gap = valuation_signals.get("valuation_gap", 0)
    
    # Convert confidence format
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bullish":
        return {
            "point": f"Valuation attractive, upside potential exists (undervalued {abs(valuation_gap)*100:.1f}%, confidence: {confidence:.1%})",
            "confidence": confidence
        }
    elif signal == "neutral":
        # Even with reasonable valuation, consider future growth potential
        return {
            "point": f"Valuation reasonable, upside potential if fundamentals improve",
            "confidence": 0.35
        }
    else:  # bearish
        # Even with high valuation, consider if there's growth support
        if valuation_gap > 0.2:  # Overvalued by more than 20%
            return {
                "point": f"Valuation high, but upside potential if earnings exceed expectations (overvalued {abs(valuation_gap)*100:.1f}%)",
                "confidence": 0.3
            }
        else:
            return {
                "point": f"Valuation slightly high, watch for performance improvement to digest valuation (overvalued {abs(valuation_gap)*100:.1f}%)",
                "confidence": 0.35
            }


def analyze_market_opportunity_factors(fundamental_signals: dict, technical_signals: dict, 
                                    valuation_signals: dict, risk_data: dict = None, 
                                    macro_data: dict = None) -> float:
    """Analyze overall market opportunity factors, return additional bullish confidence adjustment"""
    opportunity_adjustment = 0.0
    
    # 1. Valuation opportunity
    if valuation_signals.get("valuation_gap", 0) > 0.2:  # Undervalued by more than 20%
        opportunity_adjustment += 0.15
    
    # 2. Overall market opportunity
    if risk_data:
        risk_score = risk_data.get("risk_score", 5)
        if risk_score <= 3:  # Low risk environment
            opportunity_adjustment += 0.1
        
        # Volatility environment
        volatility_regime = risk_data.get("risk_metrics", {}).get("volatility_regime", "medium")
        if volatility_regime == "low":
            opportunity_adjustment += 0.05
    
    # 3. Macro environment
    if macro_data:
        macro_env = macro_data.get("macro_environment", "neutral")
        stock_impact = macro_data.get("impact_on_stock", "neutral")
        
        if macro_env == "positive" and stock_impact == "positive":
            opportunity_adjustment += 0.15
        elif macro_env == "positive" or stock_impact == "positive":
            opportunity_adjustment += 0.08
    
    # 4. Fundamental improvement signs
    if fundamental_signals.get("reasoning", {}).get("growth_signal", {}).get("signal") == "bullish":
        opportunity_adjustment += 0.1
    
    # 5. Technical indicators showing bottom pattern
    if technical_signals.get("signal") == "neutral" and technical_signals.get("confidence", 0) > 0.7:
        # High confidence neutral signal may indicate trend reversal
        opportunity_adjustment += 0.05
    
    # 6. Oversold bounce opportunity
    if technical_signals.get("signal") == "bearish" and technical_signals.get("confidence", 0) > 0.7:
        # Extreme oversold may be buying opportunity
        opportunity_adjustment += 0.08
    
    # Limit maximum adjustment
    return min(opportunity_adjustment, 0.4)