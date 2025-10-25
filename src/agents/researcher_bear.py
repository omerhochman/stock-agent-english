from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint
import json
import ast


@agent_endpoint("researcher_bear", "Bearish researcher, analyzing market data from a bearish perspective and proposing risk warnings")
def researcher_bear_agent(state: AgentState):
    """Analyzes signals from a bearish perspective and generates cautionary investment thesis."""
    show_workflow_status("Bearish Researcher")
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

    # Analyze from bearish perspective
    bearish_points = []
    confidence_scores = []

    # Technical Analysis
    tech_bear_score = analyze_bearish_technical(technical_signals)
    bearish_points.append(tech_bear_score["point"])
    confidence_scores.append(tech_bear_score["confidence"])

    # Fundamental Analysis
    fund_bear_score = analyze_bearish_fundamental(fundamental_signals)
    bearish_points.append(fund_bear_score["point"])
    confidence_scores.append(fund_bear_score["confidence"])

    # Sentiment Analysis
    sent_bear_score = analyze_bearish_sentiment(sentiment_signals)
    bearish_points.append(sent_bear_score["point"])
    confidence_scores.append(sent_bear_score["confidence"])

    # Valuation Analysis
    val_bear_score = analyze_bearish_valuation(valuation_signals)
    bearish_points.append(val_bear_score["point"])
    confidence_scores.append(val_bear_score["confidence"])

    # Market Risk Factors Analysis
    market_risk_adjustment = analyze_market_risk_factors(
        fundamental_signals, technical_signals, valuation_signals, risk_data, macro_data)

    # Calculate overall bearish confidence with market risk adjustment
    base_confidence = sum(confidence_scores) / len(confidence_scores)
    final_confidence = min(base_confidence + market_risk_adjustment, 0.9)  # Cap at 90%

    # Add risk adjustment reasoning
    if market_risk_adjustment > 0:
        bearish_points.append(f"Market risk factor analysis increases bearish confidence: +{market_risk_adjustment:.2f}")

    message_content = {
        "perspective": "bearish",
        "confidence": final_confidence,
        "base_confidence": base_confidence,
        "risk_adjustment": market_risk_adjustment,
        "thesis_points": bearish_points,
        "reasoning": f"Bearish perspective based on comprehensive analysis of technical, fundamental, sentiment and valuation factors, considering overall market risk conditions",
        "risk_factors": {
            "market_risk_score": risk_data.get("risk_score") if risk_data else None,
            "macro_environment": macro_data.get("macro_environment") if macro_data else None,
            "volatility_regime": risk_data.get("risk_metrics", {}).get("volatility_regime") if risk_data else None
        }
    }

    message = HumanMessage(
        content=json.dumps(message_content),
        name="researcher_bear_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "Bearish Researcher")
        # Save reasoning information to metadata for API use
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("Bearish Researcher", "completed")
    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
        "metadata": state["metadata"],
    }


def analyze_bearish_technical(technical_signals: dict) -> dict:
    """Analyze bearish factors in technical indicators"""
    signal = technical_signals.get("signal", "neutral")
    confidence = technical_signals.get("confidence", 0.5)
    
    # Convert confidence format
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bearish":
        return {
            "point": f"Technical indicators show bearish trend, confidence: {confidence:.1%}",
            "confidence": confidence
        }
    elif signal == "neutral":
        # Neutral signal may indicate trend change
        confidence_adjustment = 0.1 if confidence > 0.7 else 0.05
        return {
            "point": f"Technical indicators neutral, but may indicate trend reversal (base confidence: {confidence:.1%})",
            "confidence": 0.4 + confidence_adjustment
        }
    else:  # bullish
        # Extreme optimism may be a reversal signal
        if confidence > 0.8:
            return {
                "point": f"Technical indicators extremely bullish, overbought risk exists, possible reversal (confidence: {confidence:.1%})",
                "confidence": 0.6
            }
        elif confidence > 0.6:
            return {
                "point": f"Technical indicators bullish, but need to watch for pullback risk (confidence: {confidence:.1%})",
                "confidence": 0.4
            }
        else:
            return {
                "point": f"Technical indicators mildly bullish, limited bearish opportunity (confidence: {confidence:.1%})",
                "confidence": 0.25
            }


def analyze_bearish_fundamental(fundamental_signals: dict) -> dict:
    """Analyze bearish factors in fundamental indicators"""
    signal = fundamental_signals.get("signal", "neutral")
    confidence = fundamental_signals.get("confidence", 0.5)
    reasoning = fundamental_signals.get("reasoning", {})
    
    # Convert confidence format
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bearish":
        return {
            "point": f"Fundamentals show multiple risk factors, confidence: {confidence:.1%}",
            "confidence": confidence
        }
    elif signal == "neutral":
        # Analyze specific risk indicators
        risk_indicators = []
        if reasoning:
            # Check cash flow issues
            if "cash_flow" in str(reasoning).lower() and "negative" in str(reasoning).lower():
                risk_indicators.append("Deteriorating cash flow")
            # Check debt issues
            if "debt" in str(reasoning).lower() and ("high" in str(reasoning).lower() or "increase" in str(reasoning).lower()):
                risk_indicators.append("Increasing debt burden")
            # Check declining profit margins
            if "margin" in str(reasoning).lower() and ("declin" in str(reasoning).lower() or "compress" in str(reasoning).lower()):
                risk_indicators.append("Declining profit margins")
        
        risk_count = len(risk_indicators)
        confidence_adjustment = risk_count * 0.1
        
        return {
            "point": f"Fundamentals neutral but with {risk_count} potential risk indicators: {', '.join(risk_indicators)}",
            "confidence": 0.35 + confidence_adjustment
        }
    else:  # bullish
        # Even with good fundamentals, watch for valuation risk
        if confidence > 0.8:
            return {
                "point": f"Strong fundamentals but valuation already fully reflected, limited upside (confidence: {confidence:.1%})",
                "confidence": 0.35
            }
        else:
            return {
                "point": f"Good fundamentals, but need to watch future growth sustainability (confidence: {confidence:.1%})",
                "confidence": 0.3
            }


def analyze_bearish_sentiment(sentiment_signals: dict) -> dict:
    """Analyze bearish factors in market sentiment"""
    signal = sentiment_signals.get("signal", "neutral")
    confidence = sentiment_signals.get("confidence", 0.5)
    
    # Convert confidence format
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bearish":
        return {
            "point": f"Market sentiment turning pessimistic, confidence: {confidence:.1%}",
            "confidence": confidence
        }
    elif signal == "neutral":
        return {
            "point": f"Market sentiment neutral, insufficient investor confidence, downside risk exists",
            "confidence": 0.4
        }
    else:  # bullish
        # Extreme optimism may be a top signal
        if confidence > 0.8:
            return {
                "point": f"Market sentiment overly optimistic, may be near top (confidence: {confidence:.1%})",
                "confidence": 0.55
            }
        elif confidence > 0.6:
            return {
                "point": f"Market sentiment quite optimistic, need to guard against sentiment reversal (confidence: {confidence:.1%})",
                "confidence": 0.35
            }
        else:
            return {
                "point": f"Market sentiment mildly optimistic, watch for sustainability (confidence: {confidence:.1%})",
                "confidence": 0.25
            }


def analyze_bearish_valuation(valuation_signals: dict) -> dict:
    """Analyze bearish factors in valuation"""
    signal = valuation_signals.get("signal", "neutral")
    confidence = valuation_signals.get("confidence", 0.5)
    valuation_gap = valuation_signals.get("valuation_gap", 0)
    
    # Convert confidence format
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bearish":
        return {
            "point": f"Valuation too high, significant downside risk (overvalued {abs(valuation_gap)*100:.1f}%, confidence: {confidence:.1%})",
            "confidence": confidence
        }
    elif signal == "neutral":
        # Even with reasonable valuation, consider if future growth can support it
        return {
            "point": f"Valuation reasonable, but need to watch if future growth can maintain current levels",
            "confidence": 0.35
        }
    else:  # bullish
        # Even with low valuation, consider if there are fundamental issues
        if valuation_gap < -0.3:  # Undervalued by more than 30%
            return {
                "point": f"Valuation appears cheap, but may have hidden fundamental issues (undervalued {abs(valuation_gap)*100:.1f}%)",
                "confidence": 0.4
            }
        else:
            return {
                "point": f"Valuation reasonably low, but upside potential may be limited (undervalued {abs(valuation_gap)*100:.1f}%)",
                "confidence": 0.25
            }


def analyze_market_risk_factors(fundamental_signals: dict, technical_signals: dict, 
                              valuation_signals: dict, risk_data: dict = None, 
                              macro_data: dict = None) -> float:
    """Analyze overall market risk factors, return additional bearish confidence adjustment"""
    risk_adjustment = 0.0
    
    # 1. Valuation risk
    if valuation_signals.get("valuation_gap", 0) < -0.2:  # Overvalued by more than 20%
        risk_adjustment += 0.15
    
    # 2. Overall market risk
    if risk_data:
        risk_score = risk_data.get("risk_score", 5)
        if risk_score >= 8:  # High risk environment
            risk_adjustment += 0.1
        
        # Volatility environment
        volatility_regime = risk_data.get("risk_metrics", {}).get("volatility_regime", "medium")
        if volatility_regime == "high":
            risk_adjustment += 0.05
    
    # 3. Macro environment
    if macro_data:
        macro_env = macro_data.get("macro_environment", "neutral")
        stock_impact = macro_data.get("impact_on_stock", "neutral")
        
        if macro_env == "negative" and stock_impact == "negative":
            risk_adjustment += 0.15
        elif macro_env == "negative" or stock_impact == "negative":
            risk_adjustment += 0.08
    
    # 4. Fundamental deterioration signs
    if fundamental_signals.get("reasoning", {}).get("financial_health_signal", {}).get("signal") == "bearish":
        risk_adjustment += 0.1
    
    # 5. Technical indicators showing top pattern
    if technical_signals.get("signal") == "neutral" and technical_signals.get("confidence", 0) > 0.7:
        # High confidence neutral signal may indicate trend reversal
        risk_adjustment += 0.05
    
    # Limit maximum adjustment
    return min(risk_adjustment, 0.4)