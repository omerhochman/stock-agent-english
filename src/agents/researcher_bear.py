from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint
import json
import ast


@agent_endpoint("researcher_bear", "空方研究员，从看空角度分析市场数据并提出风险警示")
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
        bearish_points.append(f"市场风险因素分析提升空头置信度: +{market_risk_adjustment:.2f}")

    message_content = {
        "perspective": "bearish",
        "confidence": final_confidence,
        "base_confidence": base_confidence,
        "risk_adjustment": market_risk_adjustment,
        "thesis_points": bearish_points,
        "reasoning": f"空头观点基于技术、基本面、情绪和估值因素的综合分析，并考虑了市场整体风险状况",
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
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("Bearish Researcher", "completed")
    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
        "metadata": state["metadata"],
    }


def analyze_bearish_technical(technical_signals: dict) -> dict:
    """分析技术指标的空头因素"""
    signal = technical_signals.get("signal", "neutral")
    confidence = technical_signals.get("confidence", 0.5)
    
    # 转换confidence格式
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bearish":
        return {
            "point": f"技术指标显示看空趋势，置信度: {confidence:.1%}",
            "confidence": confidence
        }
    elif signal == "neutral":
        # 中性信号可能暗示趋势转变
        confidence_adjustment = 0.1 if confidence > 0.7 else 0.05
        return {
            "point": f"技术指标中性，但可能预示趋势反转 (基础置信度: {confidence:.1%})",
            "confidence": 0.4 + confidence_adjustment
        }
    else:  # bullish
        # 极度乐观可能是反转信号
        if confidence > 0.8:
            return {
                "point": f"技术指标极度看多，存在超买风险，可能反转 (置信度: {confidence:.1%})",
                "confidence": 0.6
            }
        elif confidence > 0.6:
            return {
                "point": f"技术指标偏多，但需警惕回调风险 (置信度: {confidence:.1%})",
                "confidence": 0.4
            }
        else:
            return {
                "point": f"技术指标轻度看多，空头机会有限 (置信度: {confidence:.1%})",
                "confidence": 0.25
            }


def analyze_bearish_fundamental(fundamental_signals: dict) -> dict:
    """分析基本面的空头因素"""
    signal = fundamental_signals.get("signal", "neutral")
    confidence = fundamental_signals.get("confidence", 0.5)
    reasoning = fundamental_signals.get("reasoning", {})
    
    # 转换confidence格式
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bearish":
        return {
            "point": f"基本面显示多项风险因素，置信度: {confidence:.1%}",
            "confidence": confidence
        }
    elif signal == "neutral":
        # 分析具体的风险指标
        risk_indicators = []
        if reasoning:
            # 检查现金流问题
            if "cash_flow" in str(reasoning).lower() and "negative" in str(reasoning).lower():
                risk_indicators.append("现金流恶化")
            # 检查负债问题
            if "debt" in str(reasoning).lower() and ("high" in str(reasoning).lower() or "increase" in str(reasoning).lower()):
                risk_indicators.append("债务负担加重")
            # 检查利润率下降
            if "margin" in str(reasoning).lower() and ("declin" in str(reasoning).lower() or "compress" in str(reasoning).lower()):
                risk_indicators.append("利润率下降")
        
        risk_count = len(risk_indicators)
        confidence_adjustment = risk_count * 0.1
        
        return {
            "point": f"基本面中性但存在{risk_count}个潜在风险指标: {', '.join(risk_indicators)}",
            "confidence": 0.35 + confidence_adjustment
        }
    else:  # bullish
        # 即使基本面看好，也要关注估值风险
        if confidence > 0.8:
            return {
                "point": f"基本面强劲但估值已充分反映，上行空间有限 (置信度: {confidence:.1%})",
                "confidence": 0.35
            }
        else:
            return {
                "point": f"基本面良好，但需关注未来增长可持续性 (置信度: {confidence:.1%})",
                "confidence": 0.3
            }


def analyze_bearish_sentiment(sentiment_signals: dict) -> dict:
    """分析市场情绪的空头因素"""
    signal = sentiment_signals.get("signal", "neutral")
    confidence = sentiment_signals.get("confidence", 0.5)
    
    # 转换confidence格式
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bearish":
        return {
            "point": f"市场情绪转向悲观，置信度: {confidence:.1%}",
            "confidence": confidence
        }
    elif signal == "neutral":
        return {
            "point": f"市场情绪中性，投资者信心不足，存在下行风险",
            "confidence": 0.4
        }
    else:  # bullish
        # 极度乐观的情绪可能是顶部信号
        if confidence > 0.8:
            return {
                "point": f"市场情绪过度乐观，可能已接近顶部 (置信度: {confidence:.1%})",
                "confidence": 0.55
            }
        elif confidence > 0.6:
            return {
                "point": f"市场情绪较为乐观，需提防情绪反转 (置信度: {confidence:.1%})",
                "confidence": 0.35
            }
        else:
            return {
                "point": f"市场情绪温和乐观，关注持续性 (置信度: {confidence:.1%})",
                "confidence": 0.25
            }


def analyze_bearish_valuation(valuation_signals: dict) -> dict:
    """分析估值的空头因素"""
    signal = valuation_signals.get("signal", "neutral")
    confidence = valuation_signals.get("confidence", 0.5)
    valuation_gap = valuation_signals.get("valuation_gap", 0)
    
    # 转换confidence格式
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bearish":
        return {
            "point": f"估值过高，下行风险显著 (高估{abs(valuation_gap)*100:.1f}%，置信度: {confidence:.1%})",
            "confidence": confidence
        }
    elif signal == "neutral":
        # 即使估值合理，也要考虑未来增长是否能支撑
        return {
            "point": f"估值合理，但需关注未来增长能否维系当前水平",
            "confidence": 0.35
        }
    else:  # bullish
        # 即使估值偏低，也要考虑是否有基本面问题
        if valuation_gap < -0.3:  # 低估超过30%
            return {
                "point": f"估值看似便宜，但可能存在隐藏的基本面问题 (低估{abs(valuation_gap)*100:.1f}%)",
                "confidence": 0.4
            }
        else:
            return {
                "point": f"估值合理偏低，但上行空间可能有限 (低估{abs(valuation_gap)*100:.1f}%)",
                "confidence": 0.25
            }


def analyze_market_risk_factors(fundamental_signals: dict, technical_signals: dict, 
                              valuation_signals: dict, risk_data: dict = None, 
                              macro_data: dict = None) -> float:
    """分析市场整体风险因素，返回额外的空头置信度调整"""
    risk_adjustment = 0.0
    
    # 1. 估值风险
    if valuation_signals.get("valuation_gap", 0) < -0.2:  # 高估超过20%
        risk_adjustment += 0.15
    
    # 2. 市场整体风险
    if risk_data:
        risk_score = risk_data.get("risk_score", 5)
        if risk_score >= 8:  # 高风险环境
            risk_adjustment += 0.1
        
        # 波动率环境
        volatility_regime = risk_data.get("risk_metrics", {}).get("volatility_regime", "medium")
        if volatility_regime == "high":
            risk_adjustment += 0.05
    
    # 3. 宏观环境
    if macro_data:
        macro_env = macro_data.get("macro_environment", "neutral")
        stock_impact = macro_data.get("impact_on_stock", "neutral")
        
        if macro_env == "negative" and stock_impact == "negative":
            risk_adjustment += 0.15
        elif macro_env == "negative" or stock_impact == "negative":
            risk_adjustment += 0.08
    
    # 4. 基本面恶化迹象
    if fundamental_signals.get("reasoning", {}).get("financial_health_signal", {}).get("signal") == "bearish":
        risk_adjustment += 0.1
    
    # 5. 技术指标显示顶部形态
    if technical_signals.get("signal") == "neutral" and technical_signals.get("confidence", 0) > 0.7:
        # 高置信度的中性信号可能暗示趋势反转
        risk_adjustment += 0.05
    
    # 限制最大调整幅度
    return min(risk_adjustment, 0.4)