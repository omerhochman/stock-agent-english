from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint
import json
import ast


@agent_endpoint("researcher_bull", "多方研究员，从看多角度分析市场数据并提出投资论点")
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
        bullish_points.append(f"市场机会因素分析提升多头置信度: +{market_opportunity_adjustment:.2f}")

    message_content = {
        "perspective": "bullish",
        "confidence": final_confidence,
        "base_confidence": base_confidence,
        "opportunity_adjustment": market_opportunity_adjustment,
        "thesis_points": bullish_points,
        "reasoning": f"多头观点基于技术、基本面、情绪和估值因素的综合分析，并考虑了市场整体机会环境",
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
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("Bullish Researcher", "completed")
    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
        "metadata": state["metadata"],
    }


def analyze_bullish_technical(technical_signals: dict) -> dict:
    """分析技术指标的多头因素"""
    signal = technical_signals.get("signal", "neutral")
    confidence = technical_signals.get("confidence", 0.5)
    
    # 转换confidence格式
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bullish":
        return {
            "point": f"技术指标显示看多趋势，置信度: {confidence:.1%}",
            "confidence": confidence
        }
    elif signal == "neutral":
        # 中性信号可能暗示突破机会
        confidence_adjustment = 0.1 if confidence > 0.7 else 0.05
        return {
            "point": f"技术指标中性，但可能预示上行突破 (基础置信度: {confidence:.1%})",
            "confidence": 0.4 + confidence_adjustment
        }
    else:  # bearish
        # 极度悲观可能是底部信号
        if confidence > 0.8:
            return {
                "point": f"技术指标极度看空，存在超卖机会，可能反转 (置信度: {confidence:.1%})",
                "confidence": 0.55
            }
        elif confidence > 0.6:
            return {
                "point": f"技术指标偏空，但可能接近支撑位 (置信度: {confidence:.1%})",
                "confidence": 0.35
            }
        else:
            return {
                "point": f"技术指标轻度看空，关注反弹机会 (置信度: {confidence:.1%})",
                "confidence": 0.3
            }


def analyze_bullish_fundamental(fundamental_signals: dict) -> dict:
    """分析基本面的多头因素"""
    signal = fundamental_signals.get("signal", "neutral")
    confidence = fundamental_signals.get("confidence", 0.5)
    reasoning = fundamental_signals.get("reasoning", {})
    
    # 转换confidence格式
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bullish":
        return {
            "point": f"基本面强劲，多项指标积极，置信度: {confidence:.1%}",
            "confidence": confidence
        }
    elif signal == "neutral":
        # 分析具体的增长指标
        growth_indicators = []
        if reasoning:
            # 检查增长指标
            if "growth" in str(reasoning).lower() and ("positive" in str(reasoning).lower() or "increas" in str(reasoning).lower()):
                growth_indicators.append("增长势头良好")
            # 检查利润质量
            if "profitability" in str(reasoning).lower() and ("improv" in str(reasoning).lower() or "strong" in str(reasoning).lower()):
                growth_indicators.append("盈利能力增强")
            # 检查现金流
            if "cash" in str(reasoning).lower() and ("flow" in str(reasoning).lower() and "positive" in str(reasoning).lower()):
                growth_indicators.append("现金流改善")
        
        growth_count = len(growth_indicators)
        confidence_adjustment = growth_count * 0.1
        
        return {
            "point": f"基本面中性但有{growth_count}个积极因素: {', '.join(growth_indicators)}",
            "confidence": 0.4 + confidence_adjustment
        }
    else:  # bearish
        # 即使基本面不佳，也要寻找转机迹象
        if confidence > 0.8:
            return {
                "point": f"基本面暂时困难，但可能已充分反映，存在修复机会 (置信度: {confidence:.1%})",
                "confidence": 0.4
            }
        else:
            return {
                "point": f"基本面待改善，关注管理层应对措施 (置信度: {confidence:.1%})",
                "confidence": 0.3
            }


def analyze_bullish_sentiment(sentiment_signals: dict) -> dict:
    """分析市场情绪的多头因素"""
    signal = sentiment_signals.get("signal", "neutral")
    confidence = sentiment_signals.get("confidence", 0.5)
    
    # 转换confidence格式
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bullish":
        return {
            "point": f"市场情绪转向乐观，置信度: {confidence:.1%}",
            "confidence": confidence
        }
    elif signal == "neutral":
        return {
            "point": f"市场情绪中性，为多头提供买入机会",
            "confidence": 0.4
        }
    else:  # bearish
        # 极度悲观的情绪可能是底部信号
        if confidence > 0.8:
            return {
                "point": f"市场情绪过度悲观，可能接近底部反转点 (置信度: {confidence:.1%})",
                "confidence": 0.6
            }
        elif confidence > 0.6:
            return {
                "point": f"市场情绪偏空，但存在情绪修复机会 (置信度: {confidence:.1%})",
                "confidence": 0.4
            }
        else:
            return {
                "point": f"市场情绪温和悲观，关注改善趋势 (置信度: {confidence:.1%})",
                "confidence": 0.3
            }


def analyze_bullish_valuation(valuation_signals: dict) -> dict:
    """分析估值的多头因素"""
    signal = valuation_signals.get("signal", "neutral")
    confidence = valuation_signals.get("confidence", 0.5)
    valuation_gap = valuation_signals.get("valuation_gap", 0)
    
    # 转换confidence格式
    if isinstance(confidence, str):
        confidence = float(confidence.replace("%", "")) / 100
    
    if signal == "bullish":
        return {
            "point": f"估值有吸引力，存在上行空间 (低估{abs(valuation_gap)*100:.1f}%，置信度: {confidence:.1%})",
            "confidence": confidence
        }
    elif signal == "neutral":
        # 即使估值合理，也要考虑未来增长潜力
        return {
            "point": f"估值合理，若基本面改善则有上行空间",
            "confidence": 0.35
        }
    else:  # bearish
        # 即使估值偏高，也要考虑是否有增长支撑
        if valuation_gap > 0.2:  # 高估超过20%
            return {
                "point": f"估值偏高，但若业绩超预期仍有上行潜力 (高估{abs(valuation_gap)*100:.1f}%)",
                "confidence": 0.3
            }
        else:
            return {
                "point": f"估值略高，关注性能改善消化估值 (高估{abs(valuation_gap)*100:.1f}%)",
                "confidence": 0.35
            }


def analyze_market_opportunity_factors(fundamental_signals: dict, technical_signals: dict, 
                                    valuation_signals: dict, risk_data: dict = None, 
                                    macro_data: dict = None) -> float:
    """分析市场整体机会因素，返回额外的多头置信度调整"""
    opportunity_adjustment = 0.0
    
    # 1. 估值机会
    if valuation_signals.get("valuation_gap", 0) > 0.2:  # 低估超过20%
        opportunity_adjustment += 0.15
    
    # 2. 市场整体机会
    if risk_data:
        risk_score = risk_data.get("risk_score", 5)
        if risk_score <= 3:  # 低风险环境
            opportunity_adjustment += 0.1
        
        # 波动率环境
        volatility_regime = risk_data.get("risk_metrics", {}).get("volatility_regime", "medium")
        if volatility_regime == "low":
            opportunity_adjustment += 0.05
    
    # 3. 宏观环境
    if macro_data:
        macro_env = macro_data.get("macro_environment", "neutral")
        stock_impact = macro_data.get("impact_on_stock", "neutral")
        
        if macro_env == "positive" and stock_impact == "positive":
            opportunity_adjustment += 0.15
        elif macro_env == "positive" or stock_impact == "positive":
            opportunity_adjustment += 0.08
    
    # 4. 基本面改善迹象
    if fundamental_signals.get("reasoning", {}).get("growth_signal", {}).get("signal") == "bullish":
        opportunity_adjustment += 0.1
    
    # 5. 技术指标显示底部形态
    if technical_signals.get("signal") == "neutral" and technical_signals.get("confidence", 0) > 0.7:
        # 高置信度的中性信号可能暗示趋势反转
        opportunity_adjustment += 0.05
    
    # 6. 超卖回调机会
    if technical_signals.get("signal") == "bearish" and technical_signals.get("confidence", 0) > 0.7:
        # 极度超卖可能是买入机会
        opportunity_adjustment += 0.08
    
    # 限制最大调整幅度
    return min(opportunity_adjustment, 0.4)