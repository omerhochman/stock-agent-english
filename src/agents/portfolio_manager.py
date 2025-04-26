from langchain_core.messages import HumanMessage
import json

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.openrouter_config import get_chat_completion
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.calc.portfolio_optimization import optimize_portfolio
from src.calc.covariance_estimation import estimate_covariance_ewma


##### Portfolio Management Agent #####
@agent_endpoint("portfolio_management", "负责投资组合管理和最终交易决策")
def portfolio_management_agent(state: AgentState):
    """负责投资组合管理和交易决策"""
    show_workflow_status("Portfolio Manager")
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    ticker = data.get("ticker", "")
    prices = data.get("prices", [])
    
    # 获取最新价格
    current_price = prices[-1]['close'] if prices else 0
    
    # 获取各个分析师的信号
    technical_message = next(
        msg for msg in state["messages"] if msg.name == "technical_analyst_agent")
    fundamentals_message = next(
        msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    sentiment_message = next(
        msg for msg in state["messages"] if msg.name == "sentiment_agent")
    valuation_message = next(
        msg for msg in state["messages"] if msg.name == "valuation_agent")
    risk_message = next(
        msg for msg in state["messages"] if msg.name == "risk_management_agent")

    # 创建系统消息
    system_message = {
        "role": "system",
        "content": """You are a portfolio manager making final trading decisions.
            Your job is to make a trading decision based on the team's analysis while strictly adhering
            to risk management constraints.

            RISK MANAGEMENT CONSTRAINTS:
            - You MUST NOT exceed the max_position_size specified by the risk manager
            - You MUST follow the trading_action (buy/sell/hold) recommended by risk management
            - These are hard constraints that cannot be overridden by other signals

            When weighing the different signals for direction and timing:
            1. Valuation Analysis (35% weight)
               - Primary driver of fair value assessment
               - Determines if price offers good entry/exit point
            
            2. Fundamental Analysis (30% weight)
               - Business quality and growth assessment
               - Determines conviction in long-term potential
            
            3. Technical Analysis (25% weight)
               - Secondary confirmation
               - Helps with entry/exit timing
            
            4. Sentiment Analysis (10% weight)
               - Final consideration
               - Can influence sizing within risk limits
            
            The decision process should be:
            1. First check risk management constraints
            2. Then evaluate valuation signal
            3. Then evaluate fundamentals signal
            4. Use technical analysis for timing
            5. Consider sentiment for final adjustment
            
            Provide the following in your output:
            - "action": "buy" | "sell" | "hold",
            - "quantity": <positive integer>
            - "confidence": <float between 0 and 1>
            - "agent_signals": <list of agent signals including agent name, signal (bullish | bearish | neutral), and their confidence>
            - "reasoning": <concise explanation of the decision including how you weighted the signals>

            Trading Rules:
            - Never exceed risk management position limits
            - Only buy if you have available cash
            - Only sell if you have shares to sell
            - Quantity must be ≤ current position for sells
            - Quantity must be ≤ max_position_size from risk management"""
    }

    # 创建用户消息，包含分析师信号
    user_message = {
        "role": "user",
        "content": f"""Based on the team's analysis below, make your trading decision.

            Technical Analysis Trading Signal: {technical_message.content}
            Fundamental Analysis Trading Signal: {fundamentals_message.content}
            Sentiment Analysis Trading Signal: {sentiment_message.content}
            Valuation Analysis Trading Signal: {valuation_message.content}
            Risk Management Trading Signal: {risk_message.content}

            Here is the current portfolio:
            Portfolio:
            Cash: {portfolio['cash']:.2f}
            Current Position: {portfolio['stock']} shares
            Current Price: {current_price:.2f}

            Only include the action, quantity, reasoning, confidence, and agent_signals in your output as JSON.  Do not include any JSON markdown.

            Remember, the action must be either buy, sell, or hold.
            You can only buy if you have available cash.
            You can only sell if you have shares in the portfolio to sell."""
    }

    # 记录LLM请求
    request_data = {
        "system_message": system_message,
        "user_message": user_message
    }

    # 获取LLM分析结果
    result = get_chat_completion([system_message, user_message])

    # 记录LLM交互
    state["metadata"]["current_agent_name"] = "portfolio_management"
    log_llm_interaction(state)(
        lambda: result
    )()

    # 解析LLM响应
    try:
        llm_decision = json.loads(result)
    except Exception:
        # 如果API调用失败或解析错误，使用默认的保守决策
        llm_decision = {
            "action": "hold",
            "quantity": 0,
            "confidence": 0.7,
            "agent_signals": [
                {"agent_name": "technical_analysis", "signal": "neutral", "confidence": 0.0},
                {"agent_name": "fundamental_analysis", "signal": "bullish", "confidence": 1.0},
                {"agent_name": "sentiment_analysis", "signal": "bullish", "confidence": 0.6},
                {"agent_name": "valuation_analysis", "signal": "bearish", "confidence": 0.67},
                {"agent_name": "risk_management", "signal": "hold", "confidence": 1.0}
            ],
            "reasoning": "API error occurred. Following risk management signal to hold."
        }

    # 应用投资组合优化
    optimized_decision = optimize_portfolio_decision(
        llm_decision=llm_decision,
        portfolio=portfolio,
        current_price=current_price,
        risk_message=risk_message.content
    )

    # 创建投资组合管理消息
    message = HumanMessage(
        content=json.dumps(optimized_decision),
        name="portfolio_management",
    )

    # 保存推理信息
    if show_reasoning:
        show_agent_reasoning(optimized_decision, "Portfolio Management Agent")
        state["metadata"]["agent_reasoning"] = optimized_decision

    show_workflow_status("Portfolio Manager", "completed")
    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
        "metadata": state["metadata"],
    }


def optimize_portfolio_decision(llm_decision, portfolio, current_price, risk_message):
    """
    优化投资组合决策，应用现代投资组合理论和风险管理原则
    
    Args:
        llm_decision: LLM生成的初始决策
        portfolio: 当前投资组合状态
        current_price: 当前股票价格
        risk_message: 风险管理代理的信息
        
    Returns:
        dict: 优化后的投资决策
    """
    # 解析基本决策
    action = llm_decision.get("action", "hold")
    original_quantity = llm_decision.get("quantity", 0)
    confidence = llm_decision.get("confidence", 0.5)
    reasoning = llm_decision.get("reasoning", "")
    agent_signals = llm_decision.get("agent_signals", [])
    
    # 解析风险管理信息
    try:
        risk_data = json.loads(risk_message)
        max_position_size = risk_data.get("max_position_size", 0)
        risk_score = risk_data.get("risk_score", 5)
        trading_action = risk_data.get("trading_action", "hold")
    except Exception:
        # 解析错误时使用保守默认值
        max_position_size = portfolio['cash'] / current_price * 0.25
        risk_score = 5
        trading_action = "hold"
    
    # 当前投资组合价值
    total_portfolio_value = portfolio['cash'] + (portfolio['stock'] * current_price)
    
    # 应用仓位管理规则 - Kelly准则的变体
    # 1. 计算风险调整的头寸规模
    risk_factor = 1 - (risk_score / 10)  # 风险分数越高，风险系数越低
    kelly_fraction = confidence * 2 - 1  # 转换置信度为Kelly分数(-1到1)
    kelly_fraction = max(0, kelly_fraction) * risk_factor  # 应用风险调整
    
    # 2. 计算建议仓位
    # Kelly建议的仓位 = 投资组合价值 * Kelly分数 * 风险保守系数
    conservative_factor = 0.5  # 半Kelly，更保守
    suggested_position_value = total_portfolio_value * kelly_fraction * conservative_factor
    
    # 确保不超过风险管理指定的最大头寸
    suggested_position_value = min(suggested_position_value, max_position_size)
    
    # 转换为股数
    suggested_quantity = int(suggested_position_value / current_price) if current_price > 0 else 0
    
    # 3. 应用止损和止盈逻辑
    stop_loss_level = 0.05  # 5%止损
    take_profit_level = 0.15  # 15%止盈
    
    # 计算当前持仓的盈亏情况 (假设)
    position_profit_pct = 0  # 默认假设无盈亏
    
    if position_profit_pct <= -stop_loss_level and portfolio['stock'] > 0:
        # 触发止损
        action = "sell"
        new_quantity = portfolio['stock']  # 全部卖出
        reasoning = f"{reasoning}\n此外，触发止损机制(亏损超过{stop_loss_level*100:.0f}%)，保护投资组合。"
    elif position_profit_pct >= take_profit_level and portfolio['stock'] > 0:
        # 触发止盈 - 部分利润了结
        action = "sell"
        new_quantity = max(int(portfolio['stock'] * 0.5), 1)  # 卖出一半仓位
        reasoning = f"{reasoning}\n部分利润了结策略：盈利达到{take_profit_level*100:.0f}%，减仓锁定收益。"
    else:
        # 正常情况下，遵循信号和风险管理建议
        
        # 首先，确保遵循风险管理建议的交易方向
        if trading_action == "reduce" and action != "sell" and portfolio['stock'] > 0:
            action = "sell"
            new_quantity = max(int(portfolio['stock'] * 0.3), 1)  # 减仓30%
            reasoning = f"{reasoning}\n根据风险管理建议减仓，降低风险暴露。"
        elif trading_action == "hold" and action == "buy":
            action = "hold"
            new_quantity = 0
            reasoning = f"{reasoning}\n风险管理指示持有观望，暂停买入操作。"
        else:
            new_quantity = suggested_quantity
    
    # 4. 应用投资组合约束
    if action == "buy":
        # 买入时的约束
        max_affordable = int(portfolio['cash'] / current_price) if current_price > 0 else 0
        new_quantity = min(new_quantity, max_affordable)
        
        # 设置最小交易量，避免过小订单
        min_transaction = 100  # 最小交易100元
        min_shares = max(1, int(min_transaction / current_price))
        
        if new_quantity < min_shares:
            # 交易量太小，改为持有
            action = "hold"
            new_quantity = 0
            reasoning = f"{reasoning}\n建议买入量过小（低于最小交易量），保持现金持有。"
            
    elif action == "sell":
        # 卖出时的约束
        new_quantity = min(new_quantity, portfolio['stock'])
        
        # 如果剩余持仓太小，全部卖出
        if portfolio['stock'] - new_quantity < 10:
            new_quantity = portfolio['stock']
            reasoning = f"{reasoning}\n剩余持仓量过小，选择全部卖出以优化仓位。"
            
    # 5. 平滑交易 - 避免频繁小额交易
    last_action = "hold"  # 假设上一次操作为持有
    last_price = current_price  # 假设上一次价格等于当前价格
    
    # 如果上次交易和这次建议相同，但价格变化小于阈值，则减少交易量或持有
    price_change_threshold = 0.03  # 3%价格变化阈值
    actual_price_change = abs(current_price - last_price) / last_price if last_price > 0 else 0
    
    if action == last_action and actual_price_change < price_change_threshold:
        if action == "buy":
            new_quantity = int(new_quantity * 0.5)  # 减半买入量
            if new_quantity == 0:
                action = "hold"
        elif action == "sell":
            new_quantity = int(new_quantity * 0.5)  # 减半卖出量
            if new_quantity == 0:
                action = "hold"
    
    # 6. 最终决策整合
    optimized_decision = {
        "action": action,
        "quantity": new_quantity,
        "confidence": confidence,
        "agent_signals": agent_signals,
        "reasoning": reasoning,
        "portfolio_optimization": {
            "risk_score": risk_score,
            "kelly_fraction": kelly_fraction,
            "risk_factor": risk_factor,
            "max_position_size": max_position_size,
            "suggested_position_value": suggested_position_value,
            "total_portfolio_value": total_portfolio_value
        }
    }
    
    return optimized_decision


def format_decision(action: str, quantity: int, confidence: float, agent_signals: list, reasoning: str) -> dict:
    """
    格式化交易决策为标准化输出格式。
    用英文思考但用中文输出分析。
    """

    # 获取各个agent的信号
    fundamental_signal = next(
        (signal for signal in agent_signals if signal["agent_name"] == "fundamental_analysis"), None)
    valuation_signal = next(
        (signal for signal in agent_signals if signal["agent_name"] == "valuation_analysis"), None)
    technical_signal = next(
        (signal for signal in agent_signals if signal["agent_name"] == "technical_analysis"), None)
    sentiment_signal = next(
        (signal for signal in agent_signals if signal["agent_name"] == "sentiment_analysis"), None)
    risk_signal = next(
        (signal for signal in agent_signals if signal["agent_name"] == "risk_management"), None)

    # 转换信号为中文
    def signal_to_chinese(signal):
        if not signal:
            return "无数据"
        if signal["signal"] == "bullish":
            return "看多"
        elif signal["signal"] == "bearish":
            return "看空"
        return "中性"

    # 创建详细分析报告
    detailed_analysis = f"""
====================================
          投资分析报告
====================================

一、策略分析

1. 基本面分析 (权重30%):
   信号: {signal_to_chinese(fundamental_signal)}
   置信度: {fundamental_signal['confidence']*100:.0f}%
   要点: 
   - 盈利能力: {fundamental_signal.get('reasoning', {}).get('profitability_signal', {}).get('details', '无数据')}
   - 增长情况: {fundamental_signal.get('reasoning', {}).get('growth_signal', {}).get('details', '无数据')}
   - 财务健康: {fundamental_signal.get('reasoning', {}).get('financial_health_signal', {}).get('details', '无数据')}
   - 估值水平: {fundamental_signal.get('reasoning', {}).get('price_ratios_signal', {}).get('details', '无数据')}

2. 估值分析 (权重35%):
   信号: {signal_to_chinese(valuation_signal)}
   置信度: {valuation_signal['confidence']*100:.0f}%
   要点:
   - DCF估值: {valuation_signal.get('reasoning', {}).get('dcf_analysis', {}).get('details', '无数据')}
   - 所有者收益法: {valuation_signal.get('reasoning', {}).get('owner_earnings_analysis', {}).get('details', '无数据')}
   - 相对估值: {valuation_signal.get('reasoning', {}).get('relative_valuation', {}).get('details', '无数据')}

3. 技术分析 (权重25%):
   信号: {signal_to_chinese(technical_signal)}
   置信度: {technical_signal['confidence']*100:.0f}%
   要点:
   - 趋势跟踪: {technical_signal.get('strategy_signals', {}).get('trend_following', {}).get('metrics', {}).get('adx', '无数据')}
   - 均值回归: RSI(14)={technical_signal.get('strategy_signals', {}).get('mean_reversion', {}).get('metrics', {}).get('rsi_14', '无数据')}
   - 动量指标: 
     * 1月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_1m', '无数据'):.2%}
     * 3月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_3m', '无数据'):.2%}
     * 6月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_6m', '无数据'):.2%}
   - 波动性: {technical_signal.get('strategy_signals', {}).get('volatility', {}).get('metrics', {}).get('historical_volatility', '无数据'):.2%}

4. 情绪分析 (权重10%):
   信号: {signal_to_chinese(sentiment_signal)}
   置信度: {sentiment_signal['confidence']*100:.0f}%
   分析: {sentiment_signal.get('reasoning', '无详细分析')}

二、风险评估
风险评分: {risk_signal.get('risk_score', '无数据')}/10
主要指标:
- 波动率: {risk_signal.get('risk_metrics', {}).get('volatility', '无数据')*100:.1f}%
- 最大回撤: {risk_signal.get('risk_metrics', {}).get('max_drawdown', '无数据')*100:.1f}%
- VaR(95%): {risk_signal.get('risk_metrics', {}).get('value_at_risk_95', '无数据')*100:.1f}%
- CVaR(95%): {risk_signal.get('risk_metrics', {}).get('conditional_var_95', '无数据')*100:.1f}%
- 市场风险: {risk_signal.get('risk_metrics', {}).get('market_risk_score', '无数据')}/10

三、投资建议
操作建议: {'买入' if action == 'buy' else '卖出' if action == 'sell' else '持有'}
交易数量: {quantity}股
决策置信度: {confidence*100:.0f}%

四、决策依据
{reasoning}

===================================="""

    return {
        "action": action,
        "quantity": quantity,
        "confidence": confidence,
        "agent_signals": agent_signals,
        "分析报告": detailed_analysis
    }