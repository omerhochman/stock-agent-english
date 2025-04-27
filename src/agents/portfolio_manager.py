from langchain_core.messages import HumanMessage
import json
import numpy as np
import pandas as pd

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.openrouter_config import get_chat_completion
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.tools.api import prices_to_df
from src.tools.factor_data_api import get_multiple_index_data, get_risk_free_rate
from src.calc.portfolio_optimization import portfolio_volatility
from src.utils.logging_config import setup_logger

# 设置日志记录器
logger = setup_logger('portfolio_management_agent')

@agent_endpoint("portfolio_management", "负责投资组合管理和最终交易决策")
def portfolio_management_agent(state: AgentState):
    """负责投资组合管理和交易决策"""
    show_workflow_status("Portfolio Manager")
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    ticker = data.get("ticker", "")
    prices = data.get("prices", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    
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
    except Exception as e:
        logger.error(f"解析LLM响应失败: {e}")
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

    # 使用更先进的投资组合优化方法
    optimized_decision = optimize_portfolio_decision_advanced(
        llm_decision=llm_decision,
        portfolio=portfolio,
        current_price=current_price,
        risk_message=risk_message.content,
        ticker=ticker,
        prices_df=prices_to_df(prices),
        start_date=start_date,
        end_date=end_date
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


def optimize_portfolio_decision_advanced(llm_decision, portfolio, current_price, risk_message, ticker, prices_df, start_date, end_date):
    """
    使用现代投资组合理论和先进的风险管理原则优化投资组合决策
    
    Args:
        llm_decision: LLM生成的初始决策
        portfolio: 当前投资组合状态
        current_price: 当前股票价格
        risk_message: 风险管理代理的信息
        ticker: 股票代码
        prices_df: 历史价格数据DataFrame
        start_date: 开始日期
        end_date: 结束日期
        
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
        # 提取GARCH模型结果(如果有)
        volatility_model = risk_data.get("volatility_model", {})
    except Exception as e:
        logger.error(f"解析风险管理信息失败: {e}")
        # 解析错误时使用保守默认值
        max_position_size = portfolio['cash'] / current_price * 0.25 if current_price > 0 else 0
        risk_score = 5
        trading_action = "hold"
        volatility_model = {}
    
    # 当前投资组合价值
    total_portfolio_value = portfolio['cash'] + (portfolio['stock'] * current_price)
    
    # 1. 获取实际市场数据
    # 获取股票和市场指数数据
    market_data = {}
    market_volatility = 0.15  # 默认值，仅在无法获取实际数据时使用
    
    try:
        # 获取主要市场指数数据(沪深300、深证成指、创业板指)
        indices = ["000300", "399001", "399006"]  # 沪深300、深证成指、创业板指
        market_indices = get_multiple_index_data(indices, start_date, end_date)
        
        # 创建市场指数的收益率序列
        market_returns_dict = {}
        for idx_name, idx_data in market_indices.items():
            if isinstance(idx_data, pd.DataFrame) and 'close' in idx_data.columns:
                idx_df = idx_data.copy()
                idx_df['return'] = idx_df['close'].pct_change()
                market_returns_dict[idx_name] = idx_df['return'].dropna()
        
        # 使用沪深300作为主要市场基准
        if '000300' in market_returns_dict:
            market_returns = market_returns_dict['000300']
            # 计算实际市场波动率
            market_volatility = market_returns.std() * np.sqrt(252)
            market_data['market_returns'] = market_returns
            market_data['market_volatility'] = market_volatility
            logger.info(f"成功获取市场数据 - 市场波动率: {market_volatility:.2%}")
        else:
            logger.warning("无法获取沪深300指数数据，使用默认市场波动率")
    except Exception as e:
        logger.error(f"获取市场数据失败: {e}")
    
    # 2. 获取实际无风险利率数据
    try:
        # 使用factor_data_api获取无风险利率数据
        risk_free_data = get_risk_free_rate(start_date=start_date, end_date=end_date)
        
        # 计算平均无风险利率
        if not risk_free_data.empty:
            risk_free_rate = risk_free_data.mean()
            logger.info(f"成功获取无风险利率数据: {risk_free_rate:.2%}")
        else:
            # 如果数据为空，使用默认值
            risk_free_rate = 0.03
            logger.warning(f"无风险利率数据为空，使用默认值: {risk_free_rate:.2%}")
    except Exception as e:
        # 如果获取失败，使用合理的默认值
        risk_free_rate = 0.03
        logger.error(f"获取无风险利率失败: {e}，使用默认值: {risk_free_rate:.2%}")
    
    # 3. 计算股票收益率和波动率
    stock_returns = None
    stock_volatility = None
    if not prices_df.empty:
        try:
            stock_returns = prices_df['close'].pct_change().dropna()
            # 使用 portfolio_volatility 函数计算实际波动率
            stock_volatility = portfolio_volatility(np.ones(1), np.array([[stock_returns.var()]])) * np.sqrt(252)
            market_data['stock_returns'] = stock_returns
            market_data['stock_volatility'] = stock_volatility
            logger.info(f"成功计算股票波动率: {stock_volatility:.2%}")
        except Exception as e:
            logger.error(f"计算股票波动率失败: {e}")
            stock_volatility = 0.3  # 默认值
    
    # 4. 计算相对波动率(Beta)和Beta调整的投资组合优化
    beta = 1.0  # 默认值
    if stock_returns is not None and 'market_returns' in market_data:
        market_returns = market_data['market_returns']
        common_dates = stock_returns.index.intersection(market_returns.index)
        
        if len(common_dates) > 30:  # 确保有足够的数据点
            # 计算协方差
            covariance = stock_returns[common_dates].cov(market_returns[common_dates])
            # 计算市场方差
            market_variance = market_returns[common_dates].var()
            # 计算Beta
            if market_variance > 0:
                beta = covariance / market_variance
                market_data['beta'] = beta
                logger.info(f"成功计算股票Beta: {beta:.2f}")
    
    # 5. 应用计算模块中的优化函数增强决策
    portfolio_optimization_results = {}
    try:
        if stock_returns is not None and len(stock_returns) > 30:
            # 使用EWMA方法估计协方差矩阵(虽然是单个资产，但可为未来扩展打基础)
            stock_return_mean = stock_returns.mean()
            stock_return_std = stock_returns.std()
            
            # 计算风险调整期望收益率 (使用技术和基本面信号)
            expected_return_multiplier = 1.0
            tech_signal = next((s for s in agent_signals if s.get("agent_name", "").lower() == "technical_analysis"), None)
            fund_signal = next((s for s in agent_signals if s.get("agent_name", "").lower() == "fundamental_analysis"), None)
            
            if tech_signal and fund_signal:
                tech_value = {"bullish": 1, "neutral": 0, "bearish": -1}.get(tech_signal.get("signal", "neutral").lower(), 0)
                fund_value = {"bullish": 1, "neutral": 0, "bearish": -1}.get(fund_signal.get("signal", "neutral").lower(), 0)
                signal_value = (tech_value * 0.4 + fund_value * 0.6)  # 基本面权重更高
                expected_return_multiplier += signal_value * 0.5  # 调整基准预期收益率
            
            # 使用GARCH预测调整波动率预期
            volatility_adjustment = 1.0
            if volatility_model and "forecast" in volatility_model:
                try:
                    garch_forecast = volatility_model["forecast"]
                    if isinstance(garch_forecast, list) and len(garch_forecast) > 0:
                        avg_forecast = sum(garch_forecast) / len(garch_forecast)
                        current_vol = stock_return_std
                        # 如果预测波动率上升，降低仓位；如果预测下降，增加仓位
                        volatility_adjustment = current_vol / avg_forecast if avg_forecast > 0 else 1.0
                        volatility_adjustment = min(max(volatility_adjustment, 0.7), 1.3)  # 限制调整范围
                except Exception as e:
                    logger.error(f"波动率调整计算错误: {e}")
            
            # 调整后的预期收益率和波动率
            adjusted_expected_return = stock_return_mean * expected_return_multiplier * 252  # 年化
            adjusted_volatility = stock_return_std * np.sqrt(252) / volatility_adjustment  # 年化并调整
            
            # 使用实际的无风险利率计算Sharpe比率
            sharpe_ratio = (adjusted_expected_return - risk_free_rate) / adjusted_volatility if adjusted_volatility > 0 else 0
            
            # 使用实际计算的Beta调整优化
            beta_adjusted_return = risk_free_rate + beta * (adjusted_expected_return - risk_free_rate)
            
            # 创建模拟投资组合进行优化
            # 注意：由于我们只有一个资产，这是简化的单资产优化
            weights = np.array([1.0])  # 100%投资于这一个资产
            asset_returns = np.array([beta_adjusted_return])
            asset_cov = np.array([[adjusted_volatility**2]])
            
            # 保存优化结果
            portfolio_optimization_results = {
                "expected_annual_return": float(adjusted_expected_return),
                "expected_annual_volatility": float(adjusted_volatility),
                "beta_adjusted_return": float(beta_adjusted_return),
                "sharpe_ratio": float(sharpe_ratio),
                "volatility_adjustment": float(volatility_adjustment),
                "return_multiplier": float(expected_return_multiplier),
                "beta": float(beta),
                "market_volatility": float(market_volatility),
                "risk_free_rate": float(risk_free_rate)
            }
    except Exception as e:
        logger.error(f"投资组合优化计算失败: {e}")
        portfolio_optimization_results = {"error": str(e)}
        
    # 6. 应用仓位管理规则 - 改进的凯利准则
    # 使用投资组合优化结果调整凯利分数
    kelly_fraction = confidence * 2 - 1  # 转换置信度为Kelly分数(-1到1)
    kelly_fraction = max(0, kelly_fraction)  # 确保非负
    
    # 调整凯利分数基于投资组合优化结果
    if "sharpe_ratio" in portfolio_optimization_results:
        sharpe_ratio = portfolio_optimization_results["sharpe_ratio"]
        # 使用Sharpe比率调整凯利分数
        if sharpe_ratio > 1.0:  # 很好的风险调整回报
            kelly_fraction = min(kelly_fraction * 1.2, 1.0)  # 提高仓位但不超过1
        elif sharpe_ratio < 0:  # 负的风险调整回报
            kelly_fraction = kelly_fraction * 0.5  # 减半仓位
    
    # 使用Beta和波动率进一步调整仓位
    if "beta" in portfolio_optimization_results:
        beta_value = portfolio_optimization_results["beta"]
        # Beta高于1表示股票比市场更波动，需要减少仓位
        if beta_value > 1.2:
            kelly_fraction = kelly_fraction * (1 / beta_value)  # 按比例减少仓位
        
    # 风险调整
    risk_factor = 1 - (risk_score / 10)  # 风险分数越高，风险系数越低
    
    # 7. 计算建议仓位
    # Kelly建议的仓位 = 投资组合价值 * Kelly分数 * 风险保守系数
    conservative_factor = 0.5  # 半Kelly，更保守
    suggested_position_value = total_portfolio_value * kelly_fraction * conservative_factor * risk_factor
    
    # 确保不超过风险管理指定的最大头寸
    suggested_position_value = min(suggested_position_value, max_position_size)
    
    # 转换为股数
    suggested_quantity = int(suggested_position_value / current_price) if current_price > 0 else 0
    
    # 8. 应用止损和止盈逻辑
    stop_loss_level = 0.05  # 5%止损
    take_profit_level = 0.15  # 15%止盈
    
    # 尝试从价格数据计算当前持仓的盈亏情况
    position_profit_pct = 0  # 默认假设无盈亏
    if not prices_df.empty and portfolio['stock'] > 0:
        # 简单假设：使用最近20天平均价格作为持仓成本
        recent_period = min(20, len(prices_df))
        avg_price = prices_df['close'].iloc[-recent_period:].mean() if recent_period > 0 else current_price
        if avg_price > 0:
            position_profit_pct = (current_price - avg_price) / avg_price
    
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
    
    # 9. 应用投资组合约束
    if action == "buy":
        # 买入时的约束
        max_affordable = int(portfolio['cash'] / current_price) if current_price > 0 else 0
        new_quantity = min(new_quantity, max_affordable)
        
        # 设置最小交易量，避免过小订单
        min_transaction = 100  # 最小交易100元
        min_shares = max(1, int(min_transaction / current_price)) if current_price > 0 else 1
        
        if new_quantity < min_shares:
            # 交易量太小，改为持有
            action = "hold"
            new_quantity = 0
            reasoning = f"{reasoning}\n建议买入量过小（低于最小交易量），保持现金持有。"
            
    elif action == "sell":
        # 卖出时的约束
        new_quantity = min(new_quantity, portfolio['stock'])
        
        # 如果剩余持仓太小，全部卖出
        if portfolio['stock'] - new_quantity < 10 and portfolio['stock'] > 0:
            new_quantity = portfolio['stock']
            reasoning = f"{reasoning}\n剩余持仓量过小，选择全部卖出以优化仓位。"
            
    # 10. 平滑交易 - 避免频繁小额交易
    last_action = "hold"  # 假设上一次操作为持有
    last_price = current_price  # 假设上一次价格等于当前价格
    
    # 如果有历史价格数据，尝试从中获取上一交易日价格
    if not prices_df.empty and len(prices_df) > 1:
        last_price = prices_df['close'].iloc[-2]
    
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
    
    # 11. 最终决策整合
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
            "total_portfolio_value": total_portfolio_value,
            "position_profit_pct": position_profit_pct,
            "analytics": portfolio_optimization_results,
            "market_data": {k: str(v) if isinstance(v, pd.Series) else v for k, v in market_data.items()}
        }
    }
    
    return optimized_decision