import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.api import prices_to_df
from src.utils.api_utils import agent_endpoint
from src.utils.logging_config import setup_logger
from src.calc.tail_risk_measures import calculate_historical_var, calculate_conditional_var
from src.calc.volatility_models import fit_garch, forecast_garch_volatility

import json
import ast

##### Risk Management Agent #####

logger = setup_logger('risk_management_agent')

@agent_endpoint("risk_management", "风险管理专家，评估投资风险并给出风险调整后的交易建议")
def risk_management_agent(state: AgentState):
    """负责风险管理和风险调整后的交易决策"""
    show_workflow_status("Risk Manager")
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]

    # 获取资产列表
    tickers = data.get("tickers", [])
    if isinstance(tickers, str):
        tickers = [ticker.strip() for ticker in tickers.split(',')]
    
    # 如果没有提供tickers但提供了ticker，则使用单一ticker
    if not tickers and data.get("ticker"):
        tickers = [data["ticker"]]
    
    # 主要资产的价格数据
    prices_df = prices_to_df(data["prices"])
    
    # 多资产投资组合风险分析
    portfolio_risk_analysis = {}
    
    # 如果有多资产数据，计算组合风险
    all_stock_data = data.get("all_stock_data", {})
    if len(tickers) > 1 and all_stock_data:
        # 提取所有股票的收益率
        returns_dict = {}
        for ticker in tickers:
            if ticker in all_stock_data:
                stock_prices = prices_to_df(all_stock_data[ticker]["prices"])
                if not stock_prices.empty:
                    returns_dict[ticker] = stock_prices['close'].pct_change().dropna()
        
        # 如果有足够的数据，计算投资组合风险
        if len(returns_dict) > 1:
            returns_df = pd.DataFrame(returns_dict)
            
            # 计算相关系数矩阵
            correlation_matrix = returns_df.corr()
            
            # 计算等权重投资组合收益率
            portfolio_return = returns_df.mean(axis=1)  # 等权重投资组合
            
            # 使用投资组合收益率计算风险指标
            daily_vol = portfolio_return.ewm(span=20, adjust=False).std().iloc[-1]
            volatility = daily_vol * (252 ** 0.5)
            
            # 计算VaR和CVaR
            var_95 = calculate_historical_var(portfolio_return, confidence_level=0.95)
            cvar_95 = calculate_conditional_var(portfolio_return, confidence_level=0.95)
            
            # 计算最大回撤
            cum_returns = (1 + portfolio_return).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max - 1)
            max_drawdown = drawdown.min()
            
            # 计算偏度和峰度
            skewness = portfolio_return.skew()
            kurtosis = portfolio_return.kurt()
            
            # 保存多资产组合风险分析结果
            portfolio_risk_analysis = {
                "tickers": tickers,
                "volatility": float(volatility),
                "var_95": float(var_95),
                "cvar_95": float(cvar_95),
                "max_drawdown": float(max_drawdown),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "correlation_matrix": correlation_matrix.to_dict()
            }
            
            # 提供多资产风险分散化建议
            diversification_tips = []
            
            # 基于相关性的建议
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > 0.7:  # 高相关性阈值
                        high_corr_pairs.append(f"{correlation_matrix.columns[i]}-{correlation_matrix.columns[j]}")
            
            if high_corr_pairs:
                diversification_tips.append(f"发现高相关性资产对: {', '.join(high_corr_pairs)}，考虑降低其中一个资产的配置以降低集中风险")
            
            avg_corr = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            if avg_corr > 0.6:
                diversification_tips.append(f"投资组合平均相关性较高({avg_corr:.2f})，可能需要增加其他资产类别以提高分散化效果")
            
            portfolio_risk_analysis["diversification_tips"] = diversification_tips

    # 获取辩论室评估
    debate_message = next(
        msg for msg in state["messages"] if msg.name == "debate_room_agent")

    try:
        debate_results = json.loads(debate_message.content)
    except Exception as e:
        debate_results = ast.literal_eval(debate_message.content)

    # 1. 计算基础风险指标
    returns = prices_df['close'].pct_change().dropna()
    
    # 使用EWMA优化波动率计算
    daily_vol = returns.ewm(span=20, adjust=False).std().iloc[-1]  # EWMA波动率
    volatility = daily_vol * (252 ** 0.5)  # 年化
    
    # 计算波动率的历史分布
    rolling_std = returns.rolling(window=120).std() * (252 ** 0.5)
    volatility_mean = rolling_std.mean()
    volatility_std = rolling_std.std()
    volatility_percentile = (volatility - volatility_mean) / volatility_std if volatility_std != 0 else 0

    # 2. 高级风险指标计算 - 使用 calc 模块中的函数
    # 2.1 历史VaR (95%置信度) - 使用 calc/tail_risk_measures 模块
    var_95 = calculate_historical_var(returns, confidence_level=0.95)
    
    # 2.2 条件VaR/Expected Shortfall
    cvar_95 = calculate_conditional_var(returns, confidence_level=0.95)
    
    # 2.3 最大回撤计算
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max - 1)
    max_drawdown = drawdown.min()
    
    # 2.4 偏度和峰度 (检测非正态分布)
    skewness = returns.skew()
    kurtosis = returns.kurt()
    
    # 2.5 Sortino比率 (仅考虑下行风险)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    mean_return = returns.mean() * 252
    sortino_ratio = mean_return / downside_deviation if downside_deviation != 0 else 0

    # 3. 市场风险评估 (评分系统)
    market_risk_score = 0

    # 波动率评分 (基于百分位)
    if volatility_percentile > 1.5:     # 高于1.5个标准差
        market_risk_score += 2
    elif volatility_percentile > 1.0:   # 高于1个标准差
        market_risk_score += 1

    # VaR评分 (基于历史分布)
    # 注意：var_95通常为正值（我们返回绝对值）
    if var_95 > 0.03:
        market_risk_score += 2
    elif var_95 > 0.02:
        market_risk_score += 1

    # 最大回撤评分
    if abs(max_drawdown) > 0.20:  # 严重回撤
        market_risk_score += 2
    elif abs(max_drawdown) > 0.10:
        market_risk_score += 1
        
    # 分布异常性评分 (显著非正态)
    if abs(skewness) > 1.0 or kurtosis > 5.0:
        market_risk_score += 1

    # 4. 头寸规模计算 - 使用凯利准则的优化版本
    # 考虑总组合价值，非仅现金
    current_stock_value = portfolio['stock'] * prices_df['close'].iloc[-1]
    total_portfolio_value = portfolio['cash'] + current_stock_value

    # 基于凯利准则的头寸规模
    # 从辩论结果调整胜率
    bull_confidence = debate_results.get("bull_confidence", 0.5)
    bear_confidence = debate_results.get("bear_confidence", 0.5)
    if bull_confidence > bear_confidence:
        win_rate = bull_confidence
    else:
        win_rate = 1 - bear_confidence
        
    # 结合波动率模型优化收益预期
    try:
        # 使用GARCH模型预测未来波动率
        if len(returns) >= 100:  # 确保有足够的数据
            garch_params, _ = fit_garch(returns.values)
            volatility_forecast = forecast_garch_volatility(returns.values, garch_params, 
                                                         forecast_horizon=10)
            
            # 根据波动率预测调整胜率
            # 如果预测波动率上升，降低胜率；如果预测波动率下降，提高胜率
            avg_forecast = np.mean(volatility_forecast)
            volatility_trend = avg_forecast / volatility - 1  # 正值表示波动率上升趋势
            
            # 调整胜率
            if volatility_trend > 0.1:  # 波动率预期显著上升
                win_rate = max(0.3, win_rate - 0.1)  # 最低设为0.3
            elif volatility_trend < -0.1:  # 波动率预期显著下降
                win_rate = min(0.8, win_rate + 0.05)  # 最高设为0.8
    except Exception as e:
        logger.warning(f"GARCH模型计算失败: {e}")
        
    # 计算平均胜利收益和亏损比率 - 使用历史数据优化
    # 分析历史正负收益比例
    avg_gain = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.005
    avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.005
    win_loss_ratio = avg_gain / avg_loss if avg_loss != 0 else 1.5
    
    # 应用改进的凯利公式
    kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
    kelly_fraction = max(0.05, kelly_fraction)
    
    # 保守系数和风险调整
    conservative_factor = 0.5  # 只用一半凯利建议
    risk_adjustment = max(0, 1 - (market_risk_score / 10))
    
    # 最终头寸大小
    max_position_size = total_portfolio_value * kelly_fraction * conservative_factor * risk_adjustment

    max_position_size = max(max_position_size, total_portfolio_value * 0.02)  # 设置最小仓位
    
    # 安全检查 - 头寸不能太高
    max_position_size = min(max_position_size, total_portfolio_value * 0.4)

    # 5. 压力测试 (更全面的情景)
    stress_test_scenarios = {
        "market_crash": -0.20,            # 市场崩溃
        "moderate_decline": -0.10,        # 中度下跌
        "slight_decline": -0.05,          # 轻微下跌
        "volatility_spike": -0.15,        # 波动率飙升
        "liquidity_crisis": -0.25,        # 流动性危机
        "sector_rotation": -0.12,         # 行业轮换
    }

    stress_test_results = {}
    current_position_value = current_stock_value
    if current_position_value > 0:
        for scenario, decline in stress_test_scenarios.items():
            potential_loss = current_position_value * decline
            portfolio_impact = potential_loss / total_portfolio_value if total_portfolio_value != 0 else float('nan')
            
            # 对每种情景进行VaR分析
            scenario_var = potential_loss * (var_95 / 0.05)  # 调整VaR以反映情景严重程度
            stress_test_results[scenario] = {
                "potential_loss": float(potential_loss),
                "portfolio_impact": float(portfolio_impact),
                "scenario_var": float(scenario_var)
            }
    else:
        # 空持仓时使用空字典或None
        stress_test_results = {"no_position": True}

    # 6. 风险调整信号分析
    # 考虑辩论室置信度
    bull_confidence = debate_results.get("bull_confidence", 0)
    bear_confidence = debate_results.get("bear_confidence", 0)
    debate_confidence = debate_results.get("confidence", 0)

    # 评估辩论结果的确定性
    confidence_diff = abs(bull_confidence - bear_confidence)
    if confidence_diff < 0.1:  # 辩论接近
        market_risk_score += 1
    if debate_confidence < 0.3:  # 总体低置信度
        market_risk_score += 1

    # 上限风险分数为10
    risk_score = min(round(market_risk_score), 10)

    # 7. 生成交易行动
    debate_signal = debate_results.get("signal", "neutral")

    # 获取宏观环境信息
    macro_message = next(
        (msg for msg in state["messages"] if msg.name == "macro_analyst_agent"), None)
    
    macro_environment_positive = False
    if macro_message:
        try:
            macro_data = json.loads(macro_message.content)
            # 检查宏观分析是否积极
            macro_impact = macro_data.get("impact_on_stock", "neutral")
            macro_environment = macro_data.get("macro_environment", "neutral")
            
            # 如果宏观环境和对股票的影响都是积极的，降低风险评分
            if macro_impact == "positive" and macro_environment == "positive":
                macro_environment_positive = True
                # 降低风险评分，但确保不小于0
                risk_score = max(risk_score - 2, 0)
                logger.info(f"基于积极的宏观环境降低风险评分至 {risk_score}")
        except Exception as e:
            logger.warning(f"解析宏观分析数据失败: {e}")
            
    # 增加对市场近期表现的评估
    recent_period = min(20, len(returns))
    if recent_period > 5:  # 确保有足够的数据
        recent_returns = returns[-recent_period:]
        recent_positive_days = sum(1 for r in recent_returns if r > 0)
        recent_positive_ratio = recent_positive_days / len(recent_returns)
        
        # 如果近期大部分交易日收益为正，降低风险评分
        if recent_positive_ratio > 0.65:  # 超过65%的天数为正收益
            risk_score = max(risk_score - 1, 0)  # 再降低1分
            logger.info(f"基于近期积极市场表现降低风险评分至 {risk_score}")

    logger.info(f"风险分数risk_score为：{risk_score}")
    # 基于风险分数和辩论信号的决策规则
    if risk_score >= 10:  
        trading_action = "hold"  # 非常高风险，持有观望
    elif risk_score >= 8:  
        if debate_signal == "bearish":
            trading_action = "sell"  # 高风险 + 看空 = 卖出
        else:
            trading_action = "reduce"  # 高风险但非看空 = 减仓
    else:
        if debate_signal == "bullish" and debate_confidence > 0.4: 
            trading_action = "buy"  # 低风险 + 看多 = 买入
        elif debate_signal == "bearish" and debate_confidence > 0.6:
            trading_action = "sell"  # 低风险 + 强看空 = 卖出
        else:
            trading_action = "hold"  # 其他情况 = 持有

    # 8. GARCH模型拟合和预测
    garch_results = {}
    try:
        # 获取足够长的数据进行GARCH拟合
        if len(returns) >= 100:  # 确保有足够的数据
            garch_params, log_likelihood = fit_garch(returns.values)
            volatility_forecast = forecast_garch_volatility(returns.values, garch_params, 
                                                         forecast_horizon=10)
            
            # 将结果添加到分析中
            market_risk_score += 1 if garch_params['persistence'] > 0.95 else 0  # 高持续性表示更高风险
            
            # 保存GARCH结果
            garch_results = {
                'model_type': 'GARCH(1,1)',
                'parameters': {
                    'omega': float(garch_params['omega']),
                    'alpha': float(garch_params['alpha']),
                    'beta': float(garch_params['beta']),
                    'persistence': float(garch_params['persistence'])
                },
                'log_likelihood': float(log_likelihood),
                'forecast': [float(v) for v in volatility_forecast],
                'forecast_annualized': [float(v * np.sqrt(252)) for v in volatility_forecast]
            }
    except Exception as e:
        logger.warning(f"GARCH模型拟合失败: {e}")
        garch_results = {"error": str(e)}

    # 9. 构建输出消息
    message_content = {
        "max_position_size": float(max_position_size),
        "risk_score": risk_score,
        "trading_action": trading_action,
        "risk_metrics": {
            "volatility": float(volatility),
            "value_at_risk_95": float(var_95),
            "conditional_var_95": float(cvar_95),  
            "max_drawdown": float(max_drawdown),
            "skewness": float(skewness),           
            "kurtosis": float(kurtosis),           
            "sortino_ratio": float(sortino_ratio), 
            "market_risk_score": market_risk_score,
            "stress_test_results": stress_test_results,
            "macro_environment_assessment": { 
                "global_risks": market_risk_score > 5,
                "liquidity_concerns": market_risk_score > 7,
                "volatility_regime": "high" if volatility > 0.3 else "medium" if volatility > 0.2 else "low"
            }
        },
        "position_sizing": {
            "kelly_fraction": float(kelly_fraction),
            "win_rate": float(win_rate),
            "win_loss_ratio": float(win_loss_ratio),
            "risk_adjustment": float(risk_adjustment),
            "total_portfolio_value": float(total_portfolio_value)
        },
        "debate_analysis": {
            "bull_confidence": bull_confidence,
            "bear_confidence": bear_confidence,
            "debate_confidence": debate_confidence,
            "debate_signal": debate_signal
        },
        "volatility_model": garch_results,
        "reasoning": f"风险评分 {risk_score}/10: 市场风险={market_risk_score}, "
                     f"波动率={volatility:.2%}, VaR={var_95:.2%}, CVaR={cvar_95:.2%}, "
                     f"最大回撤={max_drawdown:.2%}, 偏度={skewness:.2f}, "
                     f"辩论信号={debate_signal}, Kelly建议占比={kelly_fraction:.2f}"
    }
    
    # 如果有多资产分析结果，添加到输出中
    if portfolio_risk_analysis:
        message_content["portfolio_risk_analysis"] = portfolio_risk_analysis
        
        # 添加多资产风险提示到reasoning
        if "diversification_tips" in portfolio_risk_analysis and portfolio_risk_analysis["diversification_tips"]:
            message_content["reasoning"] += "\n多资产风险分析: " + " ".join(portfolio_risk_analysis["diversification_tips"])

    # 创建风险管理消息
    message = HumanMessage(
        content=json.dumps(message_content),
        name="risk_management_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "Risk Management Agent")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("Risk Manager", "completed")
    return {
        "messages": state["messages"] + [message],
        "data": {
            **data,
            "risk_analysis": message_content
        },
        "metadata": state["metadata"],
    }