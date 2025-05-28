import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.agents.regime_detector import AdvancedRegimeDetector
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
    """
    基于2024-2025研究的区制感知风险管理系统
    集成FINSABER、FLAG-Trader、RLMF等框架的风险控制技术
    """
    show_workflow_status("Risk Manager")
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]

    # 初始化区制检测器
    regime_detector = AdvancedRegimeDetector()
    
    # 获取资产列表
    tickers = data.get("tickers", [])
    if isinstance(tickers, str):
        tickers = [ticker.strip() for ticker in tickers.split(',')]
    
    # 如果没有提供tickers但提供了ticker，则使用单一ticker
    if not tickers and data.get("ticker"):
        tickers = [data["ticker"]]
    
    # 主要资产的价格数据
    prices_df = prices_to_df(data["prices"])
    
    # 进行区制分析
    regime_features = regime_detector.extract_regime_features(prices_df)
    regime_model_results = regime_detector.fit_regime_model(regime_features)
    current_regime = regime_detector.predict_current_regime(regime_features)
    
    logger.info(f"风险管理器检测到市场区制: {current_regime.get('regime_name', 'unknown')} (置信度: {current_regime.get('confidence', 0):.2f})")
    
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
            
            # 基于区制的分散化建议
            diversification_tips = _generate_regime_aware_diversification_tips(
                correlation_matrix, current_regime, high_corr_pairs=[]
            )
            
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

    # 3. 区制感知风险评估 (基于2024-2025研究)
    regime_risk_score = _calculate_regime_risk_score(
        current_regime, volatility_percentile, var_95, max_drawdown, skewness, kurtosis
    )

    # 4. 动态头寸规模计算 - 基于FLAG-Trader 2025框架
    # 考虑总组合价值，非仅现金
    current_stock_value = portfolio['stock'] * prices_df['close'].iloc[-1]
    total_portfolio_value = portfolio['cash'] + current_stock_value

    # 区制感知的凯利准则优化
    position_sizing_result = _calculate_regime_aware_position_sizing(
        debate_results, current_regime, returns, total_portfolio_value
    )

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
        regime_risk_score += 1
    if debate_confidence < 0.3:  # 总体低置信度
        regime_risk_score += 1

    # 上限风险分数为10
    risk_score = min(round(regime_risk_score), 10)

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
    logger.info(f"辩论信号debate_signal为：{debate_signal}")
    logger.info(f"辩论置信度debate_confidence为：{debate_confidence}")
    # 基于风险分数和辩论信号的决策规则
    if risk_score >= 9:  
        trading_action = "hold"  
        # 理由：极高风险(9-10/10)，市场极其危险，
        # 无论辩论结果如何，都应立即停止所有交易操作，保持观望，
        # 避免在高风险环境中遭受重大损失
        
    elif risk_score >= 7:  
        if debate_signal == "bearish":
            trading_action = "sell"  
            # 理由：高风险（7-8/10）且看空信号，双重警示，
            # 应及时卖出以避免潜在损失，保护投资组合
        else:
            trading_action = "reduce"  
            # 理由：高风险环境但非看空，市场可能有不确定性，
            # 通过减仓来降低风险暴露，但不是全部清仓，
            # 保留部分仓位以防市场反转
            
    elif risk_score >= 5:  # 新增中高风险区间
        if debate_signal == "bearish" and debate_confidence >= 0.3:
            trading_action = "sell"
            # 理由：中高风险（5-6/10）且看空信号，即使置信度不是很高（30%），
            # 在这个风险水平下也应该及时卖出，避免承担过多风险
        elif debate_signal == "bullish" and debate_confidence >= 0.4:
            trading_action = "buy"
            # 理由：中高风险但看多信号强烈（置信度≥40%），
            # 在风险可控的情况下可以谨慎买入
        else:
            trading_action = "reduce"
            # 理由：中高风险环境下信号不明确，
            # 通过减仓来降低整体风险暴露
            
    elif risk_score >= 3:  # 新增中等风险区间
        if debate_signal == "bearish" and debate_confidence >= 0.25:
            trading_action = "sell"
            # 理由：中等风险（3-4/10）下看空信号，
            # 即使置信度较低（25%），也应该考虑卖出，
            # 及早规避可能的下跌风险
        elif debate_signal == "bearish" and debate_confidence < 0.25:
            trading_action = "reduce"
            # 理由：中等风险下看空但置信度很低（<25%），
            # 不确定性很大，减仓但不全部清仓，保留后续操作空间
        elif debate_signal == "bullish" and debate_confidence >= 0.35:
            trading_action = "buy"
            # 理由：中等风险下看多信号较强（置信度≥35%），
            # 可以考虑买入，但要求比低风险时更高的置信度
        else:
            trading_action = "hold"
            # 理由：中等风险下信号不明确，维持现状最安全
            
    else:  # risk_score < 3：低风险区间
        if debate_signal == "bullish" and debate_confidence >= 0.3:
            trading_action = "buy"
            # 理由：低风险环境且看多信号强烈（置信度≥30%），
            # 适合积极布局，抓住市场上涨机会
            
        elif debate_signal == "bearish" and debate_confidence >= 0.35:
            trading_action = "sell"
            # 理由：即使在低风险环境，如果有一定把握的看空信号（置信度≥35%），
            # 也应该果断卖出，避免错过逃顶机会
            
        elif debate_signal == "neutral" and debate_confidence >= 0.3:
            # 信号中性但置信度较高，需要结合市场风险评分进行细化决策
            
            if regime_risk_score <= 2:  # 市场风险极低
                trading_action = "buy"
                # 理由：中性信号，但市场风险极低（≤2/10），
                # 适合积极建仓，利用低风险环境获取收益
                
            elif regime_risk_score >= 6:  # 市场风险偏高
                trading_action = "sell"
                # 理由：虽然信号中性，但市场风险偏高（≥6/10），
                # 应该主动减少仓位，防范潜在风险
                
            else:  # 市场风险适中（3-5）
                trading_action = "hold"
                # 理由：中性信号，中等市场风险，维持现状最安全，
                # 等待更明确的信号再做决定
                
        elif debate_signal == "bullish" and debate_confidence < 0.3:
            trading_action = "hold"
            # 理由：看多信号但置信度太低（<30%），
            # 分析师团队意见不一致，信息不足，
            # 保持观望等待更明确信号
            
        elif debate_signal == "bearish" and debate_confidence < 0.35:
            trading_action = "hold"
            # 理由：看空信号但置信度不够高（<35%），
            # 在低风险环境下不确定性太大，
            # 维持现状避免过早离场
            
        elif debate_signal == "neutral" and debate_confidence < 0.3:
            trading_action = "hold"
            # 理由：中性信号且置信度很低，信息极其不明确，
            # 分析师团队无法形成一致意见，
            # 任何操作都可能是错误的，保持观望最安全
            
        else:
            # 捕获所有未预料到的情况
            trading_action = "hold"
            # 理由：遇到未定义的情况（如非标准的 debate_signal 值），
            # 或其他未覆盖的边缘情况，
            # 保守起见选择持有，避免在不明确情况下做决定

    # 8. GARCH模型拟合和预测
    garch_results = {}
    try:
        # 获取足够长的数据进行GARCH拟合
        if len(returns) >= 100:  # 确保有足够的数据
            garch_params, log_likelihood = fit_garch(returns.values)
            volatility_forecast = forecast_garch_volatility(returns.values, garch_params, 
                                                         forecast_horizon=10)
            
            # 将结果添加到分析中
            regime_risk_score += 1 if garch_params['persistence'] > 0.95 else 0  # 高持续性表示更高风险
            
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
        "max_position_size": float(position_sizing_result["max_position_size"]),
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
            "regime_risk_score": regime_risk_score,
            "stress_test_results": stress_test_results,
            "macro_environment_assessment": { 
                "global_risks": regime_risk_score > 5,
                "liquidity_concerns": regime_risk_score > 7,
                "volatility_regime": "high" if volatility > 0.3 else "medium" if volatility > 0.2 else "low"
            }
        },
        "position_sizing": {
            "kelly_fraction": float(position_sizing_result["kelly_fraction"]),
            "win_rate": float(position_sizing_result["win_rate"]),
            "win_loss_ratio": float(position_sizing_result["win_loss_ratio"]),
            "risk_adjustment": float(position_sizing_result["risk_adjustment"]),
            "total_portfolio_value": float(total_portfolio_value)
        },
        "debate_analysis": {
            "bull_confidence": bull_confidence,
            "bear_confidence": bear_confidence,
            "debate_confidence": debate_confidence,
            "debate_signal": debate_signal
        },
        "volatility_model": garch_results,
        "reasoning": f"风险评分 {risk_score}/10: 市场风险={regime_risk_score}, "
                     f"波动率={volatility:.2%}, VaR={var_95:.2%}, CVaR={cvar_95:.2%}, "
                     f"最大回撤={max_drawdown:.2%}, 偏度={skewness:.2f}, "
                     f"辩论信号={debate_signal}, Kelly建议占比={position_sizing_result['kelly_fraction']:.2f}"
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


def _generate_regime_aware_diversification_tips(correlation_matrix, current_regime, high_corr_pairs):
    """基于市场区制生成分散化建议"""
    tips = []
    regime_name = current_regime.get('regime_name', 'unknown')
    
    if regime_name == "crisis_regime":
        tips.append("危机区制下，传统分散化效果减弱，考虑增加避险资产(如债券、黄金)配置")
        tips.append("危机期间资产相关性趋于1，建议降低整体风险敞口")
    elif regime_name == "high_volatility_mean_reverting":
        tips.append("高波动震荡市场中，可适当增加对冲策略和波动率交易")
        tips.append("震荡市场适合区间交易，建议设置更严格的止损止盈")
    elif regime_name == "low_volatility_trending":
        tips.append("低波动趋势市场适合增加动量策略配置")
        tips.append("趋势市场中可适当提高单一方向的敞口")
    
    return tips


def _calculate_regime_risk_score(current_regime, volatility_percentile, var_95, max_drawdown, skewness, kurtosis):
    """基于市场区制计算风险评分"""
    regime_name = current_regime.get('regime_name', 'unknown')
    regime_confidence = current_regime.get('confidence', 0.5)
    
    # 基础风险评分
    risk_score = 0
    
    # 区制特定风险调整
    if regime_name == "crisis_regime":
        risk_score += 3  # 危机区制基础风险高
        if regime_confidence > 0.7:
            risk_score += 1  # 高置信度的危机预测
    elif regime_name == "high_volatility_mean_reverting":
        risk_score += 2  # 高波动区制中等风险
    elif regime_name == "low_volatility_trending":
        risk_score += 1  # 低波动趋势区制较低风险
    
    # 波动率评分 (基于百分位)
    if volatility_percentile > 1.5:     # 高于1.5个标准差
        risk_score += 2
    elif volatility_percentile > 1.0:   # 高于1个标准差
        risk_score += 1

    # VaR评分 (基于历史分布)
    if var_95 > 0.03:
        risk_score += 2
    elif var_95 > 0.02:
        risk_score += 1

    # 最大回撤评分
    if abs(max_drawdown) > 0.20:  # 严重回撤
        risk_score += 2
    elif abs(max_drawdown) > 0.10:
        risk_score += 1
        
    # 分布异常性评分 (显著非正态)
    if abs(skewness) > 1.0 or kurtosis > 5.0:
        risk_score += 1
    
    return risk_score


def _calculate_regime_aware_position_sizing(debate_results, current_regime, returns, total_portfolio_value):
    """基于区制的动态头寸规模计算"""
    regime_name = current_regime.get('regime_name', 'unknown')
    regime_confidence = current_regime.get('confidence', 0.5)
    
    # 从辩论结果调整胜率
    bull_confidence = debate_results.get("bull_confidence", 0.5)
    bear_confidence = debate_results.get("bear_confidence", 0.5)
    if bull_confidence > bear_confidence:
        win_rate = bull_confidence
    else:
        win_rate = 1 - bear_confidence
    
    # 区制特定的胜率调整
    if regime_name == "crisis_regime":
        win_rate *= 0.7  # 危机期间降低胜率预期
    elif regime_name == "low_volatility_trending":
        win_rate *= 1.1  # 趋势市场提高胜率预期
        win_rate = min(win_rate, 0.8)  # 上限
    
    # 结合波动率模型优化收益预期
    try:
        # 使用GARCH模型预测未来波动率
        if len(returns) >= 100:  # 确保有足够的数据
            garch_params, _ = fit_garch(returns.values)
            volatility_forecast = forecast_garch_volatility(returns.values, garch_params, 
                                                         forecast_horizon=10)
            
            # 根据波动率预测调整胜率
            avg_forecast = np.mean(volatility_forecast)
            current_vol = returns.std()
            volatility_trend = avg_forecast / current_vol - 1 if current_vol != 0 else 0
            
            # 调整胜率
            if volatility_trend > 0.1:  # 波动率预期显著上升
                win_rate = max(0.3, win_rate - 0.1)  # 最低设为0.3
            elif volatility_trend < -0.1:  # 波动率预期显著下降
                win_rate = min(0.8, win_rate + 0.05)  # 最高设为0.8
    except Exception as e:
        logger.warning(f"GARCH模型计算失败: {e}")
    
    # 计算平均胜利收益和亏损比率
    avg_gain = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.005
    avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.005
    win_loss_ratio = avg_gain / avg_loss if avg_loss != 0 else 1.5
    
    # 应用改进的凯利公式
    kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
    kelly_fraction = max(0.05, kelly_fraction)
    
    # 区制特定的保守系数
    if regime_name == "crisis_regime":
        conservative_factor = 0.3  # 危机期间更保守
    elif regime_name == "high_volatility_mean_reverting":
        conservative_factor = 0.4  # 高波动期间保守
    else:
        conservative_factor = 0.5  # 正常情况
    
    # 风险调整基于区制置信度
    risk_adjustment = regime_confidence if regime_confidence > 0.6 else 0.5
    
    # 最终头寸大小
    max_position_size = total_portfolio_value * kelly_fraction * conservative_factor * risk_adjustment
    max_position_size = max(max_position_size, total_portfolio_value * 0.02)  # 设置最小仓位
    max_position_size = min(max_position_size, total_portfolio_value * 0.4)   # 设置最大仓位
    
    return {
        "max_position_size": max_position_size,
        "kelly_fraction": kelly_fraction,
        "win_rate": win_rate,
        "win_loss_ratio": win_loss_ratio,
        "risk_adjustment": risk_adjustment,
        "conservative_factor": conservative_factor,
        "regime_adjustment": f"Applied {regime_name} specific adjustments"
    }