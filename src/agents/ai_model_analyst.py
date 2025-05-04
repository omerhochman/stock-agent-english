from langchain_core.messages import HumanMessage
import json

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint
from src.tools.api import prices_to_df
from model.dl import MLAgent
from model.rl import RLTradingAgent
from model.deap_factors import FactorAgent
from src.utils.logging_config import setup_logger

# 设置日志记录器
logger = setup_logger('ai_model_analyst_agent')

@agent_endpoint("ai_model_analyst", "AI模型分析师，运行深度学习、强化学习和遗传编程模型进行预测")
def ai_model_analyst_agent(state: AgentState):
    """运行深度学习、强化学习和遗传编程模型，生成预测信号，支持多资产分析"""
    show_workflow_status("AI Model Analyst")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    
    # 获取资产列表
    tickers = data.get("tickers", [])
    if isinstance(tickers, str):
        tickers = [ticker.strip() for ticker in tickers.split(',')]
    
    # 如果没有提供tickers但提供了ticker，则使用单一ticker
    if not tickers and data.get("ticker"):
        tickers = [data["ticker"]]
    
    # 如果仍然没有股票代码，则返回错误
    if not tickers:
        logger.warning("未提供股票代码，无法进行AI模型分析")
        message_content = {
            "signal": "neutral",
            "confidence": 0.5,
            "reasoning": "未提供股票代码，无法进行AI模型分析"
        }
        message = HumanMessage(
            content=json.dumps(message_content),
            name="ai_model_analyst_agent",
        )
        return {
            "messages": [message],
            "data": data,
            "metadata": state["metadata"],
        }
    
    # 获取所有股票的数据
    all_stock_data = data.get("all_stock_data", {})
    
    # 单一资产或多资产处理
    if len(tickers) == 1 or not all_stock_data:
        # 单一资产处理
        ticker = tickers[0]
        prices = data.get("prices", [])
        prices_df = prices_to_df(prices)
        
        # 如果价格数据为空，则返回默认值
        if prices_df.empty:
            logger.warning(f"价格数据为空，无法运行AI模型分析")
            message_content = {
                "signal": "neutral",
                "confidence": 0.5,
                "reasoning": "缺少足够的价格数据，无法运行AI模型分析"
            }
            message = HumanMessage(
                content=json.dumps(message_content),
                name="ai_model_analyst_agent",
            )
            return {
                "messages": [message],
                "data": data,
                "metadata": state["metadata"],
            }
        
        # 初始化各AI模型
        ml_signals = run_deep_learning_model(prices_df)
        rl_signals = run_reinforcement_learning_model(prices_df)
        factor_signals = run_genetic_programming_model(prices_df)
        
        # 组合各模型信号
        combined_signal = combine_ai_signals(ml_signals, rl_signals, factor_signals)
        
        # 生成分析报告
        message_content = {
            "multi_asset": False,
            "primary_ticker": ticker,
            "signal": combined_signal["signal"],
            "confidence": combined_signal["confidence"],
            "model_signals": {
                "deep_learning": ml_signals,
                "reinforcement_learning": rl_signals,
                "genetic_programming": factor_signals
            },
            "reasoning": combined_signal["reasoning"]
        }
    else:
        # 多资产处理
        # 为每个资产运行AI模型
        all_asset_signals = {}
        
        for ticker in tickers:
            try:
                if ticker in all_stock_data:
                    # 获取当前资产的价格数据
                    asset_prices = all_stock_data[ticker].get("prices", [])
                    asset_prices_df = prices_to_df(asset_prices)
                    
                    if not asset_prices_df.empty:
                        # 运行各模型
                        ml_signals = run_deep_learning_model(asset_prices_df)
                        rl_signals = run_reinforcement_learning_model(asset_prices_df)
                        factor_signals = run_genetic_programming_model(asset_prices_df)
                        
                        # 组合信号
                        combined_signal = combine_ai_signals(ml_signals, rl_signals, factor_signals)
                        
                        # 保存该资产的信号
                        all_asset_signals[ticker] = {
                            "signal": combined_signal["signal"],
                            "confidence": combined_signal["confidence"],
                            "model_signals": {
                                "deep_learning": ml_signals,
                                "reinforcement_learning": rl_signals,
                                "genetic_programming": factor_signals
                            },
                            "reasoning": combined_signal["reasoning"]
                        }
                    else:
                        logger.warning(f"资产 {ticker} 的价格数据为空")
                        all_asset_signals[ticker] = {
                            "signal": "neutral",
                            "confidence": 0.5,
                            "model_signals": {},
                            "reasoning": "价格数据不足，无法运行AI模型分析"
                        }
                else:
                    logger.warning(f"未找到资产 {ticker} 的数据")
                    all_asset_signals[ticker] = {
                        "signal": "neutral",
                        "confidence": 0.5,
                        "model_signals": {},
                        "reasoning": "未找到该资产的数据"
                    }
            except Exception as e:
                logger.error(f"处理资产 {ticker} 时出错: {e}")
                all_asset_signals[ticker] = {
                    "signal": "neutral",
                    "confidence": 0.5,
                    "model_signals": {},
                    "reasoning": f"处理该资产时出错: {str(e)}"
                }
        
        # 分析投资组合整体AI模型信号
        portfolio_signal = analyze_portfolio_ai_signals(all_asset_signals)
        
        # 投资组合优化建议
        portfolio_allocation = optimize_portfolio_based_on_ai(all_asset_signals)
        
        # 生成分析报告
        message_content = {
            "multi_asset": True,
            "primary_ticker": tickers[0],
            "tickers": tickers,
            "signal": portfolio_signal["signal"],
            "confidence": portfolio_signal["confidence"],
            "asset_signals": all_asset_signals,
            "portfolio_allocation": portfolio_allocation,
            "reasoning": portfolio_signal["reasoning"]
        }
    
    # 创建消息
    message = HumanMessage(
        content=json.dumps(message_content),
        name="ai_model_analyst_agent",
    )
    
    # 显示推理过程
    if show_reasoning:
        show_agent_reasoning(message_content, "AI Model Analyst")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content
    
    show_workflow_status("AI Model Analyst", "completed")
    return {
        "messages": [message],
        "data": {
            **data,
            "ai_analysis": message_content
        },
        "metadata": state["metadata"],
    }

def run_deep_learning_model(prices_df):
    """运行深度学习模型，生成交易信号"""
    try:
        # 尝试加载已训练的模型
        ml_agent = MLAgent(model_dir='models')
        ml_agent.load_models()
        
        # 生成交易信号
        signals = ml_agent.generate_signals(prices_df)
        
        # 如果无法获取信号，使用默认值
        if not signals or "signal" not in signals:
            return {
                "signal": "neutral", 
                "confidence": 0.5,
                "reasoning": "深度学习模型未能生成有效信号"
            }
        
        # 返回模型生成的信号
        return {
            "signal": signals["signal"],
            "confidence": signals.get("confidence", 0.5),
            "reasoning": signals.get("reasoning", "深度学习模型基于LSTM和随机森林算法预测")
        }
    except Exception as e:
        logger.error(f"运行深度学习模型出错: {e}")
        return {
            "signal": "neutral", 
            "confidence": 0.5,
            "reasoning": f"深度学习模型运行出错: {str(e)}"
        }

def run_reinforcement_learning_model(prices_df):
    """运行强化学习模型，生成交易信号"""
    try:
        # 尝试加载已训练的模型
        rl_agent = RLTradingAgent(model_dir='models')
        rl_agent.load_model()
        
        # 生成交易信号
        signals = rl_agent.generate_signals(prices_df)
        
        # 如果无法获取信号，使用默认值
        if not signals or "signal" not in signals:
            return {
                "signal": "neutral", 
                "confidence": 0.5,
                "reasoning": "强化学习模型未能生成有效信号"
            }
        
        # 返回模型生成的信号
        return {
            "signal": signals["signal"],
            "confidence": signals.get("confidence", 0.5),
            "reasoning": signals.get("reasoning", "强化学习模型基于PPO算法优化交易决策")
        }
    except Exception as e:
        logger.error(f"运行强化学习模型出错: {e}")
        return {
            "signal": "neutral", 
            "confidence": 0.5,
            "reasoning": f"强化学习模型运行出错: {str(e)}"
        }

def run_genetic_programming_model(prices_df):
    """运行遗传编程因子模型，生成交易信号"""
    try:
        # 尝试加载已训练的模型
        factor_agent = FactorAgent(model_dir='factors')
        factor_agent.load_factors()
        
        # 生成交易信号
        signals = factor_agent.generate_signals(prices_df)
        
        # 如果无法获取信号，使用默认值
        if not signals or "signal" not in signals:
            return {
                "signal": "neutral", 
                "confidence": 0.5,
                "reasoning": "遗传编程因子模型未能生成有效信号"
            }
        
        # 返回模型生成的信号
        return {
            "signal": signals["signal"],
            "confidence": signals.get("confidence", 0.5),
            "reasoning": signals.get("reasoning", "遗传编程因子模型通过自动化因子挖掘分析市场")
        }
    except Exception as e:
        logger.error(f"运行遗传编程因子模型出错: {e}")
        return {
            "signal": "neutral", 
            "confidence": 0.5,
            "reasoning": f"遗传编程因子模型运行出错: {str(e)}"
        }

def combine_ai_signals(ml_signals, rl_signals, factor_signals):
    """组合各AI模型信号，生成最终信号"""
    # 将信号转换为数值
    signal_values = {
        'bullish': 1,
        'neutral': 0,
        'bearish': -1
    }
    
    # 各模型权重
    weights = {
        'deep_learning': 0.35,  # LSTM和随机森林组合模型
        'reinforcement_learning': 0.35,  # PPO强化学习模型
        'genetic_programming': 0.30  # 遗传编程因子模型
    }
    
    # 计算加权分数
    ml_score = signal_values.get(ml_signals["signal"], 0) * ml_signals["confidence"] * weights["deep_learning"]
    rl_score = signal_values.get(rl_signals["signal"], 0) * rl_signals["confidence"] * weights["reinforcement_learning"]
    factor_score = signal_values.get(factor_signals["signal"], 0) * factor_signals["confidence"] * weights["genetic_programming"]
    
    total_score = ml_score + rl_score + factor_score
    
    # 根据分数确定信号
    if total_score > 0.15:
        signal = "bullish"
    elif total_score < -0.15:
        signal = "bearish"
    else:
        signal = "neutral"
    
    # 计算置信度
    confidence = min(0.5 + abs(total_score), 0.9)  # 限制在0.5-0.9范围内
    
    # 创建信号一致性描述
    signals = [ml_signals["signal"], rl_signals["signal"], factor_signals["signal"]]
    signal_counts = {
        "bullish": signals.count("bullish"),
        "bearish": signals.count("bearish"),
        "neutral": signals.count("neutral")
    }
    
    # 生成推理文本
    reasoning = f"AI模型综合分析：深度学习({ml_signals['signal']}, {ml_signals['confidence']:.2f}), "
    reasoning += f"强化学习({rl_signals['signal']}, {rl_signals['confidence']:.2f}), "
    reasoning += f"遗传编程({factor_signals['signal']}, {factor_signals['confidence']:.2f}). "
    
    # 添加一致性分析
    max_signal = max(signal_counts.items(), key=lambda x: x[1])
    if max_signal[1] >= 2:
        reasoning += f"{max_signal[1]}个模型一致预测{max_signal[0]}信号，增强了预测可信度。"
    else:
        reasoning += f"模型预测不一致，降低了总体置信度。"
    
    # 添加各模型的具体推理
    reasoning += f"\n\n具体模型推理：\n- 深度学习: {ml_signals['reasoning']}\n"
    reasoning += f"- 强化学习: {rl_signals['reasoning']}\n"
    reasoning += f"- 遗传编程: {factor_signals['reasoning']}"
    
    return {
        "signal": signal,
        "confidence": confidence,
        "weighted_score": total_score,
        "signal_consistency": max_signal[1] / 3,  # 一致性比例
        "reasoning": reasoning
    }

def analyze_portfolio_ai_signals(all_asset_signals):
    """
    根据多个资产的AI模型信号分析整体投资组合信号
    
    Args:
        all_asset_signals: 所有资产的AI模型信号
        
    Returns:
        dict: 投资组合整体信号
    """
    if not all_asset_signals:
        return {
            "signal": "neutral",
            "confidence": 0.5,
            "reasoning": "未提供资产AI模型信号"
        }
    
    # 统计各类信号数量
    signal_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    total_confidence = 0
    
    # 权重为信号的置信度
    weighted_signals = 0
    total_weights = 0
    
    for ticker, signals in all_asset_signals.items():
        signal = signals.get("signal", "neutral")
        confidence = signals.get("confidence", 0.5)
        
        # 更新计数
        signal_counts[signal] += 1
        total_confidence += confidence
        
        # 计算权重信号
        signal_value = 1 if signal == "bullish" else (-1 if signal == "bearish" else 0)
        weighted_signals += signal_value * confidence
        total_weights += confidence
    
    # 计算整体信号
    if total_weights > 0:
        avg_signal = weighted_signals / total_weights
    else:
        avg_signal = 0
    
    # 确定最终信号
    if avg_signal > 0.2:
        signal = "bullish"
    elif avg_signal < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"
    
    # 计算整体置信度
    avg_confidence = total_confidence / len(all_asset_signals) if all_asset_signals else 0.5
    
    # 计算信号一致性程度
    max_signal_count = max(signal_counts.values())
    consistency = max_signal_count / len(all_asset_signals) if all_asset_signals else 0
    
    # 调整置信度
    adjusted_confidence = avg_confidence * (0.5 + 0.5 * consistency)
    
    # 生成推理说明
    reasoning = f"投资组合AI分析：共分析{len(all_asset_signals)}个资产，其中看多{signal_counts['bullish']}个，"
    reasoning += f"看空{signal_counts['bearish']}个，中性{signal_counts['neutral']}个。\n"
    reasoning += f"信号一致性：{consistency:.2f}，平均置信度：{avg_confidence:.2f}。\n\n"
    
    # 添加显著信号的资产
    strong_bullish = []
    strong_bearish = []
    
    for ticker, signals in all_asset_signals.items():
        if signals.get("signal") == "bullish" and signals.get("confidence", 0) > 0.7:
            strong_bullish.append(ticker)
        elif signals.get("signal") == "bearish" and signals.get("confidence", 0) > 0.7:
            strong_bearish.append(ticker)
    
    if strong_bullish:
        reasoning += f"强势看多资产：{', '.join(strong_bullish)}\n"
    if strong_bearish:
        reasoning += f"强势看空资产：{', '.join(strong_bearish)}\n"
    
    return {
        "signal": signal,
        "confidence": adjusted_confidence,
        "weighted_score": avg_signal,
        "signal_consistency": consistency,
        "signal_counts": signal_counts,
        "reasoning": reasoning
    }

def optimize_portfolio_based_on_ai(all_asset_signals):
    """
    基于AI模型信号优化投资组合配置
    
    Args:
        all_asset_signals: 所有资产的AI模型信号
        
    Returns:
        dict: 优化后的资产配置建议
    """
    if not all_asset_signals:
        return {}
    
    # 计算基于AI信号的权重
    signal_scores = {}
    total_positive_score = 0
    
    for ticker, signals in all_asset_signals.items():
        signal = signals.get("signal", "neutral")
        confidence = signals.get("confidence", 0.5)
        
        # 计算信号分数
        if signal == "bullish":
            score = confidence
        elif signal == "bearish":
            score = -confidence
        else:
            score = 0
        
        signal_scores[ticker] = score
        
        # 计算正分数总和（用于后续计算权重）
        if score > 0:
            total_positive_score += score
    
    # 如果没有正分数，使用均等权重
    if total_positive_score <= 0:
        weights = {ticker: 1.0 / len(all_asset_signals) for ticker in all_asset_signals}
    else:
        # 计算权重（只考虑信号为正的资产）
        weights = {}
        for ticker, score in signal_scores.items():
            if score > 0:
                weights[ticker] = score / total_positive_score
            else:
                weights[ticker] = 0
    
    # 输出资产配置建议
    allocation = {}
    recommended_assets = []
    
    for ticker, weight in weights.items():
        if weight > 0:
            allocation[ticker] = round(weight, 4)
            recommended_assets.append(ticker)
    
    return {
        "allocation": allocation,
        "recommended_assets": recommended_assets,
        "reasoning": "基于AI模型信号强度和置信度计算最优资产权重"
    }