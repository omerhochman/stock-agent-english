from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.openrouter_config import get_chat_completion
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.logging_config import setup_logger
from src.agents.regime_detector import AdvancedRegimeDetector, adaptive_signal_aggregation
from src.tools.api import prices_to_df
import json
import ast
import pandas as pd

# 获取日志记录器
logger = setup_logger('debate_room')


@agent_endpoint("debate_room", "辩论室，分析多空双方观点，得出平衡的投资结论")
def debate_room_agent(state: AgentState):
    """
    Enhanced debate room with regime-aware signal aggregation
    Based on 2024-2025 research: FLAG-Trader, FINSABER, Lopez-Lira frameworks
    """
    try:
        show_workflow_status("Debate Room")
        show_reasoning = state["metadata"]["show_reasoning"]
        logger.info("开始分析研究员观点并进行辩论...")

        # 初始化高级区制检测器
        regime_detector = AdvancedRegimeDetector()
        
        # 获取价格数据进行区制分析
        data = state["data"]
        prices = data["prices"]
        prices_df = prices_to_df(prices)
        
        # 提取区制特征并进行分析
        regime_features = regime_detector.extract_regime_features(prices_df)
        regime_model_results = regime_detector.fit_regime_model(regime_features)
        current_regime = regime_detector.predict_current_regime(regime_features)
        
        logger.info(f"检测到市场区制: {current_regime.get('regime_name', 'unknown')} (置信度: {current_regime.get('confidence', 0):.2f})")

        # 收集所有agent信号 - 向前兼容设计（添加防御性检查）
        agent_signals = {}
        researcher_messages = {}
        
        for msg in state["messages"]:
            # 添加防御性检查，确保 msg 和 msg.name 不为 None
            if msg is None:
                continue
            if not hasattr(msg, 'name') or msg.name is None:
                continue
                
            # 收集各类agent信号
            if msg.name.endswith("_agent"):
                try:
                    if hasattr(msg, 'content') and msg.content is not None:
                        content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                        
                        # 映射agent名称到信号类型
                        agent_type_mapping = {
                            'technical_analyst_agent': 'technical',
                            'fundamentals_agent': 'fundamental', 
                            'sentiment_agent': 'sentiment',
                            'valuation_agent': 'valuation',
                            'ai_model_analyst_agent': 'ai_model',
                            'macro_analyst_agent': 'macro'
                        }
                        
                        if msg.name in agent_type_mapping:
                            signal_type = agent_type_mapping[msg.name]
                            agent_signals[signal_type] = {
                                'signal': content.get('signal', 0),
                                'confidence': content.get('confidence', 0.5),
                                'raw_data': content
                            }
                            logger.debug(f"收集到{signal_type}信号: {content.get('signal', 0)} (置信度: {content.get('confidence', 0.5)})")
                        
                        # 同时保持原有的研究员逻辑
                        if msg.name.startswith("researcher_"):
                            researcher_messages[msg.name] = msg
                            logger.debug(f"收集到研究员信息: {msg.name}")
                            
                except (json.JSONDecodeError, TypeError, AttributeError) as e:
                    logger.warning(f"无法解析 {msg.name} 的消息内容: {e}")
                    continue

        # 使用新的自适应信号聚合系统
        if agent_signals and current_regime.get('regime_name') != 'unknown':
            aggregated_result = adaptive_signal_aggregation(
                signals=agent_signals,
                regime_info=current_regime,
                confidence_threshold=0.6
            )
            
            final_signal = aggregated_result['aggregated_signal']
            final_confidence = aggregated_result['aggregated_confidence']
            
            logger.info(f"自适应信号聚合结果: {final_signal:.3f} (置信度: {final_confidence:.3f})")
            
            # 创建增强的分析报告
            enhanced_analysis = {
                "signal": final_signal,
                "confidence": final_confidence,
                "aggregation_method": "regime_aware_adaptive",
                "market_regime": current_regime,
                "regime_adjusted_weights": aggregated_result['regime_adjusted_weights'],
                "signal_contributions": aggregated_result['signal_contributions'],
                "dynamic_threshold": aggregated_result['dynamic_threshold'],
                "regime_model_performance": {
                    "model_score": regime_model_results.get('model_score', 0),
                    "regime_characteristics": regime_model_results.get('regime_characteristics', {})
                }
            }
            
        else:
            # 回退到原有逻辑（保持向后兼容）
            logger.info("使用传统辩论逻辑（信号不足或区制检测失败）")
            enhanced_analysis = _traditional_debate_logic(state, researcher_messages, logger)

        # 确保至少有看多和看空两个研究员（保持原有验证逻辑）
        if "researcher_bull_agent" not in researcher_messages or "researcher_bear_agent" not in researcher_messages:
            logger.error("缺少必要的研究员数据: researcher_bull_agent 或 researcher_bear_agent")
            # 如果没有研究员数据但有其他信号，仍然可以继续
            if not agent_signals:
                raise ValueError("Missing required researcher_bull_agent or researcher_bear_agent messages")

        # 处理研究员数据（保持原有逻辑用于LLM分析）
        researcher_data = {}
        for name, msg in researcher_messages.items():
            if not hasattr(msg, 'content') or msg.content is None:
                logger.warning(f"研究员 {name} 的消息内容为空")
                continue
            try:
                data_content = json.loads(msg.content)
                logger.debug(f"成功解析 {name} 的 JSON 内容")
            except (json.JSONDecodeError, TypeError):
                try:
                    data_content = ast.literal_eval(msg.content)
                    logger.debug(f"通过 ast.literal_eval 解析 {name} 的内容")
                except (ValueError, SyntaxError, TypeError):
                    logger.warning(f"无法解析 {name} 的消息内容，已跳过")
                    continue
            researcher_data[name] = data_content

        # 如果有研究员数据，进行LLM增强分析
        if len(researcher_data) >= 2:
            llm_enhanced_analysis = _get_llm_enhanced_analysis(
                researcher_data, agent_signals, current_regime, state, logger
            )
            
            # 融合自适应聚合结果和LLM分析
            if 'signal' in enhanced_analysis and 'llm_score' in llm_enhanced_analysis:
                # 使用加权平均融合两种方法的结果
                regime_confidence = current_regime.get('confidence', 0.5)
                adaptive_weight = 0.7 if regime_confidence > 0.6 else 0.5
                llm_weight = 1 - adaptive_weight
                
                enhanced_analysis['signal'] = (
                    adaptive_weight * enhanced_analysis['signal'] + 
                    llm_weight * llm_enhanced_analysis['llm_score']
                )
                enhanced_analysis['llm_analysis'] = llm_enhanced_analysis
                enhanced_analysis['fusion_weights'] = {
                    'adaptive_aggregation': adaptive_weight,
                    'llm_analysis': llm_weight
                }

        # 创建最终消息
        message = HumanMessage(
            content=json.dumps(enhanced_analysis),
            name="debate_room_agent",
        )

        if show_reasoning:
            show_agent_reasoning(enhanced_analysis, "Enhanced Debate Room")
            state["metadata"]["agent_reasoning"] = enhanced_analysis

        show_workflow_status("Debate Room", "completed")
        return {
            "messages": [message],
            "data": data,
            "metadata": state["metadata"],
        }
        
    except Exception as e:
        logger.error(f"辩论室处理过程中发生错误: {e}")
        # 返回默认中性结果
        default_analysis = {
            "signal": 0,
            "confidence": 0.3,
            "error": str(e),
            "aggregation_method": "error_fallback"
        }
        
        message = HumanMessage(
            content=json.dumps(default_analysis),
            name="debate_room_agent",
        )
        
        return {
            "messages": [message],
            "data": state["data"],
            "metadata": state["metadata"],
        }


def _traditional_debate_logic(state: AgentState, researcher_messages: dict, logger) -> dict:
    """传统辩论逻辑的异步版本"""
    # 这里实现原有的简单平均逻辑作为回退
    if len(researcher_messages) < 2:
        return {
            "signal": 0,
            "confidence": 0.3,
            "aggregation_method": "insufficient_data"
        }
    
    # 转换研究员观点为数值信号
    perspective_values = {
        'bullish': 1.0,
        'neutral': 0.0,
        'bearish': -1.0
    }
    
    # 简单平均研究员信号
    total_signal = 0
    total_confidence = 0
    count = 0
    
    for name, msg in researcher_messages.items():
        try:
            data = json.loads(msg.content)
            # 研究员使用 perspective 而不是 signal
            perspective = data.get('perspective', 'neutral')
            confidence = data.get('confidence', 0.5)
            
            # 转换观点为数值
            numeric_signal = perspective_values.get(perspective, 0.0)
            
            total_signal += numeric_signal * confidence
            total_confidence += confidence
            count += 1
        except:
            continue
    
    if count > 0:
        avg_signal = total_signal / total_confidence if total_confidence > 0 else 0
        avg_confidence = total_confidence / count if count > 0 else 0.3
    else:
        avg_signal = 0
        avg_confidence = 0.3
    
    return {
        "signal": avg_signal,
        "confidence": avg_confidence,
        "aggregation_method": "traditional_average"
    }


def _get_llm_enhanced_analysis(researcher_data: dict, agent_signals: dict, current_regime: dict, state: AgentState, logger) -> dict:
    """获取LLM增强分析"""
    # 构建发送给 LLM 的提示（保持原有逻辑但增强）
    llm_prompt = f"""
You are a professional financial analyst. Analyze the following investment research and provide your third-party analysis.

Current Market Regime: {current_regime.get('regime_name', 'unknown')} (Confidence: {current_regime.get('confidence', 0):.2f})

RESEARCH PERSPECTIVES:
"""
    
    # 添加研究员观点
    for name, data in researcher_data.items():
        perspective = name.replace("researcher_", "").replace("_agent", "").upper()
        llm_prompt += f"\n{perspective} VIEW (Confidence: {data.get('confidence', 0)}):\n"
        for point in data.get("thesis_points", []):
            llm_prompt += f"- {point}\n"
    
    # 添加量化信号摘要
    if agent_signals:
        llm_prompt += f"\nQUANTITATIVE SIGNALS:\n"
        for signal_type, signal_data in agent_signals.items():
            llm_prompt += f"- {signal_type.title()}: {signal_data.get('signal', 0):.2f} (Confidence: {signal_data.get('confidence', 0):.2f})\n"
    
    llm_prompt += f"""
MARKET REGIME CONTEXT:
- Detected regime: {current_regime.get('regime_name', 'unknown')}
- Regime confidence: {current_regime.get('confidence', 0):.2f}

Please provide your analysis in the following JSON format:
{{
    "analysis": "Your detailed analysis evaluating the strengths and weaknesses of each perspective",
    "score": 0.5,  // Your score from -1.0 (extremely bearish) to 1.0 (extremely bullish), 0 = neutral
    "reasoning": "Brief reasoning for your score",
    "regime_considerations": "How the current market regime affects your analysis",
    "macro_factors": ["List 1-3 most important macro factors"]
}}

Ensure your response is valid JSON format and includes all fields above. Respond in English only.
"""

    try:
        logger.info("开始调用 LLM 获取增强分析...")
        messages = [
            {"role": "system", "content": "You are a professional financial analyst. Provide analysis in English only."},
            {"role": "user", "content": llm_prompt}
        ]

        llm_response = log_llm_interaction(state)(
            lambda: get_chat_completion(messages)
        )()

        if llm_response:
            # 解析 LLM 返回的 JSON
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                llm_analysis = json.loads(json_str)
                llm_score = float(llm_analysis.get("score", 0))
                llm_score = max(min(llm_score, 1.0), -1.0)  # 确保在有效范围内
                
                logger.info(f"LLM增强分析完成，评分: {llm_score}")
                return {
                    "llm_score": llm_score,
                    "llm_analysis": llm_analysis,
                    "llm_response": llm_response
                }
    except Exception as e:
        logger.error(f"LLM增强分析失败: {e}")
    
    return {"llm_score": 0, "error": "LLM analysis failed"}