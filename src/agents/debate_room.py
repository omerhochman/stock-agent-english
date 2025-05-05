from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.openrouter_config import get_chat_completion
from src.utils.api_utils import agent_endpoint, log_llm_interaction
import json
import ast
import logging

# 获取日志记录器
logger = logging.getLogger('debate_room')


@agent_endpoint("debate_room", "辩论室，分析多空双方观点，得出平衡的投资结论")
def debate_room_agent(state: AgentState):
    """Facilitates debate between bull and bear researchers to reach a balanced conclusion."""
    try:
        show_workflow_status("Debate Room")
        show_reasoning = state["metadata"]["show_reasoning"]
        logger.info("开始分析研究员观点并进行辩论...")

        # 收集所有研究员信息 - 向前兼容设计（添加防御性检查）
        researcher_messages = {}
        for msg in state["messages"]:
            # 添加防御性检查，确保 msg 和 msg.name 不为 None
            if msg is None:
                continue
            if not hasattr(msg, 'name') or msg.name is None:
                continue
            if isinstance(msg.name, str) and msg.name.startswith("researcher_") and msg.name.endswith("_agent"):
                researcher_messages[msg.name] = msg
                logger.debug(f"收集到研究员信息: {msg.name}")

        # 确保至少有看多和看空两个研究员
        if "researcher_bull_agent" not in researcher_messages or "researcher_bear_agent" not in researcher_messages:
            logger.error(
                "缺少必要的研究员数据: researcher_bull_agent 或 researcher_bear_agent")
            raise ValueError(
                "Missing required researcher_bull_agent or researcher_bear_agent messages")

        # 处理研究员数据
        researcher_data = {}
        for name, msg in researcher_messages.items():
            # 添加防御性检查，确保 msg.content 不为 None
            if not hasattr(msg, 'content') or msg.content is None:
                logger.warning(f"研究员 {name} 的消息内容为空")
                continue
            try:
                data = json.loads(msg.content)
                logger.debug(f"成功解析 {name} 的 JSON 内容")
            except (json.JSONDecodeError, TypeError):
                try:
                    data = ast.literal_eval(msg.content)
                    logger.debug(f"通过 ast.literal_eval 解析 {name} 的内容")
                except (ValueError, SyntaxError, TypeError):
                    # 如果无法解析内容，跳过此消息
                    logger.warning(f"无法解析 {name} 的消息内容，已跳过")
                    continue
            researcher_data[name] = data

        # 获取看多和看空研究员数据（为了兼容原有逻辑）
        if "researcher_bull_agent" not in researcher_data or "researcher_bear_agent" not in researcher_data:
            logger.error("无法解析必要的研究员数据")
            raise ValueError(
                "Could not parse required researcher_bull_agent or researcher_bear_agent messages")

        bull_thesis = researcher_data["researcher_bull_agent"]
        bear_thesis = researcher_data["researcher_bear_agent"]
        logger.info(
            f"已获取看多观点(置信度: {bull_thesis.get('confidence', 0)})和看空观点(置信度: {bear_thesis.get('confidence', 0)})")

        # 在获取研究员信息之后，获取AI模型分析结果
        ai_model_message = next(
            (msg for msg in state["messages"] if msg.name == "ai_model_analyst_agent"), None)

        if ai_model_message:
            try:
                ai_model_results = json.loads(ai_model_message.content)
                logger.info(
                    f"已获取AI模型分析结果(置信度: {ai_model_results.get('confidence', 0)})")
            except (json.JSONDecodeError, TypeError):
                try:
                    ai_model_results = ast.literal_eval(ai_model_message.content)
                    logger.debug(f"通过 ast.literal_eval 解析AI模型分析内容")
                except (ValueError, SyntaxError, TypeError):
                    logger.warning(f"无法解析AI模型分析结果，已跳过")
                    ai_model_results = None
        else:
            ai_model_results = None

        # 比较置信度级别
        bull_confidence = bull_thesis.get("confidence", 0)
        bear_confidence = bear_thesis.get("confidence", 0)

        # 分析辩论观点
        debate_summary = []
        debate_summary.append("Bullish Arguments:")
        for point in bull_thesis.get("thesis_points", []):
            debate_summary.append(f"+ {point}")

        debate_summary.append("\nBearish Arguments:")
        for point in bear_thesis.get("thesis_points", []):
            debate_summary.append(f"- {point}")

        # 收集所有研究员的论点，准备发给 LLM
        all_perspectives = {}
        for name, data in researcher_data.items():
            perspective = data.get("perspective", name.replace(
                "researcher_", "").replace("_agent", ""))
            all_perspectives[perspective] = {
                "confidence": data.get("confidence", 0),
                "thesis_points": data.get("thesis_points", [])
            }

        logger.info(f"准备让 LLM 分析 {len(all_perspectives)} 个研究员的观点")

        # 构建发送给 LLM 的提示
        llm_prompt = """
    你是一位专业的金融分析师，请分析以下投资研究员的观点，并给出你的第三方分析:

    """
        for perspective, data in all_perspectives.items():
            llm_prompt += f"\n{perspective.upper()} 观点 (置信度: {data['confidence']}):\n"
            for point in data["thesis_points"]:
                llm_prompt += f"- {point}\n"

        # 在构建发送给LLM的提示时，添加AI模型分析结果
        if ai_model_results:
            llm_prompt += f"\nAI模型分析 (置信度: {ai_model_results.get('confidence', 0)}):\n"
            
            # 处理多资产情况
            if ai_model_results.get("multi_asset", False):
                llm_prompt += f"投资组合AI分析: {ai_model_results.get('signal', 'neutral')} (置信度: {ai_model_results.get('confidence', 0)})\n"
                
                # 添加部分资产信息
                asset_signals = ai_model_results.get("asset_signals", {})
                for ticker, signals in list(asset_signals.items())[:3]:  # 只显示前3个资产
                    llm_prompt += f"- {ticker}: {signals.get('signal', 'neutral')} (置信度: {signals.get('confidence', 0)})\n"
                
                if len(asset_signals) > 3:
                    llm_prompt += f"- 以及其他 {len(asset_signals) - 3} 个资产...\n"
            else:
                # 单资产情况
                model_signals = ai_model_results.get("model_signals", {})
                for model_name, model_data in model_signals.items():
                    llm_prompt += f"- {model_name}: {model_data.get('signal', 'neutral')} (置信度: {model_data.get('confidence', 0)})\n"
                llm_prompt += f"综合AI预测: {ai_model_results.get('signal', 'neutral')}\n"

        llm_prompt += """
        在分析过程中，也请考虑以下宏观因素：
        1. 当前的市场环境和风险偏好
        2. 行业政策和监管走向
        3. 全球经济形势的可能影响

        请提供以下格式的 JSON 回复:
        {
            "analysis": "你的详细分析，评估各方观点的优劣，并指出你认为最有说服力的论点",
            "score": 0.5,  // 你的评分，从 -1.0(极度看空) 到 1.0(极度看多)，0 表示中性
            "reasoning": "你给出这个评分的简要理由",
            "macro_factors": ["列出1-3个你认为最重要的宏观因素"]
        }

        务必确保你的回复是有效的 JSON 格式，且包含上述所有字段。回复必须使用英文，不要使用中文或其他语言。
        """

        # 调用 LLM 获取第三方观点
        llm_response = None
        llm_analysis = None
        llm_score = 0  # 默认为中性
        try:
            logger.info("开始调用 LLM 获取第三方分析...")
            messages = [
                {"role": "system", "content": "You are a professional financial analyst. Please provide your analysis in English only, not in Chinese or any other language."},
                {"role": "user", "content": llm_prompt}
            ]

            # 使用log_llm_interaction装饰器记录LLM交互
            llm_response = log_llm_interaction(state)(
                lambda: get_chat_completion(messages)
            )()

            logger.info("LLM 返回响应完成")

            # 解析 LLM 返回的 JSON
            if llm_response:
                try:
                    # 尝试提取 JSON 部分
                    json_start = llm_response.find('{')
                    json_end = llm_response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = llm_response[json_start:json_end]
                        llm_analysis = json.loads(json_str)
                        llm_score = float(llm_analysis.get("score", 0))
                        # 确保分数在有效范围内
                        llm_score = max(min(llm_score, 1.0), -1.0)
                        logger.info(f"成功解析 LLM 回复，评分: {llm_score}")
                        logger.debug(
                            f"LLM 分析内容: {llm_analysis.get('analysis', '未提供分析')[:100]}...")
                except Exception as e:
                    # 如果解析失败，记录错误并使用默认值
                    logger.error(f"解析 LLM 回复失败: {e}")
                    llm_analysis = {"analysis": "Failed to parse LLM response",
                                    "score": 0, "reasoning": "Parsing error"}
        except Exception as e:
            logger.error(f"调用 LLM 失败: {e}")
            llm_analysis = {"analysis": "LLM API call failed",
                            "score": 0, "reasoning": "API error"}

        # 计算混合置信度差异
        confidence_diff = bull_confidence - bear_confidence

        # 将LLM权重调整为25%，AI模型权重为20%，研究员占55%
        llm_weight = 0.25
        ai_weight = 0.20
        researcher_weight = 1 - llm_weight - ai_weight
        
        # 修改混合置信度计算
        if ai_model_results:
            ai_confidence = ai_model_results.get("confidence", 0.5)
            ai_signal_value = 1 if ai_model_results.get("signal") == "bullish" else (-1 if ai_model_results.get("signal") == "bearish" else 0)
            mixed_confidence_diff = (researcher_weight * confidence_diff + 
                                    llm_weight * llm_score + 
                                    ai_weight * ai_signal_value * ai_confidence)
        else:
            # 没有AI模型结果时的默认计算
            mixed_confidence_diff = (1 - llm_weight) * confidence_diff + llm_weight * llm_score

        logger.info(
            f"计算混合置信度差异: 原始差异={confidence_diff:.4f}, LLM评分={llm_score:.4f}, 混合差异={mixed_confidence_diff:.4f}")

        # 基于混合置信度差异确定最终建议
        if abs(mixed_confidence_diff) < 0.1:  # 接近争论
            final_signal = "neutral"
            reasoning = "Balanced debate with strong arguments on both sides"
            confidence = max(bull_confidence, bear_confidence)
        elif mixed_confidence_diff > 0:  # 看多胜出
            final_signal = "bullish"
            reasoning = "Bullish arguments more convincing"
            confidence = bull_confidence
        else:  # 看空胜出
            final_signal = "bearish"
            reasoning = "Bearish arguments more convincing"
            confidence = bear_confidence

        logger.info(f"最终投资信号: {final_signal}, 置信度: {confidence}")

        # 构建返回消息，包含 LLM 分析
        message_content = {
            "signal": final_signal,
            "confidence": confidence,
            "bull_confidence": bull_confidence,
            "bear_confidence": bear_confidence,
            "confidence_diff": confidence_diff,
            "llm_score": llm_score if llm_analysis else None,
            "llm_analysis": llm_analysis["analysis"] if llm_analysis and "analysis" in llm_analysis else None,
            "llm_reasoning": llm_analysis["reasoning"] if llm_analysis and "reasoning" in llm_analysis else None,
            "mixed_confidence_diff": mixed_confidence_diff,
            "debate_summary": debate_summary,
            "reasoning": reasoning,
            "ai_model_contribution": {
                "included": ai_model_results is not None,
                "signal": ai_model_results.get("signal", "neutral") if ai_model_results else None,
                "confidence": ai_model_results.get("confidence", 0) if ai_model_results else None,
                "weight": ai_weight
            } if ai_model_results else None
        }

        message = HumanMessage(
            content=json.dumps(message_content, ensure_ascii=False),
            name="debate_room_agent",
        )

        if show_reasoning:
            show_agent_reasoning(message_content, "Debate Room")
            # 保存推理信息到metadata供API使用
            state["metadata"]["agent_reasoning"] = message_content

        show_workflow_status("Debate Room", "completed")
        return {
            "messages": state["messages"] + [message],
            "data": {
                **state["data"],
                "debate_analysis": message_content
            },
            "metadata": state["metadata"],
        }
    except ValueError as e:
        # 添加更多的错误恢复和记录
        logger.error(f"辩论室发生错误: {str(e)}")
        # 提供一个默认的回退响应，避免流程中断
        message_content = {
            "signal": "neutral",
            "confidence": 0.5,
            "reasoning": f"辩论过程中发生错误: {str(e)}"
        }
        message = HumanMessage(
            content=json.dumps(message_content),
            name="debate_room_agent",
        )
        # 继续执行流程
        return {
            "messages": state["messages"] + [message],
            "data": {**state["data"], "debate_analysis": message_content},
            "metadata": state["metadata"],
        }