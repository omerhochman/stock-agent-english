"""
Agent Collector Module - 收集和存储Agent分析结果

此模块负责存储和管理Agent的分析结果，以便生成汇总报告。
"""

import json
import logging
from typing import Dict, Any, List, Optional

# 日志设置
logger = logging.getLogger(__name__)

# 用于存储所有Agent的最终状态
_collected_states = {}

# 用于存储最终的增强状态（包含所有整合数据）
_enhanced_final_state = None

def store_final_state(state: Dict[str, Any]) -> None:
    """
    存储Agent的最终状态
    
    Args:
        state: Agent运行后的状态字典
    """
    global _collected_states, _enhanced_final_state
    
    if not state:
        logger.warning("尝试存储空状态, 已忽略")
        return
    
    # 提取关键元数据
    run_id = state.get("metadata", {}).get("run_id", "unknown")
    ticker = state.get("data", {}).get("ticker", "unknown")
    
    # 存储状态
    _collected_states[run_id] = state
    
    # 构建增强状态
    _enhanced_final_state = _build_enhanced_state(state)
    
    logger.debug(f"已存储股票 {ticker} (Run ID: {run_id}) 的最终状态")

def get_enhanced_final_state() -> Dict[str, Any]:
    """
    获取增强的最终状态，包含所有整合的数据
    
    Returns:
        包含所有Agent分析结果的增强状态字典
    """
    global _enhanced_final_state
    
    if not _enhanced_final_state:
        logger.warning("尝试获取增强状态但尚未存储任何状态")
        return {}
    
    return _enhanced_final_state

def clear_state() -> None:
    """清除所有存储的状态"""
    global _collected_states, _enhanced_final_state
    _collected_states = {}
    _enhanced_final_state = None
    logger.debug("已清除所有状态")

def _build_enhanced_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    构建增强状态，整合所有Agent的分析结果
    
    Args:
        state: 最终的状态字典
    
    Returns:
        增强的状态字典，包含所有整合数据
    """
    # 从state提取基本信息
    enhanced = {
        "ticker": state.get("data", {}).get("ticker", "unknown"),
        "run_id": state.get("metadata", {}).get("run_id", "unknown"),
        "date_range": {
            "start": state.get("data", {}).get("start_date", "unknown"),
            "end": state.get("data", {}).get("end_date", "unknown")
        },
        "analysis": {}
    }
    
    try:
        # 提取投资组合管理的最终决策
        last_message = state.get("messages", [])[-1]
        content = last_message.content if hasattr(last_message, 'content') else last_message
        if isinstance(content, str):
            try:
                # 尝试解析JSON内容
                decision = json.loads(content)
                enhanced["final_decision"] = decision
            except json.JSONDecodeError:
                # 如果不是有效的JSON，直接使用字符串
                enhanced["final_decision"] = {"summary": content}
        elif isinstance(content, dict):
            enhanced["final_decision"] = content
        else:
            enhanced["final_decision"] = {"summary": str(content)}
            
        # 提取各Agent的分析结果
        for key, value in state.items():
            if key.endswith("_analysis") and isinstance(value, dict):
                agent_name = key.replace("_analysis", "")
                enhanced["analysis"][agent_name] = value
                
        # 提取元数据中的Agent推理记录
        agent_reasoning = state.get("metadata", {}).get("agent_reasoning", {})
        if agent_reasoning:
            for agent_name, reasoning in agent_reasoning.items():
                if agent_name not in enhanced["analysis"]:
                    enhanced["analysis"][agent_name] = {}
                enhanced["analysis"][agent_name]["reasoning"] = reasoning
                
    except Exception as e:
        logger.error(f"构建增强状态时出错: {str(e)}")
        
    return enhanced