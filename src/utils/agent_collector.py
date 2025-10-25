"""
Agent Collector Module - Collect and store Agent analysis results

This module is responsible for storing and managing Agent analysis results for generating summary reports.
"""

import json
from typing import Dict, Any

from src.utils.logging_config import setup_logger

# Logging setup
logger = setup_logger('agent_collector')

# Used to store final states of all Agents
_collected_states = {}

# Used to store final enhanced state (containing all integrated data)
_enhanced_final_state = None

def store_final_state(state: Dict[str, Any]) -> None:
    """
    Store Agent's final state
    
    Args:
        state: State dictionary after Agent execution
    """
    global _collected_states, _enhanced_final_state
    
    if not state:
        logger.warning("Attempting to store empty state, ignored")
        return
    
    # Extract key metadata
    run_id = state.get("metadata", {}).get("run_id", "unknown")
    ticker = state.get("data", {}).get("ticker", "unknown")
    
    # Store state
    _collected_states[run_id] = state
    
    # Build enhanced state
    _enhanced_final_state = _build_enhanced_state(state)
    
    logger.debug(f"Stored final state for stock {ticker} (Run ID: {run_id})")

def get_enhanced_final_state() -> Dict[str, Any]:
    """
    Get enhanced final state containing all integrated data
    
    Returns:
        Enhanced state dictionary containing all Agent analysis results
    """
    global _enhanced_final_state
    
    if not _enhanced_final_state:
        logger.warning("Attempting to get enhanced state but no state has been stored yet")
        return {}
    
    return _enhanced_final_state

def clear_state() -> None:
    """Clear all stored states"""
    global _collected_states, _enhanced_final_state
    _collected_states = {}
    _enhanced_final_state = None
    logger.debug("All states cleared")

def _build_enhanced_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build enhanced state integrating all Agent analysis results
    
    Args:
        state: Final state dictionary
        
    Returns:
        Enhanced state dictionary containing all integrated data
    """
    # Extract basic information from state
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
        # Extract final decision from portfolio management
        last_message = state.get("messages", [])[-1]
        content = last_message.content if hasattr(last_message, 'content') else last_message
        if isinstance(content, str):
            try:
                # Try to parse JSON content
                decision = json.loads(content)
                enhanced["final_decision"] = decision
            except json.JSONDecodeError:
                # If not valid JSON, use string directly
                enhanced["final_decision"] = {"summary": content}
        elif isinstance(content, dict):
            enhanced["final_decision"] = content
        else:
            enhanced["final_decision"] = {"summary": str(content)}
            
        # Extract analysis results from each Agent
        for key, value in state.items():
            if key.endswith("_analysis") and isinstance(value, dict):
                agent_name = key.replace("_analysis", "")
                enhanced["analysis"][agent_name] = value
                
        # Extract Agent reasoning records from metadata
        agent_reasoning = state.get("metadata", {}).get("agent_reasoning", {})
        if agent_reasoning:
            for agent_name, reasoning in agent_reasoning.items():
                if agent_name not in enhanced["analysis"]:
                    enhanced["analysis"][agent_name] = {}
                enhanced["analysis"][agent_name]["reasoning"] = reasoning
                
    except Exception as e:
        logger.error(f"Error building enhanced state: {str(e)}")
        
    return enhanced