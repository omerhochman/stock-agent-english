import json
import re
from typing import Any

from src.utils.logging_config import SUCCESS_ICON, ERROR_ICON

def ensure_ascii_false(obj: Any) -> str:
    """
    Serialize object to JSON string, ensuring Chinese characters are not encoded as Unicode escape sequences
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string ensuring Chinese characters display normally
    """
    return json.dumps(obj, ensure_ascii=False, indent=2)

def decode_unicode_escaped_string(s: str) -> str:
    """
    Decode Unicode escape sequences in string
    
    Args:
        s: String that may contain Unicode escape sequences
        
    Returns:
        Decoded string
    """
    if not isinstance(s, str):
        return s
        
    # Match Unicode escape sequences like \u4e2d\u6587
    def replace_unicode(match):
        try:
            return bytes(match.group(0), 'utf-8').decode('unicode_escape')
        except UnicodeDecodeError:
            return match.group(0)
            
    return re.sub(r'\\u[0-9a-fA-F]{4}', replace_unicode, s)

def decode_unicode_in_obj(obj: Any) -> Any:
    """
    Recursively decode Unicode escape sequences in all strings in object
    
    Args:
        obj: Object to process, can be dictionary, list, string, etc.
        
    Returns:
        Processed object with all Unicode escape sequences in strings decoded
    """
    if isinstance(obj, dict):
        return {k: decode_unicode_in_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_unicode_in_obj(item) for item in obj]
    elif isinstance(obj, str):
        return decode_unicode_escaped_string(obj)
    else:
        return obj

def patch_json_dumps():
    """
    Modify default json.dumps behavior to ensure Chinese characters are not encoded as Unicode escape sequences
    """
    original_dumps = json.dumps
    
    def patched_dumps(*args, **kwargs):
        # If ensure_ascii parameter is not explicitly set, set it to False
        if 'ensure_ascii' not in kwargs:
            kwargs['ensure_ascii'] = False
        return original_dumps(*args, **kwargs)
    
    json.dumps = patched_dumps

def patch_agent_json_methods(agent_class: Any):
    """
    Patch agent class JSON serialization methods to ensure Chinese characters display normally
    
    Args:
        agent_class: Agent class to patch
    """
    if not hasattr(agent_class, 'to_json'):
        return
        
    original_to_json = agent_class.to_json
    
    def patched_to_json(self, *args, **kwargs):
        result = original_to_json(self, *args, **kwargs)
        if isinstance(result, str):
            # If return value is string, first try to parse as object, process then serialize
            try:
                obj = json.loads(result)
                decoded_obj = decode_unicode_in_obj(obj)
                return json.dumps(decoded_obj, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                # If not valid JSON string, directly decode string
                return decode_unicode_escaped_string(result)
        else:
            # If return value is object, process directly
            return decode_unicode_in_obj(result)
    
    agent_class.to_json = patched_to_json

def monkey_patch_all_agents():
    """
    Traverse and patch JSON serialization methods for all agent classes in the project
    """
    # Modify default json.dumps behavior
    patch_json_dumps()
    
    # Try to import all agent modules
    try:
        from src.agents import (
            valuation, fundamentals, sentiment, risk_manager, 
            technicals, portfolio_manager, market_data,
            researcher_bull, researcher_bear, debate_room, macro_analyst,
            portfolio_analyzer, ai_model_analyst,
        )
        
        # Patch JSON methods for each agent class
        agent_modules = [
            valuation, fundamentals, sentiment, risk_manager, 
            technicals, portfolio_manager, market_data,
            researcher_bull, researcher_bear, debate_room, macro_analyst,
            portfolio_analyzer, ai_model_analyst,
        ]
        
        for module in agent_modules:
            for name in dir(module):
                if name.endswith('Agent') or name.endswith('_agent'):
                    attr = getattr(module, name)
                    if callable(attr):
                        patch_agent_json_methods(attr)
                        
        print(f"{SUCCESS_ICON} Successfully patched JSON serialization methods for all agent classes")
        
    except ImportError as e:
        print(f"{ERROR_ICON} Failed to import agent modules: {e}")
        print("Please ensure project structure is correct and run this script from project root directory")