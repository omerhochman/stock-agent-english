import json
import re
from typing import Any

from src.utils.logging_config import SUCCESS_ICON, ERROR_ICON

def ensure_ascii_false(obj: Any) -> str:
    """
    将对象序列化为JSON字符串，确保中文字符不会被编码为Unicode转义序列
    
    Args:
        obj: 要序列化的对象
        
    Returns:
        确保中文正常显示的JSON字符串
    """
    return json.dumps(obj, ensure_ascii=False, indent=2)

def decode_unicode_escaped_string(s: str) -> str:
    """
    解码字符串中的Unicode转义序列
    
    Args:
        s: 可能包含Unicode转义序列的字符串
        
    Returns:
        解码后的字符串
    """
    if not isinstance(s, str):
        return s
        
    # 匹配Unicode转义序列如 \u4e2d\u6587
    def replace_unicode(match):
        try:
            return bytes(match.group(0), 'utf-8').decode('unicode_escape')
        except UnicodeDecodeError:
            return match.group(0)
            
    return re.sub(r'\\u[0-9a-fA-F]{4}', replace_unicode, s)

def decode_unicode_in_obj(obj: Any) -> Any:
    """
    递归解码对象中所有字符串的Unicode转义序列
    
    Args:
        obj: 要处理的对象，可以是字典、列表、字符串等
        
    Returns:
        处理后的对象，所有字符串中的Unicode转义序列都被解码
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
    修改默认的json.dumps行为，确保中文字符不会被编码为Unicode转义序列
    """
    original_dumps = json.dumps
    
    def patched_dumps(*args, **kwargs):
        # 如果没有显式设置ensure_ascii参数，则设置为False
        if 'ensure_ascii' not in kwargs:
            kwargs['ensure_ascii'] = False
        return original_dumps(*args, **kwargs)
    
    json.dumps = patched_dumps

def patch_agent_json_methods(agent_class: Any):
    """
    修补代理类的JSON序列化方法，确保中文正常显示
    
    Args:
        agent_class: 要修补的代理类
    """
    if not hasattr(agent_class, 'to_json'):
        return
        
    original_to_json = agent_class.to_json
    
    def patched_to_json(self, *args, **kwargs):
        result = original_to_json(self, *args, **kwargs)
        if isinstance(result, str):
            # 如果返回值是字符串，先尝试解析为对象，处理后再序列化
            try:
                obj = json.loads(result)
                decoded_obj = decode_unicode_in_obj(obj)
                return json.dumps(decoded_obj, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                # 如果不是有效的JSON字符串，直接解码字符串
                return decode_unicode_escaped_string(result)
        else:
            # 如果返回值是对象，直接处理
            return decode_unicode_in_obj(result)
    
    agent_class.to_json = patched_to_json

def monkey_patch_all_agents():
    """
    遍历并修补项目中所有代理类的JSON序列化方法
    """
    # 修改默认的json.dumps行为
    patch_json_dumps()
    
    # 尝试导入所有代理模块
    try:
        from src.agents import (
            valuation, fundamentals, sentiment, risk_manager, 
            technicals, portfolio_manager, market_data,
            researcher_bull, researcher_bear, debate_room, macro_analyst,
            portfolio_analyzer, ai_model_analyst,
        )
        
        # 修补各个代理类的JSON方法
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
                        
        print(f"{SUCCESS_ICON} 已成功修补所有代理类的JSON序列化方法")
        
    except ImportError as e:
        print(f"{ERROR_ICON} 导入代理模块失败: {e}")
        print("请确保项目结构正确，并在项目根目录运行此脚本")