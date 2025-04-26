import os
import json
import time
import logging
import traceback
import pandas as pd
from typing import Any, Callable

logger = logging.getLogger(__name__)

# 数据缓存目录设置
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cached_data(key: str, fetch_func: Callable, *args, ttl_days=1, **kwargs) -> Any:
    """
    从缓存获取数据，如果缓存过期或不存在则调用fetch_func获取
    
    Args:
        key: 缓存键
        fetch_func: 获取数据的函数
        ttl_days: 缓存有效期（天）
        args, kwargs: 传递给fetch_func的参数
    
    Returns:
        获取的数据
    """
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    
    # 检查缓存是否存在且有效
    if os.path.exists(cache_file):
        # 检查文件修改时间
        file_time = os.path.getmtime(cache_file)
        file_age = (time.time() - file_time) / (60 * 60 * 24)  # 转换为天
        
        if file_age < ttl_days:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    logger.info(f"Using cached data for {key}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache file: {e}")
    
    # 缓存不存在、过期或无效，调用获取函数
    logger.info(f"Fetching fresh data for {key}")
    data = fetch_func(*args, **kwargs)
    
    # 保存到缓存
    try:
        # 处理DataFrame对象，转换为可序列化的字典列表
        if isinstance(data, pd.DataFrame):
            # 确保日期列被正确处理
            df_dict = data.copy()
            for col in df_dict.columns:
                if pd.api.types.is_datetime64_any_dtype(df_dict[col]):
                    df_dict[col] = df_dict[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 转换为字典列表
            serializable_data = df_dict.to_dict(orient='records')
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False)
        else:
            # 非DataFrame对象直接保存
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
                
        logger.info(f"Data cached to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save cache file: {e}")
        logger.debug(f"Cache error details: {traceback.format_exc()}")
    
    return data