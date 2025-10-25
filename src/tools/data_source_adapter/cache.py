import os
import json
import time
import traceback
import pandas as pd
from typing import Any, Callable

from src.utils.logging_config import setup_logger

logger = setup_logger('cache')

# Data cache directory setup
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cached_data(key: str, fetch_func: Callable, *args, ttl_days=1, **kwargs) -> Any:
    """
    Get data from cache, if cache is expired or doesn't exist, call fetch_func to get data
    
    Args:
        key: Cache key
        fetch_func: Function to fetch data
        ttl_days: Cache validity period (days)
        args, kwargs: Parameters passed to fetch_func
    
    Returns:
        Retrieved data
    """
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    
    # Check if cache exists and is valid
    if os.path.exists(cache_file):
        # Check file modification time
        file_time = os.path.getmtime(cache_file)
        file_age = (time.time() - file_time) / (60 * 60 * 24)  # Convert to days
        
        if file_age < ttl_days:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    logger.info(f"Using cached data for {key}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache file: {e}")
    
    # Cache doesn't exist, expired or invalid, call fetch function
    logger.info(f"Fetching fresh data for {key}")
    data = fetch_func(*args, **kwargs)
    
    # Save to cache
    try:
        # Handle DataFrame objects, convert to serializable dictionary list
        if isinstance(data, pd.DataFrame):
            # Ensure date columns are properly handled
            df_dict = data.copy()
            for col in df_dict.columns:
                if pd.api.types.is_datetime64_any_dtype(df_dict[col]):
                    df_dict[col] = df_dict[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Convert to dictionary list
            serializable_data = df_dict.to_dict(orient='records')
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False)
        else:
            # Non-DataFrame objects save directly
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
                
        logger.info(f"Data cached to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save cache file: {e}")
        logger.debug(f"Cache error details: {traceback.format_exc()}")
    
    return data