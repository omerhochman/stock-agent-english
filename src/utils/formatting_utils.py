import pandas as pd
from typing import Dict


def format_market_data_summary(market_data: Dict) -> Dict:
    """
    格式化市场数据，提供简洁的摘要统计
    
    Args:
        market_data: 原始市场数据
        
    Returns:
        简化后的市场数据摘要
    """
    formatted_data = {}
    
    # 处理市场收益率
    if "market_returns" in market_data:
        if isinstance(market_data["market_returns"], str):
            # 如果已经是字符串，尝试解析为Series
            try:
                import io
                data = io.StringIO(market_data["market_returns"])
                series = pd.read_csv(data, header=None, index_col=0, squeeze=True)
                formatted_data["market_returns_mean"] = float(series.mean())
                formatted_data["market_returns_std"] = float(series.std())
            except:
                # 如果解析失败，只保留前5个字符提示
                formatted_data["market_returns"] = "[数据过长，已省略]"
        elif isinstance(market_data["market_returns"], pd.Series):
            formatted_data["market_returns_mean"] = float(market_data["market_returns"].mean())
            formatted_data["market_returns_std"] = float(market_data["market_returns"].std())
    
    # 处理股票收益率
    if "stock_returns" in market_data:
        if isinstance(market_data["stock_returns"], str):
            try:
                import io
                data = io.StringIO(market_data["stock_returns"])
                series = pd.read_csv(data, header=None, index_col=0, squeeze=True)
                formatted_data["stock_returns_mean"] = float(series.mean())
                formatted_data["stock_returns_std"] = float(series.std())
            except:
                formatted_data["stock_returns"] = "[数据过长，已省略]"
        elif isinstance(market_data["stock_returns"], pd.Series):
            formatted_data["stock_returns_mean"] = float(market_data["stock_returns"].mean())
            formatted_data["stock_returns_std"] = float(market_data["stock_returns"].std())
    
    # 保留其他重要的市场数据
    for key in ["market_volatility", "stock_volatility", "beta"]:
        if key in market_data:
            formatted_data[key] = market_data[key]
    
    return formatted_data


def format_float_as_percentage(value: float, decimal_places: int = 2) -> str:
    """将浮点数格式化为百分比字符串"""
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value*100:.{decimal_places}f}%"


def format_currency(value: float, decimal_places: int = 2, currency_symbol: str = "$") -> str:
    """将数值格式化为货币字符串"""
    if not isinstance(value, (int, float)):
        return str(value)
    
    # 对大数使用更友好的表示
    if abs(value) >= 1_000_000_000:  # 十亿
        return f"{currency_symbol}{value/1_000_000_000:.{decimal_places}f}B"
    elif abs(value) >= 1_000_000:  # 百万
        return f"{currency_symbol}{value/1_000_000:.{decimal_places}f}M"
    elif abs(value) >= 1_000:  # 千
        return f"{currency_symbol}{value/1_000:.{decimal_places}f}K"
    else:
        return f"{currency_symbol}{value:.{decimal_places}f}"