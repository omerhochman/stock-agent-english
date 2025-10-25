import pandas as pd
from typing import Dict


def format_market_data_summary(market_data: Dict) -> Dict:
    """
    Format market data to provide concise summary statistics
    
    Args:
        market_data: Raw market data
        
    Returns:
        Simplified market data summary
    """
    formatted_data = {}
    
    # Process market returns
    if "market_returns" in market_data:
        if isinstance(market_data["market_returns"], str):
            # If already a string, try to parse as Series
            try:
                import io
                data = io.StringIO(market_data["market_returns"])
                series = pd.read_csv(data, header=None, index_col=0, squeeze=True)
                formatted_data["market_returns_mean"] = float(series.mean())
                formatted_data["market_returns_std"] = float(series.std())
            except:
                # If parsing fails, keep only first 5 characters as hint
                formatted_data["market_returns"] = "[Data too long, omitted]"
        elif isinstance(market_data["market_returns"], pd.Series):
            formatted_data["market_returns_mean"] = float(market_data["market_returns"].mean())
            formatted_data["market_returns_std"] = float(market_data["market_returns"].std())
    
    # Process stock returns
    if "stock_returns" in market_data:
        if isinstance(market_data["stock_returns"], str):
            try:
                import io
                data = io.StringIO(market_data["stock_returns"])
                series = pd.read_csv(data, header=None, index_col=0, squeeze=True)
                formatted_data["stock_returns_mean"] = float(series.mean())
                formatted_data["stock_returns_std"] = float(series.std())
            except:
                formatted_data["stock_returns"] = "[Data too long, omitted]"
        elif isinstance(market_data["stock_returns"], pd.Series):
            formatted_data["stock_returns_mean"] = float(market_data["stock_returns"].mean())
            formatted_data["stock_returns_std"] = float(market_data["stock_returns"].std())
    
    # Keep other important market data
    for key in ["market_volatility", "stock_volatility", "beta"]:
        if key in market_data:
            formatted_data[key] = market_data[key]
    
    return formatted_data


def format_float_as_percentage(value: float, decimal_places: int = 2) -> str:
    """Format float as percentage string"""
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value*100:.{decimal_places}f}%"


def format_currency(value: float, decimal_places: int = 2, currency_symbol: str = "$") -> str:
    """Format value as currency string"""
    if not isinstance(value, (int, float)):
        return str(value)
    
    # Use more friendly representation for large numbers
    if abs(value) >= 1_000_000_000:  # Billion
        return f"{currency_symbol}{value/1_000_000_000:.{decimal_places}f}B"
    elif abs(value) >= 1_000_000:  # Million
        return f"{currency_symbol}{value/1_000_000:.{decimal_places}f}M"
    elif abs(value) >= 1_000:  # Thousand
        return f"{currency_symbol}{value/1_000:.{decimal_places}f}K"
    else:
        return f"{currency_symbol}{value:.{decimal_places}f}"