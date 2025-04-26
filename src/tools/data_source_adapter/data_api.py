import pandas as pd
from typing import Dict, Any
from .adapter import DataSourceAdapter

class DataAPI:
    """统一的数据API接口，封装内部数据源适配器实现"""
    
    def __init__(self):
        self.adapter = DataSourceAdapter()
    
    def get_price_data(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取股票价格数据
        
        Args:
            ticker: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        Returns:
            包含价格数据的DataFrame
        """
        return self.adapter.get_price_history(ticker, start_date, end_date)
    
    def get_financial_metrics(self, ticker: str) -> Dict[str, Any]:
        """
        获取财务指标数据
        
        Args:
            ticker: 股票代码
            
        Returns:
            包含财务指标的字典
        """
        return self.adapter.get_financial_metrics(ticker)
    
    def get_financial_statements(self, ticker: str) -> Dict[str, Any]:
        """
        获取财务报表数据
        
        Args:
            ticker: 股票代码
            
        Returns:
            包含财务报表数据的字典
        """
        return self.adapter.get_financial_statements(ticker)
    
    def get_market_data(self, ticker: str) -> Dict[str, Any]:
        """
        获取市场数据
        
        Args:
            ticker: 股票代码
            
        Returns:
            包含市场数据的字典
        """
        return self.adapter.get_market_data(ticker)