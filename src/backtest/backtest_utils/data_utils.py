import pandas as pd
import numpy as np
from typing import List

class DataProcessor:
    """
    数据处理器
    提供数据清理、转换和预处理功能
    """
    
    def __init__(self):
        pass
    
    def clean_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清理价格数据
        
        Args:
            data: 原始价格数据
            
        Returns:
            pd.DataFrame: 清理后的数据
        """
        if data is None or data.empty:
            return pd.DataFrame()
        
        # 删除重复行
        data = data.drop_duplicates()
        
        # 处理缺失值
        data = data.dropna()
        
        # 确保价格列为数值类型
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in price_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 删除异常值（价格为0或负数）
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                data = data[data[col] > 0]
        
        # 确保高价≥低价
        if 'high' in data.columns and 'low' in data.columns:
            data = data[data['high'] >= data['low']]
        
        # 按日期排序
        if data.index.name != 'date' and 'date' in data.columns:
            data = data.set_index('date')
        
        data = data.sort_index()
        
        return data
    
    def calculate_returns(self, prices: pd.Series, method: str = 'simple') -> pd.Series:
        """
        计算收益率
        
        Args:
            prices: 价格序列
            method: 计算方法 ('simple' 或 'log')
            
        Returns:
            pd.Series: 收益率序列
        """
        if method == 'simple':
            returns = prices.pct_change()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError("method must be 'simple' or 'log'")
        
        return returns.dropna()
    
    def align_data(self, *dataframes: pd.DataFrame) -> List[pd.DataFrame]:
        """
        对齐多个数据框的时间索引
        
        Args:
            *dataframes: 要对齐的数据框
            
        Returns:
            List[pd.DataFrame]: 对齐后的数据框列表
        """
        if len(dataframes) < 2:
            return list(dataframes)
        
        # 找到共同的时间索引
        common_index = dataframes[0].index
        for df in dataframes[1:]:
            common_index = common_index.intersection(df.index)
        
        # 对齐所有数据框
        aligned_dfs = []
        for df in dataframes:
            aligned_df = df.reindex(common_index)
            aligned_dfs.append(aligned_df)
        
        return aligned_dfs
    
    def resample_data(self, data: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
        重采样数据到指定频率
        
        Args:
            data: 原始数据
            frequency: 目标频率 ('D', 'W', 'M' 等)
            
        Returns:
            pd.DataFrame: 重采样后的数据
        """
        if data.empty:
            return data
        
        # 价格数据的聚合规则
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # 只使用存在的列
        available_rules = {col: rule for col, rule in agg_rules.items() 
                          if col in data.columns}
        
        resampled = data.resample(frequency).agg(available_rules)
        
        return resampled.dropna()