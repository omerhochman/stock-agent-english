import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Union
import numpy as np

class DataProcessor:
    """数据处理类，提供数据清洗、转换和增强功能"""
    
    def process_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理价格数据，包括清洗、标准化和增强
        
        Args:
            df: 包含价格数据的DataFrame
            
        Returns:
            处理后的DataFrame
        """
        if df.empty or len(df) < 5:
            return df
        
        # 创建副本避免修改原始数据
        processed_df = df.copy()
        
        # 1. 标准化日期列
        processed_df = self._standardize_date_column(processed_df)
        
        # 2. 处理缺失值
        processed_df = self._handle_missing_values(processed_df)
        
        # 3. 计算派生指标
        processed_df = self._calculate_derived_metrics(processed_df)
        
        # 4. 最终数据清理
        processed_df = self._final_data_cleanup(processed_df)
        
        # 5. 排序数据
        if 'date' in processed_df.columns:
            processed_df = processed_df.sort_values('date', ascending=True)
        
        return processed_df
    
    def _standardize_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化日期列"""
        if 'date' not in df.columns:
            # 尝试查找日期列的其他可能名称
            date_columns = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', '日期'])]
            if date_columns:
                df = df.rename(columns={date_columns[0]: 'date'})
        
        # 确保日期列为datetime类型
        if 'date' in df.columns:
            if df['date'].dtype != 'datetime64[ns]':
                try:
                    df['date'] = pd.to_datetime(df['date'])
                except Exception as e:
                    print(f"警告：日期转换失败 - {e}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 价格列的缺失值处理
        price_columns = ['open', 'high', 'low', 'close']
        available_columns = [col for col in price_columns if col in df.columns]
        
        if available_columns:
            # 对于价格列，使用前向填充
            df[available_columns] = df[available_columns].ffill()
            
            # 如果仍有缺失值（比如序列开头），使用后向填充
            df[available_columns] = df[available_columns].bfill()
        
        # 成交量列的缺失值处理
        volume_columns = ['volume', 'amount']
        available_vol_columns = [col for col in volume_columns if col in df.columns]
        
        if available_vol_columns:
            # 对于成交量，缺失值用0填充
            df[available_vol_columns] = df[available_vol_columns].fillna(0)
        
        return df
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算派生指标"""
        # 检查必要的列是否存在
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            return df
        
        # 计算每日收益率
        if 'daily_return' not in df.columns:
            df['daily_return'] = df['close'].pct_change()
        
        # 计算真实波动幅度 - 修复潜在的广播错误
        if 'true_range' not in df.columns:
            high_low = df['high'] - df['low']
            high_close_prev = (df['high'] - df['close'].shift(1)).abs()
            low_close_prev = (df['low'] - df['close'].shift(1)).abs()
            
            # 确保所有序列长度一致，处理NaN值
            high_low = high_low.fillna(0)
            high_close_prev = high_close_prev.fillna(0)
            low_close_prev = low_close_prev.fillna(0)
            
            # 使用numpy的maximum函数而不是pandas的concat和max
            df['true_range'] = np.maximum.reduce([high_low, high_close_prev, low_close_prev])
        
        # 计算典型价格
        if 'typical_price' not in df.columns:
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # 计算移动平均线（包括系统需要的ma10）
        if 'ma5' not in df.columns:
            df['ma5'] = df['close'].rolling(window=5).mean()
        
        if 'ma10' not in df.columns:
            df['ma10'] = df['close'].rolling(window=10).mean()
        
        if 'ma20' not in df.columns:
            df['ma20'] = df['close'].rolling(window=20).mean()
        
        # 计算MACD指标
        if 'macd' not in df.columns:
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 计算RSI指标（使用正确的命名）
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            # 填充NaN值
            df['rsi'] = df['rsi'].fillna(50)
        
        return df
    
    def process_financial_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        处理财务数据，清洗并增强财务指标和报表数据
        
        Args:
            data: 财务数据，可以是字典或字典列表
            
        Returns:
            处理后的财务数据
        """
        if isinstance(data, dict):
            return self._process_single_financial_item(data)
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return [self._process_single_financial_item(item) for item in data]
        else:
            return data
    
    def _process_single_financial_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个财务数据项"""
        processed_item = item.copy()
        
        # 确保所有数值为float类型
        for key, value in processed_item.items():
            if isinstance(value, (int, float)):
                processed_item[key] = float(value)
            elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                processed_item[key] = float(value)
        
        # 计算派生指标（如果有足够的基础数据）
        if all(k in processed_item for k in ['operating_revenue', 'operating_profit']):
            processed_item['operating_margin'] = (
                processed_item['operating_profit'] / processed_item['operating_revenue'] 
                if processed_item['operating_revenue'] != 0 else 0
            )
        
        if all(k in processed_item for k in ['net_income', 'operating_revenue']):
            processed_item['net_margin'] = (
                processed_item['net_income'] / processed_item['operating_revenue']
                if processed_item['operating_revenue'] != 0 else 0
            )
        
        # 添加时间戳
        if 'timestamp' not in processed_item:
            processed_item['timestamp'] = datetime.now().isoformat()
        
        return processed_item
    
    def enrich_data(self, df: pd.DataFrame, technical_indicators: bool = True) -> pd.DataFrame:
        """
        增强数据，添加额外的技术指标
        
        Args:
            df: 原始数据DataFrame
            technical_indicators: 是否添加技术指标
            
        Returns:
            增强后的DataFrame
        """
        if df.empty:
            return df
        
        enhanced_df = df.copy()
        
        # 添加技术指标
        if technical_indicators and 'close' in enhanced_df.columns:
            # 计算MACD
            enhanced_df = self._add_macd(enhanced_df)
            
            # 计算RSI
            enhanced_df = self._add_rsi(enhanced_df)
            
            # 计算布林带
            enhanced_df = self._add_bollinger_bands(enhanced_df)
        
        return enhanced_df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加MACD指标"""
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df
    
    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加RSI指标"""
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            # 填充NaN值
            df['rsi'] = df['rsi'].fillna(50)
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加布林带指标"""
        df['20ma'] = df['close'].rolling(window=20).mean()
        df['upper_band'] = df['20ma'] + (df['close'].rolling(window=20).std() * 2)
        df['lower_band'] = df['20ma'] - (df['close'].rolling(window=20).std() * 2)
        return df
    
    def _final_data_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """最终数据清理，处理无穷大值和异常值"""
        # 替换无穷大值
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 处理数值列的异常值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df.columns:
                # 计算合理的范围（使用分位数）
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                
                # 限制极端值
                if pd.notna(q1) and pd.notna(q99) and q99 > q1:
                    df[col] = df[col].clip(lower=q1, upper=q99)
                
                # 填充剩余的NaN值
                df[col] = df[col].fillna(df[col].median())
        
        return df

# 创建单例实例
data_processor = DataProcessor()