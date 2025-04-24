import pandas as pd
import numpy as np
from typing import Dict, List
import logging

# 设置日志记录
logger = logging.getLogger('data_processor')

class DataProcessor:
    """
    数据处理类，用于增强数据质量
    """
    def __init__(self):
        self.outlier_methods = {
            "zscore": self._detect_outliers_zscore,
            "iqr": self._detect_outliers_iqr,
            "percentile": self._detect_outliers_percentile
        }
        self.imputation_methods = {
            "ffill": lambda df, col: df[col].fillna(method='ffill'),
            "bfill": lambda df, col: df[col].fillna(method='bfill'),
            "mean": lambda df, col: df[col].fillna(df[col].mean()),
            "median": lambda df, col: df[col].fillna(df[col].median()),
            "interpolate": lambda df, col: df[col].interpolate(method='linear')
        }
    
    def process_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理价格数据，提高质量
        
        Args:
            df: 原始价格DataFrame
            
        Returns:
            处理后的价格DataFrame
        """
        if df is None or df.empty:
            logger.warning("输入数据为空，无法处理")
            return pd.DataFrame()
        
        # 创建副本以避免修改原始数据
        processed_df = df.copy()
        
        # 1. 检查并修复日期索引
        processed_df = self._fix_date_index(processed_df)
        
        # 2. 处理缺失值
        missing_columns = processed_df.columns[processed_df.isna().any()].tolist()
        if missing_columns:
            logger.info(f"检测到以下列有缺失值：{missing_columns}")
            for col in missing_columns:
                if col in ['open', 'high', 'low', 'close']:
                    # 价格列使用插值
                    processed_df[col] = self.imputation_methods["interpolate"](processed_df, col)
                elif col == 'volume':
                    # 成交量使用0填充
                    processed_df[col] = processed_df[col].fillna(0)
                else:
                    # 其他列使用前向填充
                    processed_df[col] = self.imputation_methods["ffill"](processed_df, col)
            
            # 检查是否仍有缺失值
            if processed_df.isna().any().any():
                logger.warning("部分缺失值无法修复，使用向前填充")
                processed_df = processed_df.fillna(method='ffill').fillna(method='bfill')
        
        # 3. 检测并处理异常值
        for col in ['open', 'high', 'low', 'close']:
            if col in processed_df.columns:
                # 使用IQR方法检测异常值
                outliers = self.outlier_methods["iqr"](processed_df[col])
                if len(outliers) > 0:
                    logger.info(f"列 {col} 中检测到 {len(outliers)} 个异常值")
                    # 使用插值替换异常值
                    for idx in outliers:
                        processed_df.loc[idx, col] = np.nan
                    processed_df[col] = processed_df[col].interpolate(method='linear')
        
        # 4. 验证OHLC数据合理性
        processed_df = self._validate_ohlc(processed_df)
        
        # 5. 添加日期特征
        processed_df = self._add_date_features(processed_df)
        
        # 6. 确保数据类型正确
        processed_df = self._ensure_data_types(processed_df)
        
        return processed_df
    
    def process_financial_data(self, data: List[Dict]) -> List[Dict]:
        """
        处理财务数据，提高质量
        
        Args:
            data: 财务数据列表
            
        Returns:
            处理后的财务数据列表
        """
        if not data:
            logger.warning("输入财务数据为空，无法处理")
            return []
        
        processed_data = []
        
        for item in data:
            # 创建副本避免修改原始数据
            processed_item = item.copy()
            
            # 1. 处理无效值 (None, NaN, 极端值)
            for key, value in processed_item.items():
                # 跳过非数值字段
                if not isinstance(value, (int, float)):
                    continue
                
                # 替换None和NaN
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    processed_item[key] = 0
                    continue
                
                # 检查是否为极端值 (针对财务比率)
                if key in ['return_on_equity', 'net_margin', 'operating_margin', 
                          'revenue_growth', 'earnings_growth', 'book_value_growth']:
                    # 财务比率通常不应超过±100%
                    if abs(value) > 1:
                        logger.warning(f"检测到极端财务比率: {key}={value}, 调整为有效范围")
                        processed_item[key] = np.clip(value, -1, 1)
                
                # 特定指标的合理性检查
                if key == 'pe_ratio' and value < 0:
                    # 负PE意味着亏损，设置为较高值表示风险
                    processed_item[key] = 100
                elif key == 'pe_ratio' and value > 200:
                    # 超高PE可能是数据错误
                    processed_item[key] = 200
                    
                if key == 'price_to_book' and value < 0:
                    # 负PB通常是错误，设为合理的低值
                    processed_item[key] = 0.1
                elif key == 'price_to_book' and value > 50:
                    # 超高PB可能是数据错误
                    processed_item[key] = 50
            
            # 2. 计算缺失的派生指标
            # 如果能计算但缺失，则计算
            if ('net_income' in processed_item and 'total_revenue' in processed_item and
                processed_item.get('total_revenue', 0) != 0 and
                'net_margin' not in processed_item):
                processed_item['net_margin'] = (processed_item['net_income'] / 
                                              processed_item['total_revenue'])
            
            processed_data.append(processed_item)
        
        return processed_data
    
    def _fix_date_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """修复日期索引，确保连续性"""
        if 'date' in df.columns:
            # 如果日期在列中，转换为索引
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        elif not isinstance(df.index, pd.DatetimeIndex):
            # 如果索引不是日期类型
            logger.warning("数据没有日期索引，无法修复日期连续性")
            return df
        
        # 确保日期升序排序
        df = df.sort_index()
        
        # 检查是否有重复的日期索引
        if df.index.duplicated().any():
            logger.warning("检测到重复的日期索引，保留最后一条记录")
            df = df[~df.index.duplicated(keep='last')]
        
        # 检查交易日期的连续性
        # 注意：只在工作日填充（周一至周五）
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        if len(date_range) > len(df):
            logger.info(f"检测到 {len(date_range) - len(df)} 个缺失的交易日")
            # 使用完整的交易日索引重新索引，并向前填充缺失值
            df = df.reindex(date_range).fillna(method='ffill')
        
        return df
    
    def _validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证并修复OHLC数据的合理性"""
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            logger.warning("缺少OHLC列，跳过OHLC验证")
            return df
        
        # 创建副本以避免修改原始数据
        validated_df = df.copy()
        
        # 验证high应该是当天最高价
        inconsistent_high = validated_df[validated_df['high'] < validated_df[['open', 'close']].max(axis=1)]
        if not inconsistent_high.empty:
            logger.warning(f"检测到 {len(inconsistent_high)} 行的high值不是当天最高")
            # 修复：设置high为open和close的最大值
            validated_df.loc[inconsistent_high.index, 'high'] = validated_df.loc[
                inconsistent_high.index, ['open', 'close']].max(axis=1)
        
        # 验证low应该是当天最低价
        inconsistent_low = validated_df[validated_df['low'] > validated_df[['open', 'close']].min(axis=1)]
        if not inconsistent_low.empty:
            logger.warning(f"检测到 {len(inconsistent_low)} 行的low值不是当天最低")
            # 修复：设置low为open和close的最小值
            validated_df.loc[inconsistent_low.index, 'low'] = validated_df.loc[
                inconsistent_low.index, ['open', 'close']].min(axis=1)
        
        # 验证成交量不应为负
        if 'volume' in validated_df.columns:
            negative_volume = validated_df[validated_df['volume'] < 0]
            if not negative_volume.empty:
                logger.warning(f"检测到 {len(negative_volume)} 行的成交量为负")
                # 修复：将负成交量设为0
                validated_df.loc[negative_volume.index, 'volume'] = 0
        
        return validated_df
    
    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加日期相关特征"""
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("数据没有日期索引，无法添加日期特征")
            return df
        
        # 创建副本以避免修改原始数据
        enhanced_df = df.copy()
        
        # 添加日期特征
        enhanced_df['year'] = enhanced_df.index.year
        enhanced_df['month'] = enhanced_df.index.month
        enhanced_df['day'] = enhanced_df.index.day
        enhanced_df['day_of_week'] = enhanced_df.index.dayofweek
        enhanced_df['is_month_end'] = enhanced_df.index.is_month_end.astype(int)
        enhanced_df['is_month_start'] = enhanced_df.index.is_month_start.astype(int)
        
        return enhanced_df
    
    def _ensure_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保数据类型正确"""
        type_map = {
            'open': float, 'high': float, 'low': float, 'close': float,
            'volume': float, 'amount': float, 'change_percent': float
        }
        
        for col, dtype in type_map.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"转换 {col} 为 {dtype} 失败: {e}")
        
        return df
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> List[int]:
        """使用Z-score方法检测异常值"""
        if series.empty:
            return []
        
        z_scores = (series - series.mean()) / series.std()
        return series.index[np.abs(z_scores) > threshold].tolist()
    
    def _detect_outliers_iqr(self, series: pd.Series, k: float = 1.5) -> List:
        """使用IQR方法检测异常值"""
        if series.empty:
            return []
        
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        return series.index[(series < lower_bound) | (series > upper_bound)].tolist()
    
    def _detect_outliers_percentile(self, series: pd.Series, 
                                   lower: float = 0.01, upper: float = 0.99) -> List:
        """使用百分位方法检测异常值"""
        if series.empty:
            return []
        
        lower_bound = series.quantile(lower)
        upper_bound = series.quantile(upper)
        
        return series.index[(series < lower_bound) | (series > upper_bound)].tolist()


# 创建单例实例
data_processor = DataProcessor()