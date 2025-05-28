import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from src.tools.api import get_price_history

class RiskFactorAPI:
    """风险因子API，提供风险因子数据的计算和获取功能"""
    
    def calculate_beta(self, symbol: str, market_index: str = 'sh000300',
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> float:
        """
        计算股票相对于市场指数的贝塔系数
        
        Args:
            symbol: 股票代码
            market_index: 市场指数代码，默认沪深300
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        Returns:
            贝塔系数
        """
        # 处理日期参数
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - 
                        timedelta(days=365)).strftime('%Y-%m-%d')
        
        # 获取股票价格数据
        stock_df = get_price_history(symbol, start_date, end_date)
        
        # 获取指数价格数据
        market_df = get_price_history(market_index, start_date, end_date)
        
        if stock_df.empty or market_df.empty:
            return 0.0
        
        # 计算股票日收益率
        if 'daily_return' not in stock_df.columns and 'close' in stock_df.columns:
            stock_df['daily_return'] = stock_df['close'].pct_change().fillna(0)
        
        # 计算市场指数日收益率
        if 'daily_return' not in market_df.columns and 'close' in market_df.columns:
            market_df['daily_return'] = market_df['close'].pct_change().fillna(0)
        
        # 确保数据对齐
        merged_df = pd.DataFrame()
        merged_df['stock_return'] = stock_df['daily_return'].reset_index(drop=True)
        merged_df['market_return'] = market_df['daily_return'].reset_index(drop=True)
        
        # 移除缺失值
        merged_df = merged_df.dropna()
        
        # 计算贝塔系数
        if len(merged_df) > 30:  # 确保有足够的数据点
            # 使用协方差法计算贝塔
            covariance = merged_df['stock_return'].cov(merged_df['market_return'])
            market_variance = merged_df['market_return'].var()
            
            # 避免除零错误
            if market_variance != 0:
                beta = covariance / market_variance
                return beta
        
        return 1.0  # 默认贝塔为1
    
    def calculate_alpha(self, symbol: str, market_index: str = 'sh000300',
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      risk_free_rate: float = 0.03) -> float:
        """
        计算股票的阿尔法值
        
        Args:
            symbol: 股票代码
            market_index: 市场指数代码，默认沪深300
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            risk_free_rate: 无风险利率，默认3%
            
        Returns:
            阿尔法值
        """
        # 处理日期参数
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - 
                        timedelta(days=365)).strftime('%Y-%m-%d')
        
        # 获取股票价格数据
        stock_df = get_price_history(symbol, start_date, end_date)
        
        # 获取指数价格数据
        market_df = get_price_history(market_index, start_date, end_date)
        
        if stock_df.empty or market_df.empty:
            return 0.0
        
        # 计算股票累积收益率
        if 'close' in stock_df.columns:
            start_price = stock_df['close'].iloc[0]
            end_price = stock_df['close'].iloc[-1]
            stock_return = (end_price / start_price) - 1
        else:
            return 0.0
        
        # 计算市场累积收益率
        if 'close' in market_df.columns:
            start_price = market_df['close'].iloc[0]
            end_price = market_df['close'].iloc[-1]
            market_return = (end_price / start_price) - 1
        else:
            return 0.0
        
        # 计算贝塔系数
        beta = self.calculate_beta(symbol, market_index, start_date, end_date)
        
        # 计算阿尔法
        days = (stock_df['date'].iloc[-1] - stock_df['date'].iloc[0]).days
        annualized_rf = (1 + risk_free_rate) ** (days / 365) - 1
        
        alpha = stock_return - (annualized_rf + beta * (market_return - annualized_rf))
        
        return alpha
    
    def calculate_volatility(self, symbol: str, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           window: int = 20) -> float:
        """
        计算股票的波动率
        
        Args:
            symbol: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            window: 滚动窗口大小，默认20
            
        Returns:
            年化波动率
        """
        # 获取价格数据
        df = get_price_history(symbol, start_date, end_date)
        
        if df.empty or 'close' not in df.columns:
            return 0.0
        
        # 计算日收益率
        df['daily_return'] = df['close'].pct_change().fillna(0)
        
        # 计算滚动波动率
        df['volatility'] = df['daily_return'].rolling(window=window).std() * np.sqrt(252)
        
        # 返回最新的波动率
        latest_volatility = df['volatility'].dropna().iloc[-1] if not df['volatility'].dropna().empty else 0.0
        
        return latest_volatility
    
    def calculate_sharpe_ratio(self, symbol: str,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             risk_free_rate: float = 0.03) -> float:
        """
        计算股票的夏普比率
        
        Args:
            symbol: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            risk_free_rate: 无风险利率，默认3%
            
        Returns:
            夏普比率
        """
        # 获取价格数据
        df = get_price_history(symbol, start_date, end_date)
        
        if df.empty or 'close' not in df.columns:
            return 0.0
        
        # 计算日收益率
        df['daily_return'] = df['close'].pct_change().fillna(0)
        
        # 计算年化收益率
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        annualized_return = ((1 + df['daily_return']).prod()) ** (252 / len(df)) - 1
        
        # 计算年化波动率
        annualized_volatility = df['daily_return'].std() * np.sqrt(252)
        
        # 计算夏普比率
        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
        excess_return = annualized_return - risk_free_rate
        
        # 检查数值稳定性
        if annualized_volatility == 0 or np.isclose(annualized_volatility, 0, atol=1e-10):
            return 0.0
        
        if not np.isfinite(annualized_volatility) or not np.isfinite(excess_return):
            return 0.0
        
        sharpe_ratio = excess_return / annualized_volatility
        
        # 确保返回有限值
        if not np.isfinite(sharpe_ratio):
            return 0.0
        
        return sharpe_ratio
    
    def calculate_maximum_drawdown(self, symbol: str,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> float:
        """
        计算股票的最大回撤
        
        Args:
            symbol: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        Returns:
            最大回撤比例
        """
        # 获取价格数据
        df = get_price_history(symbol, start_date, end_date)
        
        if df.empty or 'close' not in df.columns:
            return 0.0
        
        # 计算累积最大值
        df['cum_max'] = df['close'].cummax()
        
        # 计算回撤比例
        df['drawdown'] = (df['cum_max'] - df['close']) / df['cum_max']
        
        # 获取最大回撤
        max_drawdown = df['drawdown'].max()
        
        return max_drawdown