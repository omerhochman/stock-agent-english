"""
模型估计API - 提供各类金融模型的估计功能
"""

import pandas as pd
import numpy as np
import traceback
from typing import Dict, List

from .base import logger
from .market_data_api import get_market_returns, get_stock_returns, get_multi_stock_returns
from .fama_french_api import get_fama_french_factors
from .risk_free_api import get_risk_free_rate

def estimate_capm_for_stock(stock_symbol: str,
                           start_date: str = None,
                           end_date: str = None,
                           freq: str = 'D') -> Dict[str, float]:
    """
    为单个股票估计CAPM模型参数
    
    Args:
        stock_symbol: 股票代码
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        CAPM模型参数字典
    """
    logger.info(f"为股票 {stock_symbol} 估计CAPM模型参数: {start_date} 至 {end_date}, 频率: {freq}")
    
    try:
        # 1. 获取股票收益率
        stock_returns_dict = get_stock_returns(stock_symbol, start_date, end_date, freq)
        if not stock_returns_dict or stock_symbol not in stock_returns_dict:
            logger.warning(f"无法获取股票 {stock_symbol} 收益率数据")
            return {}
        
        stock_returns = stock_returns_dict[stock_symbol]
        
        # 2. 获取市场收益率 (使用沪深300)
        market_returns = get_market_returns("000300", start_date, end_date, freq)
        
        # 3. 获取无风险利率
        risk_free_rate = get_risk_free_rate(start_date, end_date, freq)
        
        # 4. 确保所有数据使用相同的日期索引
        common_index = stock_returns.index.intersection(market_returns.index)
        if not risk_free_rate.empty:
            common_index = common_index.intersection(risk_free_rate.index)
        
        if len(common_index) < 20:
            logger.warning(f"数据点太少，无法可靠估计CAPM模型: {len(common_index)} 条记录")
            return {}
        
        stock_returns = stock_returns.loc[common_index]
        market_returns = market_returns.loc[common_index]
        
        # 5. 估计CAPM模型
        from src.calc.factor_models import estimate_capm
        
        if risk_free_rate.empty:
            capm_results = estimate_capm(stock_returns, market_returns)
        else:
            risk_free_rate = risk_free_rate.loc[common_index]
            capm_results = estimate_capm(stock_returns, market_returns, risk_free_rate)
        
        logger.info(f"成功为股票 {stock_symbol} 估计CAPM模型参数")
        return capm_results
        
    except Exception as e:
        logger.error(f"估计CAPM模型参数时出错: {e}")
        logger.error(traceback.format_exc())
        return {}

def estimate_fama_french_for_stock(stock_symbol: str,
                                 start_date: str = None,
                                 end_date: str = None,
                                 freq: str = 'D') -> Dict[str, float]:
    """
    为单个股票估计Fama-French三因子模型参数
    
    Args:
        stock_symbol: 股票代码
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        Fama-French三因子模型参数字典
    """
    logger.info(f"为股票 {stock_symbol} 估计Fama-French三因子模型参数: {start_date} 至 {end_date}, 频率: {freq}")
    
    try:
        # 1. 获取股票收益率
        stock_returns_dict = get_stock_returns(stock_symbol, start_date, end_date, freq)
        if not stock_returns_dict or stock_symbol not in stock_returns_dict:
            logger.warning(f"无法获取股票 {stock_symbol} 收益率数据")
            return {}
        
        stock_returns = stock_returns_dict[stock_symbol]
        
        # 2. 获取Fama-French三因子
        ff_factors = get_fama_french_factors(start_date, end_date, freq)
        
        if not ff_factors:
            logger.warning("无法获取Fama-French三因子数据")
            return {}
        
        # 3. 确保所有数据使用相同的日期索引
        common_index = stock_returns.index.intersection(ff_factors['market_returns'].index)
        
        if len(common_index) < 20:
            logger.warning(f"数据点太少，无法可靠估计Fama-French模型: {len(common_index)} 条记录")
            return {}
        
        stock_returns = stock_returns.loc[common_index]
        market_returns = ff_factors['market_returns'].loc[common_index]
        smb = ff_factors['smb'].loc[common_index]
        hml = ff_factors['hml'].loc[common_index]
        risk_free_rate = ff_factors['risk_free_rate'].loc[common_index]
        
        # 4. 估计Fama-French模型
        from src.calc.factor_models import estimate_fama_french
        
        ff_results = estimate_fama_french(
            stock_returns, market_returns, smb, hml, risk_free_rate
        )
        
        logger.info(f"成功为股票 {stock_symbol} 估计Fama-French三因子模型参数")
        return ff_results
        
    except Exception as e:
        logger.error(f"估计Fama-French模型参数时出错: {e}")
        logger.error(traceback.format_exc())
        return {}

def estimate_beta_for_stocks(symbols: List[str],
                            start_date: str = None,
                            end_date: str = None,
                            freq: str = 'D') -> pd.DataFrame:
    """
    估计多个股票的贝塔系数
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        包含贝塔系数的DataFrame
    """
    logger.info(f"估计多个股票的贝塔系数: {start_date} 至 {end_date}, 股票数量: {len(symbols)}, 频率: {freq}")
    
    # 获取股票收益率数据
    returns_df = get_multi_stock_returns(symbols, start_date, end_date, freq)
    
    if returns_df.empty:
        logger.warning("无法获取股票收益率数据")
        return pd.DataFrame()
    
    # 获取市场收益率
    market_returns = get_market_returns("000300", start_date, end_date, freq)
    
    if market_returns.empty:
        logger.warning("无法获取市场收益率数据")
        return pd.DataFrame()
    
    # 确保收益率和市场收益率使用相同的日期
    common_index = returns_df.index.intersection(market_returns.index)
    returns_df = returns_df.loc[common_index]
    market_returns = market_returns.loc[common_index]
    
    # 计算贝塔系数
    betas = {}
    r_squareds = {}
    
    for symbol in returns_df.columns:
        # 使用线性回归计算贝塔系数
        import statsmodels.api as sm
        
        X = sm.add_constant(market_returns)
        y = returns_df[symbol]
        
        try:
            model = sm.OLS(y, X).fit()
            betas[symbol] = model.params.iloc[1]  # 贝塔系数
            r_squareds[symbol] = model.rsquared  # R平方
        except Exception as e:
            logger.warning(f"计算股票 {symbol} 的贝塔系数时出错: {e}")
            betas[symbol] = np.nan
            r_squareds[symbol] = np.nan
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'symbol': list(betas.keys()),
        'beta': list(betas.values()),
        'r_squared': list(r_squareds.values())
    })
    
    logger.info(f"成功估计 {len(result_df)} 只股票的贝塔系数")
    return result_df

def calculate_rolling_beta(stock_symbol: str,
                          window: int = 60,
                          start_date: str = None,
                          end_date: str = None,
                          freq: str = 'D') -> pd.Series:
    """
    计算股票的滚动贝塔系数
    
    Args:
        stock_symbol: 股票代码
        window: 滚动窗口大小（交易日数量）
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        滚动贝塔系数Series
    """
    logger.info(f"计算股票 {stock_symbol} 的滚动贝塔系数: {start_date} 至 {end_date}, 窗口大小: {window}")
    
    # 获取股票收益率数据
    stock_returns_dict = get_stock_returns(stock_symbol, start_date, end_date, freq)
    
    if not stock_returns_dict or stock_symbol not in stock_returns_dict:
        logger.warning(f"无法获取股票 {stock_symbol} 收益率数据")
        return pd.Series()
    
    stock_returns = stock_returns_dict[stock_symbol]
    
    # 获取市场收益率
    market_returns = get_market_returns("000300", start_date, end_date, freq)
    
    if market_returns.empty:
        logger.warning("无法获取市场收益率数据")
        return pd.Series()
    
    # 确保收益率和市场收益率使用相同的日期
    common_index = stock_returns.index.intersection(market_returns.index)
    stock_returns = stock_returns.loc[common_index]
    market_returns = market_returns.loc[common_index]
    
    # 确保数据长度足够
    if len(stock_returns) < window + 10:
        logger.warning(f"数据长度不足，无法计算滚动贝塔: {len(stock_returns)} < {window + 10}")
        return pd.Series()
    
    # 计算滚动贝塔系数
    rolling_betas = pd.Series(index=stock_returns.index[window-1:], dtype=float)
    
    for i in range(window-1, len(stock_returns)):
        # 获取窗口内的数据
        stock_window = stock_returns.iloc[i-window+1:i+1]
        market_window = market_returns.iloc[i-window+1:i+1]
        
        # 使用线性回归计算贝塔系数
        import statsmodels.api as sm
        
        X = sm.add_constant(market_window)
        y = stock_window
        
        try:
            model = sm.OLS(y, X).fit()
            rolling_betas.loc[stock_returns.index[i]] = model.params.iloc[1]  # 贝塔系数
        except Exception as e:
            logger.debug(f"计算 {stock_returns.index[i]} 的贝塔系数时出错: {e}")
            rolling_betas.loc[stock_returns.index[i]] = np.nan
    
    # 去除缺失值
    rolling_betas = rolling_betas.dropna()
    
    logger.info(f"成功计算股票 {stock_symbol} 的滚动贝塔系数: {len(rolling_betas)} 条记录")
    return rolling_betas