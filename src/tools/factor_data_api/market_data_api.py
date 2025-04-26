"""
市场数据API - 提供市场相关数据的获取和处理功能
"""

import os
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
from typing import Dict, Union, List

from .base import logger, data_api

def get_market_returns(index_code: str = "000300", 
                      start_date: str = None, 
                      end_date: str = None,
                      freq: str = 'D') -> pd.Series:
    """
    获取市场收益率数据（默认使用沪深300指数）
    
    Args:
        index_code: 指数代码，默认为沪深300
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        包含市场收益率的Series，索引为日期
    """
    logger.info(f"获取市场收益率数据: {start_date} 至 {end_date}, 指数代码: {index_code}, 频率: {freq}")
    
    try:
        # 尝试使用TuShare API
        try:
            import tushare as ts
            token = os.environ.get('TUSHARE_TOKEN', '')
            if not token:
                logger.warning("未找到TUSHARE_TOKEN环境变量")
                raise ValueError("TuShare token不可用")
                
            ts.set_token(token)
            pro = ts.pro_api()
            
            # 处理日期格式
            if not end_date:
                end_date = datetime.now().strftime("%Y%m%d")
            else:
                if '-' in end_date:
                    end_date = end_date.replace('-', '')
            
            if not start_date:
                # 默认获取一年的数据
                start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
            else:
                if '-' in start_date:
                    start_date = start_date.replace('-', '')
            
            # 处理指数代码
            if not index_code.endswith(('.SH', '.SZ', '.BJ')):
                if index_code.startswith('0'):
                    index_code = f"{index_code}.SH"
                elif index_code.startswith(('1', '3')):
                    index_code = f"{index_code}.SZ"
                else:
                    index_code = f"{index_code}.SH"
            
            # 获取指数数据
            index_data = None
            try:
                if freq == 'W':
                    index_data = pro.index_weekly(ts_code=index_code, start_date=start_date, end_date=end_date)
                elif freq == 'M':
                    index_data = pro.index_monthly(ts_code=index_code, start_date=start_date, end_date=end_date)
                else:  # 日频率
                    index_data = pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
            except Exception as e:
                logger.warning(f"使用TuShare获取指数数据失败: {e}")
                
            # 如果TuShare特定频率API调用失败，尝试获取日度数据然后重采样
            if index_data is None or index_data.empty:
                try:
                    logger.info("尝试获取日度数据然后重采样")
                    index_data = pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
                except Exception as e:
                    logger.warning(f"使用TuShare获取日度指数数据失败: {e}")
            
            if index_data is not None and not index_data.empty:
                # 处理数据
                index_data['trade_date'] = pd.to_datetime(index_data['trade_date'])
                index_data = index_data.sort_values('trade_date')
                index_data = index_data.set_index('trade_date')
                
                # 如果需要重采样
                if freq != 'D' and 'close' in index_data.columns:
                    if freq == 'W':
                        close_price = index_data['close'].resample('W-FRI').last()
                    elif freq == 'M':
                        close_price = index_data['close'].resample('M').last()
                    else:
                        close_price = index_data['close']
                else:
                    close_price = index_data['close']
                
                # 计算收益率
                market_returns = close_price.pct_change().dropna()
                
                logger.info(f"成功使用TuShare获取市场收益率数据: {len(market_returns)} 条记录")
                return market_returns
            else:
                logger.warning("TuShare获取的指数数据为空，尝试使用AKShare")
        except Exception as e:
            logger.warning(f"使用TuShare获取市场收益率失败: {e}")
            logger.warning("尝试使用AKShare")
        
        # 尝试使用AKShare
        try:
            import akshare as ak
            
            # 处理指数代码
            if index_code.endswith(('.SH', '.SZ', '.BJ')):
                index_code = index_code[:-3]
            
            # 为指数代码添加前缀
            if not index_code.startswith(('sh', 'sz', 'bj')):
                if index_code.startswith('0'):
                    ak_index_code = f"sh{index_code}"
                elif index_code.startswith(('1', '3')):
                    ak_index_code = f"sz{index_code}"
                else:
                    ak_index_code = f"sh{index_code}"
            else:
                ak_index_code = index_code
            
            # 处理日期格式
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
            
            # 确保日期格式为YYYY-MM-DD
            if len(start_date) == 8:
                start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            if len(end_date) == 8:
                end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            
            # 获取指数数据
            if freq == 'W':
                period = "weekly"
            elif freq == 'M':
                period = "monthly"
            else:
                period = "daily"
            
            index_data = ak.stock_zh_index_daily(symbol=ak_index_code, period=period)
            
            if not index_data.empty:
                # 处理数据
                index_data['date'] = pd.to_datetime(index_data['date'])
                index_data = index_data.sort_values('date')
                index_data = index_data.set_index('date')
                
                # 过滤日期范围
                index_data = index_data[index_data.index >= pd.to_datetime(start_date)]
                index_data = index_data[index_data.index <= pd.to_datetime(end_date)]
                
                # 计算收益率
                market_returns = index_data['close'].pct_change().dropna()
                
                logger.info(f"成功使用AKShare获取市场收益率数据: {len(market_returns)} 条记录")
                return market_returns
            else:
                logger.warning("AKShare获取的指数数据为空，使用模拟数据")
                return _generate_mock_market_returns(start_date, end_date, freq)
        except Exception as e:
            logger.warning(f"使用AKShare获取市场收益率失败: {e}")
            logger.warning("将使用模拟数据")
    
    except Exception as e:
        logger.error(f"获取市场收益率时出错: {e}")
        logger.error(traceback.format_exc())
    
    # 如果所有尝试都失败，使用模拟数据
    return _generate_mock_market_returns(start_date, end_date, freq)

def _generate_mock_market_returns(start_date: str, end_date: str, freq: str = 'D') -> pd.Series:
    """
    生成模拟的市场收益率数据
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        模拟的市场收益率Series
    """
    logger.info(f"生成模拟市场收益率数据, 频率: {freq}")
    
    # 处理日期参数
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        # 默认生成一年的数据
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # 确保日期格式一致
    if isinstance(start_date, str) and len(start_date) == 8:
        start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    if isinstance(end_date, str) and len(end_date) == 8:
        end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    
    # 生成日期范围，根据频率
    if freq == 'W':
        date_range = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
        # 均值和标准差调整为周度
        mean_return = 0.002
        std_return = 0.02
    elif freq == 'M':
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        # 均值和标准差调整为月度
        mean_return = 0.008
        std_return = 0.04
    else:  # 日频率
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        # 均值和标准差调整为日度
        mean_return = 0.0005
        std_return = 0.01
    
    # 生成模拟的市场收益率
    np.random.seed(42)  # 设置随机种子以保证可复现性
    returns = np.random.normal(mean_return, std_return, len(date_range))
    
    # 创建Series
    market_returns = pd.Series(returns, index=date_range)
    
    return market_returns

def get_stock_returns(symbols: Union[str, list], 
                     start_date: str = None, 
                     end_date: str = None,
                     freq: str = 'D') -> Dict[str, pd.Series]:
    """
    获取一个或多个股票的收益率数据
    
    Args:
        symbols: 单个股票代码或股票代码列表
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        包含股票收益率的字典，键为股票代码，值为收益率Series
    """
    logger.info(f"获取股票收益率数据: {start_date} 至 {end_date}, 频率: {freq}")
    
    # 转换为列表类型
    if isinstance(symbols, str):
        symbols = [symbols]
    
    returns_dict = {}
    
    for symbol in symbols:
        try:
            # 尝试使用TuShare获取数据
            try:
                import tushare as ts
                token = os.environ.get('TUSHARE_TOKEN', '')
                if token:
                    ts.set_token(token)
                    pro = ts.pro_api()
                    
                    # 处理日期格式
                    ts_start = start_date
                    ts_end = end_date
                    
                    if ts_start and '-' in ts_start:
                        ts_start = ts_start.replace('-', '')
                    
                    if ts_end and '-' in ts_end:
                        ts_end = ts_end.replace('-', '')
                    
                    # 处理股票代码
                    ts_code = symbol
                    if not ts_code.endswith(('.SH', '.SZ', '.BJ')):
                        if ts_code.startswith('6'):
                            ts_code = f"{ts_code}.SH"
                        elif ts_code.startswith(('0', '3')):
                            ts_code = f"{ts_code}.SZ"
                        elif ts_code.startswith('4'):
                            ts_code = f"{ts_code}.BJ"
                        else:
                            ts_code = f"{ts_code}.SH"
                    
                    # 根据频率获取不同周期的数据
                    stock_data = None
                    
                    try:
                        if freq == 'W':
                            stock_data = pro.weekly(ts_code=ts_code, start_date=ts_start, end_date=ts_end)
                        elif freq == 'M':
                            stock_data = pro.monthly(ts_code=ts_code, start_date=ts_start, end_date=ts_end)
                        else:  # 日频率
                            stock_data = pro.daily(ts_code=ts_code, start_date=ts_start, end_date=ts_end)
                    except Exception as e:
                        logger.warning(f"使用TuShare获取股票 {symbol} 数据失败: {e}")
                        
                    # 如果特定频率API调用失败，尝试获取日度数据然后重采样
                    if stock_data is None or stock_data.empty:
                        try:
                            logger.info(f"尝试获取日度数据然后重采样: {ts_code}")
                            stock_data = pro.daily(ts_code=ts_code, start_date=ts_start, end_date=ts_end)
                        except Exception as e:
                            logger.warning(f"使用TuShare获取日度股票数据失败: {e}")
                    
                    if stock_data is not None and not stock_data.empty:
                        # 处理数据
                        stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
                        stock_data = stock_data.sort_values('trade_date')
                        stock_data = stock_data.set_index('trade_date')
                        
                        # 如果需要重采样
                        if freq != 'D' and 'close' in stock_data.columns:
                            if freq == 'W':
                                close_price = stock_data['close'].resample('W-FRI').last()
                            elif freq == 'M':
                                close_price = stock_data['close'].resample('M').last()
                            else:
                                close_price = stock_data['close']
                        else:
                            close_price = stock_data['close']
                        
                        # 计算收益率
                        returns = close_price.pct_change().dropna()
                        
                        if not returns.empty:
                            returns_dict[symbol] = returns
                            logger.info(f"成功使用TuShare获取股票 {symbol} 收益率数据: {len(returns)} 条记录")
                            continue
            except ImportError:
                logger.warning("未找到tushare库，尝试使用AKShare")
            
            # 使用数据API获取股票价格数据
            prices = data_api.get_price_data(symbol, start_date, end_date)
            
            if prices.empty:
                logger.warning(f"无法获取股票 {symbol} 的价格数据")
                continue
            
            # 确保日期列为日期类型
            if 'date' in prices.columns:
                prices['date'] = pd.to_datetime(prices['date'])
                prices = prices.set_index('date')
            
            # 根据频率重采样
            if freq == 'W' and 'close' in prices.columns:
                close_price = prices['close'].resample('W-FRI').last()
            elif freq == 'M' and 'close' in prices.columns:
                close_price = prices['close'].resample('M').last()
            elif 'close' in prices.columns:
                close_price = prices['close']
            else:
                logger.warning(f"股票 {symbol} 价格数据中不包含close列")
                continue
            
            # 计算收益率
            returns = close_price.pct_change().dropna()
            
            if not returns.empty:
                returns_dict[symbol] = returns
                logger.info(f"成功获取股票 {symbol} 收益率数据: {len(returns)} 条记录")
                
        except Exception as e:
            logger.error(f"获取股票 {symbol} 收益率时出错: {e}")
            logger.error(traceback.format_exc())
    
    return returns_dict

def get_multi_stock_returns(symbols: list, 
                           start_date: str = None, 
                           end_date: str = None,
                           freq: str = 'D') -> pd.DataFrame:
    """
    获取多个股票的收益率数据并合并为DataFrame
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        包含多个股票收益率的DataFrame，列为股票代码，索引为日期
    """
    logger.info(f"获取多股票收益率数据: {start_date} 至 {end_date}, 股票数量: {len(symbols)}, 频率: {freq}")
    
    # 获取各股票收益率
    returns_dict = get_stock_returns(symbols, start_date, end_date, freq)
    
    # 如果没有获取到数据，返回空DataFrame
    if not returns_dict:
        logger.warning("未获取到任何股票收益率数据")
        return pd.DataFrame()
    
    # 合并为DataFrame
    df = pd.DataFrame(returns_dict)
    
    return df

def get_stock_covariance_matrix(symbols: list,
                               start_date: str = None,
                               end_date: str = None,
                               method: str = "sample",
                               freq: str = 'D') -> tuple:
    """
    计算多个股票收益率的协方差矩阵和平均收益率
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        method: 协方差矩阵估计方法，可选 "sample", "ewma"
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        (协方差矩阵DataFrame, 平均收益率Series)的元组
    """
    logger.info(f"计算股票协方差矩阵: {start_date} 至 {end_date}, 股票数量: {len(symbols)}, 方法: {method}")
    
    # 获取多股票收益率数据
    returns_df = get_multi_stock_returns(symbols, start_date, end_date, freq)
    
    # 如果没有获取到数据，返回空DataFrame
    if returns_df.empty:
        logger.warning("未获取到股票收益率数据，无法计算协方差矩阵")
        return pd.DataFrame(), pd.Series()
    
    try:
        # 计算协方差矩阵
        if method == "ewma":
            # 导入EWMA协方差矩阵估计函数
            from src.calc.covariance_estimation import estimate_covariance_ewma
            cov_matrix = estimate_covariance_ewma(returns_df)
        else:
            # 使用样本协方差矩阵
            cov_matrix = returns_df.cov()
        
        # 计算平均收益率
        expected_returns = returns_df.mean()
        
        # 年化处理（根据频率）
        if freq == 'D':
            annualization_factor = 252
        elif freq == 'W':
            annualization_factor = 52
        elif freq == 'M':
            annualization_factor = 12
        else:
            annualization_factor = 252
        
        expected_returns = expected_returns * annualization_factor
        cov_matrix = cov_matrix * annualization_factor
        
        return cov_matrix, expected_returns
    except Exception as e:
        logger.error(f"计算协方差矩阵时出错: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(), pd.Series()

def get_index_data(index_symbol: str = "000300", 
                  fields: list = None, 
                  start_date: str = None, 
                  end_date: str = None,
                  freq: str = 'D') -> pd.DataFrame:
    """
    获取指数数据
    
    Args:
        index_symbol: 指数代码，默认为沪深300
        fields: 要获取的字段列表，为None则获取所有可用字段
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        包含指数数据的DataFrame
    """
    logger.info(f"获取指数数据: {index_symbol}, {start_date} 至 {end_date}, 频率: {freq}")
    
    try:
        # 尝试使用TuShare API
        try:
            import tushare as ts
            token = os.environ.get('TUSHARE_TOKEN', '')
            if token:
                ts.set_token(token)
                pro = ts.pro_api()
                
                # 处理日期格式
                ts_start = start_date
                ts_end = end_date
                
                if ts_start and '-' in ts_start:
                    ts_start = ts_start.replace('-', '')
                
                if ts_end and '-' in ts_end:
                    ts_end = ts_end.replace('-', '')
                
                # 处理指数代码
                ts_code = index_symbol
                if not ts_code.endswith(('.SH', '.SZ', '.BJ')):
                    if ts_code.startswith('0'):
                        ts_code = f"{ts_code}.SH"
                    elif ts_code.startswith(('1', '3')):
                        ts_code = f"{ts_code}.SZ"
                    else:
                        ts_code = f"{ts_code}.SH"
                
                # 获取指数数据
                index_data = None
                
                try:
                    if freq == 'W':
                        index_data = pro.index_weekly(ts_code=ts_code, start_date=ts_start, end_date=ts_end)
                    elif freq == 'M':
                        index_data = pro.index_monthly(ts_code=ts_code, start_date=ts_start, end_date=ts_end)
                    else:  # 日频率
                        index_data = pro.index_daily(ts_code=ts_code, start_date=ts_start, end_date=ts_end)
                except Exception as e:
                    logger.warning(f"使用TuShare获取指数数据失败: {e}")
                    
                # 如果特定频率API调用失败，尝试获取日度数据然后重采样
                if index_data is None or index_data.empty:
                    try:
                        logger.info("尝试获取日度数据然后重采样")
                        index_data = pro.index_daily(ts_code=ts_code, start_date=ts_start, end_date=ts_end)
                    except Exception as e:
                        logger.warning(f"使用TuShare获取日度指数数据失败: {e}")
                
                if index_data is not None and not index_data.empty:
                    # 处理数据
                    index_data['trade_date'] = pd.to_datetime(index_data['trade_date'])
                    index_data = index_data.sort_values('trade_date')
                    
                    # 如果需要重采样
                    if freq != 'D' and index_data['trade_date'].dt.freq != freq:
                        index_data = index_data.set_index('trade_date')
                        
                        if freq == 'W':
                            # 对于OHLC列使用特定重采样方法
                            resampled_data = pd.DataFrame()
                            if 'open' in index_data.columns:
                                resampled_data['open'] = index_data['open'].resample('W-FRI').first()
                            if 'high' in index_data.columns:
                                resampled_data['high'] = index_data['high'].resample('W-FRI').max()
                            if 'low' in index_data.columns:
                                resampled_data['low'] = index_data['low'].resample('W-FRI').min()
                            if 'close' in index_data.columns:
                                resampled_data['close'] = index_data['close'].resample('W-FRI').last()
                            if 'vol' in index_data.columns:
                                resampled_data['vol'] = index_data['vol'].resample('W-FRI').sum()
                            if 'amount' in index_data.columns:
                                resampled_data['amount'] = index_data['amount'].resample('W-FRI').sum()
                            
                            # 添加其他字段
                            for col in index_data.columns:
                                if col not in ['open', 'high', 'low', 'close', 'vol', 'amount']:
                                    resampled_data[col] = index_data[col].resample('W-FRI').last()
                            
                            index_data = resampled_data.reset_index()
                            
                        elif freq == 'M':
                            # 对于OHLC列使用特定重采样方法
                            resampled_data = pd.DataFrame()
                            if 'open' in index_data.columns:
                                resampled_data['open'] = index_data['open'].resample('M').first()
                            if 'high' in index_data.columns:
                                resampled_data['high'] = index_data['high'].resample('M').max()
                            if 'low' in index_data.columns:
                                resampled_data['low'] = index_data['low'].resample('M').min()
                            if 'close' in index_data.columns:
                                resampled_data['close'] = index_data['close'].resample('M').last()
                            if 'vol' in index_data.columns:
                                resampled_data['vol'] = index_data['vol'].resample('M').sum()
                            if 'amount' in index_data.columns:
                                resampled_data['amount'] = index_data['amount'].resample('M').sum()
                            
                            # 添加其他字段
                            for col in index_data.columns:
                                if col not in ['open', 'high', 'low', 'close', 'vol', 'amount']:
                                    resampled_data[col] = index_data[col].resample('M').last()
                            
                            index_data = resampled_data.reset_index()
                    
                    # 如果指定了字段，只返回指定字段
                    if fields:
                        available_fields = set(index_data.columns)
                        requested_fields = set(fields)
                        missing_fields = requested_fields - available_fields
                        
                        if missing_fields:
                            logger.warning(f"请求的字段不可用: {missing_fields}")
                        
                        selected_fields = list(requested_fields & available_fields)
                        if not selected_fields:
                            logger.warning("没有可用的请求字段")
                            return pd.DataFrame()
                        
                        return index_data[selected_fields]
                    
                    logger.info(f"成功使用TuShare获取指数数据: {len(index_data)} 条记录")
                    return index_data
                else:
                    logger.warning("TuShare获取的指数数据为空，尝试使用AKShare")
        except ImportError:
            logger.warning("未找到tushare库，尝试使用AKShare")
        
        # 尝试使用AKShare
        try:
            import akshare as ak
            
            # 处理指数代码
            if index_symbol.endswith(('.SH', '.SZ', '.BJ')):
                index_symbol = index_symbol[:-3]
            
            # 为指数代码添加前缀
            if not index_symbol.startswith(('sh', 'sz', 'bj')):
                if index_symbol.startswith('0'):
                    ak_index_code = f"sh{index_symbol}"
                elif index_symbol.startswith(('1', '3')):
                    ak_index_code = f"sz{index_symbol}"
                else:
                    ak_index_code = f"sh{index_symbol}"
            else:
                ak_index_code = index_symbol
            
            # 处理日期格式
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
            
            # 确保日期格式为YYYY-MM-DD
            if len(start_date) == 8:
                start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            if len(end_date) == 8:
                end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            
            # 获取指数数据
            if freq == 'W':
                period = "weekly"
            elif freq == 'M':
                period = "monthly"
            else:
                period = "daily"
            
            index_data = ak.stock_zh_index_daily(symbol=ak_index_code, period=period)
            
            if not index_data.empty:
                # 处理日期
                index_data['date'] = pd.to_datetime(index_data['date'])
                
                # 过滤日期范围
                index_data = index_data[(index_data['date'] >= pd.to_datetime(start_date)) & 
                                      (index_data['date'] <= pd.to_datetime(end_date))]
                
                # 如果指定了字段，只返回指定字段
                if fields:
                    # 映射字段名
                    field_mapping = {
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'vol': 'volume',
                        'amount': 'amount',
                        'change': 'change',
                        'pct_chg': 'pct_change'
                    }
                    
                    # 转换字段名
                    converted_fields = []
                    for field in fields:
                        if field in field_mapping and field_mapping[field] in index_data.columns:
                            converted_fields.append(field_mapping[field])
                        elif field in index_data.columns:
                            converted_fields.append(field)
                    
                    if not converted_fields:
                        logger.warning("没有可用的请求字段")
                        return pd.DataFrame()
                    
                    if 'date' not in converted_fields:
                        converted_fields.append('date')
                    
                    return index_data[converted_fields]
                
                logger.info(f"成功使用AKShare获取指数数据: {len(index_data)} 条记录")
                return index_data
            else:
                logger.warning("AKShare获取的指数数据为空")
                return pd.DataFrame()
        except ImportError:
            logger.warning("未找到akshare库")
        except Exception as e:
            logger.warning(f"使用AKShare获取指数数据失败: {e}")
    
    except Exception as e:
        logger.error(f"获取指数数据时出错: {e}")
        logger.error(traceback.format_exc())
    
    # 如果所有尝试都失败，返回空DataFrame
    logger.warning("所有尝试获取指数数据都失败")
    return pd.DataFrame()

def get_multiple_index_data(index_symbols: List[str],
                           start_date: str = None,
                           end_date: str = None,
                           freq: str = 'D') -> Dict[str, pd.DataFrame]:
    """
    获取多个指数的数据
    
    Args:
        index_symbols: 指数代码列表
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        包含多个指数数据的字典，键为指数代码，值为指数数据DataFrame
    """
    logger.info(f"获取多指数数据: {start_date} 至 {end_date}, 指数数量: {len(index_symbols)}, 频率: {freq}")
    
    index_data_dict = {}
    
    for symbol in index_symbols:
        try:
            index_data = get_index_data(symbol, None, start_date, end_date, freq)
            
            if not index_data.empty:
                index_data_dict[symbol] = index_data
                logger.info(f"成功获取指数 {symbol} 数据: {len(index_data)} 条记录")
            else:
                logger.warning(f"无法获取指数 {symbol} 数据")
                
        except Exception as e:
            logger.error(f"获取指数 {symbol} 数据时出错: {e}")
            logger.error(traceback.format_exc())
    
    return index_data_dict