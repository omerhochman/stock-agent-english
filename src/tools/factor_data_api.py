"""
因子数据获取模块 - 提供CAPM和Fama-French因子数据的获取功能

此模块包含获取市场风险溢价、规模因子和价值因子的函数，
用于进行CAPM和Fama-French三因子模型的估计。
支持TuShare和AKShare两种数据接口。
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple, List
import logging
from datetime import datetime, timedelta
import traceback

from src.utils.logging_config import setup_logger
from src.tools.data_source_adapter import DataAPI

# 设置日志记录
logger = setup_logger('factor_data_api')

# 创建数据API实例
data_api = DataAPI()

def get_risk_free_rate(start_date: str = None, end_date: str = None, 
                      freq: str = 'D', use_cache: bool = True) -> pd.Series:
    """
    获取无风险利率数据（使用银行间同业拆借利率或国债收益率）
    
    Args:
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        use_cache: 是否使用缓存数据
        
    Returns:
        包含无风险利率的Series，索引为日期
    """
    logger.info(f"获取无风险利率数据: {start_date} 至 {end_date}, 频率: {freq}")
    
    # 缓存文件路径
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"risk_free_rate_{freq}.csv")
    
    # 检查缓存
    if use_cache and os.path.exists(cache_file):
        try:
            cache_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logger.info(f"成功从缓存加载无风险利率数据: {len(cache_data)} 条记录")
            
            # 过滤日期范围
            if start_date:
                start_date = pd.to_datetime(start_date)
                cache_data = cache_data[cache_data.index >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date)
                cache_data = cache_data[cache_data.index <= end_date]
                
            if not cache_data.empty:
                rf_series = cache_data["risk_free_rate"]
                return rf_series
        except Exception as e:
            logger.warning(f"从缓存加载无风险利率数据失败: {e}")
    
    try:
        # 尝试使用AKShare获取数据
        try:
            import akshare as ak
            logger.info("使用AKShare获取国债收益率")
            
            # 处理日期参数
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                # 默认获取一年的数据
                start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
            
            # 获取国债收益率数据
            bond_data = ak.bond_china_yield(start_date=start_date, end_date=end_date)
            
            # 提取1年期国债收益率
            if "1年期" in bond_data.columns:
                rf_data = bond_data[["日期", "1年期"]]
                rf_data = rf_data.rename(columns={"日期": "date", "1年期": "risk_free_rate"})
                
                # 转换日期格式和利率格式
                rf_data["date"] = pd.to_datetime(rf_data["date"])
                # 将百分比转换为小数
                rf_data["risk_free_rate"] = rf_data["risk_free_rate"] / 100
                
                # 设置索引并排序
                rf_data = rf_data.set_index("date").sort_index()
                
                # 重采样到指定频率
                if freq == 'W':
                    rf_data = rf_data.resample('W-FRI').last().fillna(method='ffill')
                    # 将年化利率转换为周利率
                    rf_data = rf_data / 52
                elif freq == 'M':
                    rf_data = rf_data.resample('M').last().fillna(method='ffill')
                    # 将年化利率转换为月利率
                    rf_data = rf_data / 12
                else:  # 日频率
                    # 将年化利率转换为日利率
                    rf_data = rf_data / 252
                
                # 保存到缓存
                try:
                    rf_data.to_csv(cache_file)
                    logger.info(f"无风险利率数据已保存到缓存: {cache_file}")
                except Exception as e:
                    logger.warning(f"保存无风险利率数据到缓存失败: {e}")
                
                return rf_data["risk_free_rate"]
            else:
                logger.warning("无法从AKShare获取1年期国债收益率，尝试TuShare")
        except Exception as e:
            logger.warning(f"使用AKShare获取无风险利率失败: {e}")
            logger.warning("尝试使用TuShare获取数据")
        
        # 尝试使用TuShare获取数据
        try:
            import tushare as ts
            token = os.environ.get('TUSHARE_TOKEN', '')
            if not token:
                logger.warning("未找到TUSHARE_TOKEN环境变量")
                raise ValueError("TuShare token不可用")
                
            ts.set_token(token)
            pro = ts.pro_api()
            
            # 注意：TuShare的无风险利率数据可能需要特定权限
            # 以下是使用TuShare获取LPR(贷款基础利率)的示例
            if not end_date:
                end_date = datetime.now().strftime("%Y%m%d")
            if not start_date:
                start_date = (datetime.strptime(end_date, "%Y%m%d") if len(end_date) == 8 else datetime.strptime(end_date, "%Y-%m-%d"))
                start_date = (start_date - timedelta(days=365)).strftime("%Y%m%d")
            else:
                # 确保格式为YYYYMMDD
                if len(start_date) == 10 and '-' in start_date:
                    start_date = start_date.replace('-', '')
            
            if len(end_date) == 10 and '-' in end_date:
                end_date = end_date.replace('-', '')
                
            try:
                # 获取LPR数据
                lpr_data = pro.lpr(start_date=start_date, end_date=end_date)
                
                if not lpr_data.empty:
                    # 使用1年期LPR
                    lpr_data = lpr_data[lpr_data['term'] == '1Y'][['date', 'value']]
                    lpr_data = lpr_data.rename(columns={'value': 'risk_free_rate'})
                    
                    # 转换格式
                    lpr_data['date'] = pd.to_datetime(lpr_data['date'])
                    lpr_data['risk_free_rate'] = lpr_data['risk_free_rate'] / 100  # 转换为小数
                    
                    # 设置索引并排序
                    lpr_data = lpr_data.set_index('date').sort_index()
                    
                    # 重采样到指定频率
                    if freq == 'W':
                        lpr_data = lpr_data.resample('W-FRI').last().fillna(method='ffill')
                        # 将年化利率转换为周利率
                        lpr_data = lpr_data / 52
                    elif freq == 'M':
                        lpr_data = lpr_data.resample('M').last().fillna(method='ffill')
                        # 将年化利率转换为月利率
                        lpr_data = lpr_data / 12
                    else:  # 日频率
                        # 将年化利率转换为日利率
                        lpr_data = lpr_data / 252
                    
                    # 保存到缓存
                    try:
                        lpr_data.to_csv(cache_file)
                        logger.info(f"无风险利率数据已保存到缓存: {cache_file}")
                    except Exception as e:
                        logger.warning(f"保存无风险利率数据到缓存失败: {e}")
                    
                    return lpr_data["risk_free_rate"]
                else:
                    logger.warning("TuShare LPR数据为空，尝试获取银行间同业拆借利率")
            except Exception as e:
                logger.warning(f"获取TuShare LPR数据失败: {e}")
                logger.warning("尝试获取银行间同业拆借利率")
                
            # 尝试获取银行间同业拆借利率(Shibor)
            try:
                shibor_data = pro.shibor(start_date=start_date, end_date=end_date)
                
                if not shibor_data.empty:
                    # 使用1Y Shibor
                    shibor_data = shibor_data[['date', '1y']]
                    shibor_data = shibor_data.rename(columns={'1y': 'risk_free_rate'})
                    
                    # 转换格式
                    shibor_data['date'] = pd.to_datetime(shibor_data['date'])
                    shibor_data['risk_free_rate'] = shibor_data['risk_free_rate'] / 100  # 转换为小数
                    
                    # 设置索引并排序
                    shibor_data = shibor_data.set_index('date').sort_index()
                    
                    # 重采样到指定频率
                    if freq == 'W':
                        shibor_data = shibor_data.resample('W-FRI').last().fillna(method='ffill')
                        # 将年化利率转换为周利率
                        shibor_data = shibor_data / 52
                    elif freq == 'M':
                        shibor_data = shibor_data.resample('M').last().fillna(method='ffill')
                        # 将年化利率转换为月利率
                        shibor_data = shibor_data / 12
                    else:  # 日频率
                        # 将年化利率转换为日利率
                        shibor_data = shibor_data / 252
                    
                    # 保存到缓存
                    try:
                        shibor_data.to_csv(cache_file)
                        logger.info(f"无风险利率数据已保存到缓存: {cache_file}")
                    except Exception as e:
                        logger.warning(f"保存无风险利率数据到缓存失败: {e}")
                    
                    return shibor_data["risk_free_rate"]
                else:
                    logger.warning("TuShare Shibor数据为空，使用模拟数据")
                    return _generate_mock_risk_free_rate(start_date, end_date, freq)
            except Exception as e:
                logger.warning(f"获取TuShare Shibor数据失败: {e}")
                logger.warning("将使用模拟数据")
                return _generate_mock_risk_free_rate(start_date, end_date, freq)
        except Exception as e:
            logger.warning(f"使用TuShare获取无风险利率失败: {e}")
            logger.warning("将使用模拟数据")
    
    except Exception as e:
        logger.error(f"获取无风险利率时出错: {e}")
        logger.error(traceback.format_exc())
    
    # 如果所有尝试都失败，使用模拟数据
    return _generate_mock_risk_free_rate(start_date, end_date, freq)

def _generate_mock_risk_free_rate(start_date: str, end_date: str, freq: str = 'D') -> pd.Series:
    """
    生成模拟的无风险利率数据
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        模拟的无风险利率Series
    """
    logger.info(f"生成模拟无风险利率数据, 频率: {freq}")
    
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
        # 年化利率约2.5%左右，转换为周利率
        base_rate = 0.025 / 52
    elif freq == 'M':
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        # 年化利率约2.5%左右，转换为月利率
        base_rate = 0.025 / 12
    else:  # 日频率
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        # 年化利率约2.5%左右，转换为日利率
        base_rate = 0.025 / 252
    
    # 添加一些随机波动
    np.random.seed(42)  # 设置随机种子以保证可复现性
    rates = np.random.normal(base_rate, base_rate * 0.05, len(date_range))
    
    # 创建Series
    rf_series = pd.Series(rates, index=date_range)
    
    return rf_series

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

def calculate_fama_french_factors_tushare(start_date: str, end_date: str, freq: str = 'W') -> Dict[str, pd.Series]:
    """
    使用TuShare计算Fama-French三因子数据
    
    Args:
        start_date: 开始日期，格式：YYYY-MM-DD或YYYYMMDD
        end_date: 结束日期，格式：YYYY-MM-DD或YYYYMMDD
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        包含市场风险溢价、SMB、HML因子的字典
    """
    logger.info(f"使用TuShare计算Fama-French三因子数据: {start_date} 至 {end_date}, 频率: {freq}")
    
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
        
        # 1. 获取市场指数数据作为市场收益率（使用沪深300指数）
        logger.info("获取市场指数数据...")
        
        if freq == 'W':
            market_df = pro.index_weekly(ts_code='000300.SH', start_date=start_date, end_date=end_date)
        elif freq == 'M':
            market_df = pro.index_monthly(ts_code='000300.SH', start_date=start_date, end_date=end_date)
        else:  # 日频率
            market_df = pro.index_daily(ts_code='000300.SH', start_date=start_date, end_date=end_date)
        
        if market_df.empty:
            logger.warning("无法获取市场指数数据，尝试使用日频数据并重采样")
            market_df = pro.index_daily(ts_code='000300.SH', start_date=start_date, end_date=end_date)
            if market_df.empty:
                logger.warning("仍然无法获取市场数据，将使用模拟数据")
                return _generate_mock_fama_french_factors(start_date, end_date, freq)
        
        market_df['trade_date'] = pd.to_datetime(market_df['trade_date'])
        market_df = market_df.sort_values('trade_date')
        market_df = market_df.set_index('trade_date')
        
        # 如果需要重采样
        if freq == 'W' and not market_df.empty and market_df.index.freqstr != 'W-FRI':
            close_price = market_df['close'].resample('W-FRI').last()
        elif freq == 'M' and not market_df.empty and market_df.index.freqstr != 'M':
            close_price = market_df['close'].resample('M').last()
        else:
            close_price = market_df['close']
            
        market_return = close_price.pct_change().dropna()
        
        # 2. 生成用于计算SMB和HML的日期列表（每季度一次）
        portfolio_dates = pd.date_range(
            start=pd.to_datetime(start_date, format="%Y%m%d"), 
            end=pd.to_datetime(end_date, format="%Y%m%d"), 
            freq='Q'
        ).strftime('%Y%m%d').tolist()
        
        # 3. 初始化因子数据框
        factors_weekly = pd.DataFrame(index=market_return.index)
        factors_weekly['SMB'] = np.nan
        factors_weekly['HML'] = np.nan
        
        # 4. 计算每个日期的SMB和HML因子
        for date in portfolio_dates:
            logger.info(f"计算 {date} 的SMB和HML因子...")
            
            # 获取当日所有股票的市值和PB数据
            try:
                daily_basic = pro.daily_basic(trade_date=date, fields='ts_code,total_mv,pb,close')
                if daily_basic.empty:
                    logger.info(f"日期 {date} 没有数据，跳过")
                    continue
            except Exception as e:
                logger.warning(f"获取 {date} 基础数据出错: {e}")
                continue
            
            # 过滤掉PB为负或为0的股票
            daily_basic = daily_basic[daily_basic['pb'] > 0]
            
            # 按市值分为大小两组
            daily_basic['size_group'] = pd.qcut(daily_basic['total_mv'], q=2, labels=['small', 'big'])
            
            # 按PB倒数(B/M)分为三组
            daily_basic['bm'] = 1 / daily_basic['pb']  # 计算B/M比率
            daily_basic['bm_group'] = pd.qcut(daily_basic['bm'], q=3, labels=['low', 'medium', 'high'])
            
            # 形成六个投资组合
            portfolio_groups = {}
            for size in ['small', 'big']:
                for bm in ['low', 'medium', 'high']:
                    portfolio_groups[f"{size}_{bm}"] = daily_basic[
                        (daily_basic['size_group'] == size) & 
                        (daily_basic['bm_group'] == bm)]['ts_code'].tolist()
            
            # 计算每个投资组合的收益率
            portfolio_returns = {}
            
            for port_name, stocks in portfolio_groups.items():
                if not stocks:
                    portfolio_returns[port_name] = 0
                    continue
                
                # 限制股票数量
                sample_stocks = stocks[:20] if len(stocks) > 20 else stocks
                
                # 计算投资组合收益率
                port_returns = []
                for stock in sample_stocks:
                    try:
                        # 获取股票收益率数据
                        if freq == 'W':
                            stock_data = pro.weekly(ts_code=stock, start_date=date, 
                                                 end_date=end_date, fields='ts_code,trade_date,close')
                        elif freq == 'M':
                            stock_data = pro.monthly(ts_code=stock, start_date=date, 
                                                  end_date=end_date, fields='ts_code,trade_date,close')
                        else:  # 日频率
                            stock_data = pro.daily(ts_code=stock, start_date=date, 
                                                end_date=end_date, fields='ts_code,trade_date,close')
                        
                        if not stock_data.empty and len(stock_data) > 1:
                            stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
                            stock_data = stock_data.sort_values('trade_date')
                            stock_data['return'] = stock_data['close'].pct_change()
                            stock_data = stock_data.dropna()
                            
                            if not stock_data.empty:
                                port_returns.append(stock_data[['trade_date', 'return']].set_index('trade_date'))
                    except Exception as e:
                        logger.debug(f"处理股票 {stock} 时出错: {e}")
                        continue
                
                if port_returns:
                    try:
                        # 合并投资组合收益率
                        all_returns = pd.concat(port_returns, axis=1)
                        portfolio_weekly_return = all_returns.mean(axis=1)
                        
                        # 将投资组合收益率添加到factors_weekly
                        for week_date, ret in portfolio_weekly_return.items():
                            # 找到最近的周五或月末
                            if freq == 'W':
                                period_end = week_date + pd.Timedelta(days=(4 - week_date.weekday()) % 7)
                            elif freq == 'M':
                                period_end = pd.date_range(week_date, periods=2, freq='M')[0]
                            else:
                                period_end = week_date
                                
                            if period_end in factors_weekly.index:
                                if port_name not in factors_weekly.columns:
                                    factors_weekly[port_name] = np.nan
                                factors_weekly.loc[period_end, port_name] = ret
                    except Exception as e:
                        logger.warning(f"处理投资组合 {port_name} 时出错: {e}")
                        continue
            
            # 5. 计算SMB和HML因子
            for idx in factors_weekly.index:
                # 检查是否有所有需要的投资组合数据
                port_names = ['small_low', 'small_medium', 'small_high', 'big_low', 'big_medium', 'big_high']
                if all(port in factors_weekly.columns for port in port_names):
                    try:
                        # 计算SMB
                        small_avg = (factors_weekly.loc[idx, 'small_low'] + 
                                    factors_weekly.loc[idx, 'small_medium'] + 
                                    factors_weekly.loc[idx, 'small_high']) / 3
                        
                        big_avg = (factors_weekly.loc[idx, 'big_low'] + 
                                  factors_weekly.loc[idx, 'big_medium'] + 
                                  factors_weekly.loc[idx, 'big_high']) / 3
                        
                        factors_weekly.loc[idx, 'SMB'] = small_avg - big_avg
                        
                        # 计算HML
                        high_avg = (factors_weekly.loc[idx, 'small_high'] + 
                                   factors_weekly.loc[idx, 'big_high']) / 2
                        
                        low_avg = (factors_weekly.loc[idx, 'small_low'] + 
                                  factors_weekly.loc[idx, 'big_low']) / 2
                        
                        factors_weekly.loc[idx, 'HML'] = high_avg - low_avg
                    except Exception as e:
                        logger.debug(f"计算 {idx} 的因子时出错: {e}")
                        continue
        
        # 6. 获取无风险利率
        rf_series = get_risk_free_rate(start_date=start_date, end_date=end_date, freq=freq)
        
        # 7. 最终数据整合
        final_factors = pd.DataFrame({
            'MKT': market_return,
            'SMB': factors_weekly['SMB'],
            'HML': factors_weekly['HML'],
            'RF': rf_series
        })
        
        # 对齐索引
        common_index = market_return.index.intersection(rf_series.index)
        final_factors = final_factors.loc[common_index]
        
        # 使用前向填充处理缺失值
        final_factors = final_factors.fillna(method='ffill')
        
        # 计算市场风险溢价(Mkt-RF)
        final_factors['MKT_RF'] = final_factors['MKT'] - final_factors['RF']
        
        # 将结果转换为字典
        factors_dict = {
            'market_returns': final_factors['MKT'],
            'market_excess_returns': final_factors['MKT_RF'],
            'smb': final_factors['SMB'],
            'hml': final_factors['HML'],
            'risk_free_rate': final_factors['RF']
        }
        
        # 处理NaN值
        for key in factors_dict:
            factors_dict[key] = factors_dict[key].fillna(0)
        
        logger.info(f"成功计算Fama-French三因子，共 {len(factors_dict['market_returns'])} 条记录")
        return factors_dict
        
    except Exception as e:
        logger.error(f"TuShare计算Fama-French三因子失败: {e}")
        logger.error(traceback.format_exc())
        return _generate_mock_fama_french_factors(start_date, end_date, freq)

def get_fama_french_factors(start_date: str = None, 
                           end_date: str = None, 
                           freq: str = 'D',
                           use_cache: bool = True) -> Dict[str, pd.Series]:
    """
    获取Fama-French三因子模型的因子数据
    
    Args:
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        use_cache: 是否使用缓存数据
        
    Returns:
        包含市场风险溢价、SMB、HML因子的字典
    """
    logger.info(f"获取Fama-French三因子数据: {start_date} 至 {end_date}, 频率: {freq}")
    
    # 缓存文件路径
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"fama_french_{freq}.csv")
    
    # 检查缓存
    if use_cache and os.path.exists(cache_file):
        try:
            cache_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logger.info(f"成功从缓存加载Fama-French因子数据: {len(cache_data)} 条记录")
            
            # 过滤日期范围
            if start_date:
                start_date = pd.to_datetime(start_date)
                cache_data = cache_data[cache_data.index >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date)
                cache_data = cache_data[cache_data.index <= end_date]
                
            if not cache_data.empty:
                # 转换为字典
                factors_dict = {
                    'market_returns': cache_data['MKT'],
                    'market_excess_returns': cache_data['MKT_RF'],
                    'smb': cache_data['SMB'],
                    'hml': cache_data['HML'],
                    'risk_free_rate': cache_data['RF']
                }
                return factors_dict
        except Exception as e:
            logger.warning(f"从缓存加载Fama-French因子数据失败: {e}")
    
    try:
        # 尝试使用TuShare计算Fama-French因子
        try:
            import tushare as ts
            if os.environ.get('TUSHARE_TOKEN', ''):
                factors_dict = calculate_fama_french_factors_tushare(start_date, end_date, freq)
                
                # 保存到缓存
                try:
                    # 转换为DataFrame保存
                    factors_df = pd.DataFrame({
                        'MKT': factors_dict['market_returns'],
                        'MKT_RF': factors_dict['market_excess_returns'],
                        'SMB': factors_dict['smb'],
                        'HML': factors_dict['hml'],
                        'RF': factors_dict['risk_free_rate']
                    })
                    factors_df.to_csv(cache_file)
                    logger.info(f"Fama-French因子数据已保存到缓存: {cache_file}")
                except Exception as e:
                    logger.warning(f"保存Fama-French因子数据到缓存失败: {e}")
                
                return factors_dict
            else:
                logger.warning("未找到TuShare token，无法使用TuShare计算因子")
        except ImportError:
            logger.warning("未找到tushare库，无法使用TuShare计算因子")
            
        # 如果TuShare计算失败，使用模拟数据
        return _generate_mock_fama_french_factors(start_date, end_date, freq)
            
    except Exception as e:
        logger.error(f"获取Fama-French因子时出错: {e}")
        logger.error(traceback.format_exc())
        return _generate_mock_fama_french_factors(start_date, end_date, freq)

def _generate_mock_fama_french_factors(start_date: str, end_date: str, freq: str = 'D') -> Dict[str, pd.Series]:
    """
    生成模拟的Fama-French三因子数据
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        包含市场风险溢价、SMB、HML因子的字典
    """
    logger.info(f"生成模拟Fama-French三因子数据, 频率: {freq}")
    
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
    
    # 获取市场收益率
    market_returns = get_market_returns(start_date=start_date, end_date=end_date, freq=freq)
    
    # 如果市场收益率为空，创建一个模拟的市场收益率
    if market_returns.empty:
        market_returns = _generate_mock_market_returns(start_date, end_date, freq)
    
    # 使用市场收益率的索引
    date_range = market_returns.index
    
    # 生成模拟的SMB和HML因子
    np.random.seed(42)  # 设置随机种子以保证可复现性
    
    # SMB因子，均值较小，波动适中
    # 根据频率调整参数
    if freq == 'W':
        smb_mean, smb_std = 0.0005, 0.01
        hml_mean, hml_std = 0.0008, 0.015
    elif freq == 'M':
        smb_mean, smb_std = 0.002, 0.02
        hml_mean, hml_std = 0.003, 0.025
    else:  # 日频率
        smb_mean, smb_std = 0.0001, 0.005
        hml_mean, hml_std = 0.0002, 0.006
    
    smb = pd.Series(np.random.normal(smb_mean, smb_std, len(date_range)), index=date_range)
    
    # HML因子，均值适中，波动较大
    hml = pd.Series(np.random.normal(hml_mean, hml_std, len(date_range)), index=date_range)
    
    # 获取无风险利率
    risk_free = get_risk_free_rate(start_date=start_date, end_date=end_date, freq=freq)
    
    # 如果无风险利率为空，创建一个模拟的无风险利率
    if risk_free.empty:
        risk_free = _generate_mock_risk_free_rate(start_date, end_date, freq)
    
    # 对齐所有数据的索引
    common_index = date_range
    if len(risk_free) > 0:
        common_index = date_range.intersection(risk_free.index)
    
    # 计算市场风险溢价
    market_rf = market_returns.reindex(common_index) - risk_free.reindex(common_index)
    
    # 创建结果字典
    factors = {
        'market_returns': market_returns.reindex(common_index),
        'market_excess_returns': market_rf,
        'smb': smb.reindex(common_index),
        'hml': hml.reindex(common_index),
        'risk_free_rate': risk_free.reindex(common_index)
    }
    
    # 用0填充缺失值
    for key in factors:
        factors[key] = factors[key].fillna(0)
    
    return factors

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
                               freq: str = 'D') -> Tuple[pd.DataFrame, pd.DataFrame]:
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

def get_macro_economic_data(indicator_type: str = "gdp", 
                           start_date: str = None, 
                           end_date: str = None) -> pd.DataFrame:
    """
    获取宏观经济指标数据
    
    Args:
        indicator_type: 指标类型，如 "gdp", "cpi", "interest_rate", "m2"
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        
    Returns:
        包含宏观经济数据的DataFrame
    """
    logger.info(f"获取宏观经济数据: {indicator_type}, {start_date} 至 {end_date}")
    
    try:
        # 尝试使用AKShare获取宏观经济数据
        try:
            import akshare as ak
            
            # 处理日期格式
            if start_date and len(start_date) == 8:
                start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            if end_date and len(end_date) == 8:
                end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            
            # 根据指标类型获取不同的数据
            if indicator_type == "gdp":
                # 获取GDP数据
                gdp_data = ak.macro_china_gdp()
                
                if not gdp_data.empty:
                    # 处理数据
                    gdp_data = gdp_data.rename(columns={"季度": "date"})
                    
                    # 转换日期格式
                    gdp_data["date"] = pd.to_datetime(gdp_data["date"])
                    
                    # 过滤日期范围
                    if start_date:
                        start_date = pd.to_datetime(start_date)
                        gdp_data = gdp_data[gdp_data["date"] >= start_date]
                    if end_date:
                        end_date = pd.to_datetime(end_date)
                        gdp_data = gdp_data[gdp_data["date"] <= end_date]
                    
                    logger.info(f"成功获取GDP数据: {len(gdp_data)} 条记录")
                    return gdp_data
                else:
                    logger.warning("无法获取GDP数据")
                    
            elif indicator_type == "cpi":
                # 获取CPI数据
                cpi_data = ak.macro_china_cpi()
                
                if not cpi_data.empty:
                    # 处理数据
                    cpi_data = cpi_data.rename(columns={"月份": "date"})
                    
                    # 转换日期格式
                    cpi_data["date"] = pd.to_datetime(cpi_data["date"])
                    
                    # 过滤日期范围
                    if start_date:
                        start_date = pd.to_datetime(start_date)
                        cpi_data = cpi_data[cpi_data["date"] >= start_date]
                    if end_date:
                        end_date = pd.to_datetime(end_date)
                        cpi_data = cpi_data[cpi_data["date"] <= end_date]
                    
                    logger.info(f"成功获取CPI数据: {len(cpi_data)} 条记录")
                    return cpi_data
                else:
                    logger.warning("无法获取CPI数据")
                    
            elif indicator_type == "interest_rate":
                # 获取利率数据
                interest_data = get_risk_free_rate(start_date, end_date, freq='D').to_frame(name="interest_rate")
                interest_data = interest_data.reset_index().rename(columns={"index": "date"})
                
                logger.info(f"成功获取利率数据: {len(interest_data)} 条记录")
                return interest_data
                
            elif indicator_type == "m2":
                # 获取M2数据
                m2_data = ak.macro_china_money_supply()
                
                if not m2_data.empty:
                    # 处理数据
                    m2_data = m2_data.rename(columns={"月份": "date"})
                    
                    # 转换日期格式
                    m2_data["date"] = pd.to_datetime(m2_data["date"])
                    
                    # 过滤日期范围
                    if start_date:
                        start_date = pd.to_datetime(start_date)
                        m2_data = m2_data[m2_data["date"] >= start_date]
                    if end_date:
                        end_date = pd.to_datetime(end_date)
                        m2_data = m2_data[m2_data["date"] <= end_date]
                    
                    logger.info(f"成功获取M2数据: {len(m2_data)} 条记录")
                    return m2_data
                else:
                    logger.warning("无法获取M2数据")
                    
            else:
                logger.warning(f"不支持的指标类型: {indicator_type}")
                
        except ImportError:
            logger.warning("未找到akshare库，无法获取宏观经济数据")
        except Exception as e:
            logger.warning(f"使用AKShare获取宏观数据失败: {e}")
    
    except Exception as e:
        logger.error(f"获取宏观经济数据时出错: {e}")
        logger.error(traceback.format_exc())
    
    # 如果所有尝试都失败，使用模拟数据
    return _generate_mock_macro_data(indicator_type, start_date, end_date)

def _generate_mock_macro_data(indicator_type: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    生成模拟的宏观经济数据
    
    Args:
        indicator_type: 指标类型
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        模拟的宏观经济数据DataFrame
    """
    logger.info(f"生成模拟{indicator_type}数据")
    
    # 处理日期参数
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        # 默认生成五年的数据
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365*5)).strftime("%Y-%m-%d")
    
    # 确保日期格式一致
    if isinstance(start_date, str) and len(start_date) == 8:
        start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    if isinstance(end_date, str) and len(end_date) == 8:
        end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    
    # 根据指标类型生成不同频率的日期范围
    if indicator_type == "gdp":
        # GDP数据是季度数据
        date_range = pd.date_range(start=start_date, end=end_date, freq='Q')
    elif indicator_type in ["cpi", "m2"]:
        # CPI和M2数据是月度数据
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    else:
        # 其他数据默认为月度数据
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # 生成模拟数据
    np.random.seed(42)  # 设置随机种子以保证可复现性
    
    if indicator_type == "gdp":
        # 创建GDP数据
        # 基础GDP值（单位：亿元）
        base_gdp = 100000
        # 增长率范围（6%-8%）
        growth_rates = np.random.uniform(0.06, 0.08, len(date_range))
        
        # 计算累积GDP
        gdp_values = []
        current_gdp = base_gdp
        for rate in growth_rates:
            current_gdp *= (1 + rate)
            gdp_values.append(current_gdp)
        
        # 创建DataFrame
        mock_data = pd.DataFrame({
            "date": date_range,
            "国内生产总值_当季值": gdp_values,
            "国内生产总值_累计值": np.cumsum(gdp_values),
            "国内生产总值_同比增长": growth_rates * 100
        })
        
    elif indicator_type == "cpi":
        # 创建CPI数据
        # CPI同比增长范围（1%-4%）
        cpi_yoy = np.random.uniform(0.01, 0.04, len(date_range)) * 100
        
        # 创建DataFrame
        mock_data = pd.DataFrame({
            "date": date_range,
            "全国_同比": cpi_yoy,
            "全国_环比": np.random.uniform(-0.5, 1.5, len(date_range)),
            "城市_同比": cpi_yoy + np.random.uniform(-0.5, 0.5, len(date_range)),
            "农村_同比": cpi_yoy + np.random.uniform(-0.5, 0.5, len(date_range))
        })
        
    elif indicator_type == "m2":
        # 创建M2数据
        # 基础M2值（单位：亿元）
        base_m2 = 2000000
        # 增长率范围（8%-12%）
        growth_rates = np.random.uniform(0.08, 0.12, len(date_range))
        
        # 计算累积M2
        m2_values = []
        current_m2 = base_m2
        for rate in growth_rates:
            current_m2 *= (1 + rate/12)  # 月度增长率
            m2_values.append(current_m2)
        
        # 创建DataFrame
        mock_data = pd.DataFrame({
            "date": date_range,
            "货币和准货币(M2)": m2_values,
            "M2同比增长": growth_rates * 100,
            "M1": np.array(m2_values) * 0.3,  # M1约为M2的30%
            "M1同比增长": growth_rates * 100 + np.random.uniform(-2, 2, len(date_range))
        })
        
    else:
        # 创建默认模拟数据
        mock_data = pd.DataFrame({
            "date": date_range,
            "value": np.random.normal(100, 10, len(date_range)),
            "yoy_change": np.random.uniform(-5, 8, len(date_range))
        })
    
    return mock_data

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

def get_industry_index_returns(industry_codes: Union[str, List[str]],
                              start_date: str = None,
                              end_date: str = None,
                              freq: str = 'D') -> pd.DataFrame:
    """
    获取行业指数收益率数据
    
    Args:
        industry_codes: 行业指数代码或代码列表，例如"801780"(银行)
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        包含多个行业指数收益率的DataFrame
    """
    logger.info(f"获取行业指数收益率: {start_date} 至 {end_date}, 频率: {freq}")
    
    # 转换为列表
    if isinstance(industry_codes, str):
        industry_codes = [industry_codes]
    
    # 定义行业名称映射
    industry_names = {
        '801010': '农林牧渔', '801020': '采掘', '801030': '化工', '801040': '钢铁',
        '801050': '有色金属', '801080': '电子', '801110': '家用电器', '801120': '食品饮料',
        '801130': '纺织服装', '801140': '轻工制造', '801150': '医药生物', '801160': '公用事业',
        '801170': '交通运输', '801180': '房地产', '801200': '商业贸易', '801210': '休闲服务',
        '801230': '综合', '801710': '建筑材料', '801720': '建筑装饰', '801730': '电气设备',
        '801740': '国防军工', '801750': '计算机', '801760': '传媒', '801770': '通信',
        '801780': '银行', '801790': '非银金融', '801880': '汽车', '801890': '机械设备',
        '801950': '煤炭', '801960': '石油石化', '801970': '环保', '801980': '美容护理'
    }
    
    # 获取指数收益率数据
    returns_dict = {}
    
    for code in industry_codes:
        try:
            # 获取指数数据
            index_data = get_index_data(code, None, start_date, end_date, freq)
            
            if not index_data.empty:
                # 确保日期列为日期类型
                if 'date' in index_data.columns:
                    index_data['date'] = pd.to_datetime(index_data['date'])
                    index_data = index_data.set_index('date')
                elif 'trade_date' in index_data.columns:
                    index_data['trade_date'] = pd.to_datetime(index_data['trade_date'])
                    index_data = index_data.set_index('trade_date')
                
                # 计算收益率
                if 'close' in index_data.columns:
                    returns = index_data['close'].pct_change().dropna()
                    
                    # 获取行业名称
                    industry_name = industry_names.get(code, code)
                    
                    returns_dict[industry_name] = returns
                    logger.info(f"成功获取行业 {industry_name}({code}) 收益率数据: {len(returns)} 条记录")
                else:
                    logger.warning(f"行业指数 {code} 数据中不包含close列")
            else:
                logger.warning(f"无法获取行业指数 {code} 数据")
                
        except Exception as e:
            logger.error(f"获取行业指数 {code} 收益率时出错: {e}")
            logger.error(traceback.format_exc())
    
    # 如果没有获取到数据，返回空DataFrame
    if not returns_dict:
        logger.warning("未获取到任何行业指数收益率数据")
        return pd.DataFrame()
    
    # 合并为DataFrame
    returns_df = pd.DataFrame(returns_dict)
    
    return returns_df

def get_industry_rotation_factors(start_date: str = None,
                                 end_date: str = None,
                                 freq: str = 'W') -> pd.DataFrame:
    """
    获取行业轮动因子数据 (包括行业动量、估值、成长性等)
    
    Args:
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        freq: 数据频率，推荐'W'为周度或'M'为月度
        
    Returns:
        包含行业轮动因子的DataFrame
    """
    logger.info(f"获取行业轮动因子数据: {start_date} 至 {end_date}, 频率: {freq}")
    
    try:
        # 1. 获取主要行业指数代码
        main_industries = [
            '801010', '801020', '801030', '801040', '801050', '801080', 
            '801110', '801120', '801150', '801180', '801730', '801750', 
            '801760', '801770', '801780', '801790', '801880', '801890'
        ]
        
        # 2. 获取行业收益率数据
        industry_returns = get_industry_index_returns(main_industries, start_date, end_date, freq)
        
        if industry_returns.empty:
            logger.warning("无法获取行业收益率数据")
            return pd.DataFrame()
        
        # 3. 计算行业动量因子
        factors_df = pd.DataFrame(index=industry_returns.index)
        
        # 计算各个行业的动量因子 (过去1/3/6个月的回报)
        for industry in industry_returns.columns:
            # 过去1个月动量
            factors_df[f'{industry}_MOM_1M'] = industry_returns[industry].rolling(window=4 if freq == 'W' else 1).sum()
            
            # 过去3个月动量
            factors_df[f'{industry}_MOM_3M'] = industry_returns[industry].rolling(window=12 if freq == 'W' else 3).sum()
            
            # 过去6个月动量
            factors_df[f'{industry}_MOM_6M'] = industry_returns[industry].rolling(window=24 if freq == 'W' else 6).sum()
        
        # 4. 尝试获取行业估值数据 (如PE、PB等)
        # 这部分如果TuShare和AKShare没有直接提供，可能需要构建或用模拟数据
        
        logger.info(f"成功计算行业轮动因子: {len(factors_df)} 条记录")
        return factors_df
        
    except Exception as e:
        logger.error(f"获取行业轮动因子时出错: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def get_sector_index_returns(start_date: str = None,
                            end_date: str = None,
                            freq: str = 'D') -> pd.DataFrame:
    """
    获取主要板块指数收益率数据 (上证、深证、创业板、中小板等)
    
    Args:
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        包含主要板块指数收益率的DataFrame
    """
    logger.info(f"获取主要板块指数收益率: {start_date} 至 {end_date}, 频率: {freq}")
    
    # 定义主要板块指数
    sector_indices = {
        'sh000001': '上证指数', 
        'sz399001': '深证成指',
        'sz399006': '创业板指', 
        'sz399005': '中小板指',
        'sh000300': '沪深300', 
        'sh000905': '中证500',
        'sh000016': '上证50',  
        'sh000852': '中证1000'
    }
    
    # 获取各指数收益率
    returns_dict = {}
    
    for code, name in sector_indices.items():
        try:
            # 处理指数代码
            if code.startswith('sh'):
                index_code = code[2:]
            elif code.startswith('sz'):
                index_code = code[2:]
            else:
                index_code = code
            
            # 获取指数数据
            index_data = get_index_data(index_code, None, start_date, end_date, freq)
            
            if not index_data.empty:
                # 确保日期列为日期类型
                if 'date' in index_data.columns:
                    index_data['date'] = pd.to_datetime(index_data['date'])
                    index_data = index_data.set_index('date')
                elif 'trade_date' in index_data.columns:
                    index_data['trade_date'] = pd.to_datetime(index_data['trade_date'])
                    index_data = index_data.set_index('trade_date')
                
                # 计算收益率
                if 'close' in index_data.columns:
                    returns = index_data['close'].pct_change().dropna()
                    returns_dict[name] = returns
                    logger.info(f"成功获取 {name} 收益率数据: {len(returns)} 条记录")
                else:
                    logger.warning(f"指数 {code} 数据中不包含close列")
            else:
                logger.warning(f"无法获取指数 {code} 数据")
                
        except Exception as e:
            logger.error(f"获取指数 {code} 收益率时出错: {e}")
            logger.error(traceback.format_exc())
    
    # 如果没有获取到数据，返回空DataFrame
    if not returns_dict:
        logger.warning("未获取到任何板块指数收益率数据")
        return pd.DataFrame()
    
    # 合并为DataFrame
    returns_df = pd.DataFrame(returns_dict)
    
    return returns_df

def get_style_index_returns(start_date: str = None,
                           end_date: str = None,
                           freq: str = 'D') -> pd.DataFrame:
    """
    获取风格指数收益率数据 (大盘成长、小盘价值等)
    
    Args:
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        freq: 数据频率，'D'为日度，'W'为周度，'M'为月度
        
    Returns:
        包含风格指数收益率的DataFrame
    """
    logger.info(f"获取风格指数收益率: {start_date} 至 {end_date}, 频率: {freq}")
    
    # 定义风格指数
    style_indices = {
        'sh000919': '300价值',
        'sh000918': '300成长',
        'sh000922': '中证红利',
        'sh000925': '基本面50',
        'sh000978': '医药100',
        'sh000991': '全指医药'
    }
    
    # 获取各指数收益率
    returns_dict = {}
    
    for code, name in style_indices.items():
        try:
            # 处理指数代码
            if code.startswith('sh'):
                index_code = code[2:]
            elif code.startswith('sz'):
                index_code = code[2:]
            else:
                index_code = code
            
            # 获取指数数据
            index_data = get_index_data(index_code, None, start_date, end_date, freq)
            
            if not index_data.empty:
                # 确保日期列为日期类型
                if 'date' in index_data.columns:
                    index_data['date'] = pd.to_datetime(index_data['date'])
                    index_data = index_data.set_index('date')
                elif 'trade_date' in index_data.columns:
                    index_data['trade_date'] = pd.to_datetime(index_data['trade_date'])
                    index_data = index_data.set_index('trade_date')
                
                # 计算收益率
                if 'close' in index_data.columns:
                    returns = index_data['close'].pct_change().dropna()
                    returns_dict[name] = returns
                    logger.info(f"成功获取 {name} 收益率数据: {len(returns)} 条记录")
                else:
                    logger.warning(f"指数 {code} 数据中不包含close列")
            else:
                logger.warning(f"无法获取指数 {code} 数据")
                
        except Exception as e:
            logger.error(f"获取指数 {code} 收益率时出错: {e}")
            logger.error(traceback.format_exc())
    
    # 如果没有获取到数据，返回空DataFrame
    if not returns_dict:
        logger.warning("未获取到任何风格指数收益率数据")
        return pd.DataFrame()
    
    # 合并为DataFrame
    returns_df = pd.DataFrame(returns_dict)
    
    return returns_df

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
            betas[symbol] = model.params[1]  # 贝塔系数
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
            rolling_betas.loc[stock_returns.index[i]] = model.params[1]  # 贝塔系数
        except Exception as e:
            logger.debug(f"计算 {stock_returns.index[i]} 的贝塔系数时出错: {e}")
            rolling_betas.loc[stock_returns.index[i]] = np.nan
    
    # 去除缺失值
    rolling_betas = rolling_betas.dropna()
    
    logger.info(f"成功计算股票 {stock_symbol} 的滚动贝塔系数: {len(rolling_betas)} 条记录")
    return rolling_betas