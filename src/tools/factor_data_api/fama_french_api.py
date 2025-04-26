"""
Fama-French因子API - 提供Fama-French三因子模型相关数据的计算和获取功能
"""

import os
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
from typing import Dict

from .base import logger
from .risk_free_api import get_risk_free_rate, _generate_mock_risk_free_rate
from .market_data_api import get_market_returns, _generate_mock_market_returns

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
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data', 'cache')
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