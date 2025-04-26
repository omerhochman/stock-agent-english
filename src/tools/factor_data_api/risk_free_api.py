"""
无风险利率API - 提供无风险利率数据获取功能
"""

import os
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
from .base import logger

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
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data', 'cache')
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
            # 以下是使用TuShare获取LPR(贷款基础利率)的代码
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