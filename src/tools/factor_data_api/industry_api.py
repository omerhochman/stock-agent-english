"""
行业数据API - 提供行业相关数据的获取和处理功能
"""

import pandas as pd
import traceback
from typing import Union, List

from .base import logger, data_api
from .market_data_api import get_index_data

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
    
    #  如果没有获取到数据，返回空DataFrame
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