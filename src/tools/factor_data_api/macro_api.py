"""
宏观经济数据API - 提供宏观经济数据的获取和处理功能
"""

import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta

from .base import logger
from .risk_free_api import get_risk_free_rate

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
        date_range = pd.date_range(start=start_date, end=end_date, freq='QE')
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