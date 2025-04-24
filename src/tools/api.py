from typing import Dict, Any
import pandas as pd
import os
import traceback
from datetime import datetime, timedelta
from src.utils.logging_config import setup_logger
from src.tools.data_processor import data_processor
from src.tools.data_source_adapter import DataAPI

import urllib3
urllib3.disable_warnings()
import logging
logging.getLogger("urllib3").setLevel(logging.WARNING)

# 设置日志记录
logger = setup_logger('api')

# 创建数据API实例
data_api = DataAPI()

def get_financial_metrics(symbol: str) -> Dict[str, Any]:
    """获取财务指标数据"""
    logger.info(f"Getting financial indicators for {symbol}...")
    try:
        # 使用数据API获取财务指标
        metrics = data_api.get_financial_metrics(symbol)
        
        # 使用数据处理器处理财务数据
        processed_metrics = data_processor.process_financial_data(metrics)
        
        return processed_metrics
    except Exception as e:
        logger.error(f"Error getting financial indicators: {e}")
        return [{}]


def get_financial_statements(symbol: str) -> Dict[str, Any]:
    """获取财务报表数据"""
    logger.info(f"Getting financial statements for {symbol}...")
    try:
        # 使用数据API获取财务报表
        statements = data_api.get_financial_statements(symbol)
        
        # 使用数据处理器处理财务数据
        processed_statements = data_processor.process_financial_data(statements)
        
        return processed_statements
    except Exception as e:
        logger.error(f"Error getting financial statements: {e}")
        default_item = {
            "net_income": 0,
            "operating_revenue": 0,
            "operating_profit": 0,
            "working_capital": 0,
            "depreciation_and_amortization": 0,
            "capital_expenditure": 0,
            "free_cash_flow": 0
        }
        return [default_item, default_item]


def get_market_data(symbol: str) -> Dict[str, Any]:
    """获取市场数据"""
    try:
        # 使用数据API获取市场数据
        market_data = data_api.get_market_data(symbol)
        return market_data
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return {}


def get_price_history(symbol: str, start_date: str = None, end_date: str = None, adjust: str = "qfq") -> pd.DataFrame:
    """获取历史价格数据

    Args:
        symbol: 股票代码
        start_date: 开始日期，格式：YYYY-MM-DD，如果为None则默认获取过去一年的数据
        end_date: 结束日期，格式：YYYY-MM-DD，如果为None则使用昨天作为结束日期
        adjust: 复权类型，可选值：
               - "": 不复权
               - "qfq": 前复权（默认）
               - "hfq": 后复权

    Returns:
        包含价格数据的DataFrame
    """
    try:
        # 使用数据API获取价格数据
        df = data_api.get_price_data(symbol, start_date, end_date)
        
        # 数据处理和验证
        if df is None or df.empty:
            logger.warning(f"No price history data available for {symbol}")
            # 尝试使用TuShare作为最后手段，即使适配器已经尝试过
            try:
                import tushare as ts
                token = os.environ.get('TUSHARE_TOKEN', '')
                if token:
                    ts.set_token(token)
                    pro = ts.pro_api()
                    
                    # 处理日期
                    if not end_date:
                        end_date = datetime.now() - timedelta(days=1)
                    else:
                        if isinstance(end_date, str):
                            end_date = datetime.strptime(end_date, '%Y-%m-%d')
                            
                    if not start_date:
                        start_date = end_date - timedelta(days=365)
                    else:
                        if isinstance(start_date, str):
                            start_date = datetime.strptime(start_date, '%Y-%m-%d')
                    
                    # 转换为TuShare格式
                    ts_start = start_date.strftime('%Y%m%d')
                    ts_end = end_date.strftime('%Y%m%d')
                    
                    # 添加前缀
                    if symbol.startswith(('sh', 'sz', 'bj')):
                        code = symbol[2:]
                        prefix = symbol[:2].upper()
                        ts_code = f"{code}.{prefix}"
                    else:
                        if symbol.startswith('6'):
                            ts_code = f"{symbol}.SH"
                        elif symbol.startswith(('0', '3')):
                            ts_code = f"{symbol}.SZ"
                        else:
                            ts_code = f"{symbol}.SH"
                    
                    # 直接使用TuShare API
                    logger.info(f"直接尝试TuShare获取数据: {ts_code}")
                    ts_df = pro.daily(ts_code=ts_code, start_date=ts_start, end_date=ts_end)
                    
                    if not ts_df.empty:
                        logger.info(f"TuShare最后手段成功获取到{len(ts_df)}条记录")
                        # 重命名列
                        ts_df = ts_df.rename(columns={
                            "trade_date": "date",
                            "vol": "volume",
                            "pct_chg": "pct_change",
                            "change": "change_amount"
                        })
                        
                        # 转换日期
                        ts_df["date"] = pd.to_datetime(ts_df["date"], format="%Y%m%d")
                        
                        # 排序
                        ts_df = ts_df.sort_values("date", ascending=True)
                        
                        # 处理数据
                        processed_df = data_processor.process_price_data(ts_df)
                        return processed_df
            except Exception as e:
                logger.error(f"直接使用TuShare也失败: {e}")
            
            # 如果所有尝试都失败，返回空DataFrame
            return pd.DataFrame()
        
        # 检查DataFrame中必须包含的列
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            # 添加缺失的列
            for col in missing_columns:
                df[col] = 0.0 if col != 'date' else pd.to_datetime('today')
        
        # 使用数据处理器增强数据质量
        processed_df = data_processor.process_price_data(df)
        logger.info(f"Successfully processed price data ({len(processed_df)} records)")
        
        return processed_df

    except Exception as e:
        logger.error(f"Error getting price history: {e}")
        logger.debug(f"Error details: {traceback.format_exc()}")
        return pd.DataFrame()


def prices_to_df(prices):
    """Convert price data to DataFrame with standardized column names"""
    try:
        df = pd.DataFrame(prices)

        # 标准化列名映射
        column_mapping = {
            '收盘': 'close',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'change_percent',
            '涨跌额': 'change_amount',
            '换手率': 'turnover_rate'
        }

        # 重命名列
        for cn, en in column_mapping.items():
            if cn in df.columns:
                df[en] = df[cn]

        # 确保必要的列存在
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0  # 使用0填充缺失的必要列
                
        # 使用数据处理器增强数据质量
        df = data_processor.process_price_data(df)

        return df
    except Exception as e:
        logger.error(f"Error converting price data: {str(e)}")
        # 返回一个包含必要列的空DataFrame
        return pd.DataFrame(columns=['close', 'open', 'high', 'low', 'volume'])


def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """获取股票价格数据

    Args:
        ticker: 股票代码
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD

    Returns:
        包含价格数据的DataFrame
    """
    return get_price_history(ticker, start_date, end_date)