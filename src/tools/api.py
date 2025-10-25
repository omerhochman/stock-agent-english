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

# Setup logging
logger = setup_logger('api')

# Create data API instance
data_api = DataAPI()

def handle_exception(func_name: str, exception: Exception, default_return=None):
    """Unified exception handling and return default values"""
    logger.error(f"Error in {func_name}: {exception}")
    logger.debug(f"Error details: {traceback.format_exc()}")
    return default_return

def get_financial_metrics(symbol: str) -> Dict[str, Any]:
    """Get financial metrics data"""
    logger.info(f"Getting financial indicators for {symbol}...")
    try:
        # Use data API to get financial metrics
        metrics = data_api.get_financial_metrics(symbol)
        
        # Use data processor to process financial data
        processed_metrics = data_processor.process_financial_data(metrics)
        
        return processed_metrics
    except Exception as e:
        return handle_exception("get_financial_metrics", e, [{}])


def get_financial_statements(symbol: str) -> Dict[str, Any]:
    """Get financial statements data"""
    logger.info(f"Getting financial statements for {symbol}...")
    try:
        # Use data API to get financial statements
        statements = data_api.get_financial_statements(symbol)
        
        # Use data processor to process financial data
        processed_statements = data_processor.process_financial_data(statements)
        
        return processed_statements
    except Exception as e:
        default_item = {
            "net_income": 0,
            "operating_revenue": 0,
            "operating_profit": 0,
            "working_capital": 0,
            "depreciation_and_amortization": 0,
            "capital_expenditure": 0,
            "free_cash_flow": 0
        }
        return handle_exception("get_financial_statements", e, [default_item, default_item])


def get_market_data(symbol: str) -> Dict[str, Any]:
    """Get market data"""
    try:
        # Use data API to get market data
        market_data = data_api.get_market_data(symbol)
        return market_data
    except Exception as e:
        return handle_exception("get_market_data", e, {})


def get_price_history(symbol: str, start_date: str = None, end_date: str = None, adjust: str = "qfq") -> pd.DataFrame:
    """Get historical price data

    Args:
        symbol: Stock code
        start_date: Start date, format: YYYY-MM-DD, if None defaults to past year data
        end_date: End date, format: YYYY-MM-DD, if None uses yesterday as end date
        adjust: Adjustment type, options:
               - "": No adjustment
               - "qfq": Forward adjustment (default)
               - "hfq": Backward adjustment

    Returns:
        DataFrame containing price data
    """
    try:
        # Use data API to get price data
        df = data_api.get_price_data(symbol, start_date, end_date)
        
        # Data processing and validation
        if df is None or df.empty:
            logger.warning(f"No price history data available for {symbol}")
            # Try using TuShare as last resort
            df = _try_tushare_last_resort(symbol, start_date, end_date)
            if df.empty:
                return pd.DataFrame()
        
        # Check DataFrame contains all required columns
        df = _ensure_required_columns(df)
        
        # Use data processor to enhance data quality
        processed_df = data_processor.process_price_data(df)
        logger.info(f"Successfully processed price data ({len(processed_df)} records)")
        
        return processed_df

    except Exception as e:
        return handle_exception("get_price_history", e, pd.DataFrame())


def _try_tushare_last_resort(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Try using TuShare as last resort to get data"""
    try:
        import tushare as ts
        token = os.environ.get('TUSHARE_TOKEN', '')
        if token:
            ts.set_token(token)
            pro = ts.pro_api()
            
            # Process dates
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
            
            # Convert to TuShare format
            ts_start = start_date.strftime('%Y%m%d')
            ts_end = end_date.strftime('%Y%m%d')
            
            # Add prefix
            ts_code = _convert_to_tushare_code(symbol)
            
            # Directly use TuShare API
            logger.info(f"Directly trying TuShare to get data: {ts_code}")
            ts_df = pro.daily(ts_code=ts_code, start_date=ts_start, end_date=ts_end)
            
            if not ts_df.empty:
                logger.info(f"TuShare last resort successfully obtained {len(ts_df)} records")
                # Rename columns
                ts_df = ts_df.rename(columns={
                    "trade_date": "date",
                    "vol": "volume",
                    "pct_chg": "pct_change",
                    "change": "change_amount"
                })
                
                # Convert dates
                ts_df["date"] = pd.to_datetime(ts_df["date"], format="%Y%m%d")
                
                # Sort
                ts_df = ts_df.sort_values("date", ascending=True)
                
                # Process data
                processed_df = data_processor.process_price_data(ts_df)
                return processed_df
    except Exception as e:
        logger.error(f"Directly using TuShare also failed: {e}")
    
    # If all attempts fail, return empty DataFrame
    return pd.DataFrame()


def _convert_to_tushare_code(symbol: str) -> str:
    """Convert stock code to TuShare format"""
    # Add prefix
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
    
    return ts_code


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame contains all required columns"""
    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Missing required columns: {missing_columns}")
        # Add missing columns
        for col in missing_columns:
            df[col] = 0.0 if col != 'date' else pd.to_datetime('today')
    
    return df


def prices_to_df(prices):
    """Convert price data to DataFrame with proper date index"""
    try:
        df = pd.DataFrame(prices)

        # Standardize column name mapping
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

        # Rename columns
        for cn, en in column_mapping.items():
            if cn in df.columns:
                df[en] = df[cn]

        # Ensure necessary columns exist
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0  # Use 0 to fill missing necessary columns
        
        # Ensure date column is in correct date format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            # Set date as index and sort
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
                
        # Use data processor to enhance data quality
        df = data_processor.process_price_data(df)

        return df
    except Exception as e:
        logger.error(f"prices_to_df error: {e}")
        return pd.DataFrame(columns=['close', 'open', 'high', 'low', 'volume'])


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get stock price data

    Args:
        ticker: Stock code
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD

    Returns:
        DataFrame containing price data
    """
    return get_price_history(ticker, start_date, end_date)