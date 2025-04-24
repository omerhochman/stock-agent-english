import os
import pandas as pd
import time
import json
import functools
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
import traceback

import urllib3
urllib3.disable_warnings()
logging.getLogger("urllib3").setLevel(logging.WARNING)

# 尝试导入需要的库
try:
    import akshare as ak
except ImportError:
    ak = None
    logging.warning("AKShare not found, please install with: pip install akshare")

try:
    import tushare as ts
except ImportError:
    ts = None
    logging.warning("TuShare not found, please install with: pip install tushare")

# 获取TuShare API密钥
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '')

# 如果TuShare可用且有token，则进行初始化
if ts and TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
    logging.info("TuShare API initialized with token")
else:
    logging.warning("TuShare token not found or TuShare not available")

# 数据缓存目录设置
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# 重试装饰器
def retry(max_tries=3, delay_seconds=2, backoff=2, exceptions=(Exception,)):
    """
    重试装饰器：当函数失败时自动重试
    
    Args:
        max_tries: 最大重试次数
        delay_seconds: 初始延迟秒数
        backoff: 延迟增长倍数
        exceptions: 需要重试的异常类型
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_tries, delay_seconds
            while mtries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logging.warning(f"函数 {func.__name__} 调用失败: {str(e)}, "
                                  f"剩余重试次数: {mtries-1}")
                    
                    mtries -= 1
                    if mtries == 0:
                        logging.error(f"函数 {func.__name__} 达到最大重试次数，抛出异常")
                        raise
                    
                    time.sleep(mdelay)
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator

def get_cached_data(key, fetch_func, *args, ttl_days=1, **kwargs):
    """
    从缓存获取数据，如果缓存过期或不存在则调用fetch_func获取
    
    Args:
        key: 缓存键
        fetch_func: 获取数据的函数
        ttl_days: 缓存有效期（天）
        args, kwargs: 传递给fetch_func的参数
    
    Returns:
        获取的数据
    """
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    
    # 检查缓存是否存在且有效
    if os.path.exists(cache_file):
        # 检查文件修改时间
        file_time = os.path.getmtime(cache_file)
        file_age = (time.time() - file_time) / (60 * 60 * 24)  # 转换为天
        
        if file_age < ttl_days:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    logging.info(f"Using cached data for {key}")
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load cache file: {e}")
    
    # 缓存不存在、过期或无效，调用获取函数
    logging.info(f"Fetching fresh data for {key}")
    data = fetch_func(*args, **kwargs)
    
    # 保存到缓存
    try:
        # 处理DataFrame对象，转换为可序列化的字典列表
        if isinstance(data, pd.DataFrame):
            # 确保日期列被正确处理
            df_dict = data.copy()
            for col in df_dict.columns:
                if pd.api.types.is_datetime64_any_dtype(df_dict[col]):
                    df_dict[col] = df_dict[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 转换为字典列表
            serializable_data = df_dict.to_dict(orient='records')
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False)
        else:
            # 非DataFrame对象直接保存
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
                
        logging.info(f"Data cached to {cache_file}")
    except Exception as e:
        logging.warning(f"Failed to save cache file: {e}")
        logging.debug(f"Cache error details: {traceback.format_exc()}")
    
    return data

class DataSourceAdapter:
    """数据源适配器，支持从AKShare和TuShare获取数据"""
    
    @staticmethod
    def convert_stock_code(symbol: str) -> tuple:
        """
        转换股票代码格式，返回AKShare和TuShare对应的代码格式
        返回元组: (akshare_code, tushare_code, exchange_prefix)
        """
        # 对已有代码格式的兼容
        if symbol.startswith(('sh', 'sz', 'bj')):
            # 已经带有交易所前缀的情况
            exchange_prefix = symbol[:2]
            code = symbol[2:]
        else:
            # 根据代码判断交易所
            if symbol.startswith('6'):
                exchange_prefix = 'sh'
            elif symbol.startswith(('0', '3')):
                exchange_prefix = 'sz'
            elif symbol.startswith('4'):
                exchange_prefix = 'bj'
            else:
                exchange_prefix = 'sh'  # 默认使用上交所
            code = symbol
        
        # AKShare格式
        akshare_code = symbol
        # TuShare格式: 代码.交易所缩写 (sh->SH, sz->SZ, bj->BJ)
        tushare_code = f"{code}.{exchange_prefix.upper()}"
        
        return akshare_code, tushare_code, exchange_prefix

    @retry(max_tries=3, delay_seconds=2)
    def get_price_history(self, symbol: str, start_date: str = None, end_date: str = None, adjust: str = "qfq") -> pd.DataFrame:
        """
        获取历史价格数据，优先使用AKShare，失败时切换到TuShare
        
        Args:
            symbol: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            adjust: 复权类型，qfq: 前复权, hfq: 后复权, "": 不复权
            
        Returns:
            包含价格数据的DataFrame
        """
        logging.info(f"Getting price history for {symbol} from {start_date} to {end_date}")
        
        # 处理日期参数
        current_date = datetime.now()
        yesterday = current_date - timedelta(days=1)
        
        if not end_date:
            end_date = yesterday
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date
            if end_date > yesterday:
                end_date = yesterday
                
        if not start_date:
            start_date = end_date - timedelta(days=365)
        else:
            start_date = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
            
        # 转换为字符串格式用于API调用    
        start_date_str = start_date.strftime("%Y%m%d") if isinstance(start_date, datetime) else start_date
        end_date_str = end_date.strftime("%Y%m%d") if isinstance(end_date, datetime) else end_date
        
        # 转换股票代码
        akshare_code, tushare_code, exchange_prefix = self.convert_stock_code(symbol)
        
        # 缓存键
        cache_key = f"price_hist_{symbol}_{start_date_str}_{end_date_str}_{adjust}"
        
        # 如果是当日数据，直接获取不使用缓存
        if (current_date - end_date).days < 1:
            df = self._fetch_price_history(akshare_code, tushare_code, start_date_str, end_date_str, adjust)
        else:
            # 使用缓存
            df_dict = get_cached_data(
                cache_key, 
                lambda: self._fetch_price_history(akshare_code, tushare_code, start_date_str, end_date_str, adjust), 
                ttl_days=3
            )
            
            if isinstance(df_dict, list):
                df = pd.DataFrame(df_dict)
                if not df.empty and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                logging.info(f"Successfully retrieved {len(df)} records from cache")
            else:
                df = pd.DataFrame()
                
        return df
    
    def _fetch_price_history(self, akshare_code: str, tushare_code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        """内部方法：从数据源获取价格历史数据"""
        df = pd.DataFrame()
        
        # 首先尝试使用AKShare
        if ak:
            try:
                logging.info(f"Fetching price history using AKShare for {akshare_code}")
                df = ak.stock_zh_a_hist(
                    symbol=akshare_code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )
                
                if not df.empty:
                    # 转换AKShare数据格式
                    df = df.rename(columns={
                        "日期": "date",
                        "开盘": "open",
                        "最高": "high",
                        "最低": "low",
                        "收盘": "close",
                        "成交量": "volume",
                        "成交额": "amount",
                        "振幅": "amplitude",
                        "涨跌幅": "pct_change",
                        "涨跌额": "change_amount",
                        "换手率": "turnover"
                    })
                    df["date"] = pd.to_datetime(df["date"])
                    
                    # 确保所有数值列已转换为float类型
                    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change', 'change_amount', 'turnover']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    logging.info(f"Successfully retrieved data from AKShare: {len(df)} records")
                    return df
                
            except Exception as e:
                logging.warning(f"AKShare data fetch failed: {str(e)}")
                logging.debug(f"AKShare error details: {traceback.format_exc()}")
                logging.info("Falling back to TuShare...")
        
        # 如果AKShare失败或不可用，尝试使用TuShare
        if ts and TUSHARE_TOKEN:
            try:
                logging.info(f"Fetching price history using TuShare for {tushare_code}")
                
                # 转换调整类型
                adj = None
                if adjust == 'qfq':
                    adj = 'qfq'
                elif adjust == 'hfq':
                    adj = 'hfq'
                
                # 确保日期格式正确
                if isinstance(start_date, str):
                    if '-' in start_date:
                        start_date = start_date.replace('-', '')
                    ts_start = start_date
                else:
                    ts_start = start_date.strftime('%Y%m%d')
                    
                if isinstance(end_date, str):
                    if '-' in end_date:
                        end_date = end_date.replace('-', '')
                    ts_end = end_date
                else:
                    ts_end = end_date.strftime('%Y%m%d')
                
                # 先尝试使用pro_bar获取复权数据
                try:
                    if adj:
                        df = ts.pro_bar(ts_code=tushare_code, adj=adj, start_date=ts_start, end_date=ts_end)
                    else:
                        df = ts.pro_bar(ts_code=tushare_code, start_date=ts_start, end_date=ts_end)
                except Exception as e:
                    logging.warning(f"TuShare pro_bar failed: {str(e)}, trying daily API...")
                    # 备选方法：获取日线数据
                    df = ts.pro_api().daily(ts_code=tushare_code, start_date=ts_start, end_date=ts_end)
                    
                    # 如果需要复权
                    if adj and not df.empty:
                        adj_factor = ts.pro_api().adj_factor(ts_code=tushare_code, start_date=ts_start, end_date=ts_end)
                        if not adj_factor.empty:
                            df = df.merge(adj_factor, on=['ts_code', 'trade_date'])
                            
                            # 前复权
                            if adj == 'qfq':
                                latest_factor = adj_factor['adj_factor'].iloc[0]
                                df['adj_factor'] = df['adj_factor'] / latest_factor
                                
                            # 应用复权因子
                            for col in ['open', 'high', 'low', 'close']:
                                if col in df.columns:
                                    df[col] = df[col] * df['adj_factor']
                
                # 转换TuShare数据格式
                if not df.empty:
                    # 创建统一的列名映射
                    column_mapping = {
                        "trade_date": "date",
                        "open": "open",
                        "high": "high",
                        "low": "low",
                        "close": "close",
                        "vol": "volume",
                        "volume": "volume",  # pro_bar可能使用volume列名
                        "amount": "amount",
                        "pct_chg": "pct_change",
                        "change": "change_amount",
                        "turnover_rate": "turnover",
                        "turn": "turnover"    # pro_bar可能使用turn列名
                    }
                    
                    # 只重命名存在的列
                    rename_cols = {col: new_col for col, new_col in column_mapping.items() if col in df.columns}
                    df = df.rename(columns=rename_cols)
                    
                    # 确保日期格式统一
                    # 如果date列不存在，但trade_date存在，则创建date列
                    if "date" not in df.columns and "trade_date" in df.columns:
                        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                    elif "date" in df.columns:
                        if df["date"].dtype == object:  # 如果是字符串格式
                            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
                    
                    # 确保所有数值列已转换为float类型
                    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change', 'change_amount', 'turnover']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # 排序
                    df = df.sort_values("date", ascending=True)
                    
                    logging.info(f"Successfully retrieved data from TuShare: {len(df)} records")
                    return df
                else:
                    logging.warning("TuShare returned empty DataFrame")
                    
            except Exception as e:
                logging.error(f"TuShare data fetch failed: {str(e)}")
                logging.debug(f"TuShare error details: {traceback.format_exc()}")
        
        # 如果两种数据源都失败，返回空DataFrame
        if df.empty:
            logging.warning("Both AKShare and TuShare data fetch failed, returning empty DataFrame")
        
        return df
    
    @retry(max_tries=3, delay_seconds=2)
    def get_financial_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        获取财务指标数据，优先使用AKShare，失败时切换到TuShare
        
        Args:
            symbol: 股票代码
            
        Returns:
            包含财务指标的字典
        """
        logging.info(f"Getting financial metrics for {symbol}")
        
        # 转换股票代码
        akshare_code, tushare_code, exchange_prefix = self.convert_stock_code(symbol)
        
        # 缓存键
        cache_key = f"fin_metrics_{symbol}"
        
        # 使用缓存
        metrics = get_cached_data(
            cache_key,
            lambda: self._fetch_financial_metrics(akshare_code, tushare_code, exchange_prefix),
            ttl_days=1  # 财务数据每天更新一次即可
        )
        
        return metrics
    
    def _fetch_financial_metrics(self, akshare_code: str, tushare_code: str, exchange_prefix: str) -> List[Dict[str, Any]]:
        """内部方法：从数据源获取财务指标"""
        metrics = [{}]  # 默认返回一个空字典的列表
        
        # 首先尝试使用AKShare
        if ak:
            try:
                logging.info(f"Fetching financial metrics using AKShare for {akshare_code}")
                
                # 获取实时行情数据
                realtime_data = ak.stock_zh_a_spot_em()
                if not realtime_data.empty:
                    stock_data = realtime_data[realtime_data['代码'] == akshare_code]
                    if not stock_data.empty:
                        stock_data = stock_data.iloc[0]
                        
                        # 获取新浪财务指标
                        current_year = datetime.now().year
                        financial_data = ak.stock_financial_analysis_indicator(
                            symbol=akshare_code, start_year=str(current_year-1))
                        
                        if not financial_data.empty:
                            # 按日期排序并获取最新的数据
                            financial_data['日期'] = pd.to_datetime(financial_data['日期'])
                            financial_data = financial_data.sort_values('日期', ascending=False)
                            latest_financial = financial_data.iloc[0]
                            
                            # 获取利润表数据
                            try:
                                income_statement = ak.stock_financial_report_sina(
                                    stock=f"{exchange_prefix}{akshare_code}", symbol="利润表")
                                latest_income = income_statement.iloc[0] if not income_statement.empty else pd.Series()
                            except Exception:
                                latest_income = pd.Series()
                            
                            # 构建指标数据
                            def convert_percentage(value: float) -> float:
                                """将百分比值转换为小数"""
                                try:
                                    return float(value) / 100.0 if value is not None else 0.0
                                except:
                                    return 0.0
                                    
                            agent_metrics = {
                                # 盈利能力指标
                                "return_on_equity": convert_percentage(latest_financial.get("净资产收益率(%)", 0)),
                                "net_margin": convert_percentage(latest_financial.get("销售净利率(%)", 0)),
                                "operating_margin": convert_percentage(latest_financial.get("营业利润率(%)", 0)),
                    
                                # 增长指标
                                "revenue_growth": convert_percentage(latest_financial.get("主营业务收入增长率(%)", 0)),
                                "earnings_growth": convert_percentage(latest_financial.get("净利润增长率(%)", 0)),
                                "book_value_growth": convert_percentage(latest_financial.get("净资产增长率(%)", 0)),
                    
                                # 财务健康指标
                                "current_ratio": float(latest_financial.get("流动比率", 0)),
                                "debt_to_equity": convert_percentage(latest_financial.get("资产负债率(%)", 0)),
                                "free_cash_flow_per_share": float(latest_financial.get("每股经营性现金流(元)", 0)),
                                "earnings_per_share": float(latest_financial.get("加权每股收益(元)", 0)),
                    
                                # 估值比率
                                "pe_ratio": float(stock_data.get("市盈率-动态", 0)),
                                "price_to_book": float(stock_data.get("市净率", 0)),
                                "price_to_sales": float(stock_data.get("总市值", 0)) / float(latest_income.get("营业总收入", 1)) if float(latest_income.get("营业总收入", 0)) > 0 else 0,
                            }
                            
                            return [agent_metrics]  # 返回包含一个字典的列表
            
            except Exception as e:
                logging.warning(f"AKShare financial metrics fetch failed: {str(e)}")
                logging.info("Falling back to TuShare...")
        
        # 如果AKShare失败或不可用，尝试使用TuShare
        if ts and TUSHARE_TOKEN:
            try:
                logging.info(f"Fetching financial metrics using TuShare for {tushare_code}")
                
                # 获取基本信息
                basic_info = ts.pro_api().daily_basic(ts_code=tushare_code)
                
                if not basic_info.empty:
                    basic_info = basic_info.iloc[0]
                    
                    # 获取财务指标
                    fin_indicator = ts.pro_api().fina_indicator(ts_code=tushare_code)
                    latest_fin = fin_indicator.iloc[0] if not fin_indicator.empty else pd.Series()
                    
                    # 获取利润表
                    income = ts.pro_api().income(ts_code=tushare_code)
                    latest_income = income.iloc[0] if not income.empty else pd.Series()
                    prev_income = income.iloc[1] if len(income) > 1 else pd.Series()
                    
                    # 计算增长率
                    revenue_current = float(latest_income.get('revenue', 0))
                    revenue_prev = float(prev_income.get('revenue', 0))
                    revenue_growth = (revenue_current - revenue_prev) / revenue_prev if revenue_prev > 0 else 0
                    
                    net_profit_current = float(latest_income.get('n_income', 0))
                    net_profit_prev = float(prev_income.get('n_income', 0))
                    earnings_growth = (net_profit_current - net_profit_prev) / net_profit_prev if net_profit_prev > 0 else 0
                    
                    # 构建指标字典
                    agent_metrics = {
                        # 盈利能力指标
                        "return_on_equity": float(latest_fin.get("roe", 0)),
                        "net_margin": float(latest_fin.get("netprofit_margin", 0)),
                        "operating_margin": float(latest_fin.get("profit_dedt", 0)) / float(latest_income.get("revenue", 1)) if float(latest_income.get("revenue", 0)) > 0 else 0,
                        
                        # 增长指标
                        "revenue_growth": revenue_growth,
                        "earnings_growth": earnings_growth,
                        "book_value_growth": float(latest_fin.get("equity_yoy", 0)),
                        
                        # 财务健康指标
                        "current_ratio": float(latest_fin.get("current_ratio", 0)),
                        "debt_to_equity": float(latest_fin.get("debt_to_assets", 0)),
                        "free_cash_flow_per_share": float(latest_fin.get("fcff_ps", 0)),
                        "earnings_per_share": float(latest_fin.get("eps", 0)),
                        
                        # 估值比率
                        "pe_ratio": float(basic_info.get("pe", 0)),
                        "price_to_book": float(basic_info.get("pb", 0)),
                        "price_to_sales": float(basic_info.get("ps", 0)),
                    }
                    
                    return [agent_metrics]  # 返回包含一个字典的列表
                    
            except Exception as e:
                logging.error(f"TuShare financial metrics fetch failed: {str(e)}")
        
        return metrics  # 返回默认的空指标列表
    
    @retry(max_tries=3, delay_seconds=2)
    def get_financial_statements(self, symbol: str) -> List[Dict[str, Any]]:
        """
        获取财务报表数据，优先使用AKShare，失败时切换到TuShare
        
        Args:
            symbol: 股票代码
            
        Returns:
            包含财务报表数据的字典列表
        """
        logging.info(f"Getting financial statements for {symbol}")
        
        # 转换股票代码
        akshare_code, tushare_code, exchange_prefix = self.convert_stock_code(symbol)
        
        # 缓存键
        cache_key = f"fin_statements_{symbol}"
        
        # 使用缓存
        statements = get_cached_data(
            cache_key,
            lambda: self._fetch_financial_statements(akshare_code, tushare_code, exchange_prefix),
            ttl_days=7  # 财务报表可以缓存更长时间
        )
        
        return statements
    
    def _fetch_financial_statements(self, akshare_code: str, tushare_code: str, exchange_prefix: str) -> List[Dict[str, Any]]:
        """内部方法：从数据源获取财务报表"""
        default_items = [
            {
                "net_income": 0,
                "operating_revenue": 0,
                "operating_profit": 0,
                "working_capital": 0,
                "depreciation_and_amortization": 0,
                "capital_expenditure": 0,
                "free_cash_flow": 0
            },
            {
                "net_income": 0,
                "operating_revenue": 0,
                "operating_profit": 0,
                "working_capital": 0,
                "depreciation_and_amortization": 0,
                "capital_expenditure": 0,
                "free_cash_flow": 0
            }
        ]
        
        # 首先尝试使用AKShare
        if ak:
            try:
                logging.info(f"Fetching financial statements using AKShare for {akshare_code}")
                
                # 获取资产负债表
                balance_sheet = ak.stock_financial_report_sina(
                    stock=f"{exchange_prefix}{akshare_code}", symbol="资产负债表")
                if not balance_sheet.empty:
                    latest_balance = balance_sheet.iloc[0]
                    previous_balance = balance_sheet.iloc[1] if len(balance_sheet) > 1 else balance_sheet.iloc[0]
                    
                    # 获取利润表
                    income_statement = ak.stock_financial_report_sina(
                        stock=f"{exchange_prefix}{akshare_code}", symbol="利润表")
                    latest_income = income_statement.iloc[0] if not income_statement.empty else pd.Series()
                    previous_income = income_statement.iloc[1] if len(income_statement) > 1 else income_statement.iloc[0]
                    
                    # 获取现金流量表
                    cash_flow = ak.stock_financial_report_sina(
                        stock=f"{exchange_prefix}{akshare_code}", symbol="现金流量表")
                    latest_cash_flow = cash_flow.iloc[0] if not cash_flow.empty else pd.Series()
                    previous_cash_flow = cash_flow.iloc[1] if len(cash_flow) > 1 else cash_flow.iloc[0]
                    
                    # 构建财务数据项
                    current_item = {
                        "net_income": float(latest_income.get("净利润", 0)),
                        "operating_revenue": float(latest_income.get("营业总收入", 0)),
                        "operating_profit": float(latest_income.get("营业利润", 0)),
                        "working_capital": float(latest_balance.get("流动资产合计", 0)) - float(latest_balance.get("流动负债合计", 0)),
                        "depreciation_and_amortization": float(latest_cash_flow.get("固定资产折旧、油气资产折耗、生产性生物资产折旧", 0)),
                        "capital_expenditure": abs(float(latest_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0))),
                        "free_cash_flow": float(latest_cash_flow.get("经营活动产生的现金流量净额", 0)) - abs(float(latest_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0)))
                    }
                    
                    previous_item = {
                        "net_income": float(previous_income.get("净利润", 0)),
                        "operating_revenue": float(previous_income.get("营业总收入", 0)),
                        "operating_profit": float(previous_income.get("营业利润", 0)),
                        "working_capital": float(previous_balance.get("流动资产合计", 0)) - float(previous_balance.get("流动负债合计", 0)),
                        "depreciation_and_amortization": float(previous_cash_flow.get("固定资产折旧、油气资产折耗、生产性生物资产折旧", 0)),
                        "capital_expenditure": abs(float(previous_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0))),
                        "free_cash_flow": float(previous_cash_flow.get("经营活动产生的现金流量净额", 0)) - abs(float(previous_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0)))
                    }
                    
                    return [current_item, previous_item]
                
            except Exception as e:
                logging.warning(f"AKShare financial statements fetch failed: {str(e)}")
                logging.info("Falling back to TuShare...")
        
        # 如果AKShare失败或不可用，尝试使用TuShare
        if ts and TUSHARE_TOKEN:
            try:
                logging.info(f"Fetching financial statements using TuShare for {tushare_code}")
                
                # 获取资产负债表
                balance = ts.pro_api().balancesheet(ts_code=tushare_code)
                if not balance.empty:
                    latest_balance = balance.iloc[0]
                    previous_balance = balance.iloc[1] if len(balance) > 1 else balance.iloc[0]
                    
                    # 获取利润表
                    income = ts.pro_api().income(ts_code=tushare_code)
                    latest_income = income.iloc[0] if not income.empty else pd.Series()
                    previous_income = income.iloc[1] if len(income) > 1 else income.iloc[0]
                    
                    # 获取现金流量表
                    cashflow = ts.pro_api().cashflow(ts_code=tushare_code)
                    latest_cashflow = cashflow.iloc[0] if not cashflow.empty else pd.Series()
                    previous_cashflow = cashflow.iloc[1] if len(cashflow) > 1 else cashflow.iloc[0]
                    
                    # 构建财务数据项
                    current_item = {
                        "net_income": float(latest_income.get("n_income", 0)),
                        "operating_revenue": float(latest_income.get("revenue", 0)),
                        "operating_profit": float(latest_income.get("operate_profit", 0)),
                        "working_capital": float(latest_balance.get("total_cur_assets", 0)) - float(latest_balance.get("total_cur_liab", 0)),
                        "depreciation_and_amortization": float(latest_cashflow.get("depreciation_amort_cba", 0)),
                        "capital_expenditure": abs(float(latest_cashflow.get("stot_outflows_inv_act", 0))),
                        "free_cash_flow": float(latest_cashflow.get("n_cashflow_act", 0)) - abs(float(latest_cashflow.get("stot_outflows_inv_act", 0)))
                    }
                    
                    previous_item = {
                        "net_income": float(previous_income.get("n_income", 0)),
                        "operating_revenue": float(previous_income.get("revenue", 0)),
                        "operating_profit": float(previous_income.get("operate_profit", 0)),
                        "working_capital": float(previous_balance.get("total_cur_assets", 0)) - float(previous_balance.get("total_cur_liab", 0)),
                        "depreciation_and_amortization": float(previous_cashflow.get("depreciation_amort_cba", 0)),
                        "capital_expenditure": abs(float(previous_cashflow.get("stot_outflows_inv_act", 0))),
                        "free_cash_flow": float(previous_cashflow.get("n_cashflow_act", 0)) - abs(float(previous_cashflow.get("stot_outflows_inv_act", 0)))
                    }
                    
                    return [current_item, previous_item]
                    
            except Exception as e:
                logging.error(f"TuShare financial statements fetch failed: {str(e)}")
        
        return default_items
    
    @retry(max_tries=3, delay_seconds=2)
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取市场数据，优先使用AKShare，失败时切换到TuShare
        
        Args:
            symbol: 股票代码
            
        Returns:
            包含市场数据的字典
        """
        logging.info(f"Getting market data for {symbol}")
        
        # 转换股票代码
        akshare_code, tushare_code, exchange_prefix = self.convert_stock_code(symbol)
        
        # 缓存键
        cache_key = f"market_data_{symbol}"
        
        # 市场数据更新频率高，使用较短的缓存时间
        market_data = get_cached_data(
            cache_key,
            lambda: self._fetch_market_data(akshare_code, tushare_code),
            ttl_days=0.5  # 12小时
        )
        
        return market_data
    
    def _fetch_market_data(self, akshare_code: str, tushare_code: str) -> Dict[str, Any]:
        """内部方法：从数据源获取市场数据"""
        default_data = {
            "market_cap": 0,
            "volume": 0,
            "average_volume": 0,
            "fifty_two_week_high": 0,
            "fifty_two_week_low": 0
        }
        
        # 首先尝试使用AKShare
        if ak:
            try:
                logging.info(f"Fetching market data using AKShare for {akshare_code}")
                
                # 获取实时行情
                realtime_data = ak.stock_zh_a_spot_em()
                stock_data = realtime_data[realtime_data['代码'] == akshare_code]
                
                if not stock_data.empty:
                    stock_data = stock_data.iloc[0]
                    
                    # 获取52周最高最低价(可能需要单独获取)
                    try:
                        hist_data = ak.stock_zh_a_hist(symbol=akshare_code, period="daily", adjust="qfq")
                        if not hist_data.empty:
                            # 获取近一年数据的最高和最低价
                            recent_year = hist_data.tail(250)  # 约一年的交易日
                            week_high = recent_year["最高"].max()
                            week_low = recent_year["最低"].min()
                        else:
                            week_high = stock_data.get("最高", 0)
                            week_low = stock_data.get("最低", 0)
                    except Exception:
                        week_high = stock_data.get("最高", 0)
                        week_low = stock_data.get("最低", 0)
                        
                    # 计算过去30天的平均成交量
                    try:
                        avg_volume = hist_data.tail(30)["成交量"].mean()
                    except Exception:
                        avg_volume = stock_data.get("成交量", 0)
                    
                    return {
                        "market_cap": float(stock_data.get("总市值", 0)),
                        "volume": float(stock_data.get("成交量", 0)),
                        "average_volume": float(avg_volume),
                        "fifty_two_week_high": float(week_high),
                        "fifty_two_week_low": float(week_low)
                    }
                
            except Exception as e:
                logging.warning(f"AKShare market data fetch failed: {str(e)}")
                logging.info("Falling back to TuShare...")
        
        # 如果AKShare失败或不可用，尝试使用TuShare
        if ts and TUSHARE_TOKEN:
            try:
                logging.info(f"Fetching market data using TuShare for {tushare_code}")
                
                # 获取基本行情
                daily_basic = ts.pro_api().daily_basic(ts_code=tushare_code)
                if not daily_basic.empty:
                    latest_data = daily_basic.iloc[0]
                    
                    # 获取历史数据计算52周最高最低
                    today = datetime.now()
                    start_date = (today - timedelta(days=365)).strftime('%Y%m%d')
                    end_date = today.strftime('%Y%m%d')
                    
                    hist_data = ts.pro_api().daily(ts_code=tushare_code, start_date=start_date, end_date=end_date)
                    
                    if not hist_data.empty:
                        week_high = hist_data["high"].max()
                        week_low = hist_data["low"].min()
                        avg_volume = hist_data.head(30)["vol"].mean()
                    else:
                        week_high = latest_data.get("high", 0)
                        week_low = latest_data.get("low", 0)
                        avg_volume = latest_data.get("vol", 0)
                    
                    return {
                        "market_cap": float(latest_data.get("total_mv", 0)),  # 总市值
                        "volume": float(latest_data.get("vol", 0)),           # 成交量
                        "average_volume": float(avg_volume),                   # 平均成交量
                        "fifty_two_week_high": float(week_high),              # 52周最高
                        "fifty_two_week_low": float(week_low)                 # 52周最低
                    }
                    
            except Exception as e:
                logging.error(f"TuShare market data fetch failed: {str(e)}")
        
        return default_data

# 创建统一的数据API接口
class DataAPI:
    """统一的数据API接口，封装内部数据源适配器实现"""
    
    def __init__(self):
        self.adapter = DataSourceAdapter()
    
    def get_price_data(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取股票价格数据
        
        Args:
            ticker: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD，如果为None则默认获取过去一年的数据
            end_date: 结束日期，格式：YYYY-MM-DD，如果为None则使用昨天作为结束日期
            
        Returns:
            包含价格数据的DataFrame
        """
        return self.adapter.get_price_history(ticker, start_date, end_date)
    
    def get_financial_metrics(self, ticker: str) -> Dict[str, Any]:
        """
        获取财务指标数据
        
        Args:
            ticker: 股票代码
            
        Returns:
            包含财务指标的字典
        """
        return self.adapter.get_financial_metrics(ticker)
    
    def get_financial_statements(self, ticker: str) -> Dict[str, Any]:
        """
        获取财务报表数据
        
        Args:
            ticker: 股票代码
            
        Returns:
            包含财务报表数据的字典
        """
        return self.adapter.get_financial_statements(ticker)
    
    def get_market_data(self, ticker: str) -> Dict[str, Any]:
        """
        获取市场数据
        
        Args:
            ticker: 股票代码
            
        Returns:
            包含市场数据的字典
        """
        return self.adapter.get_market_data(ticker)

# 创建全局API实例
data_api = DataAPI()