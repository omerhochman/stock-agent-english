import baostock as bs
import pandas as pd
from typing import List, Optional
import logging
from .data_source_interface import FinancialDataSource, DataSourceError, NoDataFoundError, LoginError
from .mcp_utils import baostock_login_context

# 为此模块获取一个logger实例
logger = logging.getLogger(__name__)

# K线数据的默认字段
DEFAULT_K_FIELDS = [
    "date", "code", "open", "high", "low", "close", "preclose",
    "volume", "amount", "adjustflag", "turn", "tradestatus",
    "pctChg", "isST"
]

# 基本信息的默认字段
DEFAULT_BASIC_FIELDS = [
    "code", "tradeStatus", "code_name"
    # 根据需要可以添加更多默认字段，例如 "industry", "listingDate"
]

# 辅助函数，用于减少金融数据获取中的重复代码
def _fetch_financial_data(
    bs_query_func,
    data_type_name: str,
    code: str,
    year: str,
    quarter: int
) -> pd.DataFrame:
    """
    用于获取金融数据的辅助函数
    
    参数:
        bs_query_func: Baostock查询函数
        data_type_name: 数据类型名称（用于日志）
        code: 股票代码
        year: 年份
        quarter: 季度
        
    返回:
        包含金融数据的DataFrame
    """
    logger.info(f"Fetching {data_type_name} data for {code}, year={year}, quarter={quarter}")
    try:
        with baostock_login_context():
            rs = bs_query_func(code=code, year=year, quarter=quarter)

            if rs.error_code != '0':
                logger.error(f"Baostock API error ({data_type_name}) for {code}: {rs.error_msg} (code: {rs.error_code})")
                if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                    raise NoDataFoundError(f"No {data_type_name} data found for {code}, {year}Q{quarter}. Baostock msg: {rs.error_msg}")
                else:
                    raise DataSourceError(f"Baostock API error fetching {data_type_name} data: {rs.error_msg} (code: {rs.error_code})")

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                logger.warning(f"No {data_type_name} data found for {code}, {year}Q{quarter} (empty result set from Baostock).")
                raise NoDataFoundError(f"No {data_type_name} data found for {code}, {year}Q{quarter} (empty result set).")

            result_df = pd.DataFrame(data_list, columns=rs.fields)
            logger.info(f"Retrieved {len(result_df)} {data_type_name} records for {code}, {year}Q{quarter}.")
            return result_df

    except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
        logger.warning(f"Caught known error fetching {data_type_name} data for {code}: {type(e).__name__}")
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error fetching {data_type_name} data for {code}: {e}")
        raise DataSourceError(f"Unexpected error fetching {data_type_name} data for {code}: {e}")

# 辅助函数，用于减少指数成分股数据获取中的重复代码
def _fetch_index_constituent_data(
    bs_query_func,
    index_name: str,
    date: Optional[str] = None
) -> pd.DataFrame:
    """
    用于获取指数成分股数据的辅助函数
    
    参数:
        bs_query_func: Baostock查询函数
        index_name: 指数名称（用于日志）
        date: 可选。查询日期
        
    返回:
        包含指数成分股数据的DataFrame
    """
    logger.info(f"Fetching {index_name} constituents for date={date or 'latest'}")
    try:
        with baostock_login_context():
            rs = bs_query_func(date=date)  # date是可选的，默认为最新

            if rs.error_code != '0':
                logger.error(f"Baostock API error ({index_name} Constituents) for date {date}: {rs.error_msg} (code: {rs.error_code})")
                if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                    raise NoDataFoundError(f"No {index_name} constituent data found for date {date}. Baostock msg: {rs.error_msg}")
                else:
                    raise DataSourceError(f"Baostock API error fetching {index_name} constituents: {rs.error_msg} (code: {rs.error_code})")

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                logger.warning(f"No {index_name} constituent data found for date {date} (empty result set).")
                raise NoDataFoundError(f"No {index_name} constituent data found for date {date} (empty result set).")

            result_df = pd.DataFrame(data_list, columns=rs.fields)
            logger.info(f"Retrieved {len(result_df)} {index_name} constituents for date {date or 'latest'}.")
            return result_df

    except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
        logger.warning(f"Caught known error fetching {index_name} constituents for date {date}: {type(e).__name__}")
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error fetching {index_name} constituents for date {date}: {e}")
        raise DataSourceError(f"Unexpected error fetching {index_name} constituents for date {date}: {e}")

# 辅助函数，用于减少宏观经济数据获取中的重复代码
def _fetch_macro_data(
    bs_query_func,
    data_type_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs  # 用于额外参数，如yearType
) -> pd.DataFrame:
    """
    用于获取宏观经济数据的辅助函数
    
    参数:
        bs_query_func: Baostock查询函数
        data_type_name: 数据类型名称（用于日志）
        start_date: 可选。开始日期
        end_date: 可选。结束日期
        **kwargs: 额外关键字参数
        
    返回:
        包含宏观经济数据的DataFrame
    """
    date_range_log = f"from {start_date or 'default'} to {end_date or 'default'}"
    kwargs_log = f", extra_args={kwargs}" if kwargs else ""
    logger.info(f"Fetching {data_type_name} data {date_range_log}{kwargs_log}")
    try:
        with baostock_login_context():
            rs = bs_query_func(start_date=start_date, end_date=end_date, **kwargs)

            if rs.error_code != '0':
                logger.error(f"Baostock API error ({data_type_name}): {rs.error_msg} (code: {rs.error_code})")
                if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                    raise NoDataFoundError(f"No {data_type_name} data found for the specified criteria. Baostock msg: {rs.error_msg}")
                else:
                    raise DataSourceError(f"Baostock API error fetching {data_type_name} data: {rs.error_msg} (code: {rs.error_code})")

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                logger.warning(f"No {data_type_name} data found for the specified criteria (empty result set).")
                raise NoDataFoundError(f"No {data_type_name} data found for the specified criteria (empty result set).")

            result_df = pd.DataFrame(data_list, columns=rs.fields)
            logger.info(f"Retrieved {len(result_df)} {data_type_name} records.")
            return result_df

    except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
        logger.warning(f"Caught known error fetching {data_type_name} data: {type(e).__name__}")
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error fetching {data_type_name} data: {e}")
        raise DataSourceError(f"Unexpected error fetching {data_type_name} data: {e}")

class BaostockDataSource(FinancialDataSource):
    """
    使用Baostock库实现的FinancialDataSource具体实现
    """

    def _format_fields(self, fields: Optional[List[str]], default_fields: List[str]) -> str:
        """
        将字段列表格式化为Baostock的逗号分隔字符串。
        
        参数:
            fields: 请求的字段列表
            default_fields: 默认字段列表
            
        返回:
            格式化后的字段字符串
        """
        if fields is None or not fields:
            logger.debug(f"No specific fields requested, using defaults: {default_fields}")
            return ",".join(default_fields)
        # 基本验证：确保请求的字段都是字符串
        if not all(isinstance(f, str) for f in fields):
            raise ValueError("All items in the fields list must be strings.")
        logger.debug(f"Using requested fields: {fields}")
        return ",".join(fields)

    def get_historical_k_data(
        self,
        code: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjust_flag: str = "3",
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        使用Baostock获取历史K线数据。
        
        参数:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率，默认为'd'（日线）
            adjust_flag: 复权标志，默认为'3'（不复权）
            fields: 可选的字段列表
            
        返回:
            包含K线数据的DataFrame
        """
        logger.info(f"Fetching K-data for {code} ({start_date} to {end_date}), freq={frequency}, adjust={adjust_flag}")
        try:
            formatted_fields = self._format_fields(fields, DEFAULT_K_FIELDS)
            logger.debug(f"Requesting fields from Baostock: {formatted_fields}")

            with baostock_login_context():
                rs = bs.query_history_k_data_plus(
                    code,
                    formatted_fields,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    adjustflag=adjust_flag
                )

                if rs.error_code != '0':
                    logger.error(f"Baostock API error (K-data) for {code}: {rs.error_msg} (code: {rs.error_code})")
                    # 检查常见错误代码，如没有数据
                    if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':  # 示例错误代码
                         raise NoDataFoundError(f"No historical data found for {code} in the specified range. Baostock msg: {rs.error_msg}")
                    else:
                        raise DataSourceError(f"Baostock API error fetching K-data: {rs.error_msg} (code: {rs.error_code})")

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                     logger.warning(f"No historical data found for {code} in range (empty result set from Baostock).")
                     raise NoDataFoundError(f"No historical data found for {code} in the specified range (empty result set).")

                # 关键：使用rs.fields作为列名
                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(f"Retrieved {len(result_df)} records for {code}.")
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            # 重新抛出已知错误
            logger.warning(f"Caught known error fetching K-data for {code}: {type(e).__name__}")
            raise e
        except Exception as e:
            # 包装意外错误
            logger.exception(f"Unexpected error fetching K-data for {code}: {e}")  # 使用logger.exception包含堆栈跟踪
            raise DataSourceError(f"Unexpected error fetching K-data for {code}: {e}")

    def get_stock_basic_info(self, code: str, fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        使用Baostock获取股票基本信息。
        
        参数:
            code: 股票代码
            fields: 可选的字段列表，用于选择特定列
            
        返回:
            包含股票基本信息的DataFrame
        """
        logger.info(f"Fetching basic info for {code}")
        try:
            # 注意：query_stock_basic在文档中似乎没有fields参数，
            # 但我们保持签名一致。它返回一个固定集合。
            # 如果需要，我们将在查询后使用`fields`参数选择列。
            logger.debug(f"Requesting basic info for {code}. Optional fields requested: {fields}")

            with baostock_login_context():
                # 示例：获取基本信息；根据baostock文档根据需要调整API调用
                # rs = bs.query_stock_basic(code=code, code_name=code_name)  # 如果支持名称查找
                rs = bs.query_stock_basic(code=code)

                if rs.error_code != '0':
                    logger.error(f"Baostock API error (Basic Info) for {code}: {rs.error_msg} (code: {rs.error_code})")
                    if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                        raise NoDataFoundError(f"No basic info found for {code}. Baostock msg: {rs.error_msg}")
                    else:
                        raise DataSourceError(f"Baostock API error fetching basic info: {rs.error_msg} (code: {rs.error_code})")

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(f"No basic info found for {code} (empty result set from Baostock).")
                    raise NoDataFoundError(f"No basic info found for {code} (empty result set).")

                # 关键：使用rs.fields作为列名
                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(f"Retrieved basic info for {code}. Columns: {result_df.columns.tolist()}")

                # 可选：如果提供了`fields`参数，选择列的子集
                if fields:
                    available_cols = [col for col in fields if col in result_df.columns]
                    if not available_cols:
                        raise ValueError(f"None of the requested fields {fields} are available in the basic info result.")
                    logger.debug(f"Selecting columns: {available_cols} from basic info for {code}")
                    result_df = result_df[available_cols]

                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(f"Caught known error fetching basic info for {code}: {type(e).__name__}")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error fetching basic info for {code}: {e}")
            raise DataSourceError(f"Unexpected error fetching basic info for {code}: {e}")

    def get_dividend_data(self, code: str, year: str, year_type: str = "report") -> pd.DataFrame:
        """
        使用Baostock获取分红信息。
        
        参数:
            code: 股票代码
            year: 年份
            year_type: 年份类型，'report'表示预案公告年份，'operate'表示除权除息年份
            
        返回:
            包含分红信息的DataFrame
        """
        logger.info(f"Fetching dividend data for {code}, year={year}, year_type={year_type}")
        try:
            with baostock_login_context():
                rs = bs.query_dividend_data(code=code, year=year, yearType=year_type)

                if rs.error_code != '0':
                    logger.error(f"Baostock API error (Dividend) for {code}: {rs.error_msg} (code: {rs.error_code})")
                    if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                        raise NoDataFoundError(f"No dividend data found for {code} and year {year}. Baostock msg: {rs.error_msg}")
                    else:
                        raise DataSourceError(f"Baostock API error fetching dividend data: {rs.error_msg} (code: {rs.error_code})")

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(f"No dividend data found for {code}, year {year} (empty result set from Baostock).")
                    raise NoDataFoundError(f"No dividend data found for {code}, year {year} (empty result set).")

                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(f"Retrieved {len(result_df)} dividend records for {code}, year {year}.")
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(f"Caught known error fetching dividend data for {code}: {type(e).__name__}")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error fetching dividend data for {code}: {e}")
            raise DataSourceError(f"Unexpected error fetching dividend data for {code}: {e}")

    def get_adjust_factor_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        使用Baostock获取复权因子数据。
        
        参数:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            包含复权因子数据的DataFrame
        """
        logger.info(f"Fetching adjustment factor data for {code} ({start_date} to {end_date})")
        try:
            with baostock_login_context():
                rs = bs.query_adjust_factor(code=code, start_date=start_date, end_date=end_date)

                if rs.error_code != '0':
                    logger.error(f"Baostock API error (Adjust Factor) for {code}: {rs.error_msg} (code: {rs.error_code})")
                    if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                         raise NoDataFoundError(f"No adjustment factor data found for {code} in the specified range. Baostock msg: {rs.error_msg}")
                    else:
                        raise DataSourceError(f"Baostock API error fetching adjust factor data: {rs.error_msg} (code: {rs.error_code})")

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(f"No adjustment factor data found for {code} in range (empty result set from Baostock).")
                    raise NoDataFoundError(f"No adjustment factor data found for {code} in the specified range (empty result set).")

                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(f"Retrieved {len(result_df)} adjustment factor records for {code}.")
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(f"Caught known error fetching adjust factor data for {code}: {type(e).__name__}")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error fetching adjust factor data for {code}: {e}")
            raise DataSourceError(f"Unexpected error fetching adjust factor data for {code}: {e}")

    def get_profit_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """
        使用Baostock获取季度盈利能力数据。
        
        参数:
            code: 股票代码
            year: 年份
            quarter: 季度
            
        返回:
            包含盈利能力数据的DataFrame
        """
        return _fetch_financial_data(bs.query_profit_data, "Profitability", code, year, quarter)

    def get_operation_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """
        使用Baostock获取季度运营能力数据。
        
        参数:
            code: 股票代码
            year: 年份
            quarter: 季度
            
        返回:
            包含运营能力数据的DataFrame
        """
        return _fetch_financial_data(bs.query_operation_data, "Operation Capability", code, year, quarter)

    def get_growth_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """
        使用Baostock获取季度增长能力数据。
        
        参数:
            code: 股票代码
            year: 年份
            quarter: 季度
            
        返回:
            包含增长能力数据的DataFrame
        """
        return _fetch_financial_data(bs.query_growth_data, "Growth Capability", code, year, quarter)

    def get_balance_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """
        使用Baostock获取季度资产负债表数据（偿债能力）。
        
        参数:
            code: 股票代码
            year: 年份
            quarter: 季度
            
        返回:
            包含资产负债表数据的DataFrame
        """
        return _fetch_financial_data(bs.query_balance_data, "Balance Sheet", code, year, quarter)

    def get_cash_flow_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """
        使用Baostock获取季度现金流量数据。
        
        参数:
            code: 股票代码
            year: 年份
            quarter: 季度
            
        返回:
            包含现金流量数据的DataFrame
        """
        return _fetch_financial_data(bs.query_cash_flow_data, "Cash Flow", code, year, quarter)

    def get_dupont_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """
        使用Baostock获取季度杜邦分析数据。
        
        参数:
            code: 股票代码
            year: 年份
            quarter: 季度
            
        返回:
            包含杜邦分析数据的DataFrame
        """
        return _fetch_financial_data(bs.query_dupont_data, "DuPont Analysis", code, year, quarter)

    def get_performance_express_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        使用Baostock获取业绩快报。
        
        参数:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            包含业绩快报数据的DataFrame
        """
        logger.info(f"Fetching Performance Express Report for {code} ({start_date} to {end_date})")
        try:
            with baostock_login_context():
                rs = bs.query_performance_express_report(code=code, start_date=start_date, end_date=end_date)

                if rs.error_code != '0':
                    logger.error(f"Baostock API error (Perf Express) for {code}: {rs.error_msg} (code: {rs.error_code})")
                    if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                         raise NoDataFoundError(f"No performance express report found for {code} in range {start_date}-{end_date}. Baostock msg: {rs.error_msg}")
                    else:
                        raise DataSourceError(f"Baostock API error fetching performance express report: {rs.error_msg} (code: {rs.error_code})")

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(f"No performance express report found for {code} in range {start_date}-{end_date} (empty result set).")
                    raise NoDataFoundError(f"No performance express report found for {code} in range {start_date}-{end_date} (empty result set).")

                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(f"Retrieved {len(result_df)} performance express report records for {code}.")
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(f"Caught known error fetching performance express report for {code}: {type(e).__name__}")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error fetching performance express report for {code}: {e}")
            raise DataSourceError(f"Unexpected error fetching performance express report for {code}: {e}")

    def get_forecast_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        使用Baostock获取业绩预告。
        
        参数:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            包含业绩预告数据的DataFrame
        """
        logger.info(f"Fetching Performance Forecast Report for {code} ({start_date} to {end_date})")
        try:
            with baostock_login_context():
                rs = bs.query_forecast_report(code=code, start_date=start_date, end_date=end_date)
                # 注意：Baostock文档提到此函数有分页，但Python API似乎没有直接暴露它。
                # 我们在下面的循环中获取所有可用页面。

                if rs.error_code != '0':
                    logger.error(f"Baostock API error (Forecast) for {code}: {rs.error_msg} (code: {rs.error_code})")
                    if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                         raise NoDataFoundError(f"No performance forecast report found for {code} in range {start_date}-{end_date}. Baostock msg: {rs.error_msg}")
                    else:
                        raise DataSourceError(f"Baostock API error fetching performance forecast report: {rs.error_msg} (code: {rs.error_code})")

                data_list = []
                while rs.next():  # 如果rs管理分页，循环应该隐式处理分页
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(f"No performance forecast report found for {code} in range {start_date}-{end_date} (empty result set).")
                    raise NoDataFoundError(f"No performance forecast report found for {code} in range {start_date}-{end_date} (empty result set).")

                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(f"Retrieved {len(result_df)} performance forecast report records for {code}.")
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(f"Caught known error fetching performance forecast report for {code}: {type(e).__name__}")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error fetching performance forecast report for {code}: {e}")
            raise DataSourceError(f"Unexpected error fetching performance forecast report for {code}: {e}")
    
    def get_stock_industry(self, code: Optional[str] = None, date: Optional[str] = None) -> pd.DataFrame:
        """
        使用Baostock获取行业分类数据。
        
        参数:
            code: 可选。股票代码，如果为None则获取所有股票
            date: 可选。日期，如果为None则使用最新日期
            
        返回:
            包含行业分类数据的DataFrame
        """
        log_msg = f"Fetching industry data for code={code or 'all'}, date={date or 'latest'}"
        logger.info(log_msg)
        try:
            with baostock_login_context():
                rs = bs.query_stock_industry(code=code, date=date)

                if rs.error_code != '0':
                    logger.error(f"Baostock API error (Industry) for {code}, {date}: {rs.error_msg} (code: {rs.error_code})")
                    if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                        raise NoDataFoundError(f"No industry data found for {code}, {date}. Baostock msg: {rs.error_msg}")
                    else:
                        raise DataSourceError(f"Baostock API error fetching industry data: {rs.error_msg} (code: {rs.error_code})")

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(f"No industry data found for {code}, {date} (empty result set).")
                    raise NoDataFoundError(f"No industry data found for {code}, {date} (empty result set).")

                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(f"Retrieved {len(result_df)} industry records for {code or 'all'}, {date or 'latest'}.")
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(f"Caught known error fetching industry data for {code}, {date}: {type(e).__name__}")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error fetching industry data for {code}, {date}: {e}")
            raise DataSourceError(f"Unexpected error fetching industry data for {code}, {date}: {e}")


    def get_sz50_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        使用Baostock获取上证50指数成分股。
        
        参数:
            date: 可选。日期，如果为None则使用最新日期
            
        返回:
            包含上证50指数成分股的DataFrame
        """
        return _fetch_index_constituent_data(bs.query_sz50_stocks, "SZSE 50", date)

    def get_hs300_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        使用Baostock获取沪深300指数成分股。
        
        参数:
            date: 可选。日期，如果为None则使用最新日期
            
        返回:
            包含沪深300指数成分股的DataFrame
        """
        return _fetch_index_constituent_data(bs.query_hs300_stocks, "CSI 300", date)

    def get_zz500_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        使用Baostock获取中证500指数成分股。
        
        参数:
            date: 可选。日期，如果为None则使用最新日期
            
        返回:
            包含中证500指数成分股的DataFrame
        """
        return _fetch_index_constituent_data(bs.query_zz500_stocks, "CSI 500", date)

    def get_trade_dates(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        使用Baostock获取交易日期信息。
        
        参数:
            start_date: 可选。开始日期
            end_date: 可选。结束日期
            
        返回:
            包含交易日期信息的DataFrame
        """
        logger.info(f"Fetching trade dates from {start_date or 'default'} to {end_date or 'default'}")
        try:
            with baostock_login_context():  # 对于这种情况，登录可能不是严格需要的，但保持一致
                rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)

                if rs.error_code != '0':
                    logger.error(f"Baostock API error (Trade Dates): {rs.error_msg} (code: {rs.error_code})")
                    # 日期查询不太可能有"未找到记录"，但处理API错误
                    raise DataSourceError(f"Baostock API error fetching trade dates: {rs.error_msg} (code: {rs.error_code})")

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    # 如果API返回有效范围，这种情况理论上不应该发生
                    logger.warning(f"No trade dates returned for range {start_date}-{end_date} (empty result set).")
                    raise NoDataFoundError(f"No trade dates found for range {start_date}-{end_date} (empty result set).")

                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(f"Retrieved {len(result_df)} trade date records.")
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(f"Caught known error fetching trade dates: {type(e).__name__}")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error fetching trade dates: {e}")
            raise DataSourceError(f"Unexpected error fetching trade dates: {e}")

    def get_all_stock(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        使用Baostock获取指定日期的所有股票列表。
        
        参数:
            date: 可选。日期，如果为None则使用当前日期
            
        返回:
            包含所有股票的DataFrame
        """
        logger.info(f"Fetching all stock list for date={date or 'default'}")
        try:
            with baostock_login_context():
                rs = bs.query_all_stock(day=date)

                if rs.error_code != '0':
                    logger.error(f"Baostock API error (All Stock) for date {date}: {rs.error_msg} (code: {rs.error_code})")
                    if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':  # 检查是否适用
                         raise NoDataFoundError(f"No stock data found for date {date}. Baostock msg: {rs.error_msg}")
                    else:
                        raise DataSourceError(f"Baostock API error fetching all stock list: {rs.error_msg} (code: {rs.error_code})")

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(f"No stock list returned for date {date} (empty result set).")
                    raise NoDataFoundError(f"No stock list found for date {date} (empty result set).")

                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(f"Retrieved {len(result_df)} stock records for date {date or 'default'}.")
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(f"Caught known error fetching all stock list for date {date}: {type(e).__name__}")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error fetching all stock list for date {date}: {e}")
            raise DataSourceError(f"Unexpected error fetching all stock list for date {date}: {e}")
    
    def get_deposit_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        使用Baostock获取基准存款利率数据。
        
        参数:
            start_date: 可选。开始日期
            end_date: 可选。结束日期
            
        返回:
            包含存款利率数据的DataFrame
        """
        return _fetch_macro_data(bs.query_deposit_rate_data, "Deposit Rate", start_date, end_date)

    def get_loan_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        使用Baostock获取基准贷款利率数据。
        
        参数:
            start_date: 可选。开始日期
            end_date: 可选。结束日期
            
        返回:
            包含贷款利率数据的DataFrame
        """
        return _fetch_macro_data(bs.query_loan_rate_data, "Loan Rate", start_date, end_date)

    def get_required_reserve_ratio_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None, year_type: str = '0') -> pd.DataFrame:
        """
        使用Baostock获取存款准备金率数据。
        
        参数:
            start_date: 可选。开始日期
            end_date: 可选。结束日期
            year_type: 年份类型，'0'表示公告日期，'1'表示生效日期
            
        返回:
            包含存款准备金率数据的DataFrame
        """
        # 注意通过kwargs处理额外的yearType参数
        return _fetch_macro_data(bs.query_required_reserve_ratio_data, "Required Reserve Ratio", start_date, end_date, yearType=year_type)

    def get_money_supply_data_month(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        使用Baostock获取月度货币供应量数据（M0, M1, M2）。
        
        参数:
            start_date: 可选。开始日期，格式为YYYY-MM
            end_date: 可选。结束日期，格式为YYYY-MM
            
        返回:
            包含月度货币供应量数据的DataFrame
        """
        # Baostock这里期望日期格式为YYYY-MM
        return _fetch_macro_data(bs.query_money_supply_data_month, "Monthly Money Supply", start_date, end_date)

    def get_money_supply_data_year(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        使用Baostock获取年度货币供应量数据（M0, M1, M2 - 年末余额）。
        
        参数:
            start_date: 可选。开始年份，格式为YYYY
            end_date: 可选。结束年份，格式为YYYY
            
        返回:
            包含年度货币供应量数据的DataFrame
        """
        # Baostock这里期望日期格式为YYYY
        return _fetch_macro_data(bs.query_money_supply_data_year, "Yearly Money Supply", start_date, end_date)

    def get_shibor_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Baostock不提供SHIBOR（上海银行间同业拆放利率）数据API。
        此方法用空数据帧代替，并引发NoDataFoundError。
        
        参数:
            start_date: 可选。开始日期
            end_date: 可选。结束日期
            
        返回:
            抛出NoDataFoundError
        """
        logger.warning("Baostock API不提供SHIBOR数据。尝试请求SHIBOR数据。")
        from .data_source_interface import NoDataFoundError
        raise NoDataFoundError("Baostock API不提供SHIBOR数据。请使用其他数据源获取SHIBOR数据。")