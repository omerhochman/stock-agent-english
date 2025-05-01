import logging
from typing import List, Optional

from mcp.server.fastmcp import FastMCP
from src.mcp_api.data_source_interface import FinancialDataSource, NoDataFoundError, LoginError, DataSourceError
from src.mcp_api.formatting.markdown_formatter import format_df_to_markdown

logger = logging.getLogger(__name__)


def register_stock_market_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    向MCP应用注册股票市场数据工具

    参数:
        app: FastMCP应用实例
        active_data_source: 活跃的金融数据源
    """

    @app.tool()
    def get_historical_k_data(
        code: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjust_flag: str = "3",
        fields: Optional[List[str]] = None,
    ) -> str:
        """
        获取中国A股股票的历史K线（OHLCV）数据

        参数:
            code: Baostock格式的股票代码（例如，'sh.600000'，'sz.000001'）
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'
            frequency: 数据频率。有效选项（来自Baostock）:
                         'd': 日线
                         'w': 周线
                         'm': 月线
                         '5': 5分钟线
                         '15': 15分钟线
                         '30': 30分钟线
                         '60': 60分钟线
                       默认为'd'（日线）
            adjust_flag: 价格/成交量的复权标志。有效选项（来自Baostock）:
                           '1': 后复权
                           '2': 前复权
                           '3': 不复权
                         默认为'3'（不复权）
            fields: 可选的特定数据字段列表（必须是有效的Baostock字段）
                    如果为None或空，将使用默认字段（例如，日期、代码、开盘价、最高价、最低价、收盘价、成交量、成交额、涨跌幅）

        返回:
            包含K线数据表的Markdown格式字符串，或错误信息。
            如果结果集太大，表格可能会被截断。
        """
        logger.info(
            f"Tool 'get_historical_k_data' called for {code} ({start_date}-{end_date}, freq={frequency}, adj={adjust_flag}, fields={fields})")
        try:
            # 验证频率和复权标志（如有必要）
            valid_freqs = ['d', 'w', 'm', '5', '15', '30', '60']
            valid_adjusts = ['1', '2', '3']
            if frequency not in valid_freqs:
                logger.warning(f"Invalid frequency requested: {frequency}")
                return f"Error: Invalid frequency '{frequency}'. Valid options are: {valid_freqs}"
            if adjust_flag not in valid_adjusts:
                logger.warning(f"Invalid adjust_flag requested: {adjust_flag}")
                return f"Error: Invalid adjust_flag '{adjust_flag}'. Valid options are: {valid_adjusts}"

            # 调用注入的数据源
            df = active_data_source.get_historical_k_data(
                code=code,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                adjust_flag=adjust_flag,
                fields=fields,
            )
            # 格式化结果
            logger.info(
                f"Successfully retrieved K-data for {code}, formatting to Markdown.")
            return format_df_to_markdown(df)

        except NoDataFoundError as e:
            # 未找到数据错误
            logger.warning(f"NoDataFoundError for {code}: {e}")
            return f"Error: {e}"
        except LoginError as e:
            # 登录错误
            logger.error(f"LoginError for {code}: {e}")
            return f"Error: Could not connect to data source. {e}"
        except DataSourceError as e:
            # 数据源错误
            logger.error(f"DataSourceError for {code}: {e}")
            return f"Error: An error occurred while fetching data. {e}"
        except ValueError as e:
            # 值错误
            logger.warning(f"ValueError processing request for {code}: {e}")
            return f"Error: Invalid input parameter. {e}"
        except Exception as e:
            # 捕获所有意外错误
            logger.exception(
                f"Unexpected Exception processing get_historical_k_data for {code}: {e}")
            return f"Error: An unexpected error occurred: {e}"

    @app.tool()
    def get_stock_basic_info(code: str, fields: Optional[List[str]] = None) -> str:
        """
        获取给定中国A股股票的基本信息

        参数:
            code: Baostock格式的股票代码（例如，'sh.600000'，'sz.000001'）
            fields: 可选列表，用于从可用的基本信息中选择特定列
                    （例如，['code', 'code_name', 'industry', 'listingDate']）
                    如果为None或空，则返回Baostock提供的所有可用基本信息列

        返回:
            包含基本股票信息表的Markdown格式字符串，或错误信息
        """
        logger.info(
            f"Tool 'get_stock_basic_info' called for {code} (fields={fields})")
        try:
            # 调用注入的数据源
            # 传递fields参数；BaostockDataSource实现会处理选择
            df = active_data_source.get_stock_basic_info(
                code=code, fields=fields)

            # 格式化结果（基本信息通常较小，使用默认截断）
            logger.info(
                f"Successfully retrieved basic info for {code}, formatting to Markdown.")
            # 基本信息使用较小的限制
            return format_df_to_markdown(df, max_rows=10, max_cols=10)

        except NoDataFoundError as e:
            # 未找到数据错误
            logger.warning(f"NoDataFoundError for {code}: {e}")
            return f"Error: {e}"
        except LoginError as e:
            # 登录错误
            logger.error(f"LoginError for {code}: {e}")
            return f"Error: Could not connect to data source. {e}"
        except DataSourceError as e:
            # 数据源错误
            logger.error(f"DataSourceError for {code}: {e}")
            return f"Error: An error occurred while fetching data. {e}"
        except ValueError as e:
            # 值错误
            logger.warning(f"ValueError processing request for {code}: {e}")
            return f"Error: Invalid input parameter or requested field not available. {e}"
        except Exception as e:
            # 意外异常
            logger.exception(
                f"Unexpected Exception processing get_stock_basic_info for {code}: {e}")
            return f"Error: An unexpected error occurred: {e}"

    @app.tool()
    def get_dividend_data(code: str, year: str, year_type: str = "report") -> str:
        """
        获取给定股票代码和年份的分红信息

        参数:
            code: Baostock格式的股票代码（例如，'sh.600000'，'sz.000001'）
            year: 查询年份（例如，'2023'）
            year_type: 年份类型。有效选项（来自Baostock）:
                         'report': 预案公告年份
                         'operate': 除权除息年份
                       默认为'report'（预案公告年份）

        返回:
            包含分红数据表的Markdown格式字符串，或错误信息
        """
        logger.info(
            f"Tool 'get_dividend_data' called for {code}, year={year}, year_type={year_type}")
        try:
            # 基本验证
            if year_type not in ['report', 'operate']:
                logger.warning(f"Invalid year_type requested: {year_type}")
                return f"Error: Invalid year_type '{year_type}'. Valid options are: 'report', 'operate'"
            if not year.isdigit() or len(year) != 4:
                logger.warning(f"Invalid year format requested: {year}")
                return f"Error: Invalid year '{year}'. Please provide a 4-digit year."

            df = active_data_source.get_dividend_data(
                code=code, year=year, year_type=year_type)
            logger.info(
                f"Successfully retrieved dividend data for {code}, year {year}.")
            return format_df_to_markdown(df)

        except NoDataFoundError as e:
            # 未找到数据错误
            logger.warning(f"NoDataFoundError for {code}, year {year}: {e}")
            return f"Error: {e}"
        except LoginError as e:
            # 登录错误
            logger.error(f"LoginError for {code}: {e}")
            return f"Error: Could not connect to data source. {e}"
        except DataSourceError as e:
            # 数据源错误
            logger.error(f"DataSourceError for {code}: {e}")
            return f"Error: An error occurred while fetching data. {e}"
        except ValueError as e:
            # 值错误
            logger.warning(f"ValueError processing request for {code}: {e}")
            return f"Error: Invalid input parameter. {e}"
        except Exception as e:
            # 意外异常
            logger.exception(
                f"Unexpected Exception processing get_dividend_data for {code}: {e}")
            return f"Error: An unexpected error occurred: {e}"

    @app.tool()
    def get_adjust_factor_data(code: str, start_date: str, end_date: str) -> str:
        """
        获取给定股票代码和日期范围的复权因子数据
        使用Baostock的"涨跌幅复权算法"因子。用于计算复权价格。

        参数:
            code: Baostock格式的股票代码（例如，'sh.600000'，'sz.000001'）
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'

        返回:
            包含复权因子数据表的Markdown格式字符串，或错误信息
        """
        logger.info(
            f"Tool 'get_adjust_factor_data' called for {code} ({start_date} to {end_date})")
        try:
            # 如果需要，可以在此处添加基本日期验证
            df = active_data_source.get_adjust_factor_data(
                code=code, start_date=start_date, end_date=end_date)
            logger.info(
                f"Successfully retrieved adjustment factor data for {code}.")
            return format_df_to_markdown(df)

        except NoDataFoundError as e:
            # 未找到数据错误
            logger.warning(f"NoDataFoundError for {code}: {e}")
            return f"Error: {e}"
        except LoginError as e:
            # 登录错误
            logger.error(f"LoginError for {code}: {e}")
            return f"Error: Could not connect to data source. {e}"
        except DataSourceError as e:
            # 数据源错误
            logger.error(f"DataSourceError for {code}: {e}")
            return f"Error: An error occurred while fetching data. {e}"
        except ValueError as e:
            # 值错误
            logger.warning(f"ValueError processing request for {code}: {e}")
            return f"Error: Invalid input parameter. {e}"
        except Exception as e:
            # 意外异常
            logger.exception(
                f"Unexpected Exception processing get_adjust_factor_data for {code}: {e}")
            return f"Error: An unexpected error occurred: {e}"