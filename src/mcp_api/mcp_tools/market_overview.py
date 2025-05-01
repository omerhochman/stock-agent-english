import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP
from src.mcp_api.data_source_interface import FinancialDataSource, NoDataFoundError, LoginError, DataSourceError
from src.mcp_api.formatting.markdown_formatter import format_df_to_markdown

logger = logging.getLogger(__name__)


def register_market_overview_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    向MCP应用注册市场概览工具

    参数:
        app: FastMCP应用实例
        active_data_source: 活跃的金融数据源
    """

    @app.tool()
    def get_trade_dates(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        """
        获取指定日期范围内的交易日信息

        参数:
            start_date: 可选。开始日期，格式为'YYYY-MM-DD'。如果为None，默认为2015-01-01。
            end_date: 可选。结束日期，格式为'YYYY-MM-DD'。如果为None，默认为当前日期。

        返回:
            Markdown表格，显示日期范围内每一天是否为交易日(1)或非交易日(0)。
        """
        logger.info(
            f"Tool 'get_trade_dates' called for range {start_date or 'default'} to {end_date or 'default'}")
        try:
            # 如果需要，可以添加日期验证
            df = active_data_source.get_trade_dates(
                start_date=start_date, end_date=end_date)
            logger.info("Successfully retrieved trade dates.")
            # 交易日期表可能很长，应用标准截断
            return format_df_to_markdown(df)

        except NoDataFoundError as e:
            # 未找到数据错误
            logger.warning(f"NoDataFoundError: {e}")
            return f"Error: {e}"
        except LoginError as e:
            # 登录错误
            logger.error(f"LoginError: {e}")
            return f"Error: Could not connect to data source. {e}"
        except DataSourceError as e:
            # 数据源错误
            logger.error(f"DataSourceError: {e}")
            return f"Error: An error occurred while fetching data. {e}"
        except ValueError as e:
            # 值错误
            logger.warning(f"ValueError: {e}")
            return f"Error: Invalid input parameter. {e}"
        except Exception as e:
            # 意外异常
            logger.exception(
                f"Unexpected Exception processing get_trade_dates: {e}")
            return f"Error: An unexpected error occurred: {e}"

    @app.tool()
    def get_all_stock(date: Optional[str] = None) -> str:
        """
        获取指定日期的所有股票（A股和指数）及其交易状态列表

        参数:
            date: 可选。日期，格式为'YYYY-MM-DD'。如果为None，使用当前日期。

        返回:
            Markdown表格，列出股票代码、名称及其交易状态(1=交易中，0=停牌)。
        """
        logger.info(
            f"Tool 'get_all_stock' called for date={date or 'default'}")
        try:
            # 如果需要，可以添加日期验证
            df = active_data_source.get_all_stock(date=date)
            logger.info(
                f"Successfully retrieved stock list for {date or 'default'}.")
            # 此列表可能非常长，应用标准截断
            return format_df_to_markdown(df)

        except NoDataFoundError as e:
            # 未找到数据错误
            logger.warning(f"NoDataFoundError: {e}")
            return f"Error: {e}"
        except LoginError as e:
            # 登录错误
            logger.error(f"LoginError: {e}")
            return f"Error: Could not connect to data source. {e}"
        except DataSourceError as e:
            # 数据源错误
            logger.error(f"DataSourceError: {e}")
            return f"Error: An error occurred while fetching data. {e}"
        except ValueError as e:
            # 值错误
            logger.warning(f"ValueError: {e}")
            return f"Error: Invalid input parameter. {e}"
        except Exception as e:
            # 意外异常
            logger.exception(
                f"Unexpected Exception processing get_all_stock: {e}")
            return f"Error: An unexpected error occurred: {e}"