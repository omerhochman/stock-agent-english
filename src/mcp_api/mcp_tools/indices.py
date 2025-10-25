import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP
from src.mcp_api.data_source_interface import FinancialDataSource
from src.mcp_api.mcp_tools.base import call_index_constituent_tool

logger = logging.getLogger(__name__)


def register_index_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    Register stock index related tools with MCP application

    Args:
        app: FastMCP application instance
        active_data_source: Active financial data source
    """

    @app.tool()
    def get_stock_industry(code: Optional[str] = None, date: Optional[str] = None) -> str:
        """
        Get industry classification data for specific stock or all stocks on specified date

        Args:
            code: Optional. Stock code (e.g., 'sh.600000'). If None, gets all stock data.
            date: Optional. Date in 'YYYY-MM-DD' format. If None, uses latest available date.

        Returns:
            Markdown table containing industry classification data or error message
        """
        log_msg = f"Tool 'get_stock_industry' called for code={code or 'all'}, date={date or 'latest'}"
        logger.info(log_msg)
        try:
            # Date validation can be added here if needed
            df = active_data_source.get_stock_industry(code=code, date=date)
            logger.info(
                f"Successfully retrieved industry data for {code or 'all'}, {date or 'latest'}.")
            from src.mcp_api.formatting.markdown_formatter import format_df_to_markdown
            return format_df_to_markdown(df)

        except Exception as e:
            logger.exception(
                f"Exception processing get_stock_industry: {e}")
            return f"Error: An unexpected error occurred: {e}"

    @app.tool()
    def get_sz50_stocks(date: Optional[str] = None) -> str:
        """
        Get SZSE 50 index constituent stocks for specified date

        Args:
            date: Optional. Date in 'YYYY-MM-DD' format. If None, uses latest available date.

        Returns:
            Markdown table containing SZSE 50 index constituent stocks or error message
        """
        return call_index_constituent_tool(
            "get_sz50_stocks",
            active_data_source.get_sz50_stocks,
            "SZSE 50",
            date
        )

    @app.tool()
    def get_hs300_stocks(date: Optional[str] = None) -> str:
        """
        Get CSI 300 index constituent stocks for specified date

        Args:
            date: Optional. Date in 'YYYY-MM-DD' format. If None, uses latest available date.

        Returns:
            Markdown table containing CSI 300 index constituent stocks or error message
        """
        return call_index_constituent_tool(
            "get_hs300_stocks",
            active_data_source.get_hs300_stocks,
            "CSI 300",
            date
        )

    @app.tool()
    def get_zz500_stocks(date: Optional[str] = None) -> str:
        """
        Get CSI 500 index constituent stocks for specified date

        Args:
            date: Optional. Date in 'YYYY-MM-DD' format. If None, uses latest available date.

        Returns:
            Markdown table containing CSI 500 index constituent stocks or error message
        """
        return call_index_constituent_tool(
            "get_zz500_stocks",
            active_data_source.get_zz500_stocks,
            "CSI 500",
            date
        )