import logging

from mcp.server.fastmcp import FastMCP

from src.mcp_api.data_source_interface import FinancialDataSource
from src.mcp_api.mcp_tools.base import call_financial_data_tool

logger = logging.getLogger(__name__)


def register_financial_report_tools(
    app: FastMCP, active_data_source: FinancialDataSource
):
    """
    Register financial report related tools with MCP application

    Args:
        app: FastMCP application instance
        active_data_source: Active financial data source
    """

    @app.tool()
    def get_profit_data(code: str, year: str, quarter: int) -> str:
        """
        Get quarterly profitability data for stock (e.g., ROE, net profit margin)

        Args:
            code: Stock code (e.g., 'sh.600000')
            year: 4-digit year (e.g., '2023')
            quarter: Quarter (1, 2, 3, or 4)

        Returns:
            Markdown table containing profitability data or error message
        """
        return call_financial_data_tool(
            "get_profit_data",
            active_data_source.get_profit_data,
            "Profitability",
            code,
            year,
            quarter,
        )

    @app.tool()
    def get_operation_data(code: str, year: str, quarter: int) -> str:
        """
        Get quarterly operational capability data for stock (e.g., turnover ratios)

        Args:
            code: Stock code (e.g., 'sh.600000')
            year: 4-digit year (e.g., '2023')
            quarter: Quarter (1, 2, 3, or 4)

        Returns:
            Markdown table containing operational capability data or error message
        """
        return call_financial_data_tool(
            "get_operation_data",
            active_data_source.get_operation_data,
            "Operation Capability",
            code,
            year,
            quarter,
        )

    @app.tool()
    def get_growth_data(code: str, year: str, quarter: int) -> str:
        """
        Get quarterly growth capability data for stock (e.g., year-over-year growth rates)

        Args:
            code: Stock code (e.g., 'sh.600000')
            year: 4-digit year (e.g., '2023')
            quarter: Quarter (1, 2, 3, or 4)

        Returns:
            Markdown table containing growth capability data or error message
        """
        return call_financial_data_tool(
            "get_growth_data",
            active_data_source.get_growth_data,
            "Growth Capability",
            code,
            year,
            quarter,
        )

    @app.tool()
    def get_balance_data(code: str, year: str, quarter: int) -> str:
        """
        Get quarterly balance sheet/solvency data for stock (e.g., current ratio, debt ratio)

        Args:
            code: Stock code (e.g., 'sh.600000')
            year: 4-digit year (e.g., '2023')
            quarter: Quarter (1, 2, 3, or 4)

        Returns:
            Markdown table containing balance sheet data or error message
        """
        return call_financial_data_tool(
            "get_balance_data",
            active_data_source.get_balance_data,
            "Balance Sheet",
            code,
            year,
            quarter,
        )

    @app.tool()
    def get_cash_flow_data(code: str, year: str, quarter: int) -> str:
        """
        Get quarterly cash flow data for stock (e.g., operating cash flow/revenue ratio)

        Args:
            code: Stock code (e.g., 'sh.600000')
            year: 4-digit year (e.g., '2023')
            quarter: Quarter (1, 2, 3, or 4)

        Returns:
            Markdown table containing cash flow data or error message
        """
        return call_financial_data_tool(
            "get_cash_flow_data",
            active_data_source.get_cash_flow_data,
            "Cash Flow",
            code,
            year,
            quarter,
        )

    @app.tool()
    def get_dupont_data(code: str, year: str, quarter: int) -> str:
        """
        Get quarterly DuPont analysis data for stock (ROE decomposition)

        Args:
            code: Stock code (e.g., 'sh.600000')
            year: 4-digit year (e.g., '2023')
            quarter: Quarter (1, 2, 3, or 4)

        Returns:
            Markdown table containing DuPont analysis data or error message
        """
        return call_financial_data_tool(
            "get_dupont_data",
            active_data_source.get_dupont_data,
            "DuPont Analysis",
            code,
            year,
            quarter,
        )

    @app.tool()
    def get_performance_express_report(
        code: str, start_date: str, end_date: str
    ) -> str:
        """
        Get performance express reports for stock within specified date range
        Note: Companies are not required to publish performance express reports except in specific circumstances

        Args:
            code: Stock code (e.g., 'sh.600000')
            start_date: Start date (report publication/update date), format 'YYYY-MM-DD'
            end_date: End date (report publication/update date), format 'YYYY-MM-DD'

        Returns:
            Markdown table containing performance express report data or error message
        """
        logger.info(
            f"Tool 'get_performance_express_report' called for {code} ({start_date} to {end_date})"
        )
        try:
            # Add date validation if needed
            df = active_data_source.get_performance_express_report(
                code=code, start_date=start_date, end_date=end_date
            )
            logger.info(
                f"Successfully retrieved performance express reports for {code}."
            )
            from src.mcp_api.formatting.markdown_formatter import format_df_to_markdown

            return format_df_to_markdown(df)

        except Exception as e:
            logger.exception(
                f"Exception processing get_performance_express_report for {code}: {e}"
            )
            return f"Error: An unexpected error occurred: {e}"

    @app.tool()
    def get_forecast_report(code: str, start_date: str, end_date: str) -> str:
        """
        Get performance forecast reports for stock within specified date range
        Note: Companies are not required to publish performance forecasts except in specific circumstances

        Args:
            code: Stock code (e.g., 'sh.600000')
            start_date: Start date (report publication/update date), format 'YYYY-MM-DD'
            end_date: End date (report publication/update date), format 'YYYY-MM-DD'

        Returns:
            Markdown table containing performance forecast data or error message
        """
        logger.info(
            f"Tool 'get_forecast_report' called for {code} ({start_date} to {end_date})"
        )
        try:
            # Add date validation if needed
            df = active_data_source.get_forecast_report(
                code=code, start_date=start_date, end_date=end_date
            )
            logger.info(
                f"Successfully retrieved performance forecast reports for {code}."
            )
            from src.mcp_api.formatting.markdown_formatter import format_df_to_markdown

            return format_df_to_markdown(df)

        except Exception as e:
            logger.exception(
                f"Exception processing get_forecast_report for {code}: {e}"
            )
            return f"Error: An unexpected error occurred: {e}"
