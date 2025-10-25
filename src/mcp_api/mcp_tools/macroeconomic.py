import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from src.mcp_api.data_source_interface import FinancialDataSource
from src.mcp_api.mcp_tools.base import call_macro_data_tool

logger = logging.getLogger(__name__)


def register_macroeconomic_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    Register macroeconomic data tools with MCP application

    Args:
        app: FastMCP application instance
        active_data_source: Active financial data source
    """

    @app.tool()
    def get_deposit_rate_data(
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> str:
        """
        Get benchmark deposit interest rate data (current, fixed-term) for specified date range

        Args:
            start_date: Optional. Start date in 'YYYY-MM-DD' format
            end_date: Optional. End date in 'YYYY-MM-DD' format

        Returns:
            Markdown table containing deposit interest rate data or error message
        """
        return call_macro_data_tool(
            "get_deposit_rate_data",
            active_data_source.get_deposit_rate_data,
            "Deposit Rate",
            start_date,
            end_date,
        )

    @app.tool()
    def get_loan_rate_data(
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> str:
        """
        Get benchmark loan interest rate data for specified date range

        Args:
            start_date: Optional. Start date in 'YYYY-MM-DD' format
            end_date: Optional. End date in 'YYYY-MM-DD' format

        Returns:
            Markdown table containing loan interest rate data or error message
        """
        return call_macro_data_tool(
            "get_loan_rate_data",
            active_data_source.get_loan_rate_data,
            "Loan Rate",
            start_date,
            end_date,
        )

    @app.tool()
    def get_required_reserve_ratio_data(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        year_type: str = "0",
    ) -> str:
        """
        Get required reserve ratio data for specified date range

        Args:
            start_date: Optional. Start date in 'YYYY-MM-DD' format
            end_date: Optional. End date in 'YYYY-MM-DD' format
            year_type: Optional. Date filter type. '0' for announcement date (default), '1' for effective date

        Returns:
            Markdown table containing required reserve ratio data or error message
        """
        # Basic validation for year_type parameter
        if year_type not in ["0", "1"]:
            logger.warning(f"Invalid year_type requested: {year_type}")
            return "Error: Invalid year_type '{year_type}'. Valid options are '0' (announcement date) or '1' (effective date)."

        return call_macro_data_tool(
            "get_required_reserve_ratio_data",
            active_data_source.get_required_reserve_ratio_data,
            "Required Reserve Ratio",
            start_date,
            end_date,
            yearType=year_type,  # Correctly named parameter passed to Baostock
        )

    @app.tool()
    def get_money_supply_data_month(
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> str:
        """
        Get monthly money supply data (M0, M1, M2) for specified date range

        Args:
            start_date: Optional. Start date in 'YYYY-MM' format
            end_date: Optional. End date in 'YYYY-MM' format

        Returns:
            Markdown table containing monthly money supply data or error message
        """
        # Add specific validation for YYYY-MM format if needed
        return call_macro_data_tool(
            "get_money_supply_data_month",
            active_data_source.get_money_supply_data_month,
            "Monthly Money Supply",
            start_date,
            end_date,
        )

    @app.tool()
    def get_money_supply_data_year(
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> str:
        """
        Get annual money supply data (M0, M1, M2 - year-end balance) for specified year range

        Args:
            start_date: Optional. Start year in 'YYYY' format
            end_date: Optional. End year in 'YYYY' format

        Returns:
            Markdown table containing annual money supply data or error message
        """
        # Add specific validation for YYYY format if needed
        return call_macro_data_tool(
            "get_money_supply_data_year",
            active_data_source.get_money_supply_data_year,
            "Yearly Money Supply",
            start_date,
            end_date,
        )

    @app.tool()
    def get_shibor_data(
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> str:
        """
        Get SHIBOR (Shanghai Interbank Offered Rate) data for specified date range
        Note: Current data source does not support this functionality. This tool is kept as an interface only.

        Args:
            start_date: Optional. Start date in 'YYYY-MM-DD' format
            end_date: Optional. End date in 'YYYY-MM-DD' format

        Returns:
            Information message that SHIBOR data is currently unavailable
        """
        logger.info(
            f"Tool 'get_shibor_data' called with dates from {start_date or 'default'} to {end_date or 'default'}"
        )
        return "Data source does not support SHIBOR data query functionality. You can query deposit rates (get_deposit_rate_data) or loan rates (get_loan_rate_data) as alternatives."
