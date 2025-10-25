import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from src.mcp_api.data_source_interface import (
    DataSourceError,
    FinancialDataSource,
    LoginError,
    NoDataFoundError,
)
from src.mcp_api.formatting.markdown_formatter import format_df_to_markdown

logger = logging.getLogger(__name__)


def register_market_overview_tools(
    app: FastMCP, active_data_source: FinancialDataSource
):
    """
    Register market overview tools with MCP application

    Args:
        app: FastMCP application instance
        active_data_source: Active financial data source
    """

    @app.tool()
    def get_trade_dates(
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> str:
        """
        Get trading day information for specified date range

        Args:
            start_date: Optional. Start date in 'YYYY-MM-DD' format. If None, defaults to 2015-01-01.
            end_date: Optional. End date in 'YYYY-MM-DD' format. If None, defaults to current date.

        Returns:
            Markdown table showing whether each day in the date range is a trading day (1) or non-trading day (0).
        """
        logger.info(
            f"Tool 'get_trade_dates' called for range {start_date or 'default'} to {end_date or 'default'}"
        )
        try:
            # Date validation can be added here if needed
            df = active_data_source.get_trade_dates(
                start_date=start_date, end_date=end_date
            )
            logger.info("Successfully retrieved trade dates.")
            # Trading dates table may be long, apply standard truncation
            return format_df_to_markdown(df)

        except NoDataFoundError as e:
            # No data found error
            logger.warning(f"NoDataFoundError: {e}")
            return f"Error: {e}"
        except LoginError as e:
            # Login error
            logger.error(f"LoginError: {e}")
            return f"Error: Could not connect to data source. {e}"
        except DataSourceError as e:
            # Data source error
            logger.error(f"DataSourceError: {e}")
            return f"Error: An error occurred while fetching data. {e}"
        except ValueError as e:
            # Value error
            logger.warning(f"ValueError: {e}")
            return f"Error: Invalid input parameter. {e}"
        except Exception as e:
            # Unexpected exception
            logger.exception(f"Unexpected Exception processing get_trade_dates: {e}")
            return f"Error: An unexpected error occurred: {e}"

    @app.tool()
    def get_all_stock(date: Optional[str] = None) -> str:
        """
        Get all stocks (A-shares and indices) and their trading status list for specified date

        Args:
            date: Optional. Date in 'YYYY-MM-DD' format. If None, uses current date.

        Returns:
            Markdown table listing stock codes, names and their trading status (1=trading, 0=suspended).
        """
        logger.info(f"Tool 'get_all_stock' called for date={date or 'default'}")
        try:
            # Add date validation if needed
            df = active_data_source.get_all_stock(date=date)
            logger.info(f"Successfully retrieved stock list for {date or 'default'}.")
            # This list may be very long, apply standard truncation
            return format_df_to_markdown(df)

        except NoDataFoundError as e:
            # No data found error
            logger.warning(f"NoDataFoundError: {e}")
            return f"Error: {e}"
        except LoginError as e:
            # Login error
            logger.error(f"LoginError: {e}")
            return f"Error: Could not connect to data source. {e}"
        except DataSourceError as e:
            # Data source error
            logger.error(f"DataSourceError: {e}")
            return f"Error: An error occurred while fetching data. {e}"
        except ValueError as e:
            # Value error
            logger.warning(f"ValueError: {e}")
            return f"Error: Invalid input parameter. {e}"
        except Exception as e:
            # Unexpected exception
            logger.exception(f"Unexpected Exception processing get_all_stock: {e}")
            return f"Error: An unexpected error occurred: {e}"
