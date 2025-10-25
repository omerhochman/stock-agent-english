import logging
from typing import List, Optional

from mcp.server.fastmcp import FastMCP
from src.mcp_api.data_source_interface import FinancialDataSource, NoDataFoundError, LoginError, DataSourceError
from src.mcp_api.formatting.markdown_formatter import format_df_to_markdown

logger = logging.getLogger(__name__)


def register_stock_market_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    Register stock market data tools with MCP application

    Args:
        app: FastMCP application instance
        active_data_source: Active financial data source
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
        Get historical K-line (OHLCV) data for Chinese A-share stocks

        Args:
            code: Baostock format stock code (e.g., 'sh.600000', 'sz.000001')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            frequency: Data frequency. Valid options (from Baostock):
                         'd': Daily
                         'w': Weekly
                         'm': Monthly
                         '5': 5-minute
                         '15': 15-minute
                         '30': 30-minute
                         '60': 60-minute
                        Default is 'd' (daily)
            adjust_flag: Price/volume adjustment flag. Valid options (from Baostock):
                           '1': Post-adjustment
                           '2': Pre-adjustment
                           '3': No adjustment
                          Default is '3' (no adjustment)
            fields: Optional list of specific data fields (must be valid Baostock fields)
                    If None or empty, will use default fields (e.g., date, code, open, high, low, close, volume, amount, pctChg)

        Returns:
            Markdown format string containing K-line data table, or error message.
            If result set is too large, table may be truncated.
        """
        logger.info(
            f"Tool 'get_historical_k_data' called for {code} ({start_date}-{end_date}, freq={frequency}, adj={adjust_flag}, fields={fields})")
        try:
            # Validate frequency and adjustment flag (if necessary)
            valid_freqs = ['d', 'w', 'm', '5', '15', '30', '60']
            valid_adjusts = ['1', '2', '3']
            if frequency not in valid_freqs:
                logger.warning(f"Invalid frequency requested: {frequency}")
                return f"Error: Invalid frequency '{frequency}'. Valid options are: {valid_freqs}"
            if adjust_flag not in valid_adjusts:
                logger.warning(f"Invalid adjust_flag requested: {adjust_flag}")
                return f"Error: Invalid adjust_flag '{adjust_flag}'. Valid options are: {valid_adjusts}"

            # Call injected data source
            df = active_data_source.get_historical_k_data(
                code=code,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                adjust_flag=adjust_flag,
                fields=fields,
            )
            # Format result
            logger.info(
                f"Successfully retrieved K-data for {code}, formatting to Markdown.")
            return format_df_to_markdown(df)

        except NoDataFoundError as e:
            # No data found error
            logger.warning(f"NoDataFoundError for {code}: {e}")
            return f"Error: {e}"
        except LoginError as e:
            # Login error
            logger.error(f"LoginError for {code}: {e}")
            return f"Error: Could not connect to data source. {e}"
        except DataSourceError as e:
            # Data source error
            logger.error(f"DataSourceError for {code}: {e}")
            return f"Error: An error occurred while fetching data. {e}"
        except ValueError as e:
            # Value error
            logger.warning(f"ValueError processing request for {code}: {e}")
            return f"Error: Invalid input parameter. {e}"
        except Exception as e:
            # Catch all unexpected errors
            logger.exception(
                f"Unexpected Exception processing get_historical_k_data for {code}: {e}")
            return f"Error: An unexpected error occurred: {e}"

    @app.tool()
    def get_stock_basic_info(code: str, fields: Optional[List[str]] = None) -> str:
        """
        Get basic information for a given Chinese A-share stock

        Args:
            code: Baostock format stock code (e.g., 'sh.600000', 'sz.000001')
            fields: Optional list for selecting specific columns from available basic information
                    (e.g., ['code', 'code_name', 'industry', 'listingDate'])
                    If None or empty, returns all available basic information columns provided by Baostock

        Returns:
            Markdown format string containing basic stock information table, or error message
        """
        logger.info(
            f"Tool 'get_stock_basic_info' called for {code} (fields={fields})")
        try:
            # Call injected data source
            # Pass fields parameter; BaostockDataSource implementation will handle selection
            df = active_data_source.get_stock_basic_info(
                code=code, fields=fields)

            # Format result (basic info is usually small, use default truncation)
            logger.info(
                f"Successfully retrieved basic info for {code}, formatting to Markdown.")
            # Basic info uses smaller limits
            return format_df_to_markdown(df, max_rows=10, max_cols=10)

        except NoDataFoundError as e:
            # No data found error
            logger.warning(f"NoDataFoundError for {code}: {e}")
            return f"Error: {e}"
        except LoginError as e:
            # Login error
            logger.error(f"LoginError for {code}: {e}")
            return f"Error: Could not connect to data source. {e}"
        except DataSourceError as e:
            # Data source error
            logger.error(f"DataSourceError for {code}: {e}")
            return f"Error: An error occurred while fetching data. {e}"
        except ValueError as e:
            # Value error
            logger.warning(f"ValueError processing request for {code}: {e}")
            return f"Error: Invalid input parameter or requested field not available. {e}"
        except Exception as e:
            # Unexpected exception
            logger.exception(
                f"Unexpected Exception processing get_stock_basic_info for {code}: {e}")
            return f"Error: An unexpected error occurred: {e}"

    @app.tool()
    def get_dividend_data(code: str, year: str, year_type: str = "report") -> str:
        """
        Get dividend information for a given stock code and year

        Args:
            code: Baostock format stock code (e.g., 'sh.600000', 'sz.000001')
            year: Query year (e.g., '2023')
            year_type: Year type. Valid options (from Baostock):
                         'report': Proposal announcement year
                         'operate': Ex-dividend year
                        Default is 'report' (proposal announcement year)

        Returns:
            Markdown format string containing dividend data table, or error message
        """
        logger.info(
            f"Tool 'get_dividend_data' called for {code}, year={year}, year_type={year_type}")
        try:
            # Basic validation
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
            # No data found error
            logger.warning(f"NoDataFoundError for {code}, year {year}: {e}")
            return f"Error: {e}"
        except LoginError as e:
            # Login error
            logger.error(f"LoginError for {code}: {e}")
            return f"Error: Could not connect to data source. {e}"
        except DataSourceError as e:
            # Data source error
            logger.error(f"DataSourceError for {code}: {e}")
            return f"Error: An error occurred while fetching data. {e}"
        except ValueError as e:
            # Value error
            logger.warning(f"ValueError processing request for {code}: {e}")
            return f"Error: Invalid input parameter. {e}"
        except Exception as e:
            # Unexpected exception
            logger.exception(
                f"Unexpected Exception processing get_dividend_data for {code}: {e}")
            return f"Error: An unexpected error occurred: {e}"

    @app.tool()
    def get_adjust_factor_data(code: str, start_date: str, end_date: str) -> str:
        """
        Get adjustment factor data for a given stock code and date range
        Uses Baostock's "price change adjustment algorithm" factor. Used for calculating adjusted prices.

        Args:
            code: Baostock format stock code (e.g., 'sh.600000', 'sz.000001')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            Markdown format string containing adjustment factor data table, or error message
        """
        logger.info(
            f"Tool 'get_adjust_factor_data' called for {code} ({start_date} to {end_date})")
        try:
            # Basic date validation can be added here if needed
            df = active_data_source.get_adjust_factor_data(
                code=code, start_date=start_date, end_date=end_date)
            logger.info(
                f"Successfully retrieved adjustment factor data for {code}.")
            return format_df_to_markdown(df)

        except NoDataFoundError as e:
            # No data found error
            logger.warning(f"NoDataFoundError for {code}: {e}")
            return f"Error: {e}"
        except LoginError as e:
            # Login error
            logger.error(f"LoginError for {code}: {e}")
            return f"Error: Could not connect to data source. {e}"
        except DataSourceError as e:
            # Data source error
            logger.error(f"DataSourceError for {code}: {e}")
            return f"Error: An error occurred while fetching data. {e}"
        except ValueError as e:
            # Value error
            logger.warning(f"ValueError processing request for {code}: {e}")
            return f"Error: Invalid input parameter. {e}"
        except Exception as e:
            # Unexpected exception
            logger.exception(
                f"Unexpected Exception processing get_adjust_factor_data for {code}: {e}")
            return f"Error: An unexpected error occurred: {e}"