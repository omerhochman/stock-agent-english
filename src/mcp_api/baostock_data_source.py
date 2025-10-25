import logging
from typing import List, Optional

import baostock as bs
import pandas as pd

from .data_source_interface import (
    DataSourceError,
    FinancialDataSource,
    LoginError,
    NoDataFoundError,
)
from .mcp_utils import baostock_login_context

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# Default fields for K-line data
DEFAULT_K_FIELDS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "adjustflag",
    "turn",
    "tradestatus",
    "pctChg",
    "isST",
]

# Default fields for basic information
DEFAULT_BASIC_FIELDS = [
    "code",
    "tradeStatus",
    "code_name",
    # Can add more default fields as needed, e.g., "industry", "listingDate"
]


# Helper function to reduce duplicate code in financial data fetching
def _fetch_financial_data(
    bs_query_func, data_type_name: str, code: str, year: str, quarter: int
) -> pd.DataFrame:
    """
    Helper function for fetching financial data

    Args:
        bs_query_func: Baostock query function
        data_type_name: Data type name (for logging)
        code: Stock code
        year: Year
        quarter: Quarter

    Returns:
        DataFrame containing financial data
    """
    logger.info(
        f"Fetching {data_type_name} data for {code}, year={year}, quarter={quarter}"
    )
    try:
        with baostock_login_context():
            rs = bs_query_func(code=code, year=year, quarter=quarter)

            if rs.error_code != "0":
                logger.error(
                    f"Baostock API error ({data_type_name}) for {code}: {rs.error_msg} (code: {rs.error_code})"
                )
                if (
                    "no record found" in rs.error_msg.lower()
                    or rs.error_code == "10002"
                ):
                    raise NoDataFoundError(
                        f"No {data_type_name} data found for {code}, {year}Q{quarter}. Baostock msg: {rs.error_msg}"
                    )
                else:
                    raise DataSourceError(
                        f"Baostock API error fetching {data_type_name} data: {rs.error_msg} (code: {rs.error_code})"
                    )

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                logger.warning(
                    f"No {data_type_name} data found for {code}, {year}Q{quarter} (empty result set from Baostock)."
                )
                raise NoDataFoundError(
                    f"No {data_type_name} data found for {code}, {year}Q{quarter} (empty result set)."
                )

            result_df = pd.DataFrame(data_list, columns=rs.fields)
            logger.info(
                f"Retrieved {len(result_df)} {data_type_name} records for {code}, {year}Q{quarter}."
            )
            return result_df

    except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
        logger.warning(
            f"Caught known error fetching {data_type_name} data for {code}: {type(e).__name__}"
        )
        raise e
    except Exception as e:
        logger.exception(
            f"Unexpected error fetching {data_type_name} data for {code}: {e}"
        )
        raise DataSourceError(
            f"Unexpected error fetching {data_type_name} data for {code}: {e}"
        )


# Helper function to reduce duplicate code in index constituent data fetching
def _fetch_index_constituent_data(
    bs_query_func, index_name: str, date: Optional[str] = None
) -> pd.DataFrame:
    """
    Helper function for fetching index constituent data

    Args:
        bs_query_func: Baostock query function
        index_name: Index name (for logging)
        date: Optional. Query date

    Returns:
        DataFrame containing index constituent data
    """
    logger.info(f"Fetching {index_name} constituents for date={date or 'latest'}")
    try:
        with baostock_login_context():
            rs = bs_query_func(date=date)  # date is optional, defaults to latest

            if rs.error_code != "0":
                logger.error(
                    f"Baostock API error ({index_name} Constituents) for date {date}: {rs.error_msg} (code: {rs.error_code})"
                )
                if (
                    "no record found" in rs.error_msg.lower()
                    or rs.error_code == "10002"
                ):
                    raise NoDataFoundError(
                        f"No {index_name} constituent data found for date {date}. Baostock msg: {rs.error_msg}"
                    )
                else:
                    raise DataSourceError(
                        f"Baostock API error fetching {index_name} constituents: {rs.error_msg} (code: {rs.error_code})"
                    )

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                logger.warning(
                    f"No {index_name} constituent data found for date {date} (empty result set)."
                )
                raise NoDataFoundError(
                    f"No {index_name} constituent data found for date {date} (empty result set)."
                )

            result_df = pd.DataFrame(data_list, columns=rs.fields)
            logger.info(
                f"Retrieved {len(result_df)} {index_name} constituents for date {date or 'latest'}."
            )
            return result_df

    except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
        logger.warning(
            f"Caught known error fetching {index_name} constituents for date {date}: {type(e).__name__}"
        )
        raise e
    except Exception as e:
        logger.exception(
            f"Unexpected error fetching {index_name} constituents for date {date}: {e}"
        )
        raise DataSourceError(
            f"Unexpected error fetching {index_name} constituents for date {date}: {e}"
        )


# Helper function to reduce duplicate code in macroeconomic data fetching
def _fetch_macro_data(
    bs_query_func,
    data_type_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs,  # For additional parameters, e.g., yearType
) -> pd.DataFrame:
    """
    Helper function for fetching macroeconomic data

    Args:
        bs_query_func: Baostock query function
        data_type_name: Data type name (for logging)
        start_date: Optional. Start date
        end_date: Optional. End date
        **kwargs: Additional keyword arguments

    Returns:
        DataFrame containing macroeconomic data
    """
    date_range_log = f"from {start_date or 'default'} to {end_date or 'default'}"
    kwargs_log = f", extra_args={kwargs}" if kwargs else ""
    logger.info(f"Fetching {data_type_name} data {date_range_log}{kwargs_log}")
    try:
        with baostock_login_context():
            rs = bs_query_func(start_date=start_date, end_date=end_date, **kwargs)

            if rs.error_code != "0":
                logger.error(
                    f"Baostock API error ({data_type_name}): {rs.error_msg} (code: {rs.error_code})"
                )
                if (
                    "no record found" in rs.error_msg.lower()
                    or rs.error_code == "10002"
                ):
                    raise NoDataFoundError(
                        f"No {data_type_name} data found for the specified criteria. Baostock msg: {rs.error_msg}"
                    )
                else:
                    raise DataSourceError(
                        f"Baostock API error fetching {data_type_name} data: {rs.error_msg} (code: {rs.error_code})"
                    )

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                logger.warning(
                    f"No {data_type_name} data found for the specified criteria (empty result set)."
                )
                raise NoDataFoundError(
                    f"No {data_type_name} data found for the specified criteria (empty result set)."
                )

            result_df = pd.DataFrame(data_list, columns=rs.fields)
            logger.info(f"Retrieved {len(result_df)} {data_type_name} records.")
            return result_df

    except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
        logger.warning(
            f"Caught known error fetching {data_type_name} data: {type(e).__name__}"
        )
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error fetching {data_type_name} data: {e}")
        raise DataSourceError(f"Unexpected error fetching {data_type_name} data: {e}")


class BaostockDataSource(FinancialDataSource):
    """
    FinancialDataSource concrete implementation using Baostock library
    """

    def _format_fields(
        self, fields: Optional[List[str]], default_fields: List[str]
    ) -> str:
        """
        Format field list to Baostock comma-separated string.

        Args:
            fields: Requested field list
            default_fields: Default field list

        Returns:
            Formatted field string
        """
        if fields is None or not fields:
            logger.debug(
                f"No specific fields requested, using defaults: {default_fields}"
            )
            return ",".join(default_fields)
        # Basic validation: ensure all requested fields are strings
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
        Get historical K-line data using Baostock.

        Args:
            code: Stock code
            start_date: Start date
            end_date: End date
            frequency: Data frequency, default is 'd' (daily)
            adjust_flag: Adjustment flag, default is '3' (no adjustment)
            fields: Optional field list

        Returns:
            DataFrame containing K-line data
        """
        logger.info(
            f"Fetching K-data for {code} ({start_date} to {end_date}), freq={frequency}, adjust={adjust_flag}"
        )
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
                    adjustflag=adjust_flag,
                )

                if rs.error_code != "0":
                    logger.error(
                        f"Baostock API error (K-data) for {code}: {rs.error_msg} (code: {rs.error_code})"
                    )
                    # Check common error codes, such as no data found
                    if (
                        "no record found" in rs.error_msg.lower()
                        or rs.error_code == "10002"
                    ):  # Example error code
                        raise NoDataFoundError(
                            f"No historical data found for {code} in the specified range. Baostock msg: {rs.error_msg}"
                        )
                    else:
                        raise DataSourceError(
                            f"Baostock API error fetching K-data: {rs.error_msg} (code: {rs.error_code})"
                        )

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(
                        f"No historical data found for {code} in range (empty result set from Baostock)."
                    )
                    raise NoDataFoundError(
                        f"No historical data found for {code} in the specified range (empty result set)."
                    )

                # Key: use rs.fields as column names
                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(f"Retrieved {len(result_df)} records for {code}.")
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            # Re-raise known errors
            logger.warning(
                f"Caught known error fetching K-data for {code}: {type(e).__name__}"
            )
            raise e
        except Exception as e:
            # Wrap unexpected errors
            logger.exception(
                f"Unexpected error fetching K-data for {code}: {e}"
            )  # Use logger.exception to include stack trace
            raise DataSourceError(f"Unexpected error fetching K-data for {code}: {e}")

    def get_stock_basic_info(
        self, code: str, fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get stock basic information using Baostock.

        Args:
            code: Stock code
            fields: Optional field list for selecting specific columns

        Returns:
            DataFrame containing stock basic information
        """
        logger.info(f"Fetching basic info for {code}")
        try:
            # Note: query_stock_basic doesn't seem to have fields parameter in documentation,
            # but we keep the signature consistent. It returns a fixed set.
            # If needed, we will use the `fields` parameter to select columns after query.
            logger.debug(
                f"Requesting basic info for {code}. Optional fields requested: {fields}"
            )

            with baostock_login_context():
                # Example: get basic information; adjust API call according to baostock documentation as needed
                # rs = bs.query_stock_basic(code=code, code_name=code_name)  # If name lookup is supported
                rs = bs.query_stock_basic(code=code)

                if rs.error_code != "0":
                    logger.error(
                        f"Baostock API error (Basic Info) for {code}: {rs.error_msg} (code: {rs.error_code})"
                    )
                    if (
                        "no record found" in rs.error_msg.lower()
                        or rs.error_code == "10002"
                    ):
                        raise NoDataFoundError(
                            f"No basic info found for {code}. Baostock msg: {rs.error_msg}"
                        )
                    else:
                        raise DataSourceError(
                            f"Baostock API error fetching basic info: {rs.error_msg} (code: {rs.error_code})"
                        )

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(
                        f"No basic info found for {code} (empty result set from Baostock)."
                    )
                    raise NoDataFoundError(
                        f"No basic info found for {code} (empty result set)."
                    )

                # Key: use rs.fields as column names
                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(
                    f"Retrieved basic info for {code}. Columns: {result_df.columns.tolist()}"
                )

                # Optional: if `fields` parameter is provided, select subset of columns
                if fields:
                    available_cols = [col for col in fields if col in result_df.columns]
                    if not available_cols:
                        raise ValueError(
                            f"None of the requested fields {fields} are available in the basic info result."
                        )
                    logger.debug(
                        f"Selecting columns: {available_cols} from basic info for {code}"
                    )
                    result_df = result_df[available_cols]

                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(
                f"Caught known error fetching basic info for {code}: {type(e).__name__}"
            )
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error fetching basic info for {code}: {e}")
            raise DataSourceError(
                f"Unexpected error fetching basic info for {code}: {e}"
            )

    def get_dividend_data(
        self, code: str, year: str, year_type: str = "report"
    ) -> pd.DataFrame:
        """
        Get dividend information using Baostock.

        Args:
            code: Stock code
            year: Year
            year_type: Year type, 'report' means proposal announcement year, 'operate' means ex-dividend year

        Returns:
            DataFrame containing dividend information
        """
        logger.info(
            f"Fetching dividend data for {code}, year={year}, year_type={year_type}"
        )
        try:
            with baostock_login_context():
                rs = bs.query_dividend_data(code=code, year=year, yearType=year_type)

                if rs.error_code != "0":
                    logger.error(
                        f"Baostock API error (Dividend) for {code}: {rs.error_msg} (code: {rs.error_code})"
                    )
                    if (
                        "no record found" in rs.error_msg.lower()
                        or rs.error_code == "10002"
                    ):
                        raise NoDataFoundError(
                            f"No dividend data found for {code} and year {year}. Baostock msg: {rs.error_msg}"
                        )
                    else:
                        raise DataSourceError(
                            f"Baostock API error fetching dividend data: {rs.error_msg} (code: {rs.error_code})"
                        )

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(
                        f"No dividend data found for {code}, year {year} (empty result set from Baostock)."
                    )
                    raise NoDataFoundError(
                        f"No dividend data found for {code}, year {year} (empty result set)."
                    )

                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(
                    f"Retrieved {len(result_df)} dividend records for {code}, year {year}."
                )
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(
                f"Caught known error fetching dividend data for {code}: {type(e).__name__}"
            )
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error fetching dividend data for {code}: {e}")
            raise DataSourceError(
                f"Unexpected error fetching dividend data for {code}: {e}"
            )

    def get_adjust_factor_data(
        self, code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Get adjustment factor data using Baostock.

        Args:
            code: Stock code
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame containing adjustment factor data
        """
        logger.info(
            f"Fetching adjustment factor data for {code} ({start_date} to {end_date})"
        )
        try:
            with baostock_login_context():
                rs = bs.query_adjust_factor(
                    code=code, start_date=start_date, end_date=end_date
                )

                if rs.error_code != "0":
                    logger.error(
                        f"Baostock API error (Adjust Factor) for {code}: {rs.error_msg} (code: {rs.error_code})"
                    )
                    if (
                        "no record found" in rs.error_msg.lower()
                        or rs.error_code == "10002"
                    ):
                        raise NoDataFoundError(
                            f"No adjustment factor data found for {code} in the specified range. Baostock msg: {rs.error_msg}"
                        )
                    else:
                        raise DataSourceError(
                            f"Baostock API error fetching adjust factor data: {rs.error_msg} (code: {rs.error_code})"
                        )

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(
                        f"No adjustment factor data found for {code} in range (empty result set from Baostock)."
                    )
                    raise NoDataFoundError(
                        f"No adjustment factor data found for {code} in the specified range (empty result set)."
                    )

                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(
                    f"Retrieved {len(result_df)} adjustment factor records for {code}."
                )
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(
                f"Caught known error fetching adjust factor data for {code}: {type(e).__name__}"
            )
            raise e
        except Exception as e:
            logger.exception(
                f"Unexpected error fetching adjust factor data for {code}: {e}"
            )
            raise DataSourceError(
                f"Unexpected error fetching adjust factor data for {code}: {e}"
            )

    def get_profit_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """
        Get quarterly profitability data using Baostock.

        Args:
            code: Stock code
            year: Year
            quarter: Quarter

        Returns:
            DataFrame containing profitability data
        """
        return _fetch_financial_data(
            bs.query_profit_data, "Profitability", code, year, quarter
        )

    def get_operation_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """
        Get quarterly operation capability data using Baostock.

        Args:
            code: Stock code
            year: Year
            quarter: Quarter

        Returns:
            DataFrame containing operation capability data
        """
        return _fetch_financial_data(
            bs.query_operation_data, "Operation Capability", code, year, quarter
        )

    def get_growth_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """
        Get quarterly growth capability data using Baostock.

        Args:
            code: Stock code
            year: Year
            quarter: Quarter

        Returns:
            DataFrame containing growth capability data
        """
        return _fetch_financial_data(
            bs.query_growth_data, "Growth Capability", code, year, quarter
        )

    def get_balance_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """
        Get quarterly balance sheet data (solvency) using Baostock.

        Args:
            code: Stock code
            year: Year
            quarter: Quarter

        Returns:
            DataFrame containing balance sheet data
        """
        return _fetch_financial_data(
            bs.query_balance_data, "Balance Sheet", code, year, quarter
        )

    def get_cash_flow_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """
        Get quarterly cash flow data using Baostock.

        Args:
            code: Stock code
            year: Year
            quarter: Quarter

        Returns:
            DataFrame containing cash flow data
        """
        return _fetch_financial_data(
            bs.query_cash_flow_data, "Cash Flow", code, year, quarter
        )

    def get_dupont_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """
        Get quarterly DuPont analysis data using Baostock.

        Args:
            code: Stock code
            year: Year
            quarter: Quarter

        Returns:
            DataFrame containing DuPont analysis data
        """
        return _fetch_financial_data(
            bs.query_dupont_data, "DuPont Analysis", code, year, quarter
        )

    def get_performance_express_report(
        self, code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Get performance express report using Baostock.

        Args:
            code: Stock code
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame containing performance express report data
        """
        logger.info(
            f"Fetching Performance Express Report for {code} ({start_date} to {end_date})"
        )
        try:
            with baostock_login_context():
                rs = bs.query_performance_express_report(
                    code=code, start_date=start_date, end_date=end_date
                )

                if rs.error_code != "0":
                    logger.error(
                        f"Baostock API error (Perf Express) for {code}: {rs.error_msg} (code: {rs.error_code})"
                    )
                    if (
                        "no record found" in rs.error_msg.lower()
                        or rs.error_code == "10002"
                    ):
                        raise NoDataFoundError(
                            f"No performance express report found for {code} in range {start_date}-{end_date}. Baostock msg: {rs.error_msg}"
                        )
                    else:
                        raise DataSourceError(
                            f"Baostock API error fetching performance express report: {rs.error_msg} (code: {rs.error_code})"
                        )

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(
                        f"No performance express report found for {code} in range {start_date}-{end_date} (empty result set)."
                    )
                    raise NoDataFoundError(
                        f"No performance express report found for {code} in range {start_date}-{end_date} (empty result set)."
                    )

                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(
                    f"Retrieved {len(result_df)} performance express report records for {code}."
                )
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(
                f"Caught known error fetching performance express report for {code}: {type(e).__name__}"
            )
            raise e
        except Exception as e:
            logger.exception(
                f"Unexpected error fetching performance express report for {code}: {e}"
            )
            raise DataSourceError(
                f"Unexpected error fetching performance express report for {code}: {e}"
            )

    def get_forecast_report(
        self, code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Get performance forecast report using Baostock.

        Args:
            code: Stock code
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame containing performance forecast report data
        """
        logger.info(
            f"Fetching Performance Forecast Report for {code} ({start_date} to {end_date})"
        )
        try:
            with baostock_login_context():
                rs = bs.query_forecast_report(
                    code=code, start_date=start_date, end_date=end_date
                )
                # Note: Baostock documentation mentions this function has pagination, but Python API doesn't seem to expose it directly.
                # We get all available pages in the loop below.

                if rs.error_code != "0":
                    logger.error(
                        f"Baostock API error (Forecast) for {code}: {rs.error_msg} (code: {rs.error_code})"
                    )
                    if (
                        "no record found" in rs.error_msg.lower()
                        or rs.error_code == "10002"
                    ):
                        raise NoDataFoundError(
                            f"No performance forecast report found for {code} in range {start_date}-{end_date}. Baostock msg: {rs.error_msg}"
                        )
                    else:
                        raise DataSourceError(
                            f"Baostock API error fetching performance forecast report: {rs.error_msg} (code: {rs.error_code})"
                        )

                data_list = []
                while (
                    rs.next()
                ):  # If rs manages pagination, loop should implicitly handle pagination
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(
                        f"No performance forecast report found for {code} in range {start_date}-{end_date} (empty result set)."
                    )
                    raise NoDataFoundError(
                        f"No performance forecast report found for {code} in range {start_date}-{end_date} (empty result set)."
                    )

                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(
                    f"Retrieved {len(result_df)} performance forecast report records for {code}."
                )
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(
                f"Caught known error fetching performance forecast report for {code}: {type(e).__name__}"
            )
            raise e
        except Exception as e:
            logger.exception(
                f"Unexpected error fetching performance forecast report for {code}: {e}"
            )
            raise DataSourceError(
                f"Unexpected error fetching performance forecast report for {code}: {e}"
            )

    def get_stock_industry(
        self, code: Optional[str] = None, date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get industry classification data using Baostock.

        Args:
            code: Optional. Stock code, if None then get all stocks
            date: Optional. Date, if None then use latest date

        Returns:
            DataFrame containing industry classification data
        """
        log_msg = (
            f"Fetching industry data for code={code or 'all'}, date={date or 'latest'}"
        )
        logger.info(log_msg)
        try:
            with baostock_login_context():
                rs = bs.query_stock_industry(code=code, date=date)

                if rs.error_code != "0":
                    logger.error(
                        f"Baostock API error (Industry) for {code}, {date}: {rs.error_msg} (code: {rs.error_code})"
                    )
                    if (
                        "no record found" in rs.error_msg.lower()
                        or rs.error_code == "10002"
                    ):
                        raise NoDataFoundError(
                            f"No industry data found for {code}, {date}. Baostock msg: {rs.error_msg}"
                        )
                    else:
                        raise DataSourceError(
                            f"Baostock API error fetching industry data: {rs.error_msg} (code: {rs.error_code})"
                        )

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(
                        f"No industry data found for {code}, {date} (empty result set)."
                    )
                    raise NoDataFoundError(
                        f"No industry data found for {code}, {date} (empty result set)."
                    )

                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(
                    f"Retrieved {len(result_df)} industry records for {code or 'all'}, {date or 'latest'}."
                )
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(
                f"Caught known error fetching industry data for {code}, {date}: {type(e).__name__}"
            )
            raise e
        except Exception as e:
            logger.exception(
                f"Unexpected error fetching industry data for {code}, {date}: {e}"
            )
            raise DataSourceError(
                f"Unexpected error fetching industry data for {code}, {date}: {e}"
            )

    def get_sz50_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get SZSE 50 index constituent stocks using Baostock.

        Args:
            date: Optional. Date, if None then use latest date

        Returns:
            DataFrame containing SZSE 50 index constituent stocks
        """
        return _fetch_index_constituent_data(bs.query_sz50_stocks, "SZSE 50", date)

    def get_hs300_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get CSI 300 index constituent stocks using Baostock.

        Args:
            date: Optional. Date, if None then use latest date

        Returns:
            DataFrame containing CSI 300 index constituent stocks
        """
        return _fetch_index_constituent_data(bs.query_hs300_stocks, "CSI 300", date)

    def get_zz500_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get CSI 500 index constituent stocks using Baostock.

        Args:
            date: Optional. Date, if None then use latest date

        Returns:
            DataFrame containing CSI 500 index constituent stocks
        """
        return _fetch_index_constituent_data(bs.query_zz500_stocks, "CSI 500", date)

    def get_trade_dates(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get trading date information using Baostock.

        Args:
            start_date: Optional. Start date
            end_date: Optional. End date

        Returns:
            DataFrame containing trading date information
        """
        logger.info(
            f"Fetching trade dates from {start_date or 'default'} to {end_date or 'default'}"
        )
        try:
            with baostock_login_context():  # For this case, login may not be strictly required, but keep consistent
                rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)

                if rs.error_code != "0":
                    logger.error(
                        f"Baostock API error (Trade Dates): {rs.error_msg} (code: {rs.error_code})"
                    )
                    # Date queries are unlikely to have "no record found", but handle API errors
                    raise DataSourceError(
                        f"Baostock API error fetching trade dates: {rs.error_msg} (code: {rs.error_code})"
                    )

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    # If API returns valid range, this situation should theoretically not occur
                    logger.warning(
                        f"No trade dates returned for range {start_date}-{end_date} (empty result set)."
                    )
                    raise NoDataFoundError(
                        f"No trade dates found for range {start_date}-{end_date} (empty result set)."
                    )

                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(f"Retrieved {len(result_df)} trade date records.")
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(
                f"Caught known error fetching trade dates: {type(e).__name__}"
            )
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error fetching trade dates: {e}")
            raise DataSourceError(f"Unexpected error fetching trade dates: {e}")

    def get_all_stock(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get all stock list for specified date using Baostock.

        Args:
            date: Optional. Date, if None then use current date

        Returns:
            DataFrame containing all stocks
        """
        logger.info(f"Fetching all stock list for date={date or 'default'}")
        try:
            with baostock_login_context():
                rs = bs.query_all_stock(day=date)

                if rs.error_code != "0":
                    logger.error(
                        f"Baostock API error (All Stock) for date {date}: {rs.error_msg} (code: {rs.error_code})"
                    )
                    if (
                        "no record found" in rs.error_msg.lower()
                        or rs.error_code == "10002"
                    ):  # Check if applicable
                        raise NoDataFoundError(
                            f"No stock data found for date {date}. Baostock msg: {rs.error_msg}"
                        )
                    else:
                        raise DataSourceError(
                            f"Baostock API error fetching all stock list: {rs.error_msg} (code: {rs.error_code})"
                        )

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(
                        f"No stock list returned for date {date} (empty result set)."
                    )
                    raise NoDataFoundError(
                        f"No stock list found for date {date} (empty result set)."
                    )

                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(
                    f"Retrieved {len(result_df)} stock records for date {date or 'default'}."
                )
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            logger.warning(
                f"Caught known error fetching all stock list for date {date}: {type(e).__name__}"
            )
            raise e
        except Exception as e:
            logger.exception(
                f"Unexpected error fetching all stock list for date {date}: {e}"
            )
            raise DataSourceError(
                f"Unexpected error fetching all stock list for date {date}: {e}"
            )

    def get_deposit_rate_data(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get benchmark deposit rate data using Baostock.

        Args:
            start_date: Optional. Start date
            end_date: Optional. End date

        Returns:
            DataFrame containing deposit rate data
        """
        return _fetch_macro_data(
            bs.query_deposit_rate_data, "Deposit Rate", start_date, end_date
        )

    def get_loan_rate_data(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get benchmark loan rate data using Baostock.

        Args:
            start_date: Optional. Start date
            end_date: Optional. End date

        Returns:
            DataFrame containing loan rate data
        """
        return _fetch_macro_data(
            bs.query_loan_rate_data, "Loan Rate", start_date, end_date
        )

    def get_required_reserve_ratio_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        year_type: str = "0",
    ) -> pd.DataFrame:
        """
        Get required reserve ratio data using Baostock.

        Args:
            start_date: Optional. Start date
            end_date: Optional. End date
            year_type: Year type, '0' for announcement date, '1' for effective date

        Returns:
            DataFrame containing required reserve ratio data
        """
        # Note: handle additional yearType parameter through kwargs
        return _fetch_macro_data(
            bs.query_required_reserve_ratio_data,
            "Required Reserve Ratio",
            start_date,
            end_date,
            yearType=year_type,
        )

    def get_money_supply_data_month(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get monthly money supply data (M0, M1, M2) using Baostock.

        Args:
            start_date: Optional. Start date, format: YYYY-MM
            end_date: Optional. End date, format: YYYY-MM

        Returns:
            DataFrame containing monthly money supply data
        """
        # Baostock expects date format as YYYY-MM
        return _fetch_macro_data(
            bs.query_money_supply_data_month,
            "Monthly Money Supply",
            start_date,
            end_date,
        )

    def get_money_supply_data_year(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get yearly money supply data (M0, M1, M2 - year-end balance) using Baostock.

        Args:
            start_date: Optional. Start year, format: YYYY
            end_date: Optional. End year, format: YYYY

        Returns:
            DataFrame containing yearly money supply data
        """
        # Baostock expects date format as YYYY
        return _fetch_macro_data(
            bs.query_money_supply_data_year, "Yearly Money Supply", start_date, end_date
        )

    def get_shibor_data(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Baostock does not provide SHIBOR (Shanghai Interbank Offered Rate) data API.
        This method returns empty DataFrame and raises NoDataFoundError.

        Args:
            start_date: Optional. Start date
            end_date: Optional. End date

        Returns:
            Raises NoDataFoundError
        """
        logger.warning(
            "Baostock API does not provide SHIBOR data. Attempting to request SHIBOR data."
        )
        from .data_source_interface import NoDataFoundError

        raise NoDataFoundError(
            "Baostock API does not provide SHIBOR data. Please use other data sources to get SHIBOR data."
        )
