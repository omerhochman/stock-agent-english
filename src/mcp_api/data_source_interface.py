from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List

class DataSourceError(Exception):
    """Base exception class for data source errors."""
    pass


class LoginError(DataSourceError):
    """Exception raised when login to data source fails."""
    pass


class NoDataFoundError(DataSourceError):
    """Exception raised when no data is found for a given query."""
    pass


class FinancialDataSource(ABC):
    """
    Abstract base class defining the financial data source interface.
    Implementations of this class provide access to specific financial data APIs
    (e.g., Baostock, Akshare).
    """

    @abstractmethod
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
        Get historical K-line (OHLCV) data for a given stock code.

        Args:
            code: Stock code (e.g., 'sh.600000', 'sz.000001').
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            frequency: Data frequency. Common values depend on underlying data source
                      (e.g., 'd' for daily, 'w' for weekly, 'm' for monthly,
                      '5', '15', '30', '60' for minute-level). Default is 'd'.
            adjust_flag: Adjustment flag for historical data. Common values depend on data source
                        (e.g., '1' for pre-adjustment, '2' for post-adjustment, '3' for no adjustment).
                        Default is '3'.
            fields: Optional list of specific fields to retrieve. If None,
                   retrieves implementation-defined default fields.

        Returns:
            pandas DataFrame containing historical K-line data with columns corresponding to requested fields.

        Raises:
            LoginError: If login to data source fails.
            NoDataFoundError: If no data is found for the query.
            DataSourceError: Other data source related errors.
            ValueError: If input parameters are invalid.
        """
        pass

    @abstractmethod
    def get_stock_basic_info(self, code: str) -> pd.DataFrame:
        """
        Get basic information for a given stock code.

        Args:
            code: Stock code (e.g., 'sh.600000', 'sz.000001').

        Returns:
            pandas DataFrame containing basic stock information.
            Structure and columns depend on underlying data source.
            Usually contains name, industry, listing date, etc.

        Raises:
            LoginError: If login to data source fails.
            NoDataFoundError: If no data is found for the query.
            DataSourceError: Other data source related errors.
            ValueError: If input code is invalid.
        """
        pass

    @abstractmethod
    def get_trade_dates(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Get trading date information within a certain range."""
        pass

    @abstractmethod
    def get_all_stock(self, date: Optional[str] = None) -> pd.DataFrame:
        """Get list of all stocks and their trading status for a specific date."""
        pass

    @abstractmethod
    def get_deposit_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Get benchmark deposit interest rates."""
        pass

    @abstractmethod
    def get_loan_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Get benchmark loan interest rates."""
        pass

    @abstractmethod
    def get_required_reserve_ratio_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None, year_type: str = '0') -> pd.DataFrame:
        """Get required reserve ratio data."""
        pass

    @abstractmethod
    def get_money_supply_data_month(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Get monthly money supply data (M0, M1, M2)."""
        pass

    @abstractmethod
    def get_money_supply_data_year(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Get annual money supply data (M0, M1, M2 - year-end balance)."""
        pass

    @abstractmethod
    def get_shibor_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Get SHIBOR (Shanghai Interbank Offered Rate) data."""
        pass