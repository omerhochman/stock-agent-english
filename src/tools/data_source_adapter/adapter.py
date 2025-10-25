import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd

from src.utils.logging_config import setup_logger

from .akshare_adapter import (
    get_akshare_financial_metrics,
    get_akshare_financial_statements,
    get_akshare_market_data,
    get_akshare_price_data,
)
from .cache import get_cached_data
from .retry_decorator import retry
from .tushare_adapter import (
    get_tushare_financial_metrics,
    get_tushare_financial_statements,
    get_tushare_market_data,
    get_tushare_price_data,
)

logger = setup_logger("data_source_adapter")


class DataSourceAdapter:
    """Data source adapter, supports getting data from AKShare and TuShare"""

    @staticmethod
    def convert_stock_code(symbol: str) -> tuple:
        """
        Convert stock code format, return corresponding code format for AKShare and TuShare
        Returns tuple: (akshare_code, tushare_code, exchange_prefix)
        """
        # Compatibility with existing code formats
        if symbol.startswith(("sh", "sz", "bj")):
            # Already has exchange prefix
            exchange_prefix = symbol[:2]
            code = symbol[2:]
        else:
            # Determine exchange based on code
            if symbol.startswith("6"):
                exchange_prefix = "sh"
            elif symbol.startswith(("0", "3")):
                exchange_prefix = "sz"
            elif symbol.startswith("4"):
                exchange_prefix = "bj"
            else:
                exchange_prefix = "sh"  # Default to Shanghai Stock Exchange
            code = symbol

        # AKShare format
        akshare_code = symbol
        # TuShare format: code.exchange_abbreviation (sh->SH, sz->SZ, bj->BJ)
        tushare_code = f"{code}.{exchange_prefix.upper()}"

        return akshare_code, tushare_code, exchange_prefix

    @retry(max_tries=3, delay_seconds=2)
    def get_price_history(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """
        Get historical price data, prioritize AKShare, fallback to TuShare on failure

        Args:
            symbol: Stock code
            start_date: Start date, format: YYYY-MM-DD
            end_date: End date, format: YYYY-MM-DD
            adjust: Adjustment type, qfq: forward adjustment, hfq: backward adjustment, "": no adjustment

        Returns:
            DataFrame containing price data
        """
        logger.info(
            f"Getting price history for {symbol} from {start_date} to {end_date}"
        )

        # Process date parameters
        start_date, end_date = self._process_dates(start_date, end_date)

        # Convert stock code
        akshare_code, tushare_code, exchange_prefix = self.convert_stock_code(symbol)

        # Cache key
        cache_key = f"price_hist_{symbol}_{start_date}_{end_date}_{adjust}"

        # If it's same-day data, fetch directly without using cache
        current_date = datetime.now()
        end_date_obj = (
            datetime.strptime(end_date, "%Y%m%d")
            if isinstance(end_date, str)
            else end_date
        )
        if (current_date - end_date_obj).days < 1:
            df = self._fetch_price_history(
                akshare_code, tushare_code, start_date, end_date, adjust
            )
        else:
            # Use cache
            df_dict = get_cached_data(
                cache_key,
                lambda: self._fetch_price_history(
                    akshare_code, tushare_code, start_date, end_date, adjust
                ),
                ttl_days=3,
            )

            if isinstance(df_dict, list):
                df = pd.DataFrame(df_dict)
                if not df.empty and "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                logger.info(f"Successfully retrieved {len(df)} records from cache")
            else:
                df = pd.DataFrame()

        return df

    def _process_dates(self, start_date, end_date):
        """Process date parameters"""
        current_date = datetime.now()
        yesterday = current_date - timedelta(days=1)

        if not end_date:
            end_date = yesterday
        else:
            end_date = (
                datetime.strptime(end_date, "%Y-%m-%d")
                if isinstance(end_date, str)
                else end_date
            )
            if end_date > yesterday:
                end_date = yesterday

        if not start_date:
            start_date = end_date - timedelta(days=365)
        else:
            start_date = (
                datetime.strptime(start_date, "%Y-%m-%d")
                if isinstance(start_date, str)
                else start_date
            )

        # Convert to string format for API calls
        start_date_str = (
            start_date.strftime("%Y%m%d")
            if isinstance(start_date, datetime)
            else start_date
        )
        end_date_str = (
            end_date.strftime("%Y%m%d") if isinstance(end_date, datetime) else end_date
        )

        return start_date_str, end_date_str

    def _fetch_price_history(
        self,
        akshare_code: str,
        tushare_code: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        """Internal method: get price history data from data source"""
        df = pd.DataFrame()

        # First try using AKShare
        try:
            df = get_akshare_price_data(akshare_code, start_date, end_date, adjust)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"AKShare data fetch failed: {str(e)}")
            logger.debug(f"AKShare error details: {traceback.format_exc()}")
            logger.info("Falling back to TuShare...")

        # If AKShare fails or is unavailable, try using TuShare
        try:
            df = get_tushare_price_data(tushare_code, start_date, end_date, adjust)
            if not df.empty:
                return df
            else:
                logger.warning("TuShare returned empty DataFrame")
        except Exception as e:
            logger.error(f"TuShare data fetch failed: {str(e)}")
            logger.debug(f"TuShare error details: {traceback.format_exc()}")

        # If both data sources fail, return empty DataFrame
        if df.empty:
            logger.warning(
                "Both AKShare and TuShare data fetch failed, returning empty DataFrame"
            )

        return df

    @retry(max_tries=3, delay_seconds=2)
    def get_financial_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Get financial metrics data, prioritize AKShare, fallback to TuShare on failure

        Args:
            symbol: Stock code

        Returns:
            Dictionary containing financial metrics
        """
        logger.info(f"Getting financial metrics for {symbol}")

        # Convert stock code
        akshare_code, tushare_code, exchange_prefix = self.convert_stock_code(symbol)

        # Cache key
        cache_key = f"fin_metrics_{symbol}"

        # Use cache
        metrics = get_cached_data(
            cache_key,
            lambda: self._fetch_financial_metrics(
                akshare_code, tushare_code, exchange_prefix
            ),
            ttl_days=1,  # Financial data only needs to be updated once per day
        )

        return metrics

    def _fetch_financial_metrics(
        self, akshare_code: str, tushare_code: str, exchange_prefix: str
    ) -> List[Dict[str, Any]]:
        """Internal method: get financial metrics from data source"""
        metrics = [{}]  # Default return a list with one empty dictionary

        # First try using AKShare
        try:
            metrics = get_akshare_financial_metrics(akshare_code, exchange_prefix)
            if metrics != [{}]:
                return metrics
        except Exception as e:
            logger.warning(f"AKShare financial metrics fetch failed: {str(e)}")
            logger.info("Falling back to TuShare...")

        # If AKShare fails or is unavailable, try using TuShare
        try:
            metrics = get_tushare_financial_metrics(tushare_code)
            return metrics
        except Exception as e:
            logger.error(f"TuShare financial metrics fetch failed: {str(e)}")

        return metrics  # Return default empty metrics list

    @retry(max_tries=3, delay_seconds=2)
    def get_financial_statements(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get financial statements data, prioritize AKShare, fallback to TuShare on failure

        Args:
            symbol: Stock code

        Returns:
            List of dictionaries containing financial statements data
        """
        logger.info(f"Getting financial statements for {symbol}")

        # Convert stock code
        akshare_code, tushare_code, exchange_prefix = self.convert_stock_code(symbol)

        # Cache key
        cache_key = f"fin_statements_{symbol}"

        # Use cache
        statements = get_cached_data(
            cache_key,
            lambda: self._fetch_financial_statements(
                akshare_code, tushare_code, exchange_prefix
            ),
            ttl_days=7,  # Financial statements can be cached longer
        )

        return statements

    def _fetch_financial_statements(
        self, akshare_code: str, tushare_code: str, exchange_prefix: str
    ) -> List[Dict[str, Any]]:
        """Internal method: get financial statements from data source"""
        default_items = [
            {
                "net_income": 0,
                "operating_revenue": 0,
                "operating_profit": 0,
                "working_capital": 0,
                "depreciation_and_amortization": 0,
                "capital_expenditure": 0,
                "free_cash_flow": 0,
            },
            {
                "net_income": 0,
                "operating_revenue": 0,
                "operating_profit": 0,
                "working_capital": 0,
                "depreciation_and_amortization": 0,
                "capital_expenditure": 0,
                "free_cash_flow": 0,
            },
        ]

        # First try using AKShare
        try:
            statements = get_akshare_financial_statements(akshare_code, exchange_prefix)
            if statements != default_items:
                return statements
        except Exception as e:
            logger.warning(f"AKShare financial statements fetch failed: {str(e)}")
            logger.info("Falling back to TuShare...")

        # If AKShare fails or is unavailable, try using TuShare
        try:
            statements = get_tushare_financial_statements(tushare_code)
            return statements
        except Exception as e:
            logger.error(f"TuShare financial statements fetch failed: {str(e)}")

        return default_items

    @retry(max_tries=3, delay_seconds=2)
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get market data, now includes industry metrics

        Args:
            symbol: Stock code

        Returns:
            Dictionary containing market data and industry metrics
        """
        logger.info(f"Getting market data for {symbol}")

        # Convert stock code
        akshare_code, tushare_code, exchange_prefix = self.convert_stock_code(symbol)

        # Cache key
        cache_key = f"market_data_with_industry_{symbol}"

        # Market data updates frequently, use shorter cache time
        market_data = get_cached_data(
            cache_key,
            lambda: self._fetch_market_data_with_industry(akshare_code, tushare_code),
            ttl_days=0.5,  # 12 hours
        )

        return market_data

    def _fetch_market_data_with_industry(
        self, akshare_code: str, tushare_code: str
    ) -> Dict[str, Any]:
        """Internal method: get market information including industry data"""
        # Get basic market data
        market_data = self._fetch_market_data(akshare_code, tushare_code)

        # Add industry metrics
        industry_data = self._fetch_industry_metrics(akshare_code)

        # Merge data
        combined_data = {**market_data, **industry_data}
        return combined_data

    def _fetch_market_data(
        self, akshare_code: str, tushare_code: str
    ) -> Dict[str, Any]:
        """Internal method: get market data from data source"""
        default_data = {
            "market_cap": 0,
            "volume": 0,
            "average_volume": 0,
            "fifty_two_week_high": 0,
            "fifty_two_week_low": 0,
        }

        # First try using AKShare
        try:
            market_data = get_akshare_market_data(akshare_code)
            if market_data != default_data:
                return market_data
        except Exception as e:
            logger.warning(f"AKShare market data fetch failed: {str(e)}")
            logger.info("Falling back to TuShare...")

        # If AKShare fails or is unavailable, try using TuShare
        try:
            market_data = get_tushare_market_data(tushare_code)
            return market_data
        except Exception as e:
            logger.error(f"TuShare market data fetch failed: {str(e)}")

        return default_data

    retry(max_tries=3, delay_seconds=2)

    def get_industry_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Get industry metrics data

        Args:
            symbol: Stock code

        Returns:
            Dictionary containing industry metrics
        """
        logger.info(f"Getting industry metrics for {symbol}")

        # Convert stock code
        akshare_code, tushare_code, exchange_prefix = self.convert_stock_code(symbol)

        # Cache key
        cache_key = f"industry_metrics_{symbol}"

        # Use cache
        industry_metrics = get_cached_data(
            cache_key,
            lambda: self._fetch_industry_metrics(akshare_code),
            ttl_days=1,  # Industry data only needs to be updated once per day
        )

        return industry_metrics

    def _fetch_industry_metrics(self, akshare_code: str) -> Dict[str, Any]:
        """Internal method: get industry metrics from data source"""
        from .industry import query_industry_metrics

        try:
            # Use industry module to query metrics
            industry_data = query_industry_metrics(akshare_code)
            return industry_data
        except Exception as e:
            logger.error(f"Industry metrics fetch failed: {str(e)}")
            # Return default values
            return {
                "stock": akshare_code,
                "industry": "Unknown Industry",
                "industry_avg_pe": 15,
                "industry_avg_pb": 1.5,
                "industry_growth": 0.05,
            }
