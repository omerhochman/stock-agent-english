from typing import Any, Dict

import pandas as pd

from .adapter import DataSourceAdapter


class DataAPI:
    """Unified data API interface, encapsulates internal data source adapter implementation"""

    def __init__(self):
        self.adapter = DataSourceAdapter()

    def get_price_data(
        self, ticker: str, start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        """
        Get stock price data

        Args:
            ticker: Stock code
            start_date: Start date, format: YYYY-MM-DD
            end_date: End date, format: YYYY-MM-DD

        Returns:
            DataFrame containing price data
        """
        return self.adapter.get_price_history(ticker, start_date, end_date)

    def get_financial_metrics(self, ticker: str) -> Dict[str, Any]:
        """
        Get financial metrics data

        Args:
            ticker: Stock code

        Returns:
            Dictionary containing financial metrics
        """
        return self.adapter.get_financial_metrics(ticker)

    def get_financial_statements(self, ticker: str) -> Dict[str, Any]:
        """
        Get financial statements data

        Args:
            ticker: Stock code

        Returns:
            Dictionary containing financial statements data
        """
        return self.adapter.get_financial_statements(ticker)

    def get_market_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get market data (now includes industry metrics)

        Args:
            ticker: Stock code

        Returns:
            Dictionary containing market data and industry metrics
        """
        return self.adapter.get_market_data(ticker)

    def get_industry_metrics(self, ticker: str) -> Dict[str, Any]:
        """
        Get industry metrics data

        Args:
            ticker: Stock code

        Returns:
            Dictionary containing industry metrics
        """
        return self.adapter.get_industry_metrics(ticker)
