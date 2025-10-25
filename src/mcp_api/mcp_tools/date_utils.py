import logging
from datetime import datetime
import calendar

from mcp.server.fastmcp import FastMCP
from src.mcp_api.data_source_interface import FinancialDataSource

logger = logging.getLogger(__name__)


def register_date_utils_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    Register date utility functions to MCP application

    Args:
        app: FastMCP application instance
        active_data_source: Active financial data source
    """

    @app.tool()
    def get_current_date() -> str:
        """
        Get current date, can be used to query latest data.

        Returns:
            Current date in 'YYYY-MM-DD' format.
        """
        logger.info("Tool 'get_current_date' called")
        current_date = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Returning current date: {current_date}")
        return current_date

    @app.tool()
    def get_latest_trading_date() -> str:
        """
        Get the latest trading date. If today is a trading day, return today's date; otherwise return the most recent trading day.

        Returns:
            Latest trading date in 'YYYY-MM-DD' format.
        """
        logger.info("Tool 'get_latest_trading_date' called")
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            # Get trading calendar for around one week from current date
            start_date = (datetime.now().replace(day=1)).strftime("%Y-%m-%d")
            end_date = (datetime.now().replace(day=28)).strftime("%Y-%m-%d")

            # Get trading date data from data source
            df = active_data_source.get_trade_dates(
                start_date=start_date, end_date=end_date)

            # Filter out valid trading days
            valid_trading_days = df[df['is_trading_day']
                                    == '1']['calendar_date'].tolist()

            # Find the maximum date less than or equal to today (i.e., the most recent trading day)
            latest_trading_date = None
            for date in valid_trading_days:
                if date <= today and (latest_trading_date is None or date > latest_trading_date):
                    latest_trading_date = date

            if latest_trading_date:
                logger.info(
                    f"Latest trading date found: {latest_trading_date}")
                return latest_trading_date
            else:
                logger.warning(
                    "No trading dates found before today, returning today's date")
                return today

        except Exception as e:
            # Log exception and return current date when error occurs
            logger.exception(f"Error determining latest trading date: {e}")
            return datetime.now().strftime("%Y-%m-%d")

    @app.tool()
    def get_market_analysis_timeframe(period: str = "recent") -> str:
        """
        Get appropriate time range for market analysis, based on current real date rather than training data.
        This tool should be called first when conducting market analysis or broad market analysis to ensure using the latest actual data.

        Args:
            period: Time range type, optional values:
                   "recent": Recent 1-2 months (default)
                   "quarter": Recent quarter
                   "half_year": Recent half year
                   "year": Recent year

        Returns:
            Detailed description string containing analysis time range, format "YYYY年M月-YYYY年M月" (Year-Month to Year-Month).
        """
        logger.info(
            f"Tool 'get_market_analysis_timeframe' called with period={period}")

        now = datetime.now()
        end_date = now

        # Determine start date based on requested time period
        if period == "recent":
            # Recent 1-2 months
            if now.day < 15:
                # If currently at beginning of month, look at previous two months
                if now.month == 1:
                    start_date = datetime(now.year - 1, 11, 1)  # November of previous year
                    middle_date = datetime(now.year - 1, 12, 1)  # December of previous year
                elif now.month == 2:
                    start_date = datetime(now.year, 1, 1)  # January of this year
                    middle_date = start_date
                else:
                    start_date = datetime(now.year, now.month - 2, 1)  # Two months ago
                    middle_date = datetime(now.year, now.month - 1, 1)  # Last month
            else:
                # If currently mid-month or end of month, look from last month to now
                if now.month == 1:
                    start_date = datetime(now.year - 1, 12, 1)  # December of previous year
                    middle_date = start_date
                else:
                    start_date = datetime(now.year, now.month - 1, 1)  # Last month
                    middle_date = start_date

        elif period == "quarter":
            # Recent quarter (about 3 months)
            if now.month <= 3:
                start_date = datetime(now.year - 1, now.month + 9, 1)
            else:
                start_date = datetime(now.year, now.month - 3, 1)
            middle_date = start_date

        elif period == "half_year":
            # Recent half year
            if now.month <= 6:
                start_date = datetime(now.year - 1, now.month + 6, 1)
            else:
                start_date = datetime(now.year, now.month - 6, 1)
            # Calculate middle month (half-year midpoint)
            middle_date = datetime(start_date.year, start_date.month + 3, 1) if start_date.month <= 9 else \
                datetime(start_date.year + 1, start_date.month - 9, 1)

        elif period == "year":
            # Recent year
            start_date = datetime(now.year - 1, now.month, 1)
            # Calculate middle month (year midpoint)
            middle_date = datetime(start_date.year, start_date.month + 6, 1) if start_date.month <= 6 else \
                datetime(start_date.year + 1, start_date.month - 6, 1)
        else:
            # Default to recent 1 month
            if now.month == 1:
                start_date = datetime(now.year - 1, 12, 1)
            else:
                start_date = datetime(now.year, now.month - 1, 1)
            middle_date = start_date

        # Format for user-friendly display
        def get_month_end_day(year, month):
            # Get the last day of specified year and month
            return calendar.monthrange(year, month)[1]

        # Ensure end date does not exceed current date
        end_day = min(get_month_end_day(
            end_date.year, end_date.month), end_date.day)
        end_display_date = f"{end_date.year}年{end_date.month}月"  # Year and month (年=year, 月=month)
        end_iso_date = f"{end_date.year}-{end_date.month:02d}-{end_day:02d}"

        # Start date display
        start_display_date = f"{start_date.year}年{start_date.month}月"  # Year and month (年=year, 月=month)
        start_iso_date = f"{start_date.year}-{start_date.month:02d}-01"

        # Generate appropriate date range display based on time span
        if start_date.year != end_date.year:
            # If spanning years, show years
            date_range = f"{start_date.year}年{start_date.month}月-{end_date.year}年{end_date.month}月"  # Year-month to year-month (年=year, 月=month)
        elif middle_date.month != start_date.month and middle_date.month != end_date.month:
            # If quarter or half year, show middle month
            date_range = f"{start_date.year}年{start_date.month}月-{middle_date.month}月-{end_date.month}月"  # Year-month to month to month (年=year, 月=month)
        elif start_date.month != end_date.month:
            # If different months within the same year
            date_range = f"{start_date.year}年{start_date.month}月-{end_date.month}月"  # Year-month to month (年=year, 月=month)
        else:
            # If same month
            date_range = f"{start_date.year}年{start_date.month}月"  # Year-month (年=year, 月=month)

        # Combine into final result string, including user-friendly format and ISO format
        result = f"{date_range} (ISO date range: {start_iso_date} to {end_iso_date})"  # ISO date range: from to
        logger.info(f"Generated market analysis timeframe: {result}")
        return result