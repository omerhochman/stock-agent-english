import logging
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP
from src.mcp_api.data_source_interface import FinancialDataSource

logger = logging.getLogger(__name__)


def register_analysis_tools(app: FastMCP, active_data_source: FinancialDataSource):
    @app.tool()
    def get_stock_analysis(code: str, analysis_type: str = "fundamental") -> str:
        """
        Provide data-driven stock analysis report, not investment advice.

        Args:
            code: Stock code, e.g., 'sh.600000'
            analysis_type: Analysis type, options: 'fundamental' (fundamental), 'technical' (technical) or 'comprehensive' (comprehensive)

        Returns:
            Data-driven analysis report containing key financial indicators, historical performance and peer comparison
        """
        logger.info(
            f"Tool 'get_stock_analysis' called for {code}, type={analysis_type}")

        # Collect actual data from multiple dimensions
        try:
            # Get basic information
            basic_info = active_data_source.get_stock_basic_info(code=code)

            # Get different data based on analysis type
            if analysis_type in ["fundamental", "comprehensive"]:
                # Get most recent quarter financial data
                recent_year = datetime.now().strftime("%Y")
                recent_quarter = (datetime.now().month - 1) // 3 + 1
                if recent_quarter < 1:  # Handle edge case that may occur at the beginning of the year
                    recent_year = str(int(recent_year) - 1)
                    recent_quarter = 4

                profit_data = active_data_source.get_profit_data(
                    code=code, year=recent_year, quarter=recent_quarter)
                growth_data = active_data_source.get_growth_data(
                    code=code, year=recent_year, quarter=recent_quarter)
                balance_data = active_data_source.get_balance_data(
                    code=code, year=recent_year, quarter=recent_quarter)
                dupont_data = active_data_source.get_dupont_data(
                    code=code, year=recent_year, quarter=recent_quarter)

            if analysis_type in ["technical", "comprehensive"]:
                # Get historical prices
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=180)
                              ).strftime("%Y-%m-%d")
                price_data = active_data_source.get_historical_k_data(
                    code=code, start_date=start_date, end_date=end_date
                )

            # Build objective data analysis report
            report = f"# {basic_info['code_name'].values[0] if not basic_info.empty else code} Data Analysis Report\n\n"
            report += "## Disclaimer\nThis report is generated based on public data and is for reference only, not investment advice. Investment decisions should be based on personal risk tolerance and research.\n\n"

            # Add industry information
            if not basic_info.empty:
                report += f"## Company Basic Information\n"
                report += f"- Stock Code: {code}\n"
                report += f"- Stock Name: {basic_info['code_name'].values[0]}\n"
                report += f"- Industry: {basic_info['industry'].values[0] if 'industry' in basic_info.columns else 'Unknown'}\n"
                report += f"- Listing Date: {basic_info['ipoDate'].values[0] if 'ipoDate' in basic_info.columns else 'Unknown'}\n\n"

            # Add fundamental analysis
            if analysis_type in ["fundamental", "comprehensive"] and not profit_data.empty:
                report += f"## Fundamental Indicators Analysis (Q{recent_quarter} {recent_year})\n\n"

                # Profitability
                report += "### Profitability Indicators\n"
                if not profit_data.empty and 'roeAvg' in profit_data.columns:
                    roe = profit_data['roeAvg'].values[0]
                    report += f"- ROE (Return on Equity): {roe}%\n"
                if not profit_data.empty and 'npMargin' in profit_data.columns:
                    npm = profit_data['npMargin'].values[0]
                    report += f"- Net Profit Margin: {npm}%\n"

                # Growth capability
                if not growth_data.empty:
                    report += "\n### Growth Capability Indicators\n"
                    if 'YOYEquity' in growth_data.columns:
                        equity_growth = growth_data['YOYEquity'].values[0]
                        report += f"- Year-over-year Equity Growth: {equity_growth}%\n"
                    if 'YOYAsset' in growth_data.columns:
                        asset_growth = growth_data['YOYAsset'].values[0]
                        report += f"- Year-over-year Asset Growth: {asset_growth}%\n"
                    if 'YOYNI' in growth_data.columns:
                        ni_growth = growth_data['YOYNI'].values[0]
                        report += f"- Year-over-year Net Income Growth: {ni_growth}%\n"

                # Debt paying ability
                if not balance_data.empty:
                    report += "\n### Debt Paying Ability Indicators\n"
                    if 'currentRatio' in balance_data.columns:
                        current_ratio = balance_data['currentRatio'].values[0]
                        report += f"- Current Ratio: {current_ratio}\n"
                    if 'assetLiabRatio' in balance_data.columns:
                        debt_ratio = balance_data['assetLiabRatio'].values[0]
                        report += f"- Asset-Liability Ratio: {debt_ratio}%\n"

            # Add technical analysis
            if analysis_type in ["technical", "comprehensive"] and not price_data.empty:
                report += "## Technical Analysis\n\n"

                # Calculate simple technical indicators
                # Assume price_data is already sorted by date
                if 'close' in price_data.columns and len(price_data) > 1:
                    latest_price = price_data['close'].iloc[-1]
                    start_price = price_data['close'].iloc[0]
                    price_change = (
                        (float(latest_price) / float(start_price)) - 1) * 100

                    report += f"- Latest Closing Price: {latest_price}\n"
                    report += f"- 6-month Price Change: {price_change:.2f}%\n"

                    # Calculate simple moving average
                    if len(price_data) >= 20:
                        ma20 = price_data['close'].astype(
                            float).tail(20).mean()
                        report += f"- 20-day Moving Average: {ma20:.2f}\n"
                        if float(latest_price) > ma20:
                            report += f"  (Current price above 20-day MA by {((float(latest_price)/ma20)-1)*100:.2f}%)\n"
                        else:
                            report += f"  (Current price below 20-day MA by {((ma20/float(latest_price))-1)*100:.2f}%)\n"

            # Add industry comparison analysis
            try:
                if not basic_info.empty and 'industry' in basic_info.columns:
                    industry = basic_info['industry'].values[0]
                    industry_stocks = active_data_source.get_stock_industry(
                        date=None)
                    if not industry_stocks.empty:
                        same_industry = industry_stocks[industry_stocks['industry'] == industry]
                        report += f"\n## Industry Comparison ({industry})\n"
                        report += f"- Number of stocks in same industry: {len(same_industry)}\n"

                        # More industry comparison data can be added here
            except Exception as e:
                logger.warning(f"Failed to get industry comparison data: {e}")

            report += "\n## Data Interpretation Suggestions\n"
            report += "- The above data is for reference only, suggest comprehensive analysis combining company announcements, industry trends and macro environment\n"
            report += "- Individual stock performance is affected by multiple factors, historical data does not represent future performance\n"
            report += "- Investment decisions should be based on personal risk tolerance and investment objectives\n"

            logger.info(f"Successfully generated analysis report for {code}")
            return report

        except Exception as e:
            logger.exception(f"Analysis generation failed for {code}: {e}")
            return f"Analysis generation failed: {e}"
