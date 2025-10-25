import logging
from datetime import datetime
from mcp.server.fastmcp import FastMCP

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.mcp_api.data_source_interface import FinancialDataSource
from src.mcp_api.baostock_data_source import BaostockDataSource
from src.mcp_api.mcp_utils import setup_logging
from src.mcp_api.mcp_tools.stock_market import register_stock_market_tools
from src.mcp_api.mcp_tools.financial_reports import register_financial_report_tools
from src.mcp_api.mcp_tools.indices import register_index_tools
from src.mcp_api.mcp_tools.market_overview import register_market_overview_tools
from src.mcp_api.mcp_tools.macroeconomic import register_macroeconomic_tools
from src.mcp_api.mcp_tools.date_utils import register_date_utils_tools
from src.mcp_api.mcp_tools.analysis import register_analysis_tools

# --- Logging Setup ---
# Call setup function from utils
# You can control the default level here (e.g., use logging.DEBUG for more detailed logs)
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Dependency Injection ---
# Instantiate data source - can be easily replaced later if needed
active_data_source: FinancialDataSource = BaostockDataSource()

# --- Get current date for system prompt ---
current_date = datetime.now().strftime("%Y-%m-%d")

# --- FastMCP Application Initialization ---
app = FastMCP(
    server_name="a_share_data_provider",
    description=f"""Today is {current_date}. Provides Chinese A-share market data analysis tools. This service provides objective data analysis, users need to make their own investment decisions. Data analysis is based on public market information and does not constitute investment advice, for reference only.

⚠️ Important Notes:
1. Always use get_current_date() or get_latest_trading_date() tools to get actual current date, do not rely on date cognition in training data
2. When analyzing "recent" or "recent period" market conditions, must first call get_market_analysis_timeframe() tool to determine actual analysis time range
3. Any date-related analysis must be based on actual data returned by tools, cannot use outdated or assumed dates
""",
    # If needed, specify installation dependencies (e.g., when using `mcp install`)
    # dependencies=["baostock", "pandas"]
)

# --- Register tools for each module ---
register_stock_market_tools(app, active_data_source)
register_financial_report_tools(app, active_data_source)
register_index_tools(app, active_data_source)
register_market_overview_tools(app, active_data_source)
register_macroeconomic_tools(app, active_data_source)
register_date_utils_tools(app, active_data_source)
register_analysis_tools(app, active_data_source)

# --- Main execution block ---
if __name__ == "__main__":
    logger.info(
        f"Starting A-share MCP server via stdio... Today is {current_date}")
    # Run server using stdio transport, suitable for MCP hosts like Claude Desktop
    app.run(transport='stdio')