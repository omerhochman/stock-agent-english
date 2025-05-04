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

# --- 日志设置 ---
# 从utils调用设置函数
# 您可以在此控制默认级别（例如，使用logging.DEBUG获取更详细的日志）
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 依赖注入 ---
# 实例化数据源 - 如果需要，以后可以轻松替换
active_data_source: FinancialDataSource = BaostockDataSource()

# --- 获取当前日期用于系统提示 ---
current_date = datetime.now().strftime("%Y-%m-%d")

# --- FastMCP应用初始化 ---
app = FastMCP(
    server_name="a_share_data_provider",
    description=f"""今天是{current_date}。提供中国A股市场数据分析工具。此服务提供客观数据分析，用户需自行做出投资决策。数据分析基于公开市场信息，不构成投资建议，仅供参考。

⚠️ 重要说明:
1. 请始终使用 get_current_date() 或 get_latest_trading_date() 工具获取实际当前日期，不要依赖训练数据中的日期认知
2. 当分析"最近"或"近期"市场情况时，必须首先调用 get_market_analysis_timeframe() 工具确定实际的分析时间范围
3. 任何涉及日期的分析必须基于工具返回的实际数据，不得使用过时或假设的日期
""",
    # 如果需要，指定安装依赖项（例如，当使用`mcp install`时）
    # dependencies=["baostock", "pandas"]
)

# --- 注册各模块的工具 ---
register_stock_market_tools(app, active_data_source)
register_financial_report_tools(app, active_data_source)
register_index_tools(app, active_data_source)
register_market_overview_tools(app, active_data_source)
register_macroeconomic_tools(app, active_data_source)
register_date_utils_tools(app, active_data_source)
register_analysis_tools(app, active_data_source)

# --- 主执行块 ---
if __name__ == "__main__":
    logger.info(
        f"通过stdio启动A股MCP服务器... 今天是 {current_date}")
    # 使用stdio传输运行服务器，适用于Claude Desktop等MCP主机
    app.run(transport='stdio')