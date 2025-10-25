# A-Share Data Analysis API (MCP Version)

Forked From：https://github.com/24mlight/a-share-mcp-is-just-i-need.git

## Project Overview

This is a China A-share market data analysis API based on MCP (Model Control Protocol) architecture, designed to provide AI assistants with comprehensive access to Chinese stock market data and analysis capabilities. The project uses Baostock library as the data source, providing rich stock market data, financial statement data, index constituent information, and macroeconomic data query functionality.

### Key Features

- **Comprehensive Data**: Covers A-share historical quotes, financial statements, company information, index constituents, macroeconomic data, etc.
- **Flexible Architecture**: Based on MCP protocol, easily integrates with large language models like Claude, GPT
- **Unified Interface**: Provides standardized tool function interfaces, enabling AI assistants to process financial data like professional analysts
- **Error Handling**: Robust error handling mechanism ensures stable and reliable data queries
- **User-Friendly Format**: Returns results in Markdown table format for easy reading and analysis

## System Architecture

The project adopts a modular design with main components including:

1. **Data Source Interface Layer**: Defines abstract interface for financial data sources (`FinancialDataSource`)
2. **Baostock Implementation**: Specific data source implementation based on Baostock library (`BaostockDataSource`)
3. **Tool Function Modules**: Multiple tool collections categorized by function, such as:
   - Stock market data tools
   - Financial report tools
   - Index tools
   - Market overview tools
   - Macroeconomic tools
   - Date tools
   - Analysis tools
4. **MCP Server**: FastMCP server implementation for AI assistant integration

## Installation Guide

⚠️ **Important Note**: This project must be installed using uv, MCP does not support other package management methods

```bash
# 1. Install uv (if not already installed)
pip install uv

# 2. Clone repository (if needed)
git clone https://github.com/yourusername/a-share-mcp-api.git
cd a-share-mcp-api/src/mcp_api/

# 3. Create virtual environment
uv venv

# 4. Activate environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 5. Download dependencies
uv sync
```

## Feature Overview

### 1. Stock Market Data

- **Historical K-line Data** (`get_historical_k_data`): Get stock price, trading volume and other time series data
- **Basic Information** (`get_stock_basic_info`): Get basic stock information such as stock name, industry, listing date, etc.
- **Dividend Data** (`get_dividend_data`): Get historical dividend data for stocks
- **Adjustment Factor Data** (`get_adjust_factor_data`): Get adjustment factors needed for price adjustment

### 2. Financial Report Data

- **Profitability Data** (`get_profit_data`): ROE, net profit margin and other profitability indicators
- **Operating Capability Data** (`get_operation_data`): Asset turnover and other operating indicators
- **Growth Capability Data** (`get_growth_data`): Revenue growth rate, profit growth rate, etc.
- **Solvency Data** (`get_balance_data`): Asset-liability ratio, current ratio, etc.
- **Cash Flow Data** (`get_cash_flow_data`): Operating cash flow and other cash flow indicators
- **DuPont Analysis Data** (`get_dupont_data`): ROE decomposition data
- **Performance Express Report** (`get_performance_express_report`): Company performance express report data
- **Performance Forecast Report** (`get_forecast_report`): Company performance forecast data

### 3. Index Data

- **Industry Classification Data** (`get_stock_industry`): Get stock industry information
- **Shanghai 50 Constituents** (`get_sz50_stocks`): Get Shanghai 50 index constituents
- **CSI 300 Constituents** (`get_hs300_stocks`): Get CSI 300 index constituents
- **CSI 500 Constituents** (`get_zz500_stocks`): Get CSI 500 index constituents

### 4. Market Overview Data

- **Trading Date Information** (`get_trade_dates`): Get A-share market trading calendar
- **Stock List** (`get_all_stock`): Get all A-shares and their trading status

### 5. Macroeconomic Data

- **Benchmark Deposit Rate** (`get_deposit_rate_data`): Get central bank benchmark deposit rate data
- **Benchmark Loan Rate** (`get_loan_rate_data`): Get central bank benchmark loan rate data
- **Required Reserve Ratio** (`get_required_reserve_ratio_data`): Get required reserve ratio data
- **Monthly Money Supply** (`get_money_supply_data_month`): M0, M1, M2 monthly data
- **Annual Money Supply** (`get_money_supply_data_year`): M0, M1, M2 annual data
- **SHIBOR Rate** (`get_shibor_data`): Shanghai Interbank Offered Rate data

### 6. Date Tools

- **Current Date** (`get_current_date`): Get system current date
- **Latest Trading Date** (`get_latest_trading_date`): Get the latest A-share trading date
- **Market Analysis Timeframe** (`get_market_analysis_timeframe`): Get time range suitable for market analysis

### 7. Analysis Tools

- **Stock Analysis Report** (`get_stock_analysis`): Get data-based stock analysis report

## Usage Guide

### 1. Configure MCP in Cherry Studio

First, make sure you have installed the [Cherry Studio client](https://www.cherry-ai.com/), then configure as follows:

1. Open Cherry Studio settings
2. Add new configuration in the MCP configuration section
3. Set parameters as follows:

```
--directory
project_path/src/mcp_api
run
python
mcp_server.py
```

**Configuration Screenshot:**

![mcp configuration example](../../assets/img/mcp_config.png)

#### Using through Claude/GPT and other assistants

After configuration, you can query and analyze A-share data through AI assistants. Here are some example queries:

## Usage Effect Display

![Analysis Example 1](../../assets/img/mcp_1.png)

![Analysis Example 2](../../assets/img/mcp_2.png)

![Analysis Example 3](../../assets/img/mcp_3.png)

![Analysis Example 4](../../assets/img/mcp_4.png)

### 2. Configure through Cline

Download the cline plugin in vscode, use json configuration in the mcp configuration interface:

```json
{
  "mcpServers": {
    // other mcp configurations....
    "a-share-mcp": {
      "timeout": 60,
      "command": "uv",
      "args": [
        "--directory",
        "C:\\path\\to\\stock_agent\\src\\mcp_api",
        "run",
        "python",
        "mcp_server.py"
      ],
      "transportType": "stdio",
      "disabled": false
    }
    // other mcp configurations....
  }
}
```

Then you can start conversations

## Important Notes

1. **Data Source**: This project uses Baostock as the data source, data is for reference only and does not constitute investment advice.
2. **Real-time**: Baostock data may have delays and does not guarantee real-time data.
3. **Usage Restrictions**: Please comply with Baostock's terms of use to avoid IP restrictions due to frequent requests.
4. **Disclaimer**: Users are responsible for their own investment decisions when using this project, and the author is not responsible for any losses caused by using this project.

## Development Guide

### Extending Data Sources

To add new data sources (such as TuShare, AkShare, etc.), simply implement the `FinancialDataSource` interface:

```python
from src.mcp_api.data_source_interface import FinancialDataSource

class MyNewDataSource(FinancialDataSource):
    # Implement all abstract methods
    def get_historical_k_data(self, code, start_date, end_date, frequency, adjust_flag, fields):
        # Implementation code
        pass

    # Implement other necessary methods...
```

### Adding New Tools

To add new tool functions, you can add them to existing modules or create new modules:

```python
def register_my_new_tools(app: FastMCP, active_data_source: FinancialDataSource):
    @app.tool()
    def my_new_tool(param1: str, param2: int) -> str:
        """
        Documentation string for the new tool function, describing its functionality and parameters
        """
        # Implementation code
        return "result"
```

Then register the new tool module in `mcp_server.py`:

```python
from src.mcp_api.mcp_tools.my_new_tools import register_my_new_tools

# Other imports...

# Register tools from each module
register_my_new_tools(app, active_data_source)
# Other registrations...
```
