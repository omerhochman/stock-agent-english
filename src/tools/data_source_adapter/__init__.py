"""
Data Source Adapter Module - Provides unified data access interface

This module encapsulates interfaces for multiple data sources (including AKShare, TuShare, etc.), providing a unified data access entry point,
enabling upper-level applications to seamlessly obtain data from different data sources. Main features include:
1. Data source adaptation: Shield interface differences between different data sources
2. Data caching: Improve efficiency of frequent requests
3. Error handling and retry: Enhance reliability of data acquisition
4. Data conversion: Unify data formats from different data sources

Usage Example:
```python
# Use DataAPI to get stock price data
from src.tools.data_source_adapter import DataAPI

# Create data API instance
data_api = DataAPI()

# Get stock price data
df = data_api.get_price_data("600519", "2022-01-01", "2022-12-31")
print(df.head())

# Get financial metrics data
metrics = data_api.get_financial_metrics("600519")
print(metrics)
```
"""

from .adapter import DataSourceAdapter
from .cache import get_cached_data
from .data_api import DataAPI

__all__ = ["DataSourceAdapter", "get_cached_data", "DataAPI"]

# ----------------------------------------------------------------------------#
# Data Source Adapter API Function Documentation
# ----------------------------------------------------------------------------#

# class DataSourceAdapter:
#     """
#     Data source adapter supporting data acquisition from AKShare and TuShare

#     This class is responsible for converting and standardizing data from different data sources, handling error conditions,
#     and providing a unified interface to obtain stock prices, financial metrics and market data.

#     Main methods:
#         convert_stock_code: Convert stock code format
#         get_price_history: Get historical price data
#         get_financial_metrics: Get financial metrics data
#         get_financial_statements: Get financial statement data
#         get_market_data: Get market data

#     Example:
#         >>> adapter = DataSourceAdapter()
#         >>> df = adapter.get_price_history("600519", "2022-01-01", "2022-12-31")
#         >>> print(df.head())
#                   date    open    high     low   close     volume       amount
#         0   2022-01-04  2026.0  2078.0  2019.0  2064.0  12345678.0  2517546700.0
#         ...
#     """

#     @staticmethod
#     def convert_stock_code(symbol):
#         """
#         Convert stock code format, return AKShare and TuShare corresponding code formats

#         Args:
#             symbol (str): Stock code, can be with exchange prefix (like sh600519) or without prefix

#         Returns:
#             tuple: (akshare_code, tushare_code, exchange_prefix), containing code formats suitable for different data sources

#         Example:
#             >>> DataSourceAdapter.convert_stock_code("600519")
#             ('600519', '600519.SH', 'sh')
#             >>> DataSourceAdapter.convert_stock_code("sh600519")
#             ('sh600519', '600519.SH', 'sh')
#         """
#         pass

#     def get_price_history(self, symbol, start_date=None, end_date=None, adjust="qfq"):
#         """
#         Get historical price data, prioritize AKShare, fallback to TuShare on failure

#         Args:
#             symbol (str): Stock code
#             start_date (str, optional): Start date, format: YYYY-MM-DD
#             end_date (str, optional): End date, format: YYYY-MM-DD
#             adjust (str, optional): Adjustment type, "qfq": forward adjustment, "hfq": backward adjustment, "": no adjustment, default forward adjustment

#         Returns:
#             pd.DataFrame: DataFrame containing price data, columns include:
#                          - date: Date
#                          - open: Opening price
#                          - high: Highest price
#                          - low: Lowest price
#                          - close: Closing price
#                          - volume: Trading volume
#                          - amount: Trading amount
#                          May include other columns such as price change, turnover rate, etc.

#         Example:
#             >>> adapter = DataSourceAdapter()
#             >>> df = adapter.get_price_history("600519", "2022-01-01", "2022-01-10")
#             >>> print(df)
#                       date    open    high     low   close     volume       amount
#             0   2022-01-04  2026.0  2078.0  2019.0  2064.0  12345678.0  2517546700.0
#             1   2022-01-05  2066.0  2072.0  2022.0  2043.0  11234567.0  2303456500.0
#             ...
#         """
#         pass

#     def get_financial_metrics(self, symbol):
#         """
#         Get financial metrics data, prioritize AKShare, fallback to TuShare on failure

#         Args:
#             symbol (str): Stock code

#         Returns:
#             list: List of dictionaries containing financial metrics (usually one element), keys include:
#                  - return_on_equity: Return on equity
#                  - net_margin: Net profit margin
#                  - operating_margin: Operating profit margin
#                  - revenue_growth: Revenue growth rate
#                  - earnings_growth: Earnings growth rate
#                  - book_value_growth: Book value growth rate
#                  - current_ratio: Current ratio
#                  - debt_to_equity: Debt-to-equity ratio
#                  - free_cash_flow_per_share: Free cash flow per share
#                  - earnings_per_share: Earnings per share
#                  - pe_ratio: Price-to-earnings ratio
#                  - price_to_book: Price-to-book ratio
#                  - price_to_sales: Price-to-sales ratio

#         Example:
#             >>> adapter = DataSourceAdapter()
#             >>> metrics = adapter.get_financial_metrics("600519")
#             >>> print(metrics[0])
#             {'return_on_equity': 0.325, 'net_margin': 0.518, 'operating_margin': 0.652, ...}
#         """
#         pass

#     def get_financial_statements(self, symbol):
#         """
#         Get financial statements data, prioritize AKShare, fallback to TuShare on failure

#         Args:
#             symbol (str): Stock code

#         Returns:
#             list: List of dictionaries containing financial statements data, usually includes latest two periods, keys include:
#                  - net_income: Net income
#                  - operating_revenue: Operating revenue
#                  - operating_profit: Operating profit
#                  - working_capital: Working capital
#                  - depreciation_and_amortization: Depreciation and amortization
#                  - capital_expenditure: Capital expenditure
#                  - free_cash_flow: Free cash flow

#         Example:
#             >>> adapter = DataSourceAdapter()
#             >>> statements = adapter.get_financial_statements("600519")
#             >>> print(statements[0])  # Latest financial statements data
#             {'net_income': 5000000000, 'operating_revenue': 20000000000, ...}
#             >>> print(statements[1])  # Previous period financial statements data
#             {'net_income': 4500000000, 'operating_revenue': 18000000000, ...}
#         """
#         pass

#     def get_market_data(self, symbol):
#         """
#         Get market data, prioritize AKShare, fallback to TuShare on failure

#         Args:
#             symbol (str): Stock code

#         Returns:
#             dict: Dictionary containing market data, keys include:
#                  - market_cap: Market capitalization
#                  - volume: Trading volume
#                  - average_volume: Average volume (usually 30-day)
#                  - fifty_two_week_high: 52-week high price
#                  - fifty_two_week_low: 52-week low price

#         Example:
#             >>> adapter = DataSourceAdapter()
#             >>> market_data = adapter.get_market_data("600519")
#             >>> print(market_data)
#             {'market_cap': 2500000000000, 'volume': 12345678, 'average_volume': 15482630, ...}
#         """
#         pass


# ----------------------------------------------------------------------------#
# Data Cache Function Documentation
# ----------------------------------------------------------------------------#

# def get_cached_data(key, fetch_func, *args, ttl_days=1, **kwargs):
#     """
#     Get data from cache, if cache is expired or doesn't exist, call fetch_func to get data

#     This function implements a simple data caching mechanism that can reduce repeated requests to data sources,
#     and provides cache expiration and data conversion functionality.

#     Args:
#         key (str): Cache key, used to identify data
#         fetch_func (callable): Function to fetch data, called when cache is unavailable
#         ttl_days (float, optional): Cache validity period (days), default 1 day
#         *args, **kwargs: Parameters passed to fetch_func

#     Returns:
#         Any type: Data returned by fetch_func, could be DataFrame, dict, list, etc.

#     Example:
#         >>> def fetch_stock_data(symbol, start_date, end_date):
#         ...     # Function to fetch stock data
#         ...     return pd.DataFrame(...)
#         >>>
#         >>> # Use cache to get data, cache validity period is 7 days
#         >>> data = get_cached_data(
#         ...     f"stock_data_600519_2022",
#         ...     fetch_stock_data,
#         ...     "600519", "2022-01-01", "2022-12-31",
#         ...     ttl_days=7
#         ... )
#     """
#     pass


# ----------------------------------------------------------------------------#
# Unified Data API Class Documentation
# ----------------------------------------------------------------------------#

# class DataAPI:
#     """
#     Unified data API interface, encapsulates internal data source adapter implementation

#     This class is the main interface for external calls, providing simple methods to get stock prices, financial metrics,
#     financial statements and market data, internally uses DataSourceAdapter
#     to handle data source switching and error handling.

#     Main methods:
#         get_price_data: Get stock price data
#         get_financial_metrics: Get financial metrics data
#         get_financial_statements: Get financial statements data
#         get_market_data: Get market data

#     Example:
#         >>> data_api = DataAPI()
#         >>> # Get stock price data
#         >>> df = data_api.get_price_data("600519", "2022-01-01", "2022-12-31")
#         >>> print(df.head())
#     """

#     def get_price_data(self, ticker, start_date=None, end_date=None):
#         """
#         Get stock price data

#         Args:
#             ticker (str): Stock code
#             start_date (str, optional): Start date, format: YYYY-MM-DD
#             end_date (str, optional): End date, format: YYYY-MM-DD

#         Returns:
#             pd.DataFrame: DataFrame containing price data, columns include:
#                          - date: Date
#                          - open: Opening price
#                          - high: Highest price
#                          - low: Lowest price
#                          - close: Closing price
#                          - volume: Trading volume
#                          - amount: Trading amount
#                          May include other columns such as price change, turnover rate, etc.

#         Example:
#             >>> data_api = DataAPI()
#             >>> df = data_api.get_price_data("600519", "2022-01-01", "2022-01-10")
#             >>> print(df.head())
#                       date    open    high     low   close     volume       amount
#             0   2022-01-04  2026.0  2078.0  2019.0  2064.0  12345678.0  2517546700.0
#             ...
#         """
#         pass

#     def get_financial_metrics(self, ticker):
#         """
#         Get financial metrics data

#         Args:
#             ticker (str): Stock code

#         Returns:
#             list: List of dictionaries containing financial metrics (usually one element), keys include:
#                  - return_on_equity: Return on equity
#                  - net_margin: Net profit margin
#                  - operating_margin: Operating profit margin
#                  - and other financial metrics

#         Example:
#             >>> data_api = DataAPI()
#             >>> metrics = data_api.get_financial_metrics("600519")
#             >>> print(metrics[0])
#             {'return_on_equity': 0.325, 'net_margin': 0.518, ...}
#         """
#         pass

#     def get_financial_statements(self, ticker):
#         """
#         Get financial statements data

#         Args:
#             ticker (str): Stock code

#         Returns:
#             list: List of dictionaries containing financial statements data, usually includes latest two periods

#         Example:
#             >>> data_api = DataAPI()
#             >>> statements = data_api.get_financial_statements("600519")
#             >>> print(statements[0])  # Latest financial statements data
#             {'net_income': 5000000000, 'operating_revenue': 20000000000, ...}
#         """
#         pass

#     def get_market_data(self, ticker):
#         """
#         Get market data

#         Args:
#             ticker (str): Stock code

#         Returns:
#             dict: Dictionary containing market data, including market cap, volume, 52-week high/low prices, etc.

#         Example:
#             >>> data_api = DataAPI()
#             >>> market_data = data_api.get_market_data("600519")
#             >>> print(market_data)
#             {'market_cap': 2500000000000, 'volume': 12345678, ...}
#         """
#         pass


# ----------------------------------------------------------------------------#
# Complete Usage Examples
# ----------------------------------------------------------------------------#

"""
The following are complete usage examples showing how to use the data_source_adapter module to fetch and process various types of data:

1. Basic stock price data fetching and processing

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.data_source_adapter import DataAPI

# Create data API instance
data_api = DataAPI()

# Get stock price data
ticker = "600519"  # Kweichow Moutai
start_date = "2022-01-01"
end_date = "2022-12-31"
price_data = data_api.get_price_data(ticker, start_date, end_date)

# Data preprocessing
price_data['date'] = pd.to_datetime(price_data['date'])
price_data.set_index('date', inplace=True)

# Calculate simple technical indicators
price_data['ma20'] = price_data['close'].rolling(window=20).mean()  # 20-day moving average
price_data['ma60'] = price_data['close'].rolling(window=60).mean()  # 60-day moving average
price_data['daily_return'] = price_data['close'].pct_change()  # Daily return

# Visualize stock price and moving averages
plt.figure(figsize=(12, 6))
plt.plot(price_data.index, price_data['close'], label='Closing Price')
plt.plot(price_data.index, price_data['ma20'], label='20-day MA')
plt.plot(price_data.index, price_data['ma60'], label='60-day MA')
plt.title(f"{ticker} Stock Price Chart")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print basic statistics
print(f"\n{ticker} Statistics:")
print(f"Period High: {price_data['high'].max():.2f}")
print(f"Period Low: {price_data['low'].min():.2f}")
print(f"Average Volume: {price_data['volume'].mean():.0f}")
print(f"Average Daily Return: {price_data['daily_return'].mean()*100:.4f}%")
print(f"Daily Return Std: {price_data['daily_return'].std()*100:.4f}%")
```

2. Multi-stock comparison analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.data_source_adapter import DataAPI

# Create data API instance
data_api = DataAPI()

# Get price data for multiple stocks
tickers = ["600519", "000858", "600036", "601318"]  # Moutai, Wuliangye, China Merchants Bank, Ping An Insurance
start_date = "2022-01-01"
end_date = "2022-12-31"

# Prepare DataFrame to store returns
returns_df = pd.DataFrame()

# Get data for each stock and calculate cumulative returns
for ticker in tickers:
    price_data = data_api.get_price_data(ticker, start_date, end_date)
    
    # Ensure date column is datetime type and set as index
    price_data['date'] = pd.to_datetime(price_data['date'])
    price_data.set_index('date', inplace=True)
    
    # Calculate daily returns and cumulative returns
    daily_returns = price_data['close'].pct_change().fillna(0)
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    
    # Add to DataFrame
    returns_df[ticker] = cumulative_returns

# Visualize cumulative returns comparison
plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(returns_df.index, returns_df[ticker] * 100, label=ticker)
plt.title("Multi-stock Cumulative Returns Comparison")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate correlation matrix
correlation = returns_df.pct_change().corr()
print("\nStock Returns Correlation Matrix:")
print(correlation)
```

3. Stock financial metrics analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.data_source_adapter import DataAPI

# Create data API instance
data_api = DataAPI()

# Get financial metrics for multiple stocks
tickers = ["600519", "000858", "600036", "601318"]
names = ["Kweichow Moutai", "Wuliangye", "China Merchants Bank", "Ping An Insurance"]

# Collect financial metrics
metrics_list = []
for ticker in tickers:
    metrics = data_api.get_financial_metrics(ticker)
    if metrics and len(metrics) > 0:
        metrics[0]['ticker'] = ticker
        metrics_list.append(metrics[0])

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics_list)

# Set index to stock code
metrics_df.set_index('ticker', inplace=True)

# Select metrics to compare
key_metrics = ['return_on_equity', 'net_margin', 'debt_to_equity', 'pe_ratio', 'price_to_book']
metrics_df = metrics_df[key_metrics]

# Rename columns for display
metrics_df.columns = ['Return on Equity', 'Net Profit Margin', 'Debt-to-Equity Ratio', 'P/E Ratio', 'Price-to-Book Ratio']

# Replace codes with stock names
metrics_df.index = names

# Visualize comparison - using bar charts
plt.figure(figsize=(14, 10))

for i, metric in enumerate(metrics_df.columns):
    plt.subplot(3, 2, i+1)
    plt.bar(metrics_df.index, metrics_df[metric], color='steelblue')
    plt.title(metric)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Print financial metrics table
print("\nFinancial Metrics Comparison:")
print(metrics_df)
```

4. Financial statements data processing

```python
import pandas as pd
from src.tools.data_source_adapter import DataAPI

# Create data API instance
data_api = DataAPI()

# Get financial statements data
ticker = "600519"  # Kweichow Moutai
statements = data_api.get_financial_statements(ticker)

# Convert to DataFrame for comparison
current_period = pd.Series(statements[0], name="Current Period")
previous_period = pd.Series(statements[1], name="Previous Period")

comparison_df = pd.DataFrame([current_period, previous_period])

# Calculate year-over-year changes
change = pd.Series({
    key: (statements[0][key] - statements[1][key]) / statements[1][key] * 100 if statements[1][key] != 0 else float('inf')
    for key in statements[0].keys()
}, name="YoY Change (%)")

comparison_df = comparison_df.append(change)

# Beautify display
pd.set_option('display.float_format', '{:.2f}'.format)
print(f"\n{ticker} Financial Statements YoY Comparison:")
print(comparison_df.T)  # Transpose for better display

# Calculate key financial ratios
print("\nKey Financial Ratios:")
if statements[0]["operating_revenue"] > 0:
    profit_margin = statements[0]["net_income"] / statements[0]["operating_revenue"] * 100
    print(f"Net Profit Margin: {profit_margin:.2f}%")

if statements[0]["capital_expenditure"] > 0:
    capex_to_revenue = statements[0]["capital_expenditure"] / statements[0]["operating_revenue"] * 100
    print(f"Capital Expenditure to Revenue Ratio: {capex_to_revenue:.2f}%")

if statements[0]["operating_revenue"] > 0:
    fcf_to_revenue = statements[0]["free_cash_flow"] / statements[0]["operating_revenue"] * 100
    print(f"Free Cash Flow to Revenue Ratio: {fcf_to_revenue:.2f}%")
```

5. Market data analysis and valuation

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.data_source_adapter import DataAPI

# Create data API instance
data_api = DataAPI()

# Get market data and financial metrics for multiple stocks
tickers = ["600519", "000858", "600036", "601318", "000651", "600276"]
names = ["Kweichow Moutai", "Wuliangye", "China Merchants Bank", "Ping An Insurance", "Gree Electric", "Jiangsu Hengrui Medicine"]

# Collect data
market_data_list = []
for i, ticker in enumerate(tickers):
    market_data = data_api.get_market_data(ticker)
    financial_metrics = data_api.get_financial_metrics(ticker)
    
    if market_data and financial_metrics and len(financial_metrics) > 0:
        # Combine data
        combined_data = {
            'ticker': ticker,
            'name': names[i],
            'market_cap': market_data.get('market_cap', 0) / 100000000,  # Convert to 100 million yuan
            'pe_ratio': financial_metrics[0].get('pe_ratio', 0),
            'price_to_book': financial_metrics[0].get('price_to_book', 0),
            'return_on_equity': financial_metrics[0].get('return_on_equity', 0) * 100  # Convert to percentage
        }
        market_data_list.append(combined_data)

# Convert to DataFrame
market_df = pd.DataFrame(market_data_list)

# Set index
market_df.set_index('name', inplace=True)

# Visualize relationship between market cap and PE - bubble chart
plt.figure(figsize=(12, 8))
plt.scatter(
    market_df['pe_ratio'], 
    market_df['return_on_equity'], 
    s=market_df['market_cap'] * 5,  # Bubble size determined by market cap
    alpha=0.7
)

# Add labels
for i, txt in enumerate(market_df.index):
    plt.annotate(
        txt, 
        (market_df['pe_ratio'].iloc[i], market_df['return_on_equity'].iloc[i]),
        xytext=(7, 7),
        textcoords='offset points'
    )

plt.title('Stock Valuation vs Return Rate Comparison')
plt.xlabel('Price-to-Earnings Ratio (PE)')
plt.ylabel('Return on Equity (%)')
plt.grid(True, alpha=0.3)
plt.show()

# Print market data table
print("\nMarket Data and Valuation Metrics:")
print(market_df)
```
"""
