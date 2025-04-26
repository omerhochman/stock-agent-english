"""
数据源适配器模块 - 提供统一的数据获取接口

此模块封装了多个数据源的接口（包括AKShare、TuShare等），提供统一的数据获取入口，
使上层应用能够无缝地从不同数据源获取数据。主要功能包括：
1. 数据源适配：屏蔽不同数据源的接口差异
2. 数据缓存：提高频繁请求的效率
3. 错误处理和重试：增强数据获取的可靠性
4. 数据转换：统一不同数据源的数据格式

使用示例:
```python
# 使用DataAPI获取股票价格数据
from src.tools.data_source_adapter import DataAPI

# 创建数据API实例
data_api = DataAPI()

# 获取股票价格数据
df = data_api.get_price_data("600519", "2022-01-01", "2022-12-31")
print(df.head())

# 获取财务指标数据
metrics = data_api.get_financial_metrics("600519")
print(metrics)
```
"""

from .adapter import DataSourceAdapter
from .cache import get_cached_data
from .data_api import DataAPI

__all__ = ['DataSourceAdapter', 'get_cached_data', 'DataAPI']

#----------------------------------------------------------------------------#
# 数据源适配器API函数说明
#----------------------------------------------------------------------------#

class DataSourceAdapter:
    """
    数据源适配器，支持从AKShare和TuShare获取数据
    
    此类负责转换和标准化来自不同数据源的数据，处理错误情况，
    并提供统一的接口来获取股票价格、财务指标和市场数据。
    
    主要方法:
        convert_stock_code: 转换股票代码格式
        get_price_history: 获取历史价格数据
        get_financial_metrics: 获取财务指标数据
        get_financial_statements: 获取财务报表数据
        get_market_data: 获取市场数据
    
    示例:
        >>> adapter = DataSourceAdapter()
        >>> df = adapter.get_price_history("600519", "2022-01-01", "2022-12-31")
        >>> print(df.head())
                  date    open    high     low   close     volume       amount
        0   2022-01-04  2026.0  2078.0  2019.0  2064.0  12345678.0  2517546700.0
        ...
    """
    
    @staticmethod
    def convert_stock_code(symbol):
        """
        转换股票代码格式，返回AKShare和TuShare对应的代码格式
        
        参数:
            symbol (str): 股票代码，可以是带交易所前缀(如sh600519)或不带前缀的代码
            
        返回:
            tuple: (akshare_code, tushare_code, exchange_prefix)，包含适用于不同数据源的代码格式
            
        示例:
            >>> DataSourceAdapter.convert_stock_code("600519")
            ('600519', '600519.SH', 'sh')
            >>> DataSourceAdapter.convert_stock_code("sh600519")
            ('sh600519', '600519.SH', 'sh')
        """
        pass
    
    def get_price_history(self, symbol, start_date=None, end_date=None, adjust="qfq"):
        """
        获取历史价格数据，优先使用AKShare，失败时切换到TuShare
        
        参数:
            symbol (str): 股票代码
            start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
            end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
            adjust (str, 可选): 复权类型，"qfq": 前复权, "hfq": 后复权, "": 不复权，默认前复权
            
        返回:
            pd.DataFrame: 包含价格数据的DataFrame，列包括:
                         - date: 日期
                         - open: 开盘价
                         - high: 最高价
                         - low: 最低价
                         - close: 收盘价
                         - volume: 成交量
                         - amount: 成交额
                         可能还有其他列，如涨跌幅、换手率等
            
        示例:
            >>> adapter = DataSourceAdapter()
            >>> df = adapter.get_price_history("600519", "2022-01-01", "2022-01-10")
            >>> print(df)
                      date    open    high     low   close     volume       amount
            0   2022-01-04  2026.0  2078.0  2019.0  2064.0  12345678.0  2517546700.0
            1   2022-01-05  2066.0  2072.0  2022.0  2043.0  11234567.0  2303456500.0
            ...
        """
        pass
    
    def get_financial_metrics(self, symbol):
        """
        获取财务指标数据，优先使用AKShare，失败时切换到TuShare
        
        参数:
            symbol (str): 股票代码
            
        返回:
            list: 包含财务指标的字典列表(通常只有一个元素)，键包括:
                 - return_on_equity: 净资产收益率
                 - net_margin: 销售净利率
                 - operating_margin: 营业利润率
                 - revenue_growth: 收入增长率
                 - earnings_growth: 利润增长率
                 - book_value_growth: 净资产增长率
                 - current_ratio: 流动比率
                 - debt_to_equity: 资产负债率
                 - free_cash_flow_per_share: 每股自由现金流
                 - earnings_per_share: 每股收益
                 - pe_ratio: 市盈率
                 - price_to_book: 市净率
                 - price_to_sales: 市销率
            
        示例:
            >>> adapter = DataSourceAdapter()
            >>> metrics = adapter.get_financial_metrics("600519")
            >>> print(metrics[0])
            {'return_on_equity': 0.325, 'net_margin': 0.518, 'operating_margin': 0.652, ...}
        """
        pass
    
    def get_financial_statements(self, symbol):
        """
        获取财务报表数据，优先使用AKShare，失败时切换到TuShare
        
        参数:
            symbol (str): 股票代码
            
        返回:
            list: 包含财务报表数据的字典列表，通常包含最新两期数据，键包括:
                 - net_income: 净利润
                 - operating_revenue: 营业收入
                 - operating_profit: 营业利润
                 - working_capital: 营运资金
                 - depreciation_and_amortization: 折旧和摊销
                 - capital_expenditure: 资本支出
                 - free_cash_flow: 自由现金流
            
        示例:
            >>> adapter = DataSourceAdapter()
            >>> statements = adapter.get_financial_statements("600519")
            >>> print(statements[0])  # 最新财务报表数据
            {'net_income': 5000000000, 'operating_revenue': 20000000000, ...}
            >>> print(statements[1])  # 上一期财务报表数据
            {'net_income': 4500000000, 'operating_revenue': 18000000000, ...}
        """
        pass
    
    def get_market_data(self, symbol):
        """
        获取市场数据，优先使用AKShare，失败时切换到TuShare
        
        参数:
            symbol (str): 股票代码
            
        返回:
            dict: 包含市场数据的字典，键包括:
                 - market_cap: 市值
                 - volume: 成交量
                 - average_volume: 平均成交量（通常30日）
                 - fifty_two_week_high: 52周最高价
                 - fifty_two_week_low: 52周最低价
            
        示例:
            >>> adapter = DataSourceAdapter()
            >>> market_data = adapter.get_market_data("600519")
            >>> print(market_data)
            {'market_cap': 2500000000000, 'volume': 12345678, 'average_volume': 15482630, ...}
        """
        pass


#----------------------------------------------------------------------------#
# 数据缓存函数说明
#----------------------------------------------------------------------------#

def get_cached_data(key, fetch_func, *args, ttl_days=1, **kwargs):
    """
    从缓存获取数据，如果缓存过期或不存在则调用fetch_func获取
    
    此函数实现了一个简单的数据缓存机制，能够减少对数据源的重复请求，
    并提供了缓存过期和数据转换功能。
    
    参数:
        key (str): 缓存键，用于标识数据
        fetch_func (callable): 获取数据的函数，当缓存不可用时会调用此函数
        ttl_days (float, 可选): 缓存有效期（天数），默认1天
        *args, **kwargs: 传递给fetch_func的参数
    
    返回:
        任意类型: fetch_func返回的数据，可能是DataFrame、字典、列表等
            
    示例:
        >>> def fetch_stock_data(symbol, start_date, end_date):
        ...     # 获取股票数据的函数
        ...     return pd.DataFrame(...)
        >>> 
        >>> # 使用缓存获取数据，缓存有效期为7天
        >>> data = get_cached_data(
        ...     f"stock_data_600519_2022",
        ...     fetch_stock_data,
        ...     "600519", "2022-01-01", "2022-12-31",
        ...     ttl_days=7
        ... )
    """
    pass


#----------------------------------------------------------------------------#
# 统一数据API类说明
#----------------------------------------------------------------------------#

class DataAPI:
    """
    统一的数据API接口，封装内部数据源适配器实现
    
    此类是供外部调用的主要接口，提供了获取股票价格、财务指标、
    财务报表和市场数据的简洁方法，内部使用DataSourceAdapter
    处理数据源切换和错误处理。
    
    主要方法:
        get_price_data: 获取股票价格数据
        get_financial_metrics: 获取财务指标数据
        get_financial_statements: 获取财务报表数据
        get_market_data: 获取市场数据
    
    示例:
        >>> data_api = DataAPI()
        >>> # 获取股票价格数据
        >>> df = data_api.get_price_data("600519", "2022-01-01", "2022-12-31")
        >>> print(df.head())
    """
    
    def get_price_data(self, ticker, start_date=None, end_date=None):
        """
        获取股票价格数据
        
        参数:
            ticker (str): 股票代码
            start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
            end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
            
        返回:
            pd.DataFrame: 包含价格数据的DataFrame，列包括:
                         - date: 日期
                         - open: 开盘价
                         - high: 最高价
                         - low: 最低价
                         - close: 收盘价
                         - volume: 成交量
                         - amount: 成交额
                         可能还有其他列，如涨跌幅、换手率等
            
        示例:
            >>> data_api = DataAPI()
            >>> df = data_api.get_price_data("600519", "2022-01-01", "2022-01-10")
            >>> print(df.head())
                      date    open    high     low   close     volume       amount
            0   2022-01-04  2026.0  2078.0  2019.0  2064.0  12345678.0  2517546700.0
            ...
        """
        pass
    
    def get_financial_metrics(self, ticker):
        """
        获取财务指标数据
        
        参数:
            ticker (str): 股票代码
            
        返回:
            list: 包含财务指标的字典列表(通常只有一个元素)，键包括:
                 - return_on_equity: 净资产收益率
                 - net_margin: 销售净利率
                 - operating_margin: 营业利润率
                 等多种财务指标
            
        示例:
            >>> data_api = DataAPI()
            >>> metrics = data_api.get_financial_metrics("600519")
            >>> print(metrics[0])
            {'return_on_equity': 0.325, 'net_margin': 0.518, ...}
        """
        pass
    
    def get_financial_statements(self, ticker):
        """
        获取财务报表数据
        
        参数:
            ticker (str): 股票代码
            
        返回:
            list: 包含财务报表数据的字典列表，通常包含最新两期数据
            
        示例:
            >>> data_api = DataAPI()
            >>> statements = data_api.get_financial_statements("600519")
            >>> print(statements[0])  # 最新财务报表数据
            {'net_income': 5000000000, 'operating_revenue': 20000000000, ...}
        """
        pass
    
    def get_market_data(self, ticker):
        """
        获取市场数据
        
        参数:
            ticker (str): 股票代码
            
        返回:
            dict: 包含市场数据的字典，包括市值、成交量、52周最高/最低价等
            
        示例:
            >>> data_api = DataAPI()
            >>> market_data = data_api.get_market_data("600519")
            >>> print(market_data)
            {'market_cap': 2500000000000, 'volume': 12345678, ...}
        """
        pass


#----------------------------------------------------------------------------#
# 完整使用示例
#----------------------------------------------------------------------------#

"""
以下是完整的使用示例，展示如何使用data_source_adapter模块获取和处理各类数据:

1. 基本股票价格数据获取和处理

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.data_source_adapter import DataAPI

# 创建数据API实例
data_api = DataAPI()

# 获取股票价格数据
ticker = "600519"  # 贵州茅台
start_date = "2022-01-01"
end_date = "2022-12-31"
price_data = data_api.get_price_data(ticker, start_date, end_date)

# 数据预处理
price_data['date'] = pd.to_datetime(price_data['date'])
price_data.set_index('date', inplace=True)

# 计算简单技术指标
price_data['ma20'] = price_data['close'].rolling(window=20).mean()  # 20日均线
price_data['ma60'] = price_data['close'].rolling(window=60).mean()  # 60日均线
price_data['daily_return'] = price_data['close'].pct_change()  # 日收益率

# 可视化股票价格和均线
plt.figure(figsize=(12, 6))
plt.plot(price_data.index, price_data['close'], label='收盘价')
plt.plot(price_data.index, price_data['ma20'], label='20日均线')
plt.plot(price_data.index, price_data['ma60'], label='60日均线')
plt.title(f"{ticker} 股价走势图")
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 打印基本统计信息
print(f"\n{ticker} 统计数据:")
print(f"期间最高价: {price_data['high'].max():.2f}")
print(f"期间最低价: {price_data['low'].min():.2f}")
print(f"平均成交量: {price_data['volume'].mean():.0f}")
print(f"平均日收益率: {price_data['daily_return'].mean()*100:.4f}%")
print(f"日收益率标准差: {price_data['daily_return'].std()*100:.4f}%")
```

2. 多股票对比分析

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.data_source_adapter import DataAPI

# 创建数据API实例
data_api = DataAPI()

# 获取多只股票的价格数据
tickers = ["600519", "000858", "600036", "601318"]  # 茅台、五粮液、招商银行、平安保险
start_date = "2022-01-01"
end_date = "2022-12-31"

# 准备数据框存储收益率
returns_df = pd.DataFrame()

# 获取每只股票的数据并计算累积收益率
for ticker in tickers:
    price_data = data_api.get_price_data(ticker, start_date, end_date)
    
    # 确保日期列为日期类型并设为索引
    price_data['date'] = pd.to_datetime(price_data['date'])
    price_data.set_index('date', inplace=True)
    
    # 计算日收益率和累积收益率
    daily_returns = price_data['close'].pct_change().fillna(0)
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    
    # 添加到数据框
    returns_df[ticker] = cumulative_returns

# 可视化累积收益率对比
plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(returns_df.index, returns_df[ticker] * 100, label=ticker)
plt.title("多股票累积收益率对比")
plt.xlabel("日期")
plt.ylabel("累积收益率(%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 计算相关系数矩阵
correlation = returns_df.pct_change().corr()
print("\n股票收益率相关系数矩阵:")
print(correlation)
```

3. 股票财务指标分析

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.data_source_adapter import DataAPI

# 创建数据API实例
data_api = DataAPI()

# 获取多只股票的财务指标
tickers = ["600519", "000858", "600036", "601318"]
names = ["贵州茅台", "五粮液", "招商银行", "中国平安"]

# 收集财务指标
metrics_list = []
for ticker in tickers:
    metrics = data_api.get_financial_metrics(ticker)
    if metrics and len(metrics) > 0:
        metrics[0]['ticker'] = ticker
        metrics_list.append(metrics[0])

# 转换为DataFrame
metrics_df = pd.DataFrame(metrics_list)

# 设置索引为股票代码
metrics_df.set_index('ticker', inplace=True)

# 选择要对比的指标
key_metrics = ['return_on_equity', 'net_margin', 'debt_to_equity', 'pe_ratio', 'price_to_book']
metrics_df = metrics_df[key_metrics]

# 重命名列以便显示
metrics_df.columns = ['净资产收益率', '销售净利率', '资产负债率', '市盈率', '市净率']

# 使用股票名称替换代码
metrics_df.index = names

# 可视化对比 - 使用条形图
plt.figure(figsize=(14, 10))

for i, metric in enumerate(metrics_df.columns):
    plt.subplot(3, 2, i+1)
    plt.bar(metrics_df.index, metrics_df[metric], color='steelblue')
    plt.title(metric)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# 打印财务指标表格
print("\n财务指标对比:")
print(metrics_df)
```

4. 财务报表数据处理

```python
import pandas as pd
from src.tools.data_source_adapter import DataAPI

# 创建数据API实例
data_api = DataAPI()

# 获取财务报表数据
ticker = "600519"  # 贵州茅台
statements = data_api.get_financial_statements(ticker)

# 转换为DataFrame以便比较
current_period = pd.Series(statements[0], name="当期")
previous_period = pd.Series(statements[1], name="上期")

comparison_df = pd.DataFrame([current_period, previous_period])

# 计算同比变化
change = pd.Series({
    key: (statements[0][key] - statements[1][key]) / statements[1][key] * 100 if statements[1][key] != 0 else float('inf')
    for key in statements[0].keys()
}, name="同比变化(%)")

comparison_df = comparison_df.append(change)

# 美化显示
pd.set_option('display.float_format', '{:.2f}'.format)
print(f"\n{ticker} 财务报表同比对比:")
print(comparison_df.T)  # 转置以便更好地显示

# 计算关键财务比率
print("\n关键财务比率:")
if statements[0]["operating_revenue"] > 0:
    profit_margin = statements[0]["net_income"] / statements[0]["operating_revenue"] * 100
    print(f"净利润率: {profit_margin:.2f}%")

if statements[0]["capital_expenditure"] > 0:
    capex_to_revenue = statements[0]["capital_expenditure"] / statements[0]["operating_revenue"] * 100
    print(f"资本支出占收入比例: {capex_to_revenue:.2f}%")

if statements[0]["operating_revenue"] > 0:
    fcf_to_revenue = statements[0]["free_cash_flow"] / statements[0]["operating_revenue"] * 100
    print(f"自由现金流占收入比例: {fcf_to_revenue:.2f}%")
```

5. 市场数据分析与估值

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.data_source_adapter import DataAPI

# 创建数据API实例
data_api = DataAPI()

# 获取多只股票的市场数据和财务指标
tickers = ["600519", "000858", "600036", "601318", "000651", "600276"]
names = ["贵州茅台", "五粮液", "招商银行", "中国平安", "格力电器", "恒瑞医药"]

# 收集数据
market_data_list = []
for i, ticker in enumerate(tickers):
    market_data = data_api.get_market_data(ticker)
    financial_metrics = data_api.get_financial_metrics(ticker)
    
    if market_data and financial_metrics and len(financial_metrics) > 0:
        # 组合数据
        combined_data = {
            'ticker': ticker,
            'name': names[i],
            'market_cap': market_data.get('market_cap', 0) / 100000000,  # 转换为亿元
            'pe_ratio': financial_metrics[0].get('pe_ratio', 0),
            'price_to_book': financial_metrics[0].get('price_to_book', 0),
            'return_on_equity': financial_metrics[0].get('return_on_equity', 0) * 100  # 转换为百分比
        }
        market_data_list.append(combined_data)

# 转换为DataFrame
market_df = pd.DataFrame(market_data_list)

# 设置索引
market_df.set_index('name', inplace=True)

# 可视化市值与PE的关系 - 气泡图
plt.figure(figsize=(12, 8))
plt.scatter(
    market_df['pe_ratio'], 
    market_df['return_on_equity'], 
    s=market_df['market_cap'] * 5,  # 气泡大小由市值决定
    alpha=0.7
)

# 添加标签
for i, txt in enumerate(market_df.index):
    plt.annotate(
        txt, 
        (market_df['pe_ratio'].iloc[i], market_df['return_on_equity'].iloc[i]),
        xytext=(7, 7),
        textcoords='offset points'
    )

plt.title('股票估值与回报率对比')
plt.xlabel('市盈率(PE)')
plt.ylabel('净资产收益率(%)')
plt.grid(True, alpha=0.3)
plt.show()

# 打印市场数据表格
print("\n市场数据与估值指标:")
print(market_df)
```
"""