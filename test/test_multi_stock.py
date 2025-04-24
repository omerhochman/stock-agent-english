# 多股票分析示例
from src.tools.api import get_price_history, get_financial_metrics
import pandas as pd
import matplotlib.pyplot as plt

# 股票列表
tickers = ["600519", "601318", "600036", "000858"]
results = {}

# 获取和分析多只股票
for ticker in tickers:
    print(f"\n分析股票: {ticker}")
    
    # 获取价格数据
    price_df = get_price_history(ticker, "2023-01-01", "2023-12-31")
    if price_df.empty:
        print(f"  无法获取 {ticker} 的价格数据")
        continue
        
    # 计算收益率
    price_df['return'] = price_df['close'].pct_change()
    cumulative_return = (1 + price_df['return']).cumprod() - 1
    latest_return = cumulative_return.iloc[-1]
    
    # 获取财务指标
    metrics = get_financial_metrics(ticker)
    pe_ratio = metrics[0].get("pe_ratio", 0) if metrics and metrics[0] else 0
    
    # 保存结果
    results[ticker] = {
        "return": latest_return,
        "pe_ratio": pe_ratio,
        "volatility": price_df['return'].std() * (252 ** 0.5)
    }
    
    print(f"  年化收益率: {latest_return:.2%}")
    print(f"  市盈率: {pe_ratio:.2f}")
    print(f"  年化波动率: {results[ticker]['volatility']:.2%}")

# 绘制结果图表
plt.figure(figsize=(10, 6))
tickers = list(results.keys())
returns = [results[t]["return"] for t in tickers]
volatilities = [results[t]["volatility"] for t in tickers]

plt.scatter(volatilities, returns)
for i, ticker in enumerate(tickers):
    plt.annotate(ticker, (volatilities[i], returns[i]))

plt.title("风险-收益分析")
plt.xlabel("波动率")
plt.ylabel("收益率")
plt.grid(True, alpha=0.3)
plt.savefig("risk_return_analysis.png")