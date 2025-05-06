# 数据处理示例
from src.tools.api import get_price_history, get_financial_metrics
import matplotlib.pyplot as plt

# 获取高质量价格数据
ticker = "600519"
price_df = get_price_history(ticker, "2023-01-01", "2023-12-31")

# 检查数据质量
print(f"获取了 {len(price_df)} 条价格记录")
print(f"数据开始日期: {price_df.index.min()}")
print(f"数据结束日期: {price_df.index.max()}")
print(f"数据中的缺失值数量: {price_df.isna().sum().sum()}")

# 绘制价格图表
plt.figure(figsize=(12, 6))
plt.plot(price_df.index, price_df['close'])
plt.title(f"{ticker} 收盘价")
plt.xlabel("日期")
plt.ylabel("价格")
plt.grid(True, alpha=0.3)
plt.savefig(f"{ticker}_price_chart.png")

# 获取并分析财务数据
metrics = get_financial_metrics(ticker)
print(f"财务指标数量: {len(metrics)}")
for key, value in metrics[0].items():
    print(f"  {key}: {value}")