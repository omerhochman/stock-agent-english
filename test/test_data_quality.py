# Data processing example
from src.tools.api import get_price_history, get_financial_metrics
import matplotlib.pyplot as plt

# Get high-quality price data
ticker = "600519"
price_df = get_price_history(ticker, "2023-01-01", "2023-12-31")

# Check data quality
print(f"Retrieved {len(price_df)} price records")
print(f"Data start date: {price_df.index.min()}")
print(f"Data end date: {price_df.index.max()}")
print(f"Number of missing values in data: {price_df.isna().sum().sum()}")

# Plot price chart
plt.figure(figsize=(12, 6))
plt.plot(price_df.index, price_df['close'])
plt.title(f"{ticker} Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True, alpha=0.3)
plt.savefig(f"{ticker}_price_chart.png")

# Get and analyze financial data
metrics = get_financial_metrics(ticker)
print(f"Number of financial metrics: {len(metrics)}")
for key, value in metrics[0].items():
    print(f"  {key}: {value}")