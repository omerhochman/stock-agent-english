# Multi-stock analysis example
from src.tools.api import get_price_history, get_financial_metrics
import matplotlib.pyplot as plt

# Stock list
tickers = ["600519", "601318", "600036", "000858"]
results = {}

# Get and analyze multiple stocks
for ticker in tickers:
    print(f"\nAnalyzing stock: {ticker}")
    
    # Get price data
    price_df = get_price_history(ticker, "2023-01-01", "2023-12-31")
    if price_df.empty:
        print(f"  Unable to get price data for {ticker}")
        continue
        
    # Calculate returns
    price_df['return'] = price_df['close'].pct_change()
    cumulative_return = (1 + price_df['return']).cumprod() - 1
    latest_return = cumulative_return.iloc[-1]
    
    # Get financial metrics
    metrics = get_financial_metrics(ticker)
    pe_ratio = metrics[0].get("pe_ratio", 0) if metrics and metrics[0] else 0
    
    # Save results
    results[ticker] = {
        "return": latest_return,
        "pe_ratio": pe_ratio,
        "volatility": price_df['return'].std() * (252 ** 0.5)
    }
    
    print(f"  Annualized return: {latest_return:.2%}")
    print(f"  P/E ratio: {pe_ratio:.2f}")
    print(f"  Annualized volatility: {results[ticker]['volatility']:.2%}")

# Plot results chart
plt.figure(figsize=(10, 6))
tickers = list(results.keys())
returns = [results[t]["return"] for t in tickers]
volatilities = [results[t]["volatility"] for t in tickers]

plt.scatter(volatilities, returns)
for i, ticker in enumerate(tickers):
    plt.annotate(ticker, (volatilities[i], returns[i]))

plt.title("Risk-Return Analysis")
plt.xlabel("Volatility")
plt.ylabel("Return Rate")
plt.grid(True, alpha=0.3)
plt.savefig("risk_return_analysis.png")