# A-Share Investment Agent System - Backtesting Framework User Guide

## Overview

This backtesting framework provides complete quantitative investment strategy backtesting functionality, including:
- Multiple baseline strategy implementations
- AI Agent strategy backtesting
- Statistical significance testing
- Performance metrics calculation
- Visualization chart generation
- Detailed report export

## Quick Start

### 1. Basic Usage

```bash
# Run baseline strategy backtesting
python src/main_backtester.py --ticker 000001 --baseline-only

# Run complete backtesting (including AI Agent)
python src/main_backtester.py --ticker 000001 --start-date 2023-01-01 --end-date 2023-12-31

# Multi-stock portfolio backtesting
python src/main_backtester.py --ticker 000001 --tickers "000001,000002,600036" --initial-capital 500000
```

### 2. Command Line Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--ticker` | Main stock code (required) | - |
| `--tickers` | Multiple stock codes, comma-separated | - |
| `--start-date` | Backtesting start date | One year ago |
| `--end-date` | Backtesting end date | Today |
| `--initial-capital` | Initial capital | 100000 |
| `--benchmark` | Benchmark index code | 000300 |
| `--trading-cost` | Trading cost ratio | 0.001 |
| `--slippage` | Slippage ratio | 0.001 |
| `--baseline-only` | Run baseline strategies only | False |
| `--save-results` | Save results to file | False |
| `--export-report` | Export HTML report | False |

### 3. Programming Interface Usage

```python
from src.backtest import Backtester, BacktestConfig

# Create configuration
config = BacktestConfig(
    initial_capital=100000,
    start_date="2023-01-01",
    end_date="2023-12-31",
    trading_cost=0.001,
    slippage=0.001
)

# Create backtester
backtester = Backtester(
    ticker="000001",
    config=config
)

# Run baseline strategy backtesting
baseline_results = backtester.run_baseline_backtests()

# Run comparison analysis
comparison_results = backtester.run_comprehensive_comparison()

# Generate visualization charts
chart_paths = backtester.generate_visualization()

# Export report
report_file = backtester.export_report(format='html')
```

## Framework Architecture

### Core Modules

1. **core.py** - Main backtesting engine
   - `Backtester`: Main backtesting class
   - `BacktestConfig`: Configuration class
   - `BacktestResult`: Result class

2. **baselines/** - Baseline strategies
   - `BuyHoldStrategy`: Buy and hold strategy
   - `MomentumStrategy`: Momentum strategy
   - `MeanReversionStrategy`: Mean reversion strategy
   - `MovingAverageStrategy`: Moving average strategy
   - `RandomWalkStrategy`: Random walk strategy

3. **evaluation/** - Evaluation module
   - `PerformanceMetrics`: Performance metrics calculation
   - `SignificanceTester`: Statistical significance testing
   - `StrategyComparator`: Strategy comparison
   - `BacktestVisualizer`: Visualization

4. **execution/** - Execution module
   - `TradeExecutor`: Trade executor
   - `CostModel`: Cost model

5. **backtest_utils/** - Utility modules
   - `DataProcessor`: Data processing
   - `PerformanceAnalyzer`: Performance analysis
   - `StatisticalAnalyzer`: Statistical analysis

### Baseline Strategy Description

1. **Buy and Hold Strategy**
   - Buy at the beginning of backtesting and hold until the end
   - Suitable as a benchmark for comparison

2. **Momentum Strategy**
   - Trade based on price momentum
   - Support for multiple parameter combinations

3. **Mean Reversion Strategy**
   - Trade based on the degree of price deviation from the mean
   - Use Z-score to determine buy/sell timing

4. **Moving Average Strategy**
   - Based on short-term and long-term moving average crossovers
   - Classic technical analysis strategy

5. **Random Walk Strategy**
   - Randomly generate trading signals
   - Used as a control group to validate the effectiveness of other strategies

## Performance Metrics

The framework calculates the following performance metrics:

- **Return Metrics**
  - Total Return
  - Annual Return
  - Daily Returns

- **Risk Metrics**
  - Volatility
  - Maximum Drawdown
  - VaR (Value at Risk)

- **Risk-Adjusted Returns**
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio

- **Trading Metrics**
  - Win Rate
  - Average Profit/Loss
  - Number of Trades

## Statistical Significance Testing

The framework provides multiple statistical testing methods:

1. **T-test** - Compare strategy return differences
2. **Sharpe Ratio Test** - Compare risk-adjusted returns
3. **Bootstrap Test** - Non-parametric statistical testing
4. **Power Analysis** - Evaluate statistical power of tests

## Visualization Charts

Automatically generates the following charts:

1. **Cumulative Return Curve** - Shows return performance of each strategy
2. **Drawdown Curve** - Displays maximum drawdown situation
3. **Return Distribution Chart** - Distribution characteristics of returns
4. **Correlation Heatmap** - Correlations between strategies
5. **Risk-Return Scatter Plot** - Relationship between risk and return

## Testing Framework

Run test scripts to verify framework functionality:

```bash
python test/test_backtest.py
```

Test content includes:
- Configuration validation
- Baseline strategy initialization
- Single strategy backtesting
- Multi-strategy comparison
- Statistical analysis

## Important Notes

1. **Data Dependencies**
   - Ensure stock data API is available
   - Check network connection and API limits

2. **Computational Resources**
   - Long-term backtesting may take considerable time
   - Recommend testing with short-term data first

3. **Parameter Tuning**
   - Trading costs and slippage settings should match actual conditions
   - Baseline strategy parameters can be adjusted as needed

4. **Result Interpretation**
   - Pay attention to confidence levels of statistical significance
   - Consider out-of-sample validation

## Extension Development

### Adding New Baseline Strategies

1. Inherit from `BaseStrategy` class
2. Implement `generate_signal` method
3. Add to `Backtester.initialize_baseline_strategies`

### Custom Performance Metrics

1. Add new methods to `PerformanceMetrics` class
2. Update `calculate_all_metrics` method

### Adding New Visualization Charts

1. Add new methods to `BacktestVisualizer` class
2. Update `create_comparison_charts` method

## Frequently Asked Questions

**Q: What to do if backtesting results are unstable?**
A: Check data quality, increase backtesting period length, use multiple runs and take averages.

**Q: How to handle suspended stocks?**
A: The framework will automatically skip dates when prices cannot be obtained, recommend selecting stocks with good liquidity.

**Q: What to do if statistical tests show no significance?**
A: May be due to insufficient sample size or indeed small strategy differences, consider increasing backtesting period or adjusting strategy parameters.

**Q: How to optimize backtesting speed?**
A: Reduce backtesting period, lower data acquisition frequency, use caching mechanisms.

## Update Log

- v1.0.0: Initial version with basic backtesting functionality
- Support for multiple baseline strategies
- Statistical significance testing
- Visualization chart generation
- HTML report export

---

For questions or suggestions, please contact the development team. 