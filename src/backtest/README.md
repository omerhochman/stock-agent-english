# A股投资Agent系统 - 回测框架使用指南

## 概述

本回测框架提供了完整的量化投资策略回测功能，包括：
- 多种基准策略实现
- AI Agent策略回测
- 统计显著性检验
- 性能指标计算
- 可视化图表生成
- 详细报告导出

## 快速开始

### 1. 基本使用

```bash
# 运行基准策略回测
python src/main_backtester.py --ticker 000001 --baseline-only

# 运行完整回测（包括AI Agent）
python src/main_backtester.py --ticker 000001 --start-date 2023-01-01 --end-date 2023-12-31

# 多股票组合回测
python src/main_backtester.py --ticker 000001 --tickers "000001,000002,600036" --initial-capital 500000
```

### 2. 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ticker` | 主要股票代码（必需） | - |
| `--tickers` | 多股票代码，逗号分隔 | - |
| `--start-date` | 回测开始日期 | 一年前 |
| `--end-date` | 回测结束日期 | 今天 |
| `--initial-capital` | 初始资金 | 100000 |
| `--benchmark` | 基准指数代码 | 000300 |
| `--trading-cost` | 交易成本比例 | 0.001 |
| `--slippage` | 滑点比例 | 0.001 |
| `--baseline-only` | 仅运行基准策略 | False |
| `--save-results` | 保存结果到文件 | False |
| `--export-report` | 导出HTML报告 | False |

### 3. 编程接口使用

```python
from src.backtest import Backtester, BacktestConfig

# 创建配置
config = BacktestConfig(
    initial_capital=100000,
    start_date="2023-01-01",
    end_date="2023-12-31",
    trading_cost=0.001,
    slippage=0.001
)

# 创建回测器
backtester = Backtester(
    ticker="000001",
    config=config
)

# 运行基准策略回测
baseline_results = backtester.run_baseline_backtests()

# 运行比较分析
comparison_results = backtester.run_comprehensive_comparison()

# 生成可视化图表
chart_paths = backtester.generate_visualization()

# 导出报告
report_file = backtester.export_report(format='html')
```

## 框架架构

### 核心模块

1. **core.py** - 主回测引擎
   - `Backtester`: 主回测类
   - `BacktestConfig`: 配置类
   - `BacktestResult`: 结果类

2. **baselines/** - 基准策略
   - `BuyHoldStrategy`: 买入持有策略
   - `MomentumStrategy`: 动量策略
   - `MeanReversionStrategy`: 均值回归策略
   - `MovingAverageStrategy`: 移动平均策略
   - `RandomWalkStrategy`: 随机游走策略

3. **evaluation/** - 评估模块
   - `PerformanceMetrics`: 性能指标计算
   - `SignificanceTester`: 统计显著性检验
   - `StrategyComparator`: 策略比较
   - `BacktestVisualizer`: 可视化

4. **execution/** - 执行模块
   - `TradeExecutor`: 交易执行器
   - `CostModel`: 成本模型

5. **backtest_utils/** - 工具模块
   - `DataProcessor`: 数据处理
   - `PerformanceAnalyzer`: 性能分析
   - `StatisticalAnalyzer`: 统计分析

### 基准策略说明

1. **买入持有策略 (Buy & Hold)**
   - 在回测开始时买入并持有到结束
   - 适合作为基准比较

2. **动量策略 (Momentum)**
   - 基于价格动量进行交易
   - 支持多种参数组合

3. **均值回归策略 (Mean Reversion)**
   - 基于价格偏离均值的程度进行交易
   - 使用Z-score判断买卖时机

4. **移动平均策略 (Moving Average)**
   - 基于短期和长期移动平均线交叉
   - 经典的技术分析策略

5. **随机游走策略 (Random Walk)**
   - 随机生成交易信号
   - 作为对照组验证其他策略的有效性

## 性能指标

框架计算以下性能指标：

- **收益率指标**
  - 总收益率 (Total Return)
  - 年化收益率 (Annual Return)
  - 日收益率 (Daily Returns)

- **风险指标**
  - 波动率 (Volatility)
  - 最大回撤 (Maximum Drawdown)
  - VaR (Value at Risk)

- **风险调整收益**
  - 夏普比率 (Sharpe Ratio)
  - 索提诺比率 (Sortino Ratio)
  - 卡尔马比率 (Calmar Ratio)

- **交易指标**
  - 胜率 (Win Rate)
  - 平均盈利/亏损
  - 交易次数

## 统计显著性检验

框架提供多种统计检验方法：

1. **T检验** - 比较策略收益率差异
2. **Sharpe比率检验** - 比较风险调整收益
3. **Bootstrap检验** - 非参数统计检验
4. **功效分析** - 评估检验的统计功效

## 可视化图表

自动生成以下图表：

1. **累计收益曲线** - 展示各策略的收益表现
2. **回撤曲线** - 显示最大回撤情况
3. **收益分布图** - 收益率的分布特征
4. **相关性热力图** - 策略间的相关性
5. **风险收益散点图** - 风险与收益的关系

## 测试框架

运行测试脚本验证框架功能：

```bash
python test/test_backtest.py
```

测试内容包括：
- 配置验证
- 基准策略初始化
- 单策略回测
- 多策略比较
- 统计分析

## 注意事项

1. **数据依赖**
   - 确保股票数据API可用
   - 检查网络连接和API限制

2. **计算资源**
   - 长期回测可能需要较长时间
   - 建议先用短期数据测试

3. **参数调优**
   - 交易成本和滑点设置要符合实际情况
   - 基准策略参数可根据需要调整

4. **结果解读**
   - 注意统计显著性的置信水平
   - 考虑样本外验证

## 扩展开发

### 添加新的基准策略

1. 继承 `BaseStrategy` 类
2. 实现 `generate_signal` 方法
3. 在 `Backtester.initialize_baseline_strategies` 中添加

### 自定义性能指标

1. 在 `PerformanceMetrics` 类中添加新方法
2. 更新 `calculate_all_metrics` 方法

### 新增可视化图表

1. 在 `BacktestVisualizer` 类中添加新方法
2. 更新 `create_comparison_charts` 方法

## 常见问题

**Q: 回测结果不稳定怎么办？**
A: 检查数据质量，增加回测期间长度，使用多次运行取平均值。

**Q: 如何处理停牌股票？**
A: 框架会自动跳过无法获取价格的日期，建议选择流动性好的股票。

**Q: 统计检验显示不显著怎么办？**
A: 可能是样本量不足或策略差异确实不大，考虑增加回测期间或调整策略参数。

**Q: 如何优化回测速度？**
A: 减少回测期间，降低数据获取频率，使用缓存机制。

## 更新日志

- v1.0.0: 初始版本，包含基本回测功能
- 支持多种基准策略
- 统计显著性检验
- 可视化图表生成
- HTML报告导出

---

如有问题或建议，请联系开发团队。 