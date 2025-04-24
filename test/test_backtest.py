# 回测示例
from src.backtester import Backtester
from src.main import run_hedge_fund

# 创建回测器实例
backtester = Backtester(
    agent=run_hedge_fund,
    ticker="600519",
    start_date="2023-01-01",
    end_date="2023-12-31",
    initial_capital=100000,
    num_of_news=5
)

# 运行回测
backtester.run_backtest()

# 分析性能
performance_df = backtester.analyze_performance()

# 查看关键指标
print(f"总收益率: {backtester.metrics['总收益率']}")
print(f"夏普比率: {backtester.metrics['夏普比率']}")
print(f"最大回撤: {backtester.metrics['最大回撤']}")