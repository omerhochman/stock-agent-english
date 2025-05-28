"""
A股投资Agent系统 - 回测框架

这个模块提供了完整的回测框架，包括：
- 核心回测引擎 (core.py)
- 基准策略集合 (baselines/)
- 性能评估工具 (evaluation/)
- 交易执行引擎 (execution/)
- 工具函数集合 (backtest_utils/)

主要类:
    Backtester: 主回测引擎
    BacktestConfig: 回测配置
    BacktestResult: 回测结果
    BaseStrategy: 策略基类
    
使用示例:
    from src.backtest import Backtester, BacktestConfig
    
    config = BacktestConfig(
        initial_capital=100000,
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    backtester = Backtester(config=config)
    results = backtester.run_baseline_backtests()
"""

# 核心模块
from .core import (
    Backtester,
    BacktestConfig, 
    BacktestResult
)

# 基准策略
from .baselines import (
    BaseStrategy,
    Signal,
    Portfolio,
    BuyHoldStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    MovingAverageStrategy,
    RandomWalkStrategy
)

# 评估模块
from .evaluation.metrics import PerformanceMetrics
from .evaluation.significance import SignificanceTester
from .evaluation.comparison import StrategyComparator
from .evaluation.visualization import BacktestVisualizer

# 执行模块
from .execution.trade_executor import TradeExecutor
from .execution.cost_model import CostModel

# 工具模块
from .backtest_utils.data_utils import DataProcessor
from .backtest_utils.performance import PerformanceAnalyzer
from .backtest_utils.statistics import StatisticalAnalyzer

__version__ = "1.0.0"
__author__ = "A股投资Agent团队"

__all__ = [
    # 核心类
    "Backtester",
    "BacktestConfig", 
    "BacktestResult",
    
    # 基准策略
    "BaseStrategy",
    "Signal",
    "Portfolio",
    "BuyHoldStrategy",
    "MomentumStrategy", 
    "MeanReversionStrategy",
    "MovingAverageStrategy",
    "RandomWalkStrategy",
    
    # 评估工具
    "PerformanceMetrics",
    "SignificanceTester",
    "StrategyComparator",
    "BacktestVisualizer",
    
    # 执行工具
    "TradeExecutor",
    "CostModel",
    
    # 工具函数
    "DataProcessor",
    "PerformanceAnalyzer",
    "StatisticalAnalyzer"
]
