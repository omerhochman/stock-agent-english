"""
回测评估模块

提供策略性能评估、统计显著性检验、策略比较和可视化功能。

主要组件:
- PerformanceMetrics: 性能指标计算
- SignificanceTester: 统计显著性检验
- StrategyComparator: 策略比较分析
- BacktestVisualizer: 可视化图表生成
"""

from .metrics import PerformanceMetrics
from .significance import SignificanceTester
from .comparison import StrategyComparator
from .visualization import BacktestVisualizer

__all__ = [
    "PerformanceMetrics",
    "SignificanceTester", 
    "StrategyComparator",
    "BacktestVisualizer"
] 