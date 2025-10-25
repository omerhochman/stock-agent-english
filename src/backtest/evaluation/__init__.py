"""
Backtesting evaluation module

Provides strategy performance evaluation, statistical significance testing, strategy comparison and visualization functionality.

Main Components:
- PerformanceMetrics: Performance metrics calculation
- SignificanceTester: Statistical significance testing
- StrategyComparator: Strategy comparison analysis
- BacktestVisualizer: Visualization chart generation
- BacktestTableGenerator: Table generator
"""

from .comparison import StrategyComparator
from .metrics import PerformanceMetrics
from .significance import SignificanceTester
from .table_generator import BacktestTableGenerator
from .visualization import BacktestVisualizer

__all__ = [
    "PerformanceMetrics",
    "SignificanceTester",
    "StrategyComparator",
    "BacktestVisualizer",
    "BacktestTableGenerator",
]
