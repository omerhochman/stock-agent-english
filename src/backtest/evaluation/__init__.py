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

from .metrics import PerformanceMetrics
from .significance import SignificanceTester
from .comparison import StrategyComparator
from .visualization import BacktestVisualizer
from .table_generator import BacktestTableGenerator

__all__ = [
    "PerformanceMetrics",
    "SignificanceTester", 
    "StrategyComparator",
    "BacktestVisualizer",
    "BacktestTableGenerator"
] 