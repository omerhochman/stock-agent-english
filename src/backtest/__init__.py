"""
A-Share Investment Agent System - Backtesting Framework

This module provides a complete backtesting framework, including:
- Core backtesting engine (core.py)
- Baseline strategy collection (baselines/)
- Performance evaluation tools (evaluation/)
- Trading execution engine (execution/)
- Utility function collection (backtest_utils/)

Main Classes:
    Backtester: Main backtesting engine
    BacktestConfig: Backtesting configuration
    BacktestResult: Backtesting results
    BaseStrategy: Strategy base class

Usage Example:
    from src.backtest import Backtester, BacktestConfig

    config = BacktestConfig(
        initial_capital=100000,
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

    backtester = Backtester(config=config)
    results = backtester.run_baseline_backtests()
"""

# Utility modules
from .backtest_utils.data_utils import DataProcessor
from .backtest_utils.performance import PerformanceAnalyzer
from .backtest_utils.statistics import StatisticalAnalyzer

# Baseline strategies
from .baselines import (
    BaseStrategy,
    BollingerStrategy,
    BuyHoldStrategy,
    MACDStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    MovingAverageStrategy,
    Portfolio,
    RandomWalkStrategy,
    RSIStrategy,
    Signal,
)

# Core modules
from .core import BacktestConfig, Backtester, BacktestResult
from .evaluation.comparison import StrategyComparator

# Evaluation modules
from .evaluation.metrics import PerformanceMetrics
from .evaluation.significance import SignificanceTester
from .evaluation.visualization import BacktestVisualizer
from .execution.cost_model import CostModel

# Execution modules
from .execution.trade_executor import TradeExecutor

__version__ = "1.0.0"
__author__ = "A-Share Investment Agent Team"

__all__ = [
    # Core classes
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    # Baseline strategies
    "BaseStrategy",
    "Signal",
    "Portfolio",
    "BuyHoldStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "MovingAverageStrategy",
    "RandomWalkStrategy",
    # Evaluation tools
    "PerformanceMetrics",
    "SignificanceTester",
    "StrategyComparator",
    "BacktestVisualizer",
    # Execution tools
    "TradeExecutor",
    "CostModel",
    # Utility functions
    "DataProcessor",
    "PerformanceAnalyzer",
    "StatisticalAnalyzer",
]
