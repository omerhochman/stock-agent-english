"""
Trading execution module

Provides trading execution and cost calculation functionality.

Main Components:
- TradeExecutor: Trading execution engine
- CostModel: Trading cost model
"""

from .cost_model import CostModel
from .trade_executor import TradeExecutor

__all__ = ["TradeExecutor", "CostModel"]
