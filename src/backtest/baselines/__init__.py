from typing import List

from .base_strategy import BaseStrategy, Portfolio, Signal
from .bollinger_strategy import BollingerStrategy
from .buy_hold import BuyHoldStrategy
from .macd_strategy import MACDStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .moving_average import MovingAverageStrategy
from .random_walk import RandomWalkStrategy
from .rsi_strategy import RSIStrategy

# Strategy registry
STRATEGY_REGISTRY = {
    "buy_hold": BuyHoldStrategy,
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "moving_average": MovingAverageStrategy,
    "random_walk": RandomWalkStrategy,
    "rsi_strategy": RSIStrategy,
    "bollinger_strategy": BollingerStrategy,
    "macd_strategy": MACDStrategy,
}


def get_strategy(strategy_name: str, **kwargs) -> BaseStrategy:
    """
    Get strategy instance

    Args:
        strategy_name: Strategy name
        **kwargs: Strategy parameters

    Returns:
        BaseStrategy: Strategy instance
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Available strategies: {list(STRATEGY_REGISTRY.keys())}"
        )

    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class(**kwargs)


def list_available_strategies() -> List[str]:
    """List all available strategies"""
    return list(STRATEGY_REGISTRY.keys())
