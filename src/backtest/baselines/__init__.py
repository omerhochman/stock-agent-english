from typing import List

from .base_strategy import BaseStrategy, Signal, Portfolio
from .buy_hold import BuyHoldStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .moving_average import MovingAverageStrategy
from .random_walk import RandomWalkStrategy

# 策略注册表
STRATEGY_REGISTRY = {
    'buy_hold': BuyHoldStrategy,
    'momentum': MomentumStrategy,
    'mean_reversion': MeanReversionStrategy,
    'moving_average': MovingAverageStrategy,
    'random_walk': RandomWalkStrategy,
}

def get_strategy(strategy_name: str, **kwargs) -> BaseStrategy:
    """
    获取策略实例
    
    Args:
        strategy_name: 策略名称
        **kwargs: 策略参数
        
    Returns:
        BaseStrategy: 策略实例
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略: {strategy_name}. 可用策略: {list(STRATEGY_REGISTRY.keys())}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class(**kwargs)

def list_available_strategies() -> List[str]:
    """列出所有可用策略"""
    return list(STRATEGY_REGISTRY.keys())