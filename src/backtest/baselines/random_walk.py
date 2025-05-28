import pandas as pd
import numpy as np
import time
from typing import Optional
from .base_strategy import BaseStrategy, Signal, Portfolio

class RandomWalkStrategy(BaseStrategy):
    """
    随机游走策略
    用作控制组的基准策略
    真正的随机性 - 每次运行结果都不同
    """
    
    def __init__(self, trade_probability: float = 0.1, max_position_ratio: float = 0.5,
                 truly_random: bool = True, **kwargs):
        # Extract name from kwargs if provided, otherwise use default
        name = kwargs.pop('name', 'Random-Walk')
        super().__init__(name, **kwargs)
        self.trade_probability = trade_probability
        self.max_position_ratio = max_position_ratio
        self.truly_random = truly_random
        
        if truly_random:
            # 使用当前时间的纳秒级精度作为种子，确保真正的随机性
            seed = int(time.time() * 1000000) % (2**32)
            self.rng = np.random.RandomState(seed)
            print(f"RandomWalk策略使用随机种子: {seed}")
        else:
            # 如果需要可重现的结果（用于调试），可以设置固定种子
            self.rng = np.random.RandomState(42)
            print("RandomWalk策略使用固定种子: 42 (调试模式)")
        
    def generate_signal(self, data: pd.DataFrame, portfolio: Portfolio, 
                       current_date: str, **kwargs) -> Signal:
        """
        随机游走策略逻辑：
        - 随机决定是否交易
        - 随机决定买入或卖出
        - 随机决定交易数量
        
        真正的随机性模拟市场的不可预测性
        """
        # 随机决定是否交易
        if self.rng.random() > self.trade_probability:
            return Signal(
                action='hold',
                quantity=0,
                confidence=0.5,
                reasoning="Random walk: no trade decision"
            )
        
        current_price = data['close'].iloc[-1]
        position_ratio = (portfolio.stock * current_price) / (portfolio.cash + portfolio.stock * current_price)
        
        # 随机决定买入或卖出
        if self.rng.random() < 0.5 and position_ratio < self.max_position_ratio:
            # 随机买入
            max_investment = portfolio.cash * self.rng.uniform(0.1, 0.3)
            quantity = int(max_investment / current_price)
            
            if quantity > 0:
                return Signal(
                    action='buy',
                    quantity=quantity,
                    confidence=0.5,
                    reasoning="Random walk: random buy decision",
                    metadata={'random_factor': self.rng.random()}
                )
        
        elif portfolio.stock > 0:
            # 随机卖出
            quantity = int(portfolio.stock * self.rng.uniform(0.1, 0.5))
            
            if quantity > 0:
                return Signal(
                    action='sell',
                    quantity=quantity,
                    confidence=0.5,
                    reasoning="Random walk: random sell decision",
                    metadata={'random_factor': self.rng.random()}
                )
        
        return Signal(
            action='hold',
            quantity=0,
            confidence=0.5,
            reasoning="Random walk: hold decision"
        )