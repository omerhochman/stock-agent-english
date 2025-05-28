import pandas as pd
from .base_strategy import BaseStrategy, Signal, Portfolio

class MovingAverageStrategy(BaseStrategy):
    """
    移动平均策略
    基于不同期间移动平均线交叉的技术分析策略
    """
    
    def __init__(self, short_window: int = 50, long_window: int = 200, 
                 signal_threshold: float = 0.005, **kwargs):
        # Extract name from kwargs if provided, otherwise use default
        name = kwargs.pop('name', 'Moving-Average')
        super().__init__(name, **kwargs)
        self.short_window = short_window
        self.long_window = long_window
        self.signal_threshold = signal_threshold
        self.last_signal = None
        self.signal_count = 0
        
    def generate_signal(self, data: pd.DataFrame, portfolio: Portfolio, 
                       current_date: str, **kwargs) -> Signal:
        """
        移动平均策略逻辑：
        - 短期均线上穿长期均线时买入
        - 短期均线下穿长期均线时卖出
        """
        prices = data['close']
        
        if len(prices) < self.long_window:
            return Signal(
                action='hold',
                quantity=0,
                confidence=0.5,
                reasoning=f"Insufficient data: need {self.long_window} periods"
            )
        
        # 需要额外的数据来计算前一期的移动平均线
        if len(prices) < self.long_window + 1:
            return Signal(
                action='hold',
                quantity=0,
                confidence=0.5,
                reasoning=f"Insufficient data for trend calculation: need {self.long_window + 1} periods"
            )
        
        # 计算当前移动平均线
        short_ma = prices.tail(self.short_window).mean()
        long_ma = prices.tail(self.long_window).mean()
        
        # 计算前一期移动平均线
        prev_short_ma = prices.iloc[-(self.short_window+1):-1].mean()
        prev_long_ma = prices.iloc[-(self.long_window+1):-1].mean()
        
        current_price = prices.iloc[-1]
        position_ratio = (portfolio.stock * current_price) / (portfolio.cash + portfolio.stock * current_price)
        
        # 黄金交叉 - 买入信号（简化条件）
        if (prev_short_ma <= prev_long_ma and short_ma > long_ma and position_ratio < 0.9):
            # 简单的交叉买入
            max_investment = portfolio.cash * 0.8  # 投入80%现金
            quantity = int(max_investment / current_price)
            
            if quantity > 0:
                return Signal(
                    action='buy',
                    quantity=quantity,
                    confidence=0.7,
                    reasoning=f"Golden cross: Short MA {short_ma:.2f} > Long MA {long_ma:.2f}",
                    metadata={
                        'short_ma': short_ma,
                        'long_ma': long_ma,
                        'prev_short_ma': prev_short_ma,
                        'prev_long_ma': prev_long_ma
                    }
                )
        
        # 死亡交叉 - 卖出信号（简化条件）
        elif (prev_short_ma >= prev_long_ma and short_ma < long_ma and portfolio.stock > 0):
            # 简单的交叉卖出
            quantity = int(portfolio.stock * 0.8)  # 卖出80%持仓
            
            if quantity > 0:
                return Signal(
                    action='sell',
                    quantity=quantity,
                    confidence=0.7,
                    reasoning=f"Death cross: Short MA {short_ma:.2f} < Long MA {long_ma:.2f}",
                    metadata={
                        'short_ma': short_ma,
                        'long_ma': long_ma,
                        'prev_short_ma': prev_short_ma,
                        'prev_long_ma': prev_long_ma
                    }
                )
        
        # 持有信号
        return Signal(
            action='hold',
            quantity=0,
            confidence=0.5,
            reasoning=f"No crossover: Short MA {short_ma:.2f}, Long MA {long_ma:.2f}",
            metadata={
                'short_ma': short_ma,
                'long_ma': long_ma,
                'prev_short_ma': prev_short_ma,
                'prev_long_ma': prev_long_ma
            }
        )