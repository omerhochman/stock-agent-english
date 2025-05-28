import pandas as pd
from .base_strategy import BaseStrategy, Signal, Portfolio

class MomentumStrategy(BaseStrategy):
    """
    动量策略
    基于价格趋势延续的投资策略
    参考：Jegadeesh and Titman (1993)
    """
    
    def __init__(self, lookback_period: int = 252, formation_period: int = 63,
                 holding_period: int = 21, momentum_threshold: float = 0.02, **kwargs):
        # Extract name from kwargs if provided, otherwise use default
        name = kwargs.pop('name', 'Momentum')
        super().__init__(name, **kwargs)
        self.lookback_period = lookback_period      # 历史数据回看期
        self.formation_period = formation_period    # 动量形成期
        self.holding_period = holding_period        # 持有期
        self.momentum_threshold = momentum_threshold # 动量阈值
        self.last_trade_date = None
        self.hold_until_date = None
        
    def calculate_momentum(self, prices: pd.Series) -> float:
        """
        计算价格动量
        使用多期动量的加权平均
        """
        if len(prices) < self.formation_period:
            return 0.0
            
        # 计算不同期间的动量
        momentum_1m = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) >= 21 else 0
        momentum_3m = (prices.iloc[-1] / prices.iloc[-63] - 1) if len(prices) >= 63 else 0
        momentum_6m = (prices.iloc[-1] / prices.iloc[-126] - 1) if len(prices) >= 126 else 0
        momentum_12m = (prices.iloc[-1] / prices.iloc[-252] - 1) if len(prices) >= 252 else 0
        
        # 加权动量计算（忽略最近一个月，避免短期反转）
        weighted_momentum = (
            0.2 * momentum_1m +
            0.3 * momentum_3m + 
            0.3 * momentum_6m +
            0.2 * momentum_12m
        )
        
        return weighted_momentum
    
    def generate_signal(self, data: pd.DataFrame, portfolio: Portfolio, 
                       current_date: str, **kwargs) -> Signal:
        """
        动量策略逻辑：
        - 计算价格动量
        - 当动量超过阈值时买入
        - 当动量转负时卖出
        - 考虑持有期限制
        """
        current_date_obj = pd.to_datetime(current_date)
        
        # 检查是否在强制持有期内
        if (self.hold_until_date and current_date_obj < self.hold_until_date):
            return Signal(
                action='hold',
                quantity=0,
                confidence=0.8,
                reasoning=f"Within holding period until {self.hold_until_date}"
            )
        
        # 计算动量指标
        prices = data['close']
        momentum = self.calculate_momentum(prices)
        
        # 计算附加指标
        volatility = prices.pct_change().rolling(21).std().iloc[-1] if len(prices) > 21 else 0
        volume_trend = (data['volume'].rolling(21).mean().iloc[-1] / 
                       data['volume'].rolling(63).mean().iloc[-1]) if len(data) > 63 else 1
        
        current_price = prices.iloc[-1]
        position_ratio = (portfolio.stock * current_price) / (portfolio.cash + portfolio.stock * current_price)
        
        # 动量买入信号
        if momentum > self.momentum_threshold and position_ratio < 0.8:
            # 强动量且未满仓
            volume_confirmation = volume_trend > 1.1  # 成交量确认
            volatility_filter = volatility < 0.05  # 波动率过滤
            
            if volume_confirmation or not volatility_filter:
                max_investment = portfolio.cash * 0.5  # 最大投入50%现金
                quantity = int(max_investment / current_price)
                
                if quantity > 0:
                    # 设置持有期
                    self.hold_until_date = current_date_obj + pd.Timedelta(days=self.holding_period)
                    
                    confidence = min(0.9, 0.5 + abs(momentum))
                    return Signal(
                        action='buy',
                        quantity=quantity,
                        confidence=confidence,
                        reasoning=f"Strong momentum {momentum:.2%}, volume trend {volume_trend:.2f}",
                        metadata={
                            'momentum': momentum,
                            'volume_trend': volume_trend,
                            'volatility': volatility
                        }
                    )
        
        # 动量卖出信号  
        elif momentum < -self.momentum_threshold and portfolio.stock > 0:
            # 负动量且有持仓
            quantity = min(portfolio.stock, int(portfolio.stock * 0.5))  # 部分卖出
            
            confidence = min(0.9, 0.5 + abs(momentum))
            return Signal(
                action='sell',
                quantity=quantity,
                confidence=confidence,
                reasoning=f"Negative momentum {momentum:.2%}, reducing exposure",
                metadata={
                    'momentum': momentum,
                    'volume_trend': volume_trend,
                    'volatility': volatility
                }
            )
        
        # 持有信号
        return Signal(
            action='hold',
            quantity=0,
            confidence=0.6,
            reasoning=f"Momentum {momentum:.2%} within neutral range",
            metadata={
                'momentum': momentum,
                'volume_trend': volume_trend,
                'volatility': volatility
            }
        )