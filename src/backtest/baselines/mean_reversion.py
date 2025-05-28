import pandas as pd
from .base_strategy import BaseStrategy, Signal, Portfolio

class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略
    基于价格向长期均值回归的投资策略
    参考：De Bondt and Thaler (1985)
    """
    
    def __init__(self, lookback_period: int = 252, z_threshold: float = 2.0,
                 mean_period: int = 50, exit_threshold: float = 0.5, **kwargs):
        # Extract name from kwargs if provided, otherwise use default
        name = kwargs.pop('name', 'Mean-Reversion')
        super().__init__(name, **kwargs)
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold  # Z-score阈值
        self.mean_period = mean_period  # 均值计算期
        self.exit_threshold = exit_threshold  # 退出阈值
        self.position_entry_date = None
        
    def calculate_z_score(self, prices: pd.Series) -> float:
        """计算价格的Z-score"""
        if len(prices) < self.mean_period:
            return 0.0
            
        recent_prices = prices.tail(self.mean_period)
        mean_price = recent_prices.mean()
        std_price = recent_prices.std()
        
        if std_price == 0:
            return 0.0
            
        current_price = prices.iloc[-1]
        z_score = (current_price - mean_price) / std_price
        
        return z_score
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI指标"""
        if len(prices) < period + 1:
            return 50.0
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).tail(period).mean()
        loss = (-delta.where(delta < 0, 0)).tail(period).mean()
        
        if loss == 0:
            return 100.0
            
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signal(self, data: pd.DataFrame, portfolio: Portfolio, 
                       current_date: str, **kwargs) -> Signal:
        """
        均值回归策略逻辑：
        - 计算Z-score和RSI
        - 价格严重偏离均值时进行反向操作
        - 价格回归至均值附近时平仓
        """
        prices = data['close']
        z_score = self.calculate_z_score(prices)
        rsi = self.calculate_rsi(prices)
        
        # 计算布林带
        if len(prices) >= 20:
            bb_period = 20
            sma = prices.tail(bb_period).mean()
            std = prices.tail(bb_period).std()
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            current_price = prices.iloc[-1]
            bb_position = (current_price - lower_band) / (upper_band - lower_band)
        else:
            bb_position = 0.5
            
        # 计算波动率
        volatility = prices.pct_change().tail(21).std() if len(prices) > 21 else 0
        
        position_ratio = (portfolio.stock * prices.iloc[-1]) / (portfolio.cash + portfolio.stock * prices.iloc[-1])
        
        # 超卖信号 - 买入
        if (z_score < -self.z_threshold or rsi < 30 or bb_position < 0.1) and position_ratio < 0.7:
            # 价格严重低于均值，超卖
            signal_strength = abs(z_score) / self.z_threshold
            max_investment = portfolio.cash * min(0.4, signal_strength * 0.3)
            quantity = int(max_investment / prices.iloc[-1])
            
            if quantity > 0:
                confidence = min(0.9, 0.5 + signal_strength * 0.3)
                return Signal(
                    action='buy',
                    quantity=quantity,
                    confidence=confidence,
                    reasoning=f"Oversold: Z-score {z_score:.2f}, RSI {rsi:.1f}, BB {bb_position:.2f}",
                    metadata={
                        'z_score': z_score,
                        'rsi': rsi,
                        'bb_position': bb_position,
                        'signal_strength': signal_strength
                    }
                )
        
        # 超买信号 - 卖出
        elif (z_score > self.z_threshold or rsi > 70 or bb_position > 0.9) and portfolio.stock > 0:
            # 价格严重高于均值，超买
            signal_strength = abs(z_score) / self.z_threshold
            quantity = min(portfolio.stock, int(portfolio.stock * min(0.6, signal_strength * 0.4)))
            
            if quantity > 0:
                confidence = min(0.9, 0.5 + signal_strength * 0.3)
                return Signal(
                    action='sell',
                    quantity=quantity,
                    confidence=confidence,
                    reasoning=f"Overbought: Z-score {z_score:.2f}, RSI {rsi:.1f}, BB {bb_position:.2f}",
                    metadata={
                        'z_score': z_score,
                        'rsi': rsi,
                        'bb_position': bb_position,
                        'signal_strength': signal_strength
                    }
                )
        
        # 回归信号 - 平仓
        elif abs(z_score) < self.exit_threshold and 30 < rsi < 70 and 0.3 < bb_position < 0.7:
            if portfolio.stock > 0:
                # 部分获利了结
                quantity = int(portfolio.stock * 0.3)
                return Signal(
                    action='sell',
                    quantity=quantity,
                    confidence=0.6,
                    reasoning=f"Mean reversion: Z-score {z_score:.2f} approaching neutral",
                    metadata={
                        'z_score': z_score,
                        'rsi': rsi,
                        'bb_position': bb_position
                    }
                )
        
        # 持有信号
        return Signal(
            action='hold',
            quantity=0,
            confidence=0.5,
            reasoning=f"Neutral: Z-score {z_score:.2f}, RSI {rsi:.1f}",
            metadata={
                'z_score': z_score,
                'rsi': rsi,
                'bb_position': bb_position
            }
        )