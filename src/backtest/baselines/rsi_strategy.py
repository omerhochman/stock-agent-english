import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal, Portfolio

class RSIStrategy(BaseStrategy):
    """
    RSI策略
    基于相对强弱指数(RSI)的超买超卖策略
    参考：Wilder (1978) - New Concepts in Technical Trading Systems
    """
    
    def __init__(self, rsi_period: int = 14, overbought: float = 70, 
                 oversold: float = 30, smoothing_period: int = 3, **kwargs):
        # Extract name from kwargs if provided, otherwise use default
        name = kwargs.pop('name', 'RSI-Strategy')
        super().__init__(name, **kwargs)
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.smoothing_period = smoothing_period
        self.last_signal = None
        self.signal_count = 0
        
    def calculate_rsi(self, prices: pd.Series) -> float:
        """计算RSI指标"""
        if len(prices) < self.rsi_period + 1:
            return 50.0
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def calculate_rsi_divergence(self, prices: pd.Series, rsi_values: pd.Series) -> str:
        """计算RSI背离"""
        if len(prices) < 10 or len(rsi_values) < 10:
            return "none"
            
        # 寻找最近的高点和低点
        recent_prices = prices.tail(10)
        recent_rsi = rsi_values.tail(10)
        
        price_high_idx = recent_prices.idxmax()
        price_low_idx = recent_prices.idxmin()
        rsi_high_idx = recent_rsi.idxmax()
        rsi_low_idx = recent_rsi.idxmin()
        
        # 检查看涨背离（价格创新低，RSI未创新低）
        if (price_low_idx == recent_prices.index[-1] and 
            rsi_low_idx != recent_rsi.index[-1] and
            recent_rsi.iloc[-1] > recent_rsi.min()):
            return "bullish"
            
        # 检查看跌背离（价格创新高，RSI未创新高）
        if (price_high_idx == recent_prices.index[-1] and 
            rsi_high_idx != recent_rsi.index[-1] and
            recent_rsi.iloc[-1] < recent_rsi.max()):
            return "bearish"
            
        return "none"
    
    def calculate_stochastic_rsi(self, rsi_values: pd.Series, period: int = 14) -> float:
        """计算随机RSI"""
        if len(rsi_values) < period:
            return 50.0
            
        recent_rsi = rsi_values.tail(period)
        lowest_rsi = recent_rsi.min()
        highest_rsi = recent_rsi.max()
        
        if highest_rsi == lowest_rsi:
            return 50.0
            
        stoch_rsi = (rsi_values.iloc[-1] - lowest_rsi) / (highest_rsi - lowest_rsi) * 100
        return stoch_rsi
    
    def generate_signal(self, data: pd.DataFrame, portfolio: Portfolio, 
                       current_date: str, **kwargs) -> Signal:
        """
        RSI策略逻辑：
        - RSI < 30时考虑买入（超卖）
        - RSI > 70时考虑卖出（超买）
        - 结合价格动量和成交量确认
        - 使用RSI背离增强信号
        """
        prices = data['close']
        
        # 计算RSI
        rsi_series = prices.rolling(window=self.rsi_period+1).apply(
            lambda x: self._single_rsi(x), raw=False
        )
        current_rsi = self.calculate_rsi(prices)
        
        # 计算随机RSI
        stoch_rsi = self.calculate_stochastic_rsi(rsi_series.dropna())
        
        # 计算RSI背离
        divergence = self.calculate_rsi_divergence(prices, rsi_series.dropna())
        
        # 计算价格动量
        if len(prices) >= 5:
            momentum = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100
        else:
            momentum = 0
            
        # 计算成交量趋势
        if 'volume' in data.columns and len(data) >= 10:
            volume_ma_short = data['volume'].tail(5).mean()
            volume_ma_long = data['volume'].tail(10).mean()
            volume_trend = volume_ma_short / volume_ma_long if volume_ma_long > 0 else 1
        else:
            volume_trend = 1
            
        current_price = prices.iloc[-1]
        position_ratio = (portfolio.stock * current_price) / (portfolio.cash + portfolio.stock * current_price)
        
        # 超卖买入信号
        if current_rsi < self.oversold and position_ratio < 0.8:
            # 增强条件：RSI持续下降或出现看涨背离
            signal_strength = (self.oversold - current_rsi) / self.oversold
            
            # 背离增强信号
            if divergence == "bullish":
                signal_strength *= 1.5
                
            # 成交量确认
            if volume_trend > 1.2:
                signal_strength *= 1.2
                
            # 随机RSI确认
            if stoch_rsi < 20:
                signal_strength *= 1.1
                
            max_investment = portfolio.cash * min(0.4, signal_strength * 0.3)
            quantity = int(max_investment / current_price)
            
            if quantity > 0:
                confidence = min(0.9, 0.6 + signal_strength * 0.2)
                return Signal(
                    action='buy',
                    quantity=quantity,
                    confidence=confidence,
                    reasoning=f"RSI oversold: {current_rsi:.1f}, StochRSI: {stoch_rsi:.1f}, Divergence: {divergence}",
                    metadata={
                        'rsi': current_rsi,
                        'stoch_rsi': stoch_rsi,
                        'divergence': divergence,
                        'signal_strength': signal_strength,
                        'volume_trend': volume_trend
                    }
                )
        
        # 超买卖出信号
        elif current_rsi > self.overbought and portfolio.stock > 0:
            signal_strength = (current_rsi - self.overbought) / (100 - self.overbought)
            
            # 背离增强信号
            if divergence == "bearish":
                signal_strength *= 1.5
                
            # 成交量确认
            if volume_trend > 1.2:
                signal_strength *= 1.2
                
            # 随机RSI确认
            if stoch_rsi > 80:
                signal_strength *= 1.1
                
            quantity = min(portfolio.stock, int(portfolio.stock * min(0.6, signal_strength * 0.4)))
            
            if quantity > 0:
                confidence = min(0.9, 0.6 + signal_strength * 0.2)
                return Signal(
                    action='sell',
                    quantity=quantity,
                    confidence=confidence,
                    reasoning=f"RSI overbought: {current_rsi:.1f}, StochRSI: {stoch_rsi:.1f}, Divergence: {divergence}",
                    metadata={
                        'rsi': current_rsi,
                        'stoch_rsi': stoch_rsi,
                        'divergence': divergence,
                        'signal_strength': signal_strength,
                        'volume_trend': volume_trend
                    }
                )
        
        # 中性区域 - 部分获利了结
        elif 40 < current_rsi < 60 and portfolio.stock > 0:
            # 如果RSI回到中性区域，考虑部分获利了结
            if self.last_signal == 'buy' and current_rsi > 50:
                quantity = int(portfolio.stock * 0.2)  # 获利了结20%
                return Signal(
                    action='sell',
                    quantity=quantity,
                    confidence=0.6,
                    reasoning=f"RSI neutral zone profit taking: {current_rsi:.1f}",
                    metadata={
                        'rsi': current_rsi,
                        'action_type': 'profit_taking'
                    }
                )
        
        # 记录最后信号
        if current_rsi < self.oversold:
            self.last_signal = 'buy'
        elif current_rsi > self.overbought:
            self.last_signal = 'sell'
            
        # 持有信号
        return Signal(
            action='hold',
            quantity=0,
            confidence=0.5,
            reasoning=f"RSI neutral: {current_rsi:.1f}",
            metadata={
                'rsi': current_rsi,
                'stoch_rsi': stoch_rsi,
                'divergence': divergence
            }
        )
    
    def _single_rsi(self, prices):
        """计算单个RSI值的辅助函数"""
        if len(prices) < 2:
            return 50.0
            
        delta = prices.diff().dropna()
        if len(delta) == 0:
            return 50.0
            
        gain = delta.where(delta > 0, 0).mean()
        loss = (-delta.where(delta < 0, 0)).mean()
        
        if loss == 0:
            return 100.0
        if gain == 0:
            return 0.0
            
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi 