import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal, Portfolio

class BollingerStrategy(BaseStrategy):
    """
    布林带策略
    基于布林带的突破和均值回归策略
    参考：John Bollinger (1983) - Bollinger Bands
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, 
                 strategy_mode: str = "mean_reversion", 
                 volume_confirmation: bool = True, **kwargs):
        # Extract name from kwargs if provided, otherwise use default
        name = kwargs.pop('name', f'Bollinger-{strategy_mode.title()}')
        super().__init__(name, **kwargs)
        self.period = period
        self.std_dev = std_dev
        self.strategy_mode = strategy_mode  # "mean_reversion" or "breakout"
        self.volume_confirmation = volume_confirmation
        self.last_signal = None
        self.squeeze_threshold = 0.1  # 布林带收缩阈值
        
    def calculate_bollinger_bands(self, prices: pd.Series):
        """计算布林带"""
        if len(prices) < self.period:
            return None, None, None
            
        sma = prices.rolling(window=self.period).mean()
        std = prices.rolling(window=self.period).std()
        
        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)
        
        return upper_band.iloc[-1], sma.iloc[-1], lower_band.iloc[-1]
    
    def calculate_bandwidth(self, upper_band: float, lower_band: float, middle_band: float) -> float:
        """计算布林带宽度"""
        if middle_band == 0:
            return 0
        return (upper_band - lower_band) / middle_band
    
    def calculate_bb_position(self, price: float, upper_band: float, lower_band: float) -> float:
        """计算价格在布林带中的位置 (0-1)"""
        if upper_band == lower_band:
            return 0.5
        return (price - lower_band) / (upper_band - lower_band)
    
    def detect_squeeze(self, prices: pd.Series) -> bool:
        """检测布林带收缩"""
        if len(prices) < self.period + 10:
            return False
            
        # 计算最近几天的带宽
        recent_bandwidths = []
        for i in range(5):
            end_idx = len(prices) - i
            start_idx = max(0, end_idx - self.period)
            if start_idx >= end_idx:
                continue
                
            period_prices = prices.iloc[start_idx:end_idx]
            if len(period_prices) < self.period:
                continue
                
            sma = period_prices.mean()
            std = period_prices.std()
            bandwidth = (2 * self.std_dev * std) / sma if sma != 0 else 0
            recent_bandwidths.append(bandwidth)
        
        if len(recent_bandwidths) < 3:
            return False
            
        # 如果带宽小于阈值且在收缩
        current_bandwidth = recent_bandwidths[0]
        avg_bandwidth = np.mean(recent_bandwidths)
        
        return current_bandwidth < self.squeeze_threshold and current_bandwidth <= avg_bandwidth
    
    def calculate_volume_confirmation(self, data: pd.DataFrame) -> float:
        """计算成交量确认指标"""
        if 'volume' not in data.columns or len(data) < 20:
            return 1.0
            
        volume = data['volume']
        volume_sma = volume.rolling(window=20).mean()
        current_volume_ratio = volume.iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1.0
        
        return current_volume_ratio
    
    def generate_signal(self, data: pd.DataFrame, portfolio: Portfolio, 
                       current_date: str, **kwargs) -> Signal:
        """
        布林带策略逻辑：
        均值回归模式：
        - 价格触及下轨时买入
        - 价格触及上轨时卖出
        突破模式：
        - 价格突破上轨时买入
        - 价格跌破下轨时卖出
        """
        prices = data['close']
        
        # 计算布林带
        bb_result = self.calculate_bollinger_bands(prices)
        if bb_result[0] is None:
            return Signal(
                action='hold',
                quantity=0,
                confidence=0.5,
                reasoning=f"Insufficient data for Bollinger Bands calculation"
            )
            
        upper_band, middle_band, lower_band = bb_result
        current_price = prices.iloc[-1]
        
        # 计算布林带指标
        bb_position = self.calculate_bb_position(current_price, upper_band, lower_band)
        bandwidth = self.calculate_bandwidth(upper_band, lower_band, middle_band)
        is_squeeze = self.detect_squeeze(prices)
        
        # 计算成交量确认
        volume_ratio = self.calculate_volume_confirmation(data) if self.volume_confirmation else 1.0
        
        # 计算价格动量
        if len(prices) >= 5:
            momentum = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100
        else:
            momentum = 0
            
        position_ratio = (portfolio.stock * current_price) / (portfolio.cash + portfolio.stock * current_price)
        
        if self.strategy_mode == "mean_reversion":
            return self._mean_reversion_signal(
                current_price, upper_band, middle_band, lower_band,
                bb_position, bandwidth, is_squeeze, volume_ratio,
                momentum, portfolio, position_ratio
            )
        else:  # breakout mode
            return self._breakout_signal(
                current_price, upper_band, middle_band, lower_band,
                bb_position, bandwidth, is_squeeze, volume_ratio,
                momentum, portfolio, position_ratio
            )
    
    def _mean_reversion_signal(self, current_price, upper_band, middle_band, lower_band,
                              bb_position, bandwidth, is_squeeze, volume_ratio,
                              momentum, portfolio, position_ratio):
        """均值回归信号生成"""
        
        # 超卖买入信号（价格接近或低于下轨）
        if bb_position <= 0.1 and position_ratio < 0.7:
            signal_strength = (0.1 - bb_position) / 0.1
            
            # 布林带收缩增强信号
            if is_squeeze:
                signal_strength *= 1.3
                
            # 成交量确认
            if volume_ratio > 1.2:
                signal_strength *= 1.2
                
            # 负动量确认（价格下跌但接近支撑）
            if momentum < -2:
                signal_strength *= 1.1
                
            max_investment = portfolio.cash * min(0.4, signal_strength * 0.35)
            quantity = int(max_investment / current_price)
            
            if quantity > 0:
                confidence = min(0.9, 0.6 + signal_strength * 0.25)
                return Signal(
                    action='buy',
                    quantity=quantity,
                    confidence=confidence,
                    reasoning=f"BB mean reversion buy: position {bb_position:.2f}, squeeze: {is_squeeze}",
                    metadata={
                        'bb_position': bb_position,
                        'bandwidth': bandwidth,
                        'is_squeeze': is_squeeze,
                        'volume_ratio': volume_ratio,
                        'signal_strength': signal_strength
                    }
                )
        
        # 超买卖出信号（价格接近或高于上轨）
        elif bb_position >= 0.9 and portfolio.stock > 0:
            signal_strength = (bb_position - 0.9) / 0.1
            
            # 布林带收缩增强信号
            if is_squeeze:
                signal_strength *= 1.3
                
            # 成交量确认
            if volume_ratio > 1.2:
                signal_strength *= 1.2
                
            # 正动量确认（价格上涨但接近阻力）
            if momentum > 2:
                signal_strength *= 1.1
                
            quantity = min(portfolio.stock, int(portfolio.stock * min(0.6, signal_strength * 0.4)))
            
            if quantity > 0:
                confidence = min(0.9, 0.6 + signal_strength * 0.25)
                return Signal(
                    action='sell',
                    quantity=quantity,
                    confidence=confidence,
                    reasoning=f"BB mean reversion sell: position {bb_position:.2f}, squeeze: {is_squeeze}",
                    metadata={
                        'bb_position': bb_position,
                        'bandwidth': bandwidth,
                        'is_squeeze': is_squeeze,
                        'volume_ratio': volume_ratio,
                        'signal_strength': signal_strength
                    }
                )
        
        # 回归中轨 - 部分获利了结
        elif 0.4 < bb_position < 0.6 and portfolio.stock > 0:
            if self.last_signal == 'buy':
                quantity = int(portfolio.stock * 0.25)  # 获利了结25%
                return Signal(
                    action='sell',
                    quantity=quantity,
                    confidence=0.6,
                    reasoning=f"BB mean reversion to middle: position {bb_position:.2f}",
                    metadata={
                        'bb_position': bb_position,
                        'action_type': 'profit_taking'
                    }
                )
        
        return Signal(
            action='hold',
            quantity=0,
            confidence=0.5,
            reasoning=f"BB mean reversion hold: position {bb_position:.2f}",
            metadata={
                'bb_position': bb_position,
                'bandwidth': bandwidth,
                'is_squeeze': is_squeeze
            }
        )
    
    def _breakout_signal(self, current_price, upper_band, middle_band, lower_band,
                        bb_position, bandwidth, is_squeeze, volume_ratio,
                        momentum, portfolio, position_ratio):
        """突破信号生成"""
        
        # 向上突破买入信号
        if bb_position > 1.0 and position_ratio < 0.8:
            signal_strength = min(2.0, bb_position - 1.0)
            
            # 布林带收缩后突破增强信号
            if is_squeeze:
                signal_strength *= 1.5
                
            # 成交量确认（突破需要放量）
            if volume_ratio > 1.5:
                signal_strength *= 1.3
            elif volume_ratio < 1.0:
                signal_strength *= 0.7  # 缩量突破降低信号强度
                
            # 正动量确认
            if momentum > 3:
                signal_strength *= 1.2
                
            max_investment = portfolio.cash * min(0.5, signal_strength * 0.3)
            quantity = int(max_investment / current_price)
            
            if quantity > 0:
                confidence = min(0.9, 0.6 + signal_strength * 0.2)
                return Signal(
                    action='buy',
                    quantity=quantity,
                    confidence=confidence,
                    reasoning=f"BB upward breakout: position {bb_position:.2f}, volume {volume_ratio:.1f}x",
                    metadata={
                        'bb_position': bb_position,
                        'bandwidth': bandwidth,
                        'is_squeeze': is_squeeze,
                        'volume_ratio': volume_ratio,
                        'signal_strength': signal_strength,
                        'breakout_type': 'upward'
                    }
                )
        
        # 向下突破卖出信号
        elif bb_position < 0.0 and portfolio.stock > 0:
            signal_strength = min(2.0, abs(bb_position))
            
            # 布林带收缩后突破增强信号
            if is_squeeze:
                signal_strength *= 1.5
                
            # 成交量确认
            if volume_ratio > 1.5:
                signal_strength *= 1.3
            elif volume_ratio < 1.0:
                signal_strength *= 0.7
                
            # 负动量确认
            if momentum < -3:
                signal_strength *= 1.2
                
            quantity = min(portfolio.stock, int(portfolio.stock * min(0.7, signal_strength * 0.4)))
            
            if quantity > 0:
                confidence = min(0.9, 0.6 + signal_strength * 0.2)
                return Signal(
                    action='sell',
                    quantity=quantity,
                    confidence=confidence,
                    reasoning=f"BB downward breakout: position {bb_position:.2f}, volume {volume_ratio:.1f}x",
                    metadata={
                        'bb_position': bb_position,
                        'bandwidth': bandwidth,
                        'is_squeeze': is_squeeze,
                        'volume_ratio': volume_ratio,
                        'signal_strength': signal_strength,
                        'breakout_type': 'downward'
                    }
                )
        
        # 假突破回调
        elif 0.8 < bb_position < 1.0 and self.last_signal == 'buy':
            # 可能是假突破，考虑减仓
            quantity = int(portfolio.stock * 0.3)
            return Signal(
                action='sell',
                quantity=quantity,
                confidence=0.6,
                reasoning=f"BB potential false breakout: position {bb_position:.2f}",
                metadata={
                    'bb_position': bb_position,
                    'action_type': 'false_breakout_protection'
                }
            )
        
        return Signal(
            action='hold',
            quantity=0,
            confidence=0.5,
            reasoning=f"BB breakout hold: position {bb_position:.2f}",
            metadata={
                'bb_position': bb_position,
                'bandwidth': bandwidth,
                'is_squeeze': is_squeeze
            }
        ) 