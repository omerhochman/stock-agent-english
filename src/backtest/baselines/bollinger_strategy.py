import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal, Portfolio

class BollingerStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy
    Breakout and mean reversion strategy based on Bollinger Bands
    Reference: John Bollinger (1983) - Bollinger Bands
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
        self.squeeze_threshold = 0.1  # Bollinger Bands squeeze threshold
        
    def calculate_bollinger_bands(self, prices: pd.Series):
        """Calculate Bollinger Bands"""
        if len(prices) < self.period:
            return None, None, None
            
        sma = prices.rolling(window=self.period).mean()
        std = prices.rolling(window=self.period).std()
        
        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)
        
        return upper_band.iloc[-1], sma.iloc[-1], lower_band.iloc[-1]
    
    def calculate_bandwidth(self, upper_band: float, lower_band: float, middle_band: float) -> float:
        """Calculate Bollinger Bands width"""
        if middle_band == 0:
            return 0
        return (upper_band - lower_band) / middle_band
    
    def calculate_bb_position(self, price: float, upper_band: float, lower_band: float) -> float:
        """Calculate price position within Bollinger Bands (0-1)"""
        if upper_band == lower_band:
            return 0.5
        return (price - lower_band) / (upper_band - lower_band)
    
    def detect_squeeze(self, prices: pd.Series) -> bool:
        """Detect Bollinger Bands squeeze"""
        if len(prices) < self.period + 10:
            return False
            
        # Calculate bandwidth for recent days
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
            
        # If bandwidth is below threshold and contracting
        current_bandwidth = recent_bandwidths[0]
        avg_bandwidth = np.mean(recent_bandwidths)
        
        return current_bandwidth < self.squeeze_threshold and current_bandwidth <= avg_bandwidth
    
    def calculate_volume_confirmation(self, data: pd.DataFrame) -> float:
        """Calculate volume confirmation indicator"""
        if 'volume' not in data.columns or len(data) < 20:
            return 1.0
            
        volume = data['volume']
        volume_sma = volume.rolling(window=20).mean()
        current_volume_ratio = volume.iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1.0
        
        return current_volume_ratio
    
    def generate_signal(self, data: pd.DataFrame, portfolio: Portfolio, 
                       current_date: str, **kwargs) -> Signal:
        """
        Bollinger Bands strategy logic:
        Mean reversion mode:
        - Buy when price touches lower band
        - Sell when price touches upper band
        Breakout mode:
        - Buy when price breaks above upper band
        - Sell when price breaks below lower band
        """
        prices = data['close']
        
        # Calculate Bollinger Bands
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
        
        # Calculate Bollinger Bands indicators
        bb_position = self.calculate_bb_position(current_price, upper_band, lower_band)
        bandwidth = self.calculate_bandwidth(upper_band, lower_band, middle_band)
        is_squeeze = self.detect_squeeze(prices)
        
        # Calculate volume confirmation
        volume_ratio = self.calculate_volume_confirmation(data) if self.volume_confirmation else 1.0
        
        # Calculate price momentum
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
        """Mean reversion signal generation"""
        
        # Oversold buy signal (price near or below lower band)
        if bb_position <= 0.1 and position_ratio < 0.7:
            signal_strength = (0.1 - bb_position) / 0.1
            
            # Bollinger Bands squeeze enhances signal
            if is_squeeze:
                signal_strength *= 1.3
                
            # Volume confirmation
            if volume_ratio > 1.2:
                signal_strength *= 1.2
                
            # Negative momentum confirmation (price falling but near support)
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
        
        # Overbought sell signal (price near or above upper band)
        elif bb_position >= 0.9 and portfolio.stock > 0:
            signal_strength = (bb_position - 0.9) / 0.1
            
            # Bollinger Bands squeeze enhances signal
            if is_squeeze:
                signal_strength *= 1.3
                
            # Volume confirmation
            if volume_ratio > 1.2:
                signal_strength *= 1.2
                
            # Positive momentum confirmation (price rising but near resistance)
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
        
        # Return to middle band - partial profit taking
        elif 0.4 < bb_position < 0.6 and portfolio.stock > 0:
            if self.last_signal == 'buy':
                quantity = int(portfolio.stock * 0.25)  # Take profit 25%
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
        """Breakout signal generation"""
        
        # Upward breakout buy signal
        if bb_position > 1.0 and position_ratio < 0.8:
            signal_strength = min(2.0, bb_position - 1.0)
            
            # Bollinger Bands squeeze followed by breakout enhances signal
            if is_squeeze:
                signal_strength *= 1.5
                
            # Volume confirmation (breakout needs volume)
            if volume_ratio > 1.5:
                signal_strength *= 1.3
            elif volume_ratio < 1.0:
                signal_strength *= 0.7  # Low volume breakout reduces signal strength
                
            # Positive momentum confirmation
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
        
        # Downward breakout sell signal
        elif bb_position < 0.0 and portfolio.stock > 0:
            signal_strength = min(2.0, abs(bb_position))
            
            # Bollinger Bands squeeze followed by breakout enhances signal
            if is_squeeze:
                signal_strength *= 1.5
                
            # Volume confirmation
            if volume_ratio > 1.5:
                signal_strength *= 1.3
            elif volume_ratio < 1.0:
                signal_strength *= 0.7
                
            # Negative momentum confirmation
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
        
        # False breakout pullback
        elif 0.8 < bb_position < 1.0 and self.last_signal == 'buy':
            # Possible false breakout, consider reducing position
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