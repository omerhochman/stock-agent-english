import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal, Portfolio

class MACDStrategy(BaseStrategy):
    """
    MACD Strategy
    Trend following and divergence strategy based on MACD indicator
    Reference: Gerald Appel (1979) - Moving Average Convergence Divergence
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9, histogram_threshold: float = 0.0,
                 trend_confirmation: bool = True, **kwargs):
        # Extract name from kwargs if provided, otherwise use default
        name = kwargs.pop('name', 'MACD-Strategy')
        super().__init__(name, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.histogram_threshold = histogram_threshold
        self.trend_confirmation = trend_confirmation
        self.last_signal = None
        self.last_crossover = None
        
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate exponential moving average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_macd(self, prices: pd.Series):
        """Calculate MACD indicator"""
        if len(prices) < self.slow_period:
            return None, None, None
            
        # Calculate fast and slow EMA
        ema_fast = self.calculate_ema(prices, self.fast_period)
        ema_slow = self.calculate_ema(prices, self.slow_period)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = self.calculate_ema(macd_line, self.signal_period)
        
        # Calculate MACD histogram
        histogram = macd_line - signal_line
        
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    
    def detect_macd_divergence(self, prices: pd.Series, macd_line: pd.Series) -> str:
        """Detect MACD divergence"""
        if len(prices) < 20 or len(macd_line) < 20:
            return "none"
            
        # Get recent 20 days of data
        recent_prices = prices.tail(20)
        recent_macd = macd_line.tail(20)
        
        # Find peaks and troughs in price and MACD
        price_peaks = []
        macd_peaks = []
        price_troughs = []
        macd_troughs = []
        
        # Simplified peak and trough detection
        for i in range(2, len(recent_prices) - 2):
            # Detect peaks
            if (recent_prices.iloc[i] > recent_prices.iloc[i-1] and 
                recent_prices.iloc[i] > recent_prices.iloc[i+1] and
                recent_prices.iloc[i] > recent_prices.iloc[i-2] and
                recent_prices.iloc[i] > recent_prices.iloc[i+2]):
                price_peaks.append((i, recent_prices.iloc[i]))
                
            if (recent_macd.iloc[i] > recent_macd.iloc[i-1] and 
                recent_macd.iloc[i] > recent_macd.iloc[i+1] and
                recent_macd.iloc[i] > recent_macd.iloc[i-2] and
                recent_macd.iloc[i] > recent_macd.iloc[i+2]):
                macd_peaks.append((i, recent_macd.iloc[i]))
                
            # Detect troughs
            if (recent_prices.iloc[i] < recent_prices.iloc[i-1] and 
                recent_prices.iloc[i] < recent_prices.iloc[i+1] and
                recent_prices.iloc[i] < recent_prices.iloc[i-2] and
                recent_prices.iloc[i] < recent_prices.iloc[i+2]):
                price_troughs.append((i, recent_prices.iloc[i]))
                
            if (recent_macd.iloc[i] < recent_macd.iloc[i-1] and 
                recent_macd.iloc[i] < recent_macd.iloc[i+1] and
                recent_macd.iloc[i] < recent_macd.iloc[i-2] and
                recent_macd.iloc[i] < recent_macd.iloc[i+2]):
                macd_troughs.append((i, recent_macd.iloc[i]))
        
        # Check for bullish divergence (price makes new low, MACD does not)
        if len(price_troughs) >= 2 and len(macd_troughs) >= 2:
            latest_price_trough = price_troughs[-1]
            prev_price_trough = price_troughs[-2]
            latest_macd_trough = macd_troughs[-1]
            prev_macd_trough = macd_troughs[-2]
            
            if (latest_price_trough[1] < prev_price_trough[1] and 
                latest_macd_trough[1] > prev_macd_trough[1]):
                return "bullish"
        
        # Check for bearish divergence (price makes new high, MACD does not)
        if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
            latest_price_peak = price_peaks[-1]
            prev_price_peak = price_peaks[-2]
            latest_macd_peak = macd_peaks[-1]
            prev_macd_peak = macd_peaks[-2]
            
            if (latest_price_peak[1] > prev_price_peak[1] and 
                latest_macd_peak[1] < prev_macd_peak[1]):
                return "bearish"
        
        return "none"
    
    def calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength"""
        if len(prices) < 20:
            return 0.0
            
        # Use linear regression slope to measure trend strength
        x = np.arange(len(prices.tail(20)))
        y = prices.tail(20).values
        
        # Calculate linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope
        price_range = prices.tail(20).max() - prices.tail(20).min()
        if price_range == 0:
            return 0.0
            
        normalized_slope = slope / price_range * 20  # 20-day normalized slope
        
        return normalized_slope
    
    def calculate_macd_momentum(self, macd_line: pd.Series, signal_line: pd.Series) -> float:
        """Calculate MACD momentum"""
        if len(macd_line) < 5 or len(signal_line) < 5:
            return 0.0
            
        # MACD line momentum relative to signal line
        current_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
        prev_diff = macd_line.iloc[-5] - signal_line.iloc[-5]
        
        momentum = current_diff - prev_diff
        return momentum
    
    def generate_signal(self, data: pd.DataFrame, portfolio: Portfolio, 
                       current_date: str, **kwargs) -> Signal:
        """
        MACD strategy logic:
        - Buy when MACD line crosses above signal line
        - Sell when MACD line crosses below signal line
        - Combine with divergence and trend confirmation
        - Use histogram to confirm signal strength
        """
        prices = data['close']
        
        # Calculate MACD indicator
        macd_result = self.calculate_macd(prices)
        if macd_result[0] is None:
            return Signal(
                action='hold',
                quantity=0,
                confidence=0.5,
                reasoning="Insufficient data for MACD calculation"
            )
            
        macd_line, signal_line, histogram = macd_result
        
        # Calculate historical MACD data for divergence detection
        if len(prices) >= self.slow_period + 10:
            ema_fast = self.calculate_ema(prices, self.fast_period)
            ema_slow = self.calculate_ema(prices, self.slow_period)
            macd_series = ema_fast - ema_slow
            
            # Detect divergence
            divergence = self.detect_macd_divergence(prices, macd_series)
        else:
            divergence = "none"
        
        # Calculate trend strength
        trend_strength = self.calculate_trend_strength(prices)
        
        # Calculate MACD momentum
        if len(prices) >= self.slow_period + 5:
            ema_fast = self.calculate_ema(prices, self.fast_period)
            ema_slow = self.calculate_ema(prices, self.slow_period)
            macd_series = ema_fast - ema_slow
            signal_series = self.calculate_ema(macd_series, self.signal_period)
            macd_momentum = self.calculate_macd_momentum(macd_series, signal_series)
        else:
            macd_momentum = 0.0
        
        # Detect MACD crossover
        if len(prices) >= self.slow_period + self.signal_period + 1:
            ema_fast = self.calculate_ema(prices, self.fast_period)
            ema_slow = self.calculate_ema(prices, self.slow_period)
            macd_series = ema_fast - ema_slow
            signal_series = self.calculate_ema(macd_series, self.signal_period)
            
            # Current and previous day MACD status
            current_above = macd_series.iloc[-1] > signal_series.iloc[-1]
            prev_above = macd_series.iloc[-2] > signal_series.iloc[-2]
            
            # Detect golden cross and death cross
            golden_cross = current_above and not prev_above  # Golden cross
            death_cross = not current_above and prev_above   # Death cross
        else:
            golden_cross = False
            death_cross = False
        
        current_price = prices.iloc[-1]
        position_ratio = (portfolio.stock * current_price) / (portfolio.cash + portfolio.stock * current_price)
        
        # Buy signal (golden cross)
        if golden_cross and position_ratio < 0.8:
            signal_strength = 1.0
            
            # MACD above zero line enhances signal
            if macd_line > 0:
                signal_strength *= 1.3
                
            # Histogram confirmation
            if histogram > self.histogram_threshold:
                signal_strength *= 1.2
                
            # Divergence enhances signal
            if divergence == "bullish":
                signal_strength *= 1.5
                
            # Trend confirmation
            if self.trend_confirmation and trend_strength > 0.02:
                signal_strength *= 1.2
            elif self.trend_confirmation and trend_strength < -0.02:
                signal_strength *= 0.7  # Counter-trend signal weakened
                
            # MACD momentum confirmation
            if macd_momentum > 0:
                signal_strength *= 1.1
                
            max_investment = portfolio.cash * min(0.5, signal_strength * 0.3)
            quantity = int(max_investment / current_price)
            
            if quantity > 0:
                confidence = min(0.9, 0.6 + signal_strength * 0.2)
                self.last_crossover = "golden"
                return Signal(
                    action='buy',
                    quantity=quantity,
                    confidence=confidence,
                    reasoning=f"MACD golden cross: {macd_line:.4f} > {signal_line:.4f}, divergence: {divergence}",
                    metadata={
                        'macd_line': macd_line,
                        'signal_line': signal_line,
                        'histogram': histogram,
                        'divergence': divergence,
                        'trend_strength': trend_strength,
                        'signal_strength': signal_strength,
                        'crossover_type': 'golden'
                    }
                )
        
        # Sell signal (death cross)
        elif death_cross and portfolio.stock > 0:
            signal_strength = 1.0
            
            # MACD below zero line enhances signal
            if macd_line < 0:
                signal_strength *= 1.3
                
            # Histogram confirmation
            if histogram < -self.histogram_threshold:
                signal_strength *= 1.2
                
            # Divergence enhances signal
            if divergence == "bearish":
                signal_strength *= 1.5
                
            # Trend confirmation
            if self.trend_confirmation and trend_strength < -0.02:
                signal_strength *= 1.2
            elif self.trend_confirmation and trend_strength > 0.02:
                signal_strength *= 0.7  # Counter-trend signal weakened
                
            # MACD momentum confirmation
            if macd_momentum < 0:
                signal_strength *= 1.1
                
            quantity = min(portfolio.stock, int(portfolio.stock * min(0.7, signal_strength * 0.4)))
            
            if quantity > 0:
                confidence = min(0.9, 0.6 + signal_strength * 0.2)
                self.last_crossover = "death"
                return Signal(
                    action='sell',
                    quantity=quantity,
                    confidence=confidence,
                    reasoning=f"MACD death cross: {macd_line:.4f} < {signal_line:.4f}, divergence: {divergence}",
                    metadata={
                        'macd_line': macd_line,
                        'signal_line': signal_line,
                        'histogram': histogram,
                        'divergence': divergence,
                        'trend_strength': trend_strength,
                        'signal_strength': signal_strength,
                        'crossover_type': 'death'
                    }
                )
        
        # Divergence signal (when no crossover)
        elif divergence == "bullish" and portfolio.stock == 0 and position_ratio < 0.6:
            # Bullish divergence buy signal
            max_investment = portfolio.cash * 0.25  # Smaller position
            quantity = int(max_investment / current_price)
            
            if quantity > 0:
                return Signal(
                    action='buy',
                    quantity=quantity,
                    confidence=0.7,
                    reasoning=f"MACD bullish divergence: {macd_line:.4f}",
                    metadata={
                        'macd_line': macd_line,
                        'signal_line': signal_line,
                        'divergence': divergence,
                        'signal_type': 'divergence'
                    }
                )
        
        elif divergence == "bearish" and portfolio.stock > 0:
            # Bearish divergence sell signal
            quantity = int(portfolio.stock * 0.3)  # Partial position reduction
            
            if quantity > 0:
                return Signal(
                    action='sell',
                    quantity=quantity,
                    confidence=0.7,
                    reasoning=f"MACD bearish divergence: {macd_line:.4f}",
                    metadata={
                        'macd_line': macd_line,
                        'signal_line': signal_line,
                        'divergence': divergence,
                        'signal_type': 'divergence'
                    }
                )
        
        # Zero line breakout signal
        elif macd_line > 0 and len(prices) >= self.slow_period + 1:
            ema_fast = self.calculate_ema(prices, self.fast_period)
            ema_slow = self.calculate_ema(prices, self.slow_period)
            macd_series = ema_fast - ema_slow
            
            # Check if just broke through zero line
            if macd_series.iloc[-2] <= 0 and macd_line > 0 and position_ratio < 0.6:
                max_investment = portfolio.cash * 0.2
                quantity = int(max_investment / current_price)
                
                if quantity > 0:
                    return Signal(
                        action='buy',
                        quantity=quantity,
                        confidence=0.6,
                        reasoning=f"MACD zero line breakout: {macd_line:.4f}",
                        metadata={
                            'macd_line': macd_line,
                            'signal_line': signal_line,
                            'signal_type': 'zero_breakout'
                        }
                    )
        
        # Record last signal
        if golden_cross:
            self.last_signal = 'buy'
        elif death_cross:
            self.last_signal = 'sell'
            
        # Hold signal
        return Signal(
            action='hold',
            quantity=0,
            confidence=0.5,
            reasoning=f"MACD hold: {macd_line:.4f} vs {signal_line:.4f}",
            metadata={
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram,
                'divergence': divergence,
                'trend_strength': trend_strength
            }
        ) 