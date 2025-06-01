import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal, Portfolio

class MACDStrategy(BaseStrategy):
    """
    MACD策略
    基于MACD指标的趋势跟踪和背离策略
    参考：Gerald Appel (1979) - Moving Average Convergence Divergence
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
        """计算指数移动平均线"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_macd(self, prices: pd.Series):
        """计算MACD指标"""
        if len(prices) < self.slow_period:
            return None, None, None
            
        # 计算快线和慢线EMA
        ema_fast = self.calculate_ema(prices, self.fast_period)
        ema_slow = self.calculate_ema(prices, self.slow_period)
        
        # 计算MACD线
        macd_line = ema_fast - ema_slow
        
        # 计算信号线
        signal_line = self.calculate_ema(macd_line, self.signal_period)
        
        # 计算MACD柱状图
        histogram = macd_line - signal_line
        
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    
    def detect_macd_divergence(self, prices: pd.Series, macd_line: pd.Series) -> str:
        """检测MACD背离"""
        if len(prices) < 20 or len(macd_line) < 20:
            return "none"
            
        # 获取最近20天的数据
        recent_prices = prices.tail(20)
        recent_macd = macd_line.tail(20)
        
        # 寻找价格和MACD的高点和低点
        price_peaks = []
        macd_peaks = []
        price_troughs = []
        macd_troughs = []
        
        # 简化的峰谷检测
        for i in range(2, len(recent_prices) - 2):
            # 检测峰值
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
                
            # 检测谷值
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
        
        # 检查看涨背离（价格创新低，MACD未创新低）
        if len(price_troughs) >= 2 and len(macd_troughs) >= 2:
            latest_price_trough = price_troughs[-1]
            prev_price_trough = price_troughs[-2]
            latest_macd_trough = macd_troughs[-1]
            prev_macd_trough = macd_troughs[-2]
            
            if (latest_price_trough[1] < prev_price_trough[1] and 
                latest_macd_trough[1] > prev_macd_trough[1]):
                return "bullish"
        
        # 检查看跌背离（价格创新高，MACD未创新高）
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
        """计算趋势强度"""
        if len(prices) < 20:
            return 0.0
            
        # 使用线性回归斜率衡量趋势强度
        x = np.arange(len(prices.tail(20)))
        y = prices.tail(20).values
        
        # 计算线性回归斜率
        slope = np.polyfit(x, y, 1)[0]
        
        # 标准化斜率
        price_range = prices.tail(20).max() - prices.tail(20).min()
        if price_range == 0:
            return 0.0
            
        normalized_slope = slope / price_range * 20  # 20天的标准化斜率
        
        return normalized_slope
    
    def calculate_macd_momentum(self, macd_line: pd.Series, signal_line: pd.Series) -> float:
        """计算MACD动量"""
        if len(macd_line) < 5 or len(signal_line) < 5:
            return 0.0
            
        # MACD线相对于信号线的动量
        current_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
        prev_diff = macd_line.iloc[-5] - signal_line.iloc[-5]
        
        momentum = current_diff - prev_diff
        return momentum
    
    def generate_signal(self, data: pd.DataFrame, portfolio: Portfolio, 
                       current_date: str, **kwargs) -> Signal:
        """
        MACD策略逻辑：
        - MACD线上穿信号线时买入
        - MACD线下穿信号线时卖出
        - 结合背离和趋势确认
        - 使用柱状图确认信号强度
        """
        prices = data['close']
        
        # 计算MACD指标
        macd_result = self.calculate_macd(prices)
        if macd_result[0] is None:
            return Signal(
                action='hold',
                quantity=0,
                confidence=0.5,
                reasoning="Insufficient data for MACD calculation"
            )
            
        macd_line, signal_line, histogram = macd_result
        
        # 计算历史MACD数据用于背离检测
        if len(prices) >= self.slow_period + 10:
            ema_fast = self.calculate_ema(prices, self.fast_period)
            ema_slow = self.calculate_ema(prices, self.slow_period)
            macd_series = ema_fast - ema_slow
            
            # 检测背离
            divergence = self.detect_macd_divergence(prices, macd_series)
        else:
            divergence = "none"
        
        # 计算趋势强度
        trend_strength = self.calculate_trend_strength(prices)
        
        # 计算MACD动量
        if len(prices) >= self.slow_period + 5:
            ema_fast = self.calculate_ema(prices, self.fast_period)
            ema_slow = self.calculate_ema(prices, self.slow_period)
            macd_series = ema_fast - ema_slow
            signal_series = self.calculate_ema(macd_series, self.signal_period)
            macd_momentum = self.calculate_macd_momentum(macd_series, signal_series)
        else:
            macd_momentum = 0.0
        
        # 检测MACD交叉
        if len(prices) >= self.slow_period + self.signal_period + 1:
            ema_fast = self.calculate_ema(prices, self.fast_period)
            ema_slow = self.calculate_ema(prices, self.slow_period)
            macd_series = ema_fast - ema_slow
            signal_series = self.calculate_ema(macd_series, self.signal_period)
            
            # 当前和前一天的MACD状态
            current_above = macd_series.iloc[-1] > signal_series.iloc[-1]
            prev_above = macd_series.iloc[-2] > signal_series.iloc[-2]
            
            # 检测金叉和死叉
            golden_cross = current_above and not prev_above  # 金叉
            death_cross = not current_above and prev_above   # 死叉
        else:
            golden_cross = False
            death_cross = False
        
        current_price = prices.iloc[-1]
        position_ratio = (portfolio.stock * current_price) / (portfolio.cash + portfolio.stock * current_price)
        
        # 买入信号（金叉）
        if golden_cross and position_ratio < 0.8:
            signal_strength = 1.0
            
            # MACD在零轴上方增强信号
            if macd_line > 0:
                signal_strength *= 1.3
                
            # 柱状图确认
            if histogram > self.histogram_threshold:
                signal_strength *= 1.2
                
            # 背离增强信号
            if divergence == "bullish":
                signal_strength *= 1.5
                
            # 趋势确认
            if self.trend_confirmation and trend_strength > 0.02:
                signal_strength *= 1.2
            elif self.trend_confirmation and trend_strength < -0.02:
                signal_strength *= 0.7  # 逆趋势信号减弱
                
            # MACD动量确认
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
        
        # 卖出信号（死叉）
        elif death_cross and portfolio.stock > 0:
            signal_strength = 1.0
            
            # MACD在零轴下方增强信号
            if macd_line < 0:
                signal_strength *= 1.3
                
            # 柱状图确认
            if histogram < -self.histogram_threshold:
                signal_strength *= 1.2
                
            # 背离增强信号
            if divergence == "bearish":
                signal_strength *= 1.5
                
            # 趋势确认
            if self.trend_confirmation and trend_strength < -0.02:
                signal_strength *= 1.2
            elif self.trend_confirmation and trend_strength > 0.02:
                signal_strength *= 0.7  # 逆趋势信号减弱
                
            # MACD动量确认
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
        
        # 背离信号（无交叉时）
        elif divergence == "bullish" and portfolio.stock == 0 and position_ratio < 0.6:
            # 看涨背离买入信号
            max_investment = portfolio.cash * 0.25  # 较小仓位
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
            # 看跌背离卖出信号
            quantity = int(portfolio.stock * 0.3)  # 部分减仓
            
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
        
        # 零轴突破信号
        elif macd_line > 0 and len(prices) >= self.slow_period + 1:
            ema_fast = self.calculate_ema(prices, self.fast_period)
            ema_slow = self.calculate_ema(prices, self.slow_period)
            macd_series = ema_fast - ema_slow
            
            # 检查是否刚突破零轴
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
        
        # 记录最后信号
        if golden_cross:
            self.last_signal = 'buy'
        elif death_cross:
            self.last_signal = 'sell'
            
        # 持有信号
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