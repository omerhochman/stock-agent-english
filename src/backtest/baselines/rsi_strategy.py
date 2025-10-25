import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Portfolio, Signal


class RSIStrategy(BaseStrategy):
    """
    RSI Strategy
    Overbought/oversold strategy based on Relative Strength Index (RSI)
    Reference: Wilder (1978) - New Concepts in Technical Trading Systems
    """

    def __init__(
        self,
        rsi_period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        smoothing_period: int = 3,
        **kwargs,
    ):
        # Extract name from kwargs if provided, otherwise use default
        name = kwargs.pop("name", "RSI-Strategy")
        super().__init__(name, **kwargs)
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.smoothing_period = smoothing_period
        self.last_signal = None
        self.signal_count = 0

    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI indicator"""
        if len(prices) < self.rsi_period + 1:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    def calculate_rsi_divergence(self, prices: pd.Series, rsi_values: pd.Series) -> str:
        """Calculate RSI divergence"""
        if len(prices) < 10 or len(rsi_values) < 10:
            return "none"

        # Find recent highs and lows
        recent_prices = prices.tail(10)
        recent_rsi = rsi_values.tail(10)

        price_high_idx = recent_prices.idxmax()
        price_low_idx = recent_prices.idxmin()
        rsi_high_idx = recent_rsi.idxmax()
        rsi_low_idx = recent_rsi.idxmin()

        # Check bullish divergence (price makes new low, RSI doesn't make new low)
        if (
            price_low_idx == recent_prices.index[-1]
            and rsi_low_idx != recent_rsi.index[-1]
            and recent_rsi.iloc[-1] > recent_rsi.min()
        ):
            return "bullish"

        # Check bearish divergence (price makes new high, RSI doesn't make new high)
        if (
            price_high_idx == recent_prices.index[-1]
            and rsi_high_idx != recent_rsi.index[-1]
            and recent_rsi.iloc[-1] < recent_rsi.max()
        ):
            return "bearish"

        return "none"

    def calculate_stochastic_rsi(
        self, rsi_values: pd.Series, period: int = 14
    ) -> float:
        """Calculate Stochastic RSI"""
        if len(rsi_values) < period:
            return 50.0

        recent_rsi = rsi_values.tail(period)
        lowest_rsi = recent_rsi.min()
        highest_rsi = recent_rsi.max()

        if highest_rsi == lowest_rsi:
            return 50.0

        stoch_rsi = (
            (rsi_values.iloc[-1] - lowest_rsi) / (highest_rsi - lowest_rsi) * 100
        )
        return stoch_rsi

    def generate_signal(
        self, data: pd.DataFrame, portfolio: Portfolio, current_date: str, **kwargs
    ) -> Signal:
        """
        RSI strategy logic:
        - Consider buying when RSI < 30 (oversold)
        - Consider selling when RSI > 70 (overbought)
        - Combine with price momentum and volume confirmation
        - Use RSI divergence to enhance signals
        """
        prices = data["close"]

        # Calculate RSI
        rsi_series = prices.rolling(window=self.rsi_period + 1).apply(
            lambda x: self._single_rsi(x), raw=False
        )
        current_rsi = self.calculate_rsi(prices)

        # Calculate Stochastic RSI
        stoch_rsi = self.calculate_stochastic_rsi(rsi_series.dropna())

        # Calculate RSI divergence
        divergence = self.calculate_rsi_divergence(prices, rsi_series.dropna())

        # Calculate price momentum
        if len(prices) >= 5:
            momentum = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100
        else:
            momentum = 0

        # Calculate volume trend
        if "volume" in data.columns and len(data) >= 10:
            volume_ma_short = data["volume"].tail(5).mean()
            volume_ma_long = data["volume"].tail(10).mean()
            volume_trend = volume_ma_short / volume_ma_long if volume_ma_long > 0 else 1
        else:
            volume_trend = 1

        current_price = prices.iloc[-1]
        position_ratio = (portfolio.stock * current_price) / (
            portfolio.cash + portfolio.stock * current_price
        )

        # Oversold buy signal
        if current_rsi < self.oversold and position_ratio < 0.8:
            # Enhanced condition: RSI continues to decline or bullish divergence appears
            signal_strength = (self.oversold - current_rsi) / self.oversold

            # Divergence enhances signal
            if divergence == "bullish":
                signal_strength *= 1.5

            # Volume confirmation
            if volume_trend > 1.2:
                signal_strength *= 1.2

            # Stochastic RSI confirmation
            if stoch_rsi < 20:
                signal_strength *= 1.1

            max_investment = portfolio.cash * min(0.4, signal_strength * 0.3)
            quantity = int(max_investment / current_price)

            if quantity > 0:
                confidence = min(0.9, 0.6 + signal_strength * 0.2)
                return Signal(
                    action="buy",
                    quantity=quantity,
                    confidence=confidence,
                    reasoning=f"RSI oversold: {current_rsi:.1f}, StochRSI: {stoch_rsi:.1f}, Divergence: {divergence}",
                    metadata={
                        "rsi": current_rsi,
                        "stoch_rsi": stoch_rsi,
                        "divergence": divergence,
                        "signal_strength": signal_strength,
                        "volume_trend": volume_trend,
                    },
                )

        # Overbought sell signal
        elif current_rsi > self.overbought and portfolio.stock > 0:
            signal_strength = (current_rsi - self.overbought) / (100 - self.overbought)

            # Divergence enhances signal
            if divergence == "bearish":
                signal_strength *= 1.5

            # Volume confirmation
            if volume_trend > 1.2:
                signal_strength *= 1.2

            # Stochastic RSI confirmation
            if stoch_rsi > 80:
                signal_strength *= 1.1

            quantity = min(
                portfolio.stock, int(portfolio.stock * min(0.6, signal_strength * 0.4))
            )

            if quantity > 0:
                confidence = min(0.9, 0.6 + signal_strength * 0.2)
                return Signal(
                    action="sell",
                    quantity=quantity,
                    confidence=confidence,
                    reasoning=f"RSI overbought: {current_rsi:.1f}, StochRSI: {stoch_rsi:.1f}, Divergence: {divergence}",
                    metadata={
                        "rsi": current_rsi,
                        "stoch_rsi": stoch_rsi,
                        "divergence": divergence,
                        "signal_strength": signal_strength,
                        "volume_trend": volume_trend,
                    },
                )

        # Neutral zone - partial profit taking
        elif 40 < current_rsi < 60 and portfolio.stock > 0:
            # If RSI returns to neutral zone, consider partial profit taking
            if self.last_signal == "buy" and current_rsi > 50:
                quantity = int(portfolio.stock * 0.2)  # Take profit on 20%
                return Signal(
                    action="sell",
                    quantity=quantity,
                    confidence=0.6,
                    reasoning=f"RSI neutral zone profit taking: {current_rsi:.1f}",
                    metadata={"rsi": current_rsi, "action_type": "profit_taking"},
                )

        # Record last signal
        if current_rsi < self.oversold:
            self.last_signal = "buy"
        elif current_rsi > self.overbought:
            self.last_signal = "sell"

        # Hold signal
        return Signal(
            action="hold",
            quantity=0,
            confidence=0.5,
            reasoning=f"RSI neutral: {current_rsi:.1f}",
            metadata={
                "rsi": current_rsi,
                "stoch_rsi": stoch_rsi,
                "divergence": divergence,
            },
        )

    def _single_rsi(self, prices):
        """Helper function to calculate single RSI value"""
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
