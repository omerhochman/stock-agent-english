import pandas as pd

from .base_strategy import BaseStrategy, Portfolio, Signal


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy
    Investment strategy based on price reversion to long-term mean
    Reference: De Bondt and Thaler (1985)
    """

    def __init__(
        self,
        lookback_period: int = 252,
        z_threshold: float = 2.0,
        mean_period: int = 50,
        exit_threshold: float = 0.5,
        **kwargs,
    ):
        # Extract name from kwargs if provided, otherwise use default
        name = kwargs.pop("name", "Mean-Reversion")
        super().__init__(name, **kwargs)
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold  # Z-score threshold
        self.mean_period = mean_period  # Mean calculation period
        self.exit_threshold = exit_threshold  # Exit threshold
        self.position_entry_date = None

    def calculate_z_score(self, prices: pd.Series) -> float:
        """Calculate Z-score of prices"""
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
        """Calculate RSI indicator"""
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

    def generate_signal(
        self, data: pd.DataFrame, portfolio: Portfolio, current_date: str, **kwargs
    ) -> Signal:
        """
        Mean reversion strategy logic:
        - Calculate Z-score and RSI
        - Take contrarian action when price deviates significantly from mean
        - Close position when price returns near mean
        """
        prices = data["close"]
        z_score = self.calculate_z_score(prices)
        rsi = self.calculate_rsi(prices)

        # Calculate Bollinger Bands
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

        # Calculate volatility
        volatility = prices.pct_change().tail(21).std() if len(prices) > 21 else 0

        position_ratio = (portfolio.stock * prices.iloc[-1]) / (
            portfolio.cash + portfolio.stock * prices.iloc[-1]
        )

        # Oversold signal - buy
        if (
            z_score < -self.z_threshold or rsi < 30 or bb_position < 0.1
        ) and position_ratio < 0.7:
            # Price significantly below mean, oversold
            signal_strength = abs(z_score) / self.z_threshold
            max_investment = portfolio.cash * min(0.4, signal_strength * 0.3)
            quantity = int(max_investment / prices.iloc[-1])

            if quantity > 0:
                confidence = min(0.9, 0.5 + signal_strength * 0.3)
                return Signal(
                    action="buy",
                    quantity=quantity,
                    confidence=confidence,
                    reasoning=f"Oversold: Z-score {z_score:.2f}, RSI {rsi:.1f}, BB {bb_position:.2f}",
                    metadata={
                        "z_score": z_score,
                        "rsi": rsi,
                        "bb_position": bb_position,
                        "signal_strength": signal_strength,
                    },
                )

        # Overbought signal - sell
        elif (
            z_score > self.z_threshold or rsi > 70 or bb_position > 0.9
        ) and portfolio.stock > 0:
            # Price significantly above mean, overbought
            signal_strength = abs(z_score) / self.z_threshold
            quantity = min(
                portfolio.stock, int(portfolio.stock * min(0.6, signal_strength * 0.4))
            )

            if quantity > 0:
                confidence = min(0.9, 0.5 + signal_strength * 0.3)
                return Signal(
                    action="sell",
                    quantity=quantity,
                    confidence=confidence,
                    reasoning=f"Overbought: Z-score {z_score:.2f}, RSI {rsi:.1f}, BB {bb_position:.2f}",
                    metadata={
                        "z_score": z_score,
                        "rsi": rsi,
                        "bb_position": bb_position,
                        "signal_strength": signal_strength,
                    },
                )

        # Reversion signal - close position
        elif (
            abs(z_score) < self.exit_threshold
            and 30 < rsi < 70
            and 0.3 < bb_position < 0.7
        ):
            if portfolio.stock > 0:
                # Partial profit taking
                quantity = int(portfolio.stock * 0.3)
                return Signal(
                    action="sell",
                    quantity=quantity,
                    confidence=0.6,
                    reasoning=f"Mean reversion: Z-score {z_score:.2f} approaching neutral",
                    metadata={
                        "z_score": z_score,
                        "rsi": rsi,
                        "bb_position": bb_position,
                    },
                )

        # Hold signal
        return Signal(
            action="hold",
            quantity=0,
            confidence=0.5,
            reasoning=f"Neutral: Z-score {z_score:.2f}, RSI {rsi:.1f}",
            metadata={"z_score": z_score, "rsi": rsi, "bb_position": bb_position},
        )
