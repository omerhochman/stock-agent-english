import pandas as pd

from .base_strategy import BaseStrategy, Portfolio, Signal


class BuyHoldStrategy(BaseStrategy):
    """
    Buy and Hold Strategy
    Classic long-term investment benchmark strategy
    """

    def __init__(self, allocation_ratio: float = 1.0, **kwargs):
        # Extract name from kwargs if provided, otherwise use default
        name = kwargs.pop("name", "Buy-and-Hold")
        super().__init__(name, **kwargs)
        self.allocation_ratio = allocation_ratio  # Capital allocation ratio
        self.initial_purchase_made = False

    def generate_signal(
        self, data: pd.DataFrame, portfolio: Portfolio, current_date: str, **kwargs
    ) -> Signal:
        """
        Buy and hold strategy logic:
        - Buy and hold on first trade
        - Hold all the time after that
        """
        if not self.initial_purchase_made and portfolio.cash > 0:
            # First investment: buy
            current_price = data.iloc[-1]["close"]
            max_shares = int((portfolio.cash * self.allocation_ratio) / current_price)

            if max_shares > 0:
                self.initial_purchase_made = True
                return Signal(
                    action="buy",
                    quantity=max_shares,
                    confidence=1.0,
                    reasoning="Initial buy-and-hold purchase",
                    metadata={"allocation_ratio": self.allocation_ratio},
                )

        # Already invested or cannot invest: hold
        return Signal(
            action="hold",
            quantity=0,
            confidence=1.0,
            reasoning="Buy-and-hold strategy maintains position",
        )
