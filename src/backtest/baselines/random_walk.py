import time
from typing import Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Portfolio, Signal


class RandomWalkStrategy(BaseStrategy):
    """
    Random Walk Strategy
    Used as baseline strategy for control group
    True randomness - different results each run
    """

    def __init__(
        self,
        trade_probability: float = 0.1,
        max_position_ratio: float = 0.5,
        truly_random: bool = True,
        **kwargs,
    ):
        # Extract name from kwargs if provided, otherwise use default
        name = kwargs.pop("name", "Random-Walk")
        super().__init__(name, **kwargs)
        self.trade_probability = trade_probability
        self.max_position_ratio = max_position_ratio
        self.truly_random = truly_random

        if truly_random:
            # Use nanosecond precision of current time as seed to ensure true randomness
            seed = int(time.time() * 1000000) % (2**32)
            self.rng = np.random.RandomState(seed)
            print(f"RandomWalk strategy using random seed: {seed}")
        else:
            # If reproducible results are needed (for debugging), can set fixed seed
            self.rng = np.random.RandomState(42)
            print("RandomWalk strategy using fixed seed: 42 (debug mode)")

    def generate_signal(
        self, data: pd.DataFrame, portfolio: Portfolio, current_date: str, **kwargs
    ) -> Signal:
        """
        Random walk strategy logic:
        - Randomly decide whether to trade
        - Randomly decide to buy or sell
        - Randomly decide trade quantity

        True randomness simulates market unpredictability
        """
        # Randomly decide whether to trade
        if self.rng.random() > self.trade_probability:
            return Signal(
                action="hold",
                quantity=0,
                confidence=0.5,
                reasoning="Random walk: no trade decision",
            )

        current_price = data["close"].iloc[-1]
        position_ratio = (portfolio.stock * current_price) / (
            portfolio.cash + portfolio.stock * current_price
        )

        # Randomly decide to buy or sell
        if self.rng.random() < 0.5 and position_ratio < self.max_position_ratio:
            # Random buy
            max_investment = portfolio.cash * self.rng.uniform(0.1, 0.3)
            quantity = int(max_investment / current_price)

            if quantity > 0:
                return Signal(
                    action="buy",
                    quantity=quantity,
                    confidence=0.5,
                    reasoning="Random walk: random buy decision",
                    metadata={"random_factor": self.rng.random()},
                )

        elif portfolio.stock > 0:
            # Random sell
            quantity = int(portfolio.stock * self.rng.uniform(0.1, 0.5))

            if quantity > 0:
                return Signal(
                    action="sell",
                    quantity=quantity,
                    confidence=0.5,
                    reasoning="Random walk: random sell decision",
                    metadata={"random_factor": self.rng.random()},
                )

        return Signal(
            action="hold",
            quantity=0,
            confidence=0.5,
            reasoning="Random walk: hold decision",
        )
