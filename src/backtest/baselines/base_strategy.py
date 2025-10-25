from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class Signal:
    """Trading signal class"""

    action: str  # 'buy', 'sell', 'hold'
    quantity: int
    confidence: float
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Portfolio:
    """Portfolio state"""

    cash: float
    stock: int
    total_value: float


class BaseStrategy(ABC):
    """
    Base strategy class
    All baseline strategies should inherit from this class
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.parameters = kwargs
        self.trade_history = []
        self.signal_history = []

    @abstractmethod
    def generate_signal(
        self, data: pd.DataFrame, portfolio: Portfolio, current_date: str, **kwargs
    ) -> Signal:
        """
        Generate trading signal

        Args:
            data: Historical price data
            portfolio: Current portfolio state
            current_date: Current date
            **kwargs: Other parameters

        Returns:
            Signal: Trading signal
        """
        pass

    def initialize(self, initial_capital: float):
        """Initialize strategy"""
        self.initial_capital = initial_capital
        self.trade_history = []
        self.signal_history = []

    def reset(self):
        """Reset strategy state"""
        self.trade_history = []
        self.signal_history = []
        # Reset strategy-specific state variables
        if hasattr(self, "last_trade_date"):
            self.last_trade_date = None
        if hasattr(self, "hold_until_date"):
            self.hold_until_date = None
        if hasattr(self, "position_entry_date"):
            self.position_entry_date = None
        if hasattr(self, "last_signal"):
            self.last_signal = None
        if hasattr(self, "signal_count"):
            self.signal_count = 0
        if hasattr(self, "initial_purchase_made"):
            self.initial_purchase_made = False

    def record_signal(self, signal: Signal, date: str, price: float):
        """Record signal history"""
        self.signal_history.append({"date": date, "signal": signal, "price": price})

    def record_trade(
        self,
        action: str,
        quantity: int,
        price: float,
        date: str,
        value: float,
        fees: float,
    ):
        """Record trade history"""
        self.trade_history.append(
            {
                "date": date,
                "action": action,
                "quantity": quantity,
                "price": price,
                "value": value,
                "fees": fees,
            }
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary"""
        return {
            "name": self.name,
            "parameters": self.parameters,
            "total_trades": len(self.trade_history),
            "total_signals": len(self.signal_history),
        }

    @property
    def strategy_type(self) -> str:
        """Strategy type"""
        return self.__class__.__name__.replace("Strategy", "").lower()
