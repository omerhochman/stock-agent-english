from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass

@dataclass
class Signal:
    """交易信号类"""
    action: str  # 'buy', 'sell', 'hold'
    quantity: int
    confidence: float
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass 
class Portfolio:
    """投资组合状态"""
    cash: float
    stock: int
    total_value: float
    
class BaseStrategy(ABC):
    """
    策略基类
    所有baseline策略都应该继承此类
    """
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.parameters = kwargs
        self.trade_history = []
        self.signal_history = []
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, portfolio: Portfolio, 
                       current_date: str, **kwargs) -> Signal:
        """
        生成交易信号
        
        Args:
            data: 历史价格数据
            portfolio: 当前投资组合状态
            current_date: 当前日期
            **kwargs: 其他参数
            
        Returns:
            Signal: 交易信号
        """
        pass
    
    def initialize(self, initial_capital: float):
        """初始化策略"""
        self.initial_capital = initial_capital
        self.trade_history = []
        self.signal_history = []
        
    def reset(self):
        """重置策略状态"""
        self.trade_history = []
        self.signal_history = []
        # 重置策略特定的状态变量
        if hasattr(self, 'last_trade_date'):
            self.last_trade_date = None
        if hasattr(self, 'hold_until_date'):
            self.hold_until_date = None
        if hasattr(self, 'position_entry_date'):
            self.position_entry_date = None
        if hasattr(self, 'last_signal'):
            self.last_signal = None
        if hasattr(self, 'signal_count'):
            self.signal_count = 0
        if hasattr(self, 'initial_purchase_made'):
            self.initial_purchase_made = False
        
    def record_signal(self, signal: Signal, date: str, price: float):
        """记录信号历史"""
        self.signal_history.append({
            'date': date,
            'signal': signal,
            'price': price
        })
        
    def record_trade(self, action: str, quantity: int, price: float, 
                    date: str, value: float, fees: float):
        """记录交易历史"""
        self.trade_history.append({
            'date': date,
            'action': action,
            'quantity': quantity,
            'price': price,
            'value': value,
            'fees': fees
        })
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取策略表现摘要"""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'total_trades': len(self.trade_history),
            'total_signals': len(self.signal_history)
        }
        
    @property
    def strategy_type(self) -> str:
        """策略类型"""
        return self.__class__.__name__.replace('Strategy', '').lower()