import pandas as pd
from .base_strategy import BaseStrategy, Signal, Portfolio

class MovingAverageStrategy(BaseStrategy):
    """
    Moving Average Strategy
    Technical analysis strategy based on crossover of different period moving averages
    """
    
    def __init__(self, short_window: int = 50, long_window: int = 200, 
                 signal_threshold: float = 0.005, **kwargs):
        # Extract name from kwargs if provided, otherwise use default
        name = kwargs.pop('name', 'Moving-Average')
        super().__init__(name, **kwargs)
        self.short_window = short_window
        self.long_window = long_window
        self.signal_threshold = signal_threshold
        self.last_signal = None
        self.signal_count = 0
        
    def generate_signal(self, data: pd.DataFrame, portfolio: Portfolio, 
                       current_date: str, **kwargs) -> Signal:
        """
        Moving average strategy logic:
        - Buy when short-term MA crosses above long-term MA
        - Sell when short-term MA crosses below long-term MA
        """
        prices = data['close']
        
        # Reduce data requirement: only need long_window days of data
        if len(prices) < self.long_window:
            return Signal(
                action='hold',
                quantity=0,
                confidence=0.5,
                reasoning=f"Insufficient data: need {self.long_window} periods, got {len(prices)}"
            )
        
        # Calculate current moving averages
        short_ma = prices.tail(self.short_window).mean()
        long_ma = prices.tail(self.long_window).mean()
        
        # Calculate previous period moving averages (if data is sufficient)
        if len(prices) >= self.long_window + 1:
            prev_short_ma = prices.iloc[-(self.short_window+1):-1].mean()
            prev_long_ma = prices.iloc[-(self.long_window+1):-1].mean()
            
            # Use crossover signals
            golden_cross = (prev_short_ma <= prev_long_ma and short_ma > long_ma)
            death_cross = (prev_short_ma >= prev_long_ma and short_ma < long_ma)
        else:
            # When data is insufficient, use simple positional relationship
            golden_cross = short_ma > long_ma * (1 + self.signal_threshold)
            death_cross = short_ma < long_ma * (1 - self.signal_threshold)
            prev_short_ma = short_ma
            prev_long_ma = long_ma
        
        current_price = prices.iloc[-1]
        position_ratio = (portfolio.stock * current_price) / (portfolio.cash + portfolio.stock * current_price)
        
        # Golden cross - buy signal
        if golden_cross and position_ratio < 0.9:
            # Calculate buy quantity
            max_investment = portfolio.cash * 0.8  # Invest 80% of cash
            quantity = int(max_investment / current_price)
            
            if quantity > 0:
                return Signal(
                    action='buy',
                    quantity=quantity,
                    confidence=0.7,
                    reasoning=f"Golden cross: Short MA {short_ma:.2f} > Long MA {long_ma:.2f}",
                    metadata={
                        'short_ma': short_ma,
                        'long_ma': long_ma,
                        'prev_short_ma': prev_short_ma,
                        'prev_long_ma': prev_long_ma,
                        'data_length': len(prices)
                    }
                )
        
        # Death cross - sell signal
        elif death_cross and portfolio.stock > 0:
            # Calculate sell quantity
            quantity = int(portfolio.stock * 0.8)  # Sell 80% of holdings
            
            if quantity > 0:
                return Signal(
                    action='sell',
                    quantity=quantity,
                    confidence=0.7,
                    reasoning=f"Death cross: Short MA {short_ma:.2f} < Long MA {long_ma:.2f}",
                    metadata={
                        'short_ma': short_ma,
                        'long_ma': long_ma,
                        'prev_short_ma': prev_short_ma,
                        'prev_long_ma': prev_long_ma,
                        'data_length': len(prices)
                    }
                )
        
        # Hold signal
        return Signal(
            action='hold',
            quantity=0,
            confidence=0.5,
            reasoning=f"No crossover: Short MA {short_ma:.2f}, Long MA {long_ma:.2f} (data: {len(prices)} days)",
            metadata={
                'short_ma': short_ma,
                'long_ma': long_ma,
                'prev_short_ma': prev_short_ma,
                'prev_long_ma': prev_long_ma,
                'data_length': len(prices)
            }
        )