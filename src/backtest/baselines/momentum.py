import pandas as pd
from .base_strategy import BaseStrategy, Signal, Portfolio

class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy
    Investment strategy based on price trend continuation
    Reference: Jegadeesh and Titman (1993)
    """
    
    def __init__(self, lookback_period: int = 252, formation_period: int = 63,
                 holding_period: int = 21, momentum_threshold: float = 0.02, **kwargs):
        # Extract name from kwargs if provided, otherwise use default
        name = kwargs.pop('name', 'Momentum')
        super().__init__(name, **kwargs)
        self.lookback_period = lookback_period      # Historical data lookback period
        self.formation_period = formation_period    # Momentum formation period
        self.holding_period = holding_period        # Holding period
        self.momentum_threshold = momentum_threshold # Momentum threshold
        self.last_trade_date = None
        self.hold_until_date = None
        
    def calculate_momentum(self, prices: pd.Series) -> float:
        """
        Calculate price momentum
        Use weighted average of multiple period momentum
        """
        if len(prices) < self.formation_period:
            return 0.0
            
        # Calculate momentum for different periods
        momentum_1m = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) >= 21 else 0
        momentum_3m = (prices.iloc[-1] / prices.iloc[-63] - 1) if len(prices) >= 63 else 0
        momentum_6m = (prices.iloc[-1] / prices.iloc[-126] - 1) if len(prices) >= 126 else 0
        momentum_12m = (prices.iloc[-1] / prices.iloc[-252] - 1) if len(prices) >= 252 else 0
        
        # Weighted momentum calculation (ignore recent month to avoid short-term reversal)
        weighted_momentum = (
            0.2 * momentum_1m +
            0.3 * momentum_3m + 
            0.3 * momentum_6m +
            0.2 * momentum_12m
        )
        
        return weighted_momentum
    
    def generate_signal(self, data: pd.DataFrame, portfolio: Portfolio, 
                       current_date: str, **kwargs) -> Signal:
        """
        Momentum strategy logic:
        - Calculate price momentum
        - Buy when momentum exceeds threshold
        - Sell when momentum turns negative
        - Consider holding period constraints
        """
        current_date_obj = pd.to_datetime(current_date)
        
        # Check if within forced holding period
        if (self.hold_until_date and current_date_obj < self.hold_until_date):
            return Signal(
                action='hold',
                quantity=0,
                confidence=0.8,
                reasoning=f"Within holding period until {self.hold_until_date}"
            )
        
        # Calculate momentum indicator
        prices = data['close']
        momentum = self.calculate_momentum(prices)
        
        # Calculate additional indicators
        volatility = prices.pct_change().rolling(21).std().iloc[-1] if len(prices) > 21 else 0
        volume_trend = (data['volume'].rolling(21).mean().iloc[-1] / 
                       data['volume'].rolling(63).mean().iloc[-1]) if len(data) > 63 else 1
        
        current_price = prices.iloc[-1]
        position_ratio = (portfolio.stock * current_price) / (portfolio.cash + portfolio.stock * current_price)
        
        # Momentum buy signal
        if momentum > self.momentum_threshold and position_ratio < 0.8:
            # Strong momentum and not fully invested
            volume_confirmation = volume_trend > 1.1  # Volume confirmation
            volatility_filter = volatility < 0.05  # Volatility filter
            
            if volume_confirmation or not volatility_filter:
                max_investment = portfolio.cash * 0.5  # Maximum invest 50% of cash
                quantity = int(max_investment / current_price)
                
                if quantity > 0:
                    # Set holding period
                    self.hold_until_date = current_date_obj + pd.Timedelta(days=self.holding_period)
                    
                    confidence = min(0.9, 0.5 + abs(momentum))
                    return Signal(
                        action='buy',
                        quantity=quantity,
                        confidence=confidence,
                        reasoning=f"Strong momentum {momentum:.2%}, volume trend {volume_trend:.2f}",
                        metadata={
                            'momentum': momentum,
                            'volume_trend': volume_trend,
                            'volatility': volatility
                        }
                    )
        
        # Momentum sell signal  
        elif momentum < -self.momentum_threshold and portfolio.stock > 0:
            # Negative momentum and have holdings
            quantity = min(portfolio.stock, int(portfolio.stock * 0.5))  # Partial sell
            
            confidence = min(0.9, 0.5 + abs(momentum))
            return Signal(
                action='sell',
                quantity=quantity,
                confidence=confidence,
                reasoning=f"Negative momentum {momentum:.2%}, reducing exposure",
                metadata={
                    'momentum': momentum,
                    'volume_trend': volume_trend,
                    'volatility': volatility
                }
            )
        
        # Hold signal
        return Signal(
            action='hold',
            quantity=0,
            confidence=0.6,
            reasoning=f"Momentum {momentum:.2%} within neutral range",
            metadata={
                'momentum': momentum,
                'volume_trend': volume_trend,
                'volatility': volatility
            }
        )