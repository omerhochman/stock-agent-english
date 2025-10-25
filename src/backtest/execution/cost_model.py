class CostModel:
    """
    Trading cost model
    Calculates trading fees and slippage costs
    """
    
    def __init__(self, trading_cost: float = 0.001, slippage: float = 0.001,
                 min_commission: float = 5.0):
        self.trading_cost = trading_cost    # Trading fee rate
        self.slippage = slippage           # Slippage rate
        self.min_commission = min_commission # Minimum commission
    
    def calculate_total_cost(self, trade_value: float, action: str) -> float:
        """
        Calculate total trading cost
        
        Args:
            trade_value: Trade amount
            action: Trading action ('buy' or 'sell')
            
        Returns:
            float: Total cost
        """
        # Base commission
        commission = max(trade_value * self.trading_cost, self.min_commission)
        
        # Slippage cost
        slippage_cost = trade_value * self.slippage
        
        # Stamp tax (only charged on sell)
        stamp_tax = trade_value * 0.001 if action == 'sell' else 0
        
        # Transfer fee (0.002% of transaction amount)
        transfer_fee = trade_value * 0.00002
        
        total_cost = trade_value + commission + slippage_cost + stamp_tax + transfer_fee
        
        return total_cost
    
    def calculate_net_proceeds(self, trade_value: float, action: str) -> float:
        """
        Calculate net proceeds (after deducting fees)
        
        Args:
            trade_value: Trade amount
            action: Trading action
            
        Returns:
            float: Net proceeds
        """
        # Base commission
        commission = max(trade_value * self.trading_cost, self.min_commission)
        
        # Slippage cost
        slippage_cost = trade_value * self.slippage
        
        # Stamp tax (only charged on sell)
        stamp_tax = trade_value * 0.001 if action == 'sell' else 0
        
        # Transfer fee
        transfer_fee = trade_value * 0.00002
        
        total_fees = commission + slippage_cost + stamp_tax + transfer_fee
        net_proceeds = trade_value - total_fees
        
        return net_proceeds