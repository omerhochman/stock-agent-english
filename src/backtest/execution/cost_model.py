class CostModel:
    """
    交易成本模型
    计算交易费用和滑点成本
    """
    
    def __init__(self, trading_cost: float = 0.001, slippage: float = 0.001,
                 min_commission: float = 5.0):
        self.trading_cost = trading_cost    # 交易费率
        self.slippage = slippage           # 滑点费率
        self.min_commission = min_commission # 最低佣金
    
    def calculate_total_cost(self, trade_value: float, action: str) -> float:
        """
        计算总交易成本
        
        Args:
            trade_value: 交易金额
            action: 交易动作 ('buy' 或 'sell')
            
        Returns:
            float: 总成本
        """
        # 基础佣金
        commission = max(trade_value * self.trading_cost, self.min_commission)
        
        # 滑点成本
        slippage_cost = trade_value * self.slippage
        
        # 印花税（仅卖出时收取）
        stamp_tax = trade_value * 0.001 if action == 'sell' else 0
        
        # 过户费（按成交金额的0.002%收取）
        transfer_fee = trade_value * 0.00002
        
        total_cost = trade_value + commission + slippage_cost + stamp_tax + transfer_fee
        
        return total_cost
    
    def calculate_net_proceeds(self, trade_value: float, action: str) -> float:
        """
        计算净收益（扣除费用后）
        
        Args:
            trade_value: 交易金额
            action: 交易动作
            
        Returns:
            float: 净收益
        """
        # 基础佣金
        commission = max(trade_value * self.trading_cost, self.min_commission)
        
        # 滑点成本
        slippage_cost = trade_value * self.slippage
        
        # 印花税（仅卖出时收取）
        stamp_tax = trade_value * 0.001 if action == 'sell' else 0
        
        # 过户费
        transfer_fee = trade_value * 0.00002
        
        total_fees = commission + slippage_cost + stamp_tax + transfer_fee
        net_proceeds = trade_value - total_fees
        
        return net_proceeds