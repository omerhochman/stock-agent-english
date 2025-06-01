from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class TradeResult:
    """交易结果"""
    action: str
    quantity: int
    price: float
    value: float
    fees: float
    executed_at: str
    success: bool
    message: str

class TradeExecutor:
    """
    交易执行器
    处理交易订单的执行逻辑
    """
    
    def __init__(self):
        self.execution_history = []
    
    def execute_trade(self, decision_data: dict, portfolio: dict, 
                     current_price: float, date: str, cost_model) -> List[Dict[str, Any]]:
        """
        执行交易决策
        
        Args:
            decision_data: 交易决策数据
            portfolio: 当前投资组合
            current_price: 当前价格
            date: 交易日期
            cost_model: 成本模型
            
        Returns:
            List[Dict]: 执行的交易列表
        """
        executed_trades = []
        
        action = decision_data.get("action", "hold")
        quantity = decision_data.get("quantity", 0)
        
        # 智能解析：如果action是hold但quantity>0，说明想要建仓
        if action == "hold" and quantity > 0:
            if portfolio["stock"] == 0:  # 当前无持仓，将hold+quantity解释为买入
                action = "buy"
            else:  # 当前有持仓，检查是否需要调整仓位
                current_position_value = portfolio["stock"] * current_price
                target_position_value = quantity * current_price
                
                if target_position_value > current_position_value * 1.1:  # 目标仓位比当前大10%以上
                    action = "buy"
                    quantity = quantity - portfolio["stock"]  # 计算需要额外买入的数量
                elif target_position_value < current_position_value * 0.9:  # 目标仓位比当前小10%以上
                    action = "sell"
                    quantity = portfolio["stock"] - quantity  # 计算需要卖出的数量
                else:
                    # 目标仓位与当前仓位相近，真正hold
                    action = "hold"
                    quantity = 0
        
        if action == "buy" and quantity > 0:
            trade_result = self._execute_buy_order(
                quantity, current_price, portfolio, date, cost_model
            )
            if trade_result.success:
                executed_trades.append({
                    "date": date,
                    "action": "buy",
                    "quantity": trade_result.quantity,
                    "price": trade_result.price,
                    "value": trade_result.value,
                    "fees": trade_result.fees
                })
        
        elif action == "sell" and quantity > 0:
            trade_result = self._execute_sell_order(
                quantity, current_price, portfolio, date, cost_model
            )
            if trade_result.success:
                executed_trades.append({
                    "date": date,
                    "action": "sell",
                    "quantity": trade_result.quantity,
                    "price": trade_result.price,
                    "value": trade_result.value,
                    "fees": trade_result.fees
                })
        
        return executed_trades
    
    def _execute_buy_order(self, quantity: int, price: float, portfolio: dict, 
                          date: str, cost_model) -> TradeResult:
        """执行买入订单"""
        cost_without_fees = quantity * price
        total_cost = cost_model.calculate_total_cost(cost_without_fees, 'buy')
        
        if total_cost <= portfolio["cash"]:
            portfolio["stock"] += quantity
            portfolio["cash"] -= total_cost
            
            return TradeResult(
                action="buy",
                quantity=quantity,
                price=price,
                value=cost_without_fees,
                fees=total_cost - cost_without_fees,
                executed_at=date,
                success=True,
                message="Order executed successfully"
            )
        else:
            return TradeResult(
                action="buy",
                quantity=0,
                price=price,
                value=0,
                fees=0,
                executed_at=date,
                success=False,
                message="Insufficient cash"
            )
    
    def _execute_sell_order(self, quantity: int, price: float, portfolio: dict, 
                           date: str, cost_model) -> TradeResult:
        """执行卖出订单"""
        actual_quantity = min(quantity, portfolio["stock"])
        
        if actual_quantity > 0:
            value_without_fees = actual_quantity * price
            net_proceeds = cost_model.calculate_net_proceeds(value_without_fees, 'sell')
            
            portfolio["cash"] += net_proceeds
            portfolio["stock"] -= actual_quantity
            
            return TradeResult(
                action="sell",
                quantity=actual_quantity,
                price=price,
                value=value_without_fees,
                fees=value_without_fees - net_proceeds,
                executed_at=date,
                success=True,
                message="Order executed successfully"
            )
        else:
            return TradeResult(
                action="sell",
                quantity=0,
                price=price,
                value=0,
                fees=0,
                executed_at=date,
                success=False,
                message="No shares to sell"
            )
