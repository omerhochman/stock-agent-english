from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class TradeResult:
    """Trade result"""
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
    Trade executor
    Handles trade order execution logic
    """
    
    def __init__(self):
        self.execution_history = []
    
    def execute_trade(self, decision_data: dict, portfolio: dict, 
                     current_price: float, date: str, cost_model) -> List[Dict[str, Any]]:
        """
        Execute trade decision
        
        Args:
            decision_data: Trade decision data
            portfolio: Current portfolio
            current_price: Current price
            date: Trade date
            cost_model: Cost model
            
        Returns:
            List[Dict]: List of executed trades
        """
        executed_trades = []
        
        action = decision_data.get("action", "hold")
        quantity = decision_data.get("quantity", 0)
        
        # Smart parsing: if action is hold but quantity>0, means want to establish position
        if action == "hold" and quantity > 0:
            if portfolio["stock"] == 0:  # Currently no position, interpret hold+quantity as buy
                action = "buy"
            else:  # Currently have position, check if need to adjust position
                current_position_value = portfolio["stock"] * current_price
                target_position_value = quantity * current_price
                
                if target_position_value > current_position_value * 1.1:  # Target position is 10% larger than current
                    action = "buy"
                    quantity = quantity - portfolio["stock"]  # Calculate additional quantity to buy
                elif target_position_value < current_position_value * 0.9:  # Target position is 10% smaller than current
                    action = "sell"
                    quantity = portfolio["stock"] - quantity  # Calculate quantity to sell
                else:
                    # Target position is similar to current position, truly hold
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
        """Execute buy order"""
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
        """Execute sell order"""
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
