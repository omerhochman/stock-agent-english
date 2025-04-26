import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Dict, Optional, Union

def portfolio_return(weights: np.ndarray, returns: np.ndarray) -> float:
    """
    计算投资组合预期收益率
    
    Args:
        weights: 权重数组
        returns: 各资产预期收益率数组
        
    Returns:
        投资组合预期收益率
    """
    return np.sum(returns * weights)

def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    计算投资组合波动率
    
    Args:
        weights: 权重数组
        cov_matrix: 协方差矩阵
        
    Returns:
        投资组合波动率
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def portfolio_sharpe_ratio(weights: np.ndarray, 
                          returns: np.ndarray, 
                          cov_matrix: np.ndarray, 
                          risk_free_rate: float = 0.0) -> float:
    """
    计算投资组合夏普比率
    
    Args:
        weights: 权重数组
        returns: 各资产预期收益率数组
        cov_matrix: 协方差矩阵
        risk_free_rate: 无风险利率
        
    Returns:
        投资组合夏普比率
    """
    p_ret = portfolio_return(weights, returns)
    p_vol = portfolio_volatility(weights, cov_matrix)
    
    # 计算夏普比率
    sharpe = (p_ret - risk_free_rate) / p_vol
    
    return sharpe

def negative_sharpe_ratio(weights: np.ndarray, 
                         returns: np.ndarray, 
                         cov_matrix: np.ndarray, 
                         risk_free_rate: float = 0.0) -> float:
    """
    计算夏普比率的负值（用于最小化）
    
    Args:
        weights: 权重数组
        returns: 各资产预期收益率数组
        cov_matrix: 协方差矩阵
        risk_free_rate: 无风险利率
        
    Returns:
        投资组合夏普比率的负值
    """
    return -portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate)

def optimize_portfolio(expected_returns: pd.Series, 
                      cov_matrix: pd.DataFrame, 
                      risk_free_rate: float = 0.0,
                      target_return: Optional[float] = None,
                      target_risk: Optional[float] = None,
                      objective: str = 'sharpe') -> Dict[str, Union[pd.Series, float]]:
    """
    投资组合优化函数
    
    Args:
        expected_returns: 各资产预期收益率
        cov_matrix: 协方差矩阵
        risk_free_rate: 无风险利率
        target_return: 目标收益率（如果指定）
        target_risk: 目标风险（如果指定）
        objective: 优化目标, 可选 'sharpe', 'min_risk', 'max_return'
        
    Returns:
        优化结果字典，包含权重、收益率、风险和夏普比率
    """
    # 确保资产一致性
    assets = expected_returns.index
    if not all(asset in cov_matrix.index for asset in assets):
        raise ValueError("Expected returns and covariance matrix must have the same assets")
    
    # 准备数据
    n_assets = len(assets)
    returns_array = expected_returns.values
    cov_array = cov_matrix.loc[assets, assets].values
    
    # 初始权重（均等分配）
    init_weights = np.ones(n_assets) / n_assets
    
    # 权重约束（和为1，且都为正）
    bounds = tuple((0, 1) for _ in range(n_assets))
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # 根据优化目标设置不同的优化函数
    if objective == 'sharpe':
        # 最大化夏普比率
        opt_function = lambda x: negative_sharpe_ratio(x, returns_array, cov_array, risk_free_rate)
        constraints = [weights_sum_to_1]
    elif objective == 'min_risk':
        # 最小化风险
        opt_function = lambda x: portfolio_volatility(x, cov_array)
        constraints = [weights_sum_to_1]
        
        # 如果指定了目标收益率，添加收益率约束
        if target_return is not None:
            return_constraint = {'type': 'eq', 
                               'fun': lambda x: portfolio_return(x, returns_array) - target_return}
            constraints.append(return_constraint)
    elif objective == 'max_return':
        # 最大化收益率
        opt_function = lambda x: -portfolio_return(x, returns_array)
        constraints = [weights_sum_to_1]
        
        # 如果指定了目标风险，添加风险约束
        if target_risk is not None:
            risk_constraint = {'type': 'eq', 
                             'fun': lambda x: portfolio_volatility(x, cov_array) - target_risk}
            constraints.append(risk_constraint)
    else:
        raise ValueError(f"Unsupported objective: {objective}")
    
    # 优化
    opt_result = sco.minimize(opt_function, init_weights, method='SLSQP', 
                             bounds=bounds, constraints=constraints)
    
    if not opt_result['success']:
        raise RuntimeError(f"Optimization failed: {opt_result['message']}")
    
    # 获取最优权重
    optimal_weights = opt_result['x']
    
    # 计算组合收益率、风险和夏普比率
    opt_return = portfolio_return(optimal_weights, returns_array)
    opt_volatility = portfolio_volatility(optimal_weights, cov_array)
    opt_sharpe = portfolio_sharpe_ratio(optimal_weights, returns_array, cov_array, risk_free_rate)
    
    # 创建权重Series
    weights_series = pd.Series(optimal_weights, index=assets)
    
    # 返回结果
    return {
        'weights': weights_series,
        'return': opt_return,
        'risk': opt_volatility,
        'sharpe_ratio': opt_sharpe
    }

def efficient_frontier(expected_returns: pd.Series, 
                      cov_matrix: pd.DataFrame,
                      risk_free_rate: float = 0.0,
                      points: int = 50) -> pd.DataFrame:
    """
    计算有效前沿
    
    Args:
        expected_returns: 各资产预期收益率
        cov_matrix: 协方差矩阵
        risk_free_rate: 无风险利率
        points: 有效前沿上的点数
        
    Returns:
        有效前沿DataFrame，包含收益率、风险和夏普比率
    """
    # 找到最小风险组合
    min_risk_port = optimize_portfolio(expected_returns, cov_matrix, 
                                      risk_free_rate, objective='min_risk')
    min_return = min_risk_port['return']
    min_risk = min_risk_port['risk']
    
    # 找到最大收益率组合
    max_return_port = optimize_portfolio(expected_returns, cov_matrix,
                                        risk_free_rate, objective='max_return')
    max_return = max_return_port['return']
    
    # 生成目标收益率序列
    target_returns = np.linspace(min_return, max_return, points)
    
    # 计算有效前沿上的每个点
    efficient_portfolios = []
    for target_return in target_returns:
        try:
            port = optimize_portfolio(expected_returns, cov_matrix,
                                     risk_free_rate, target_return=target_return,
                                     objective='min_risk')
            efficient_portfolios.append({
                'return': port['return'],
                'risk': port['risk'],
                'sharpe_ratio': port['sharpe_ratio'],
                'weights': port['weights']
            })
        except Exception as e:
            print(f"Optimization failed for target return {target_return}: {e}")
            continue
    
    # 创建有效前沿DataFrame
    ef_df = pd.DataFrame(efficient_portfolios)
    
    return ef_df