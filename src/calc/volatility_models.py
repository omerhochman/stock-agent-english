import numpy as np
import pandas as pd
from scipy import optimize
from typing import Tuple, Dict

def garch_likelihood(params, returns):
    """
    GARCH(1,1)模型的负对数似然函数
    
    模型: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
    
    Args:
        params: [omega, alpha, beta] 模型参数
        returns: 收益率序列
        
    Returns:
        负对数似然值
    """
    omega, alpha, beta = params
    
    # 参数约束条件
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return np.inf
    
    # 初始化
    n = len(returns)
    h = np.zeros(n)  # 条件方差
    h[0] = np.var(returns)  # 初始条件方差设为样本方差
    
    # 递归计算条件方差
    for t in range(1, n):
        h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
    
    # 计算负对数似然
    logliks = -0.5 * (np.log(2 * np.pi) + np.log(h) + returns**2 / h)
    loglik = np.sum(logliks)
    
    # 返回负对数似然（因为我们要最小化）
    return -loglik

def fit_garch(returns: np.ndarray, initial_guess=None) -> Tuple[Dict[str, float], float]:
    """
    拟合GARCH(1,1)模型
    
    Args:
        returns: 收益率序列
        initial_guess: 初始参数猜测 [omega, alpha, beta]
        
    Returns:
        模型参数和对数似然值
    """
    # 默认初始参数
    if initial_guess is None:
        var_r = np.var(returns)
        initial_guess = [0.1 * var_r, 0.1, 0.8]  # 常见的初始参数
    
    # 优化负对数似然函数
    result = optimize.minimize(garch_likelihood, initial_guess, args=(returns,), 
                              method='L-BFGS-B',
                              bounds=((1e-6, None), (0, 1), (0, 1)))
    
    # 提取参数
    omega, alpha, beta = result.x
    
    # 计算长期波动率
    long_run_var = omega / (1 - alpha - beta) if alpha + beta < 1 else None
    
    # 返回参数与对数似然值
    params = {
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'long_run_variance': long_run_var,
        'persistence': alpha + beta
    }
    
    return params, -result.fun

def forecast_garch_volatility(returns: np.ndarray, params: Dict[str, float], 
                             forecast_horizon: int = 10) -> np.ndarray:
    """
    使用GARCH(1,1)模型预测未来波动率
    
    Args:
        returns: 历史收益率序列
        params: GARCH模型参数
        forecast_horizon: 预测期数
        
    Returns:
        预测的波动率序列
    """
    omega = params['omega']
    alpha = params['alpha']
    beta = params['beta']
    
    # 初始化
    n = len(returns)
    h = np.zeros(n)
    h[0] = np.var(returns)
    
    # 计算历史条件方差
    for t in range(1, n):
        h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
    
    # 最后一期的条件方差
    last_var = h[-1]
    
    # 预测未来波动率
    forecast_var = np.zeros(forecast_horizon)
    for t in range(forecast_horizon):
        if t == 0:
            forecast_var[t] = omega + alpha * returns[-1]**2 + beta * last_var
        else:
            forecast_var[t] = omega + (alpha + beta) * forecast_var[t-1]
    
    # 返回波动率（标准差）
    return np.sqrt(forecast_var)

def calculate_realized_volatility(returns: pd.Series, window: int = 21,
                                  annualize: bool = True) -> pd.Series:
    """
    计算已实现波动率（历史波动率）
    
    Args:
        returns: 收益率序列
        window: 滑动窗口大小，通常用21表示月度波动率
        annualize: 是否年化波动率
        
    Returns:
        已实现波动率序列
    """
    # 计算滚动标准差
    realized_vol = returns.rolling(window=window).std()
    
    # 年化处理
    if annualize:
        # 假设252个交易日/年
        realized_vol = realized_vol * np.sqrt(252)
    
    return realized_vol