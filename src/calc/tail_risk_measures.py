import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, Union

def calculate_historical_var(returns: pd.Series, 
                            confidence_level: float = 0.95,
                            window: int = None) -> float:
    """
    使用历史模拟法计算VaR(Value at Risk)
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平，默认95%
        window: 如果指定，则只使用最近window个样本
        
    Returns:
        在给定置信水平下的VaR值
    """
    if window is not None and window < len(returns):
        returns = returns.iloc[-window:]
    
    # 计算分位数
    var = returns.quantile(1 - confidence_level)
    
    return abs(var)  # 通常报告为正值

def calculate_conditional_var(returns: pd.Series, 
                             confidence_level: float = 0.95,
                             window: int = None) -> float:
    """
    计算条件风险价值(CVaR/Expected Shortfall)
    CVaR是超过VaR阈值的平均亏损
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平，默认95%
        window: 如果指定，则只使用最近window个样本
        
    Returns:
        CVaR值
    """
    if window is not None and window < len(returns):
        returns = returns.iloc[-window:]
    
    # 计算VaR
    var = calculate_historical_var(returns, confidence_level)
    
    # 计算超过VaR的收益率的平均值
    cvar = returns[returns <= -var].mean()
    
    return abs(cvar)  # 通常报告为正值

def calculate_parametric_var(returns: pd.Series, 
                            confidence_level: float = 0.95,
                            distribution: str = 'normal') -> float:
    """
    使用参数法计算VaR，可假设正态分布或t分布
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平，默认95%
        distribution: 分布假设，可选'normal'或't'
        
    Returns:
        参数法计算的VaR值
    """
    # 计算均值和标准差
    mu = returns.mean()
    sigma = returns.std()
    
    if distribution == 'normal':
        # 使用正态分布
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -(mu + z_score * sigma)
    elif distribution == 't':
        # 使用t分布，首先估计自由度
        params = stats.t.fit(returns)
        df = params[0]  # 自由度
        t_score = stats.t.ppf(1 - confidence_level, df)
        var = -(mu + t_score * sigma)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    
    return abs(var)  # 通常报告为正值

def backtesting_var(returns: pd.Series, 
                   var_method: str = 'historical',
                   confidence_level: float = 0.95,
                   window: int = 252) -> Dict[str, Union[int, float]]:
    """
    回测VaR模型的准确性
    
    Args:
        returns: 收益率序列
        var_method: VaR计算方法，可选'historical'、'parametric'
        confidence_level: 置信水平
        window: 滚动窗口大小
        
    Returns:
        回测结果统计
    """
    violations = 0
    var_values = []
    
    # 确保数据量足够
    if len(returns) <= window:
        raise ValueError(f"Not enough data for backtesting. Need more than {window} observations.")
    
    # 按照滚动窗口计算VaR
    for i in range(window, len(returns)):
        rolling_returns = returns.iloc[i - window:i]
        
        # 根据指定方法计算VaR
        if var_method == 'historical':
            var = calculate_historical_var(rolling_returns, confidence_level)
        elif var_method == 'parametric':
            var = calculate_parametric_var(rolling_returns, confidence_level)
        else:
            raise ValueError(f"Unsupported VaR method: {var_method}")
        
        var_values.append(var)
        
        # 检查是否违反VaR
        actual_return = returns.iloc[i]
        if actual_return < -var:
            violations += 1
    
    # 计算违反率
    violation_rate = violations / (len(returns) - window)
    expected_rate = 1 - confidence_level
    
    # Kupiec测试统计量
    if violations > 0:
        kupiec_stat = -2 * (np.log((1 - expected_rate) ** (len(returns) - window - violations) * 
                                   expected_rate ** violations) - 
                           np.log((1 - violation_rate) ** (len(returns) - window - violations) * 
                                  violation_rate ** violations))
        
        # Kupiec测试p值（使用卡方分布的1自由度）
        kupiec_pvalue = 1 - stats.chi2.cdf(kupiec_stat, 1)
    else:
        kupiec_stat = None
        kupiec_pvalue = None
    
    return {
        "violations": violations,
        "observation_count": len(returns) - window,
        "violation_rate": violation_rate,
        "expected_rate": expected_rate,
        "kupiec_statistic": kupiec_stat,
        "kupiec_pvalue": kupiec_pvalue,
        "average_var": np.mean(var_values)
    }