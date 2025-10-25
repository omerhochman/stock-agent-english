import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, Union

def calculate_historical_var(returns: pd.Series, 
                            confidence_level: float = 0.95,
                            window: int = None) -> float:
    """
    Calculate VaR (Value at Risk) using historical simulation method
    
    Args:
        returns: return series
        confidence_level: confidence level, default 95%
        window: if specified, only use the most recent window samples
        
    Returns:
        VaR value at given confidence level
    """
    if window is not None and window < len(returns):
        returns = returns.iloc[-window:]
    
    # Calculate quantile
    var = returns.quantile(1 - confidence_level)
    
    return abs(var)  # Usually reported as positive value

def calculate_conditional_var(returns: pd.Series, 
                             confidence_level: float = 0.95,
                             window: int = None) -> float:
    """
    Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
    CVaR is the average loss exceeding the VaR threshold
    
    Args:
        returns: return series
        confidence_level: confidence level, default 95%
        window: if specified, only use the most recent window samples
        
    Returns:
        CVaR value
    """
    if window is not None and window < len(returns):
        returns = returns.iloc[-window:]
    
    # Calculate VaR
    var = calculate_historical_var(returns, confidence_level)
    
    # Calculate average of returns exceeding VaR
    cvar = returns[returns <= -var].mean()
    
    return abs(cvar)  # Usually reported as positive value

def calculate_parametric_var(returns: pd.Series, 
                            confidence_level: float = 0.95,
                            distribution: str = 'normal') -> float:
    """
    Calculate VaR using parametric method, can assume normal or t distribution
    
    Args:
        returns: return series
        confidence_level: confidence level, default 95%
        distribution: distribution assumption, optional 'normal' or 't'
        
    Returns:
        VaR value calculated by parametric method
    """
    # Calculate mean and standard deviation
    mu = returns.mean()
    sigma = returns.std()
    
    if distribution == 'normal':
        # Use normal distribution
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -(mu + z_score * sigma)
    elif distribution == 't':
        # Use t distribution, first estimate degrees of freedom
        params = stats.t.fit(returns)
        df = params[0]  # Degrees of freedom
        t_score = stats.t.ppf(1 - confidence_level, df)
        var = -(mu + t_score * sigma)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    
    return abs(var)  # Usually reported as positive value

def backtesting_var(returns: pd.Series, 
                   var_method: str = 'historical',
                   confidence_level: float = 0.95,
                   window: int = 252) -> Dict[str, Union[int, float]]:
    """
    Backtest VaR model accuracy
    
    Args:
        returns: return series
        var_method: VaR calculation method, optional 'historical', 'parametric'
        confidence_level: confidence level
        window: rolling window size
        
    Returns:
        Backtest result statistics
    """
    violations = 0
    var_values = []
    
    # Ensure sufficient data
    if len(returns) <= window:
        raise ValueError(f"Not enough data for backtesting. Need more than {window} observations.")
    
    # Calculate VaR using rolling window
    for i in range(window, len(returns)):
        rolling_returns = returns.iloc[i - window:i]
        
        # Calculate VaR using specified method
        if var_method == 'historical':
            var = calculate_historical_var(rolling_returns, confidence_level)
        elif var_method == 'parametric':
            var = calculate_parametric_var(rolling_returns, confidence_level)
        else:
            raise ValueError(f"Unsupported VaR method: {var_method}")
        
        var_values.append(var)
        
        # Check if VaR is violated
        actual_return = returns.iloc[i]
        if actual_return < -var:
            violations += 1
    
    # Calculate violation rate
    violation_rate = violations / (len(returns) - window)
    expected_rate = 1 - confidence_level
    
    # Kupiec test statistic
    if violations > 0:
        kupiec_stat = -2 * (np.log((1 - expected_rate) ** (len(returns) - window - violations) * 
                                   expected_rate ** violations) - 
                           np.log((1 - violation_rate) ** (len(returns) - window - violations) * 
                                  violation_rate ** violations))
        
        # Kupiec test p-value (using chi-square distribution with 1 degree of freedom)
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