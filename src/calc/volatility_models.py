import numpy as np
import pandas as pd
from scipy import optimize
from typing import Tuple, Dict

def garch_likelihood(params, returns):
    """
    Negative log-likelihood function for GARCH(1,1) model
    
    Model: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
    
    Args:
        params: [omega, alpha, beta] model parameters
        returns: return series
        
    Returns:
        Negative log-likelihood value
    """
    omega, alpha, beta = params
    
    # Parameter constraints
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return np.inf
    
    # Initialize
    n = len(returns)
    h = np.zeros(n)  # Conditional variance
    h[0] = np.var(returns)  # Set initial conditional variance to sample variance
    
    # Recursively calculate conditional variance
    for t in range(1, n):
        h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
    
    # Calculate negative log-likelihood
    logliks = -0.5 * (np.log(2 * np.pi) + np.log(h) + returns**2 / h)
    loglik = np.sum(logliks)
    
    # Return negative log-likelihood (because we want to minimize)
    return -loglik

def fit_garch(returns: np.ndarray, initial_guess=None) -> Tuple[Dict[str, float], float]:
    """
    Fit GARCH(1,1) model
    
    Args:
        returns: return series
        initial_guess: initial parameter guess [omega, alpha, beta]
        
    Returns:
        Model parameters and log-likelihood value
    """
    # Default initial parameters
    if initial_guess is None:
        var_r = np.var(returns)
        initial_guess = [0.1 * var_r, 0.1, 0.8]  # Common initial parameters
    
    # Optimize negative log-likelihood function
    result = optimize.minimize(garch_likelihood, initial_guess, args=(returns,), 
                              method='L-BFGS-B',
                              bounds=((1e-6, None), (0, 1), (0, 1)))
    
    # Extract parameters
    omega, alpha, beta = result.x
    
    # Calculate long-run volatility
    long_run_var = omega / (1 - alpha - beta) if alpha + beta < 1 else None
    
    # Return parameters and log-likelihood value
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
    Use GARCH(1,1) model to forecast future volatility
    
    Args:
        returns: historical return series
        params: GARCH model parameters
        forecast_horizon: forecast periods
        
    Returns:
        Forecasted volatility series
    """
    omega = params['omega']
    alpha = params['alpha']
    beta = params['beta']
    
    # Initialize
    n = len(returns)
    h = np.zeros(n)
    h[0] = np.var(returns)
    
    # Calculate historical conditional variance
    for t in range(1, n):
        h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
    
    # Last period's conditional variance
    last_var = h[-1]
    
    # Forecast future volatility
    forecast_var = np.zeros(forecast_horizon)
    for t in range(forecast_horizon):
        if t == 0:
            forecast_var[t] = omega + alpha * returns[-1]**2 + beta * last_var
        else:
            forecast_var[t] = omega + (alpha + beta) * forecast_var[t-1]
    
    # Return volatility (standard deviation)
    return np.sqrt(forecast_var)

def calculate_realized_volatility(returns: pd.Series, window: int = 21,
                                  annualize: bool = True) -> pd.Series:
    """
    Calculate realized volatility (historical volatility)
    
    Args:
        returns: return series
        window: rolling window size, typically 21 for monthly volatility
        annualize: whether to annualize volatility
        
    Returns:
        Realized volatility series
    """
    # Calculate rolling standard deviation
    realized_vol = returns.rolling(window=window).std()
    
    # Annualize
    if annualize:
        # Assume 252 trading days per year
        realized_vol = realized_vol * np.sqrt(252)
    
    return realized_vol