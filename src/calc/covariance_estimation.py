import numpy as np
import pandas as pd

def estimate_covariance_ewma(returns: pd.DataFrame, 
                            lambda_param: float = 0.94, 
                            min_periods: int = 20) -> pd.DataFrame:
    """
    Estimate covariance matrix using Exponential Weighted Moving Average (EWMA)
    
    EWMA is the standard method used by RiskMetrics, using exponentially decaying weights to calculate covariance
    Formula: σ_t^2 = (1-λ) * r_{t-1}^2 + λ * σ_{t-1}^2
    
    Args:
        returns: Returns DataFrame, each column is an asset, each row is a time point
        lambda_param: Decay factor, usually between 0.9 and 0.99, higher values indicate higher weight for historical data
        min_periods: Minimum number of samples required for calculation
        
    Returns:
        Covariance matrix DataFrame
    """
    # Remove missing values
    returns_clean = returns.fillna(0)
    
    # Number of assets
    n_assets = returns_clean.shape[1]
    
    # Prepare EWMA covariance matrix
    # Initialize with sample covariance matrix
    sample_cov = returns_clean.cov().values
    
    # Set initial covariance matrix
    ewma_cov = sample_cov.copy()
    
    # Returns matrix
    returns_array = returns_clean.values
    
    # Calculate EWMA covariance matrix
    for t in range(1, len(returns_clean)):
        # Get current returns vector
        r_t = returns_array[t-1, :]
        
        # Calculate outer product r_t * r_t^T, get instantaneous estimate of covariance matrix
        outer_product = np.outer(r_t, r_t)
        
        # Update EWMA covariance matrix
        ewma_cov = lambda_param * ewma_cov + (1 - lambda_param) * outer_product
    
    # Convert back to DataFrame
    ewma_cov_df = pd.DataFrame(
        ewma_cov, 
        index=returns_clean.columns, 
        columns=returns_clean.columns
    )
    
    return ewma_cov_df