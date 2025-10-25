"""
Financial calculation module providing a series of financial calculation tools, including asset pricing model estimation, risk measurement, portfolio optimization and volatility modeling.

Main Features
-------
1. Asset Pricing Models
   - CAPM model
   - Fama-French three-factor model
   - Time series and cross-sectional tests

2. Beta Calculation
   - Calculate stock Beta values using market data

3. Risk Measurement
   - VaR (Value at Risk)
   - CVaR (Conditional Value at Risk)
   - VaR backtesting

4. Portfolio Optimization
   - Maximize Sharpe ratio
   - Minimize risk
   - Maximize return
   - Efficient frontier calculation

5. Volatility Models
   - GARCH(1,1) model
   - Realized volatility calculation

6. Covariance Estimation
   - EWMA method for covariance matrix estimation
"""

# Import main functions from submodules
from src.calc.factor_models import (
    estimate_capm,
    estimate_fama_french,
    time_series_test,
    cross_sectional_test
)

from src.calc.calculate_beta import calculate_beta

from src.calc.tail_risk_measures import (
    calculate_historical_var,
    calculate_conditional_var,
    calculate_parametric_var,
    backtesting_var
)

from src.calc.portfolio_optimization import (
    optimize_portfolio,
    efficient_frontier,
    portfolio_return,
    portfolio_volatility,
    portfolio_sharpe_ratio
)

from src.calc.volatility_models import (
    fit_garch,
    forecast_garch_volatility,
    calculate_realized_volatility
)

from src.calc.covariance_estimation import estimate_covariance_ewma

from src.calc.correlation_analysis import (
    analyze_asset_correlations,
    calculate_optimal_weights_for_correlation,
    cluster_assets
)

# Public API
__all__ = [
    # Asset pricing models
    'estimate_capm',
    'estimate_fama_french',
    'time_series_test',
    'cross_sectional_test',
    
    # Beta calculation
    'calculate_beta',
    
    # Risk measurement
    'calculate_historical_var',
    'calculate_conditional_var',
    'calculate_parametric_var',
    'backtesting_var',
    
    # Portfolio optimization
    'optimize_portfolio',
    'efficient_frontier',
    'portfolio_return',
    'portfolio_volatility',
    'portfolio_sharpe_ratio',
    
    # Volatility models
    'fit_garch',
    'forecast_garch_volatility',
    'calculate_realized_volatility',
    
    # Covariance estimation
    'estimate_covariance_ewma',

    # Correlation analysis
    'analyze_asset_correlations',
    'calculate_optimal_weights_for_correlation',
    'cluster_assets'
]

"""
Usage Examples
-------

1. Calculate stock Beta value

```python
from src.calc import calculate_beta

# Calculate Beta value of a stock relative to CSI 300
# Parameters:
#   ticker: Stock code, e.g. "600519"
#   market_index: Market index code, default "000300" (CSI 300)
#   start_date: Start date, format "YYYY-MM-DD"
#   end_date: End date, format "YYYY-MM-DD"
# Returns:
#   float: Calculated Beta value, typically ranges from 0.2 to 3.0
#   Returns default value 1.0 if insufficient data

beta = calculate_beta(
    ticker="600519",              # Stock code
    market_index="000300",        # Market index code
    start_date="2022-01-01",      # Start date
    end_date="2022-12-31"         # End date
)
print(f"Calculated Beta value: {beta:.2f}")
```

2. Using CAPM model for estimation

```python
import pandas as pd
from src.calc import estimate_capm

# Prepare data
stock_returns = pd.Series([0.01, -0.02, 0.015, 0.008, -0.01])  # Asset return series
market_returns = pd.Series([0.005, -0.01, 0.012, 0.006, -0.008])  # Market return series
risk_free_rate = pd.Series([0.001, 0.001, 0.001, 0.001, 0.001])  # Risk-free rate series

# Estimate CAPM model parameters
# Parameters:
#   returns: Asset return series (pd.Series)
#   market_returns: Market return series (pd.Series)
#   risk_free_rate: Risk-free rate series (pd.Series, optional)
# Returns:
#   Dict[str, float]: Dictionary containing the following key-value pairs
#     - alpha: Jensen's Alpha
#     - beta: Systematic risk Beta
#     - r_squared: Coefficient of determination
#     - p_value_alpha, p_value_beta: p-values for alpha and beta
#     - information_ratio: Information ratio
#     - treynor_ratio: Treynor ratio
#     - residual_std: Residual standard deviation
#     - annualized_alpha: Annualized alpha
#     - observations: Number of observations

capm_results = estimate_capm(
    returns=stock_returns,
    market_returns=market_returns,
    risk_free_rate=risk_free_rate
)

print(f"Alpha: {capm_results['alpha']:.4f}")
print(f"Beta: {capm_results['beta']:.4f}")
print(f"R-squared: {capm_results['r_squared']:.4f}")
print(f"Information ratio: {capm_results['information_ratio']:.4f}")
```

3. Using Fama-French three-factor model

```python
from src.calc import estimate_fama_french

# Prepare data (same as above, plus SMB and HML factors)
smb = pd.Series([0.003, -0.005, 0.008, 0.002, -0.004])  # Small minus big factor returns
hml = pd.Series([0.002, 0.003, -0.001, 0.004, 0.001])   # High minus low factor returns

# Estimate Fama-French three-factor model parameters
# Parameters:
#   returns: Asset return series (pd.Series)
#   market_returns: Market return series (pd.Series)
#   smb: Small minus big factor return series (pd.Series)
#   hml: High minus low factor return series (pd.Series)
#   risk_free_rate: Risk-free rate series (pd.Series, optional)
# Returns:
#   Dict[str, float]: Dictionary containing the following key-value pairs
#     - alpha: Jensen's Alpha
#     - beta_market, beta_smb, beta_hml: Beta values for the three factors
#     - r_squared: Coefficient of determination
#     - p_value_alpha, p_value_market, p_value_smb, p_value_hml: p-values for each coefficient
#     - residual_std: Residual standard deviation
#     - annualized_alpha: Annualized alpha
#     - observations: Number of observations

ff3_results = estimate_fama_french(
    returns=stock_returns,
    market_returns=market_returns,
    smb=smb,
    hml=hml,
    risk_free_rate=risk_free_rate
)

print(f"Alpha: {ff3_results['alpha']:.4f}")
print(f"Market Beta: {ff3_results['beta_market']:.4f}")
print(f"SMB Beta: {ff3_results['beta_smb']:.4f}")
print(f"HML Beta: {ff3_results['beta_hml']:.4f}")
```

4. Calculate Value at Risk (VaR)

```python
import numpy as np
from src.calc import calculate_historical_var, calculate_conditional_var

# Create simulated daily return data
returns = pd.Series(np.random.normal(0.0005, 0.01, 1000))

# Calculate historical VaR
# Parameters:
#   returns: Return series (pd.Series)
#   confidence_level: Confidence level, default 0.95 (95%)
#   window: If specified, only use the most recent window samples
# Returns:
#   float: VaR value at given confidence level (positive value)

hist_var = calculate_historical_var(
    returns=returns,
    confidence_level=0.95,
    window=252  # Use most recent year of data
)

# Calculate conditional VaR (CVaR/Expected Shortfall)
# Parameters:
#   returns: Return series (pd.Series)
#   confidence_level: Confidence level, default 0.95 (95%)
#   window: If specified, only use the most recent window samples
# Returns:
#   float: CVaR value (positive value)

cvar = calculate_conditional_var(
    returns=returns,
    confidence_level=0.95,
    window=252
)

print(f"95% confidence level historical VaR: {hist_var:.4f} (equivalent to {hist_var*100:.2f}% loss)")
print(f"95% confidence level CVaR: {cvar:.4f} (equivalent to {cvar*100:.2f}% loss)")
```

5. Portfolio optimization

```python
import numpy as np
import pandas as pd
from src.calc import optimize_portfolio, efficient_frontier

# Simulate expected returns and covariance matrix for multiple assets
assets = ['Asset A', 'Asset B', 'Asset C', 'Asset D']
expected_returns = pd.Series([0.08, 0.12, 0.10, 0.07], index=assets)
cov_matrix = pd.DataFrame([
    [0.04, 0.02, 0.01, 0.02],
    [0.02, 0.09, 0.03, 0.01],
    [0.01, 0.03, 0.06, 0.02],
    [0.02, 0.01, 0.02, 0.05]
], index=assets, columns=assets)

# Optimize portfolio
# Parameters:
#   expected_returns: Expected returns for each asset (pd.Series)
#   cov_matrix: Covariance matrix (pd.DataFrame)
#   risk_free_rate: Risk-free rate, default 0.0
#   target_return: Target return (if specified)
#   target_risk: Target risk (if specified)
#   objective: Optimization objective, options: 'sharpe'(maximize Sharpe ratio), 'min_risk'(minimize risk), 'max_return'(maximize return)
# Returns:
#   Dict[str, Union[pd.Series, float]]: Dictionary containing the following key-value pairs
#     - weights: Optimal weights Series
#     - return: Portfolio expected return
#     - risk: Portfolio risk (volatility)
#     - sharpe_ratio: Sharpe ratio

# Portfolio with maximum Sharpe ratio
max_sharpe_port = optimize_portfolio(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    risk_free_rate=0.03,
    objective='sharpe'
)

print("Maximum Sharpe ratio portfolio:")
print(f"Optimal weights: \n{max_sharpe_port['weights']}")
print(f"Expected return: {max_sharpe_port['return']:.4f}")
print(f"Risk (standard deviation): {max_sharpe_port['risk']:.4f}")
print(f"Sharpe ratio: {max_sharpe_port['sharpe_ratio']:.4f}")

# Calculate efficient frontier
# Parameters:
#   expected_returns: Expected returns for each asset (pd.Series)
#   cov_matrix: Covariance matrix (pd.DataFrame)
#   risk_free_rate: Risk-free rate, default 0.0
#   points: Number of points on efficient frontier, default 50
# Returns:
#   pd.DataFrame: DataFrame containing returns, risk, and Sharpe ratio

ef = efficient_frontier(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    risk_free_rate=0.03,
    points=20
)

print("\nRepresentative points on efficient frontier:")
print(ef[['return', 'risk', 'sharpe_ratio']].head())
```

6. GARCH model fitting and volatility forecasting

```python
import numpy as np
from src.calc import fit_garch, forecast_garch_volatility

# Simulate return data
np.random.seed(42)
returns = np.random.normal(0, 0.01, 1000)
for i in range(1, len(returns)):
    # Add volatility clustering effect
    if i > 100 and i < 200:
        returns[i] *= 2  # Create a period of high volatility

# Fit GARCH(1,1) model
# Parameters:
#   returns: Return series (np.ndarray)
#   initial_guess: Initial parameter guess [omega, alpha, beta], default None
# Returns:
#   Tuple[Dict[str, float], float]: 
#     - Dict: Model parameters, including omega, alpha, beta, long_run_variance, persistence
#     - float: Log-likelihood value

garch_params, loglik = fit_garch(returns)

print("GARCH(1,1) model parameters:")
print(f"omega: {garch_params['omega']:.6f}")
print(f"alpha: {garch_params['alpha']:.4f}")
print(f"beta: {garch_params['beta']:.4f}")
print(f"Persistence (alpha+beta): {garch_params['persistence']:.4f}")
print(f"Long-run variance: {garch_params['long_run_variance']:.6f}")
print(f"Log-likelihood: {loglik:.2f}")

# Forecast future volatility
# Parameters:
#   returns: Historical return series (np.ndarray)
#   params: GARCH model parameters (Dict[str, float])
#   forecast_horizon: Forecast horizon, default 10
# Returns:
#   np.ndarray: Forecasted volatility series (standard deviation)

forecast_vol = forecast_garch_volatility(
    returns=returns,
    params=garch_params,
    forecast_horizon=5
)

print("\nVolatility forecast for next 5 days:")
for i, vol in enumerate(forecast_vol):
    print(f"Day {i+1}: {vol:.6f}")
```

7. Using EWMA method to estimate covariance matrix

```python
import pandas as pd
import numpy as np
from src.calc import estimate_covariance_ewma

# Create simulated multi-asset return data
np.random.seed(42)
dates = pd.date_range('2022-01-01', periods=500)
assets = ['Asset A', 'Asset B', 'Asset C']
returns_data = {}

for asset in assets:
    returns_data[asset] = np.random.normal(0, 0.01, 500)
    
returns_df = pd.DataFrame(returns_data, index=dates)

# Use EWMA method to estimate covariance matrix
# Parameters:
#   returns: Return DataFrame, each column is an asset, each row is a time point
#   lambda_param: Decay factor, typically between 0.9 and 0.99, higher values give more weight to historical data
#   min_periods: Minimum number of samples required for calculation
# Returns:
#   pd.DataFrame: Covariance matrix DataFrame

ewma_cov = estimate_covariance_ewma(
    returns=returns_df,
    lambda_param=0.94,
    min_periods=50
)

print("EWMA covariance matrix:")
print(ewma_cov)

# Calculate correlation matrix
std_dev = np.sqrt(np.diag(ewma_cov))
corr_matrix = ewma_cov.copy()
for i in range(len(assets)):
    for j in range(len(assets)):
        corr_matrix.iloc[i, j] = ewma_cov.iloc[i, j] / (std_dev[i] * std_dev[j])

print("\nEWMA correlation matrix:")
print(corr_matrix)
```
"""