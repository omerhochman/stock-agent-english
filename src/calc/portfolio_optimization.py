from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import scipy.optimize as sco


def portfolio_return(weights: np.ndarray, returns: np.ndarray) -> float:
    """
    Calculate portfolio expected return

    Args:
        weights: weight array
        returns: expected return array for each asset

    Returns:
        Portfolio expected return
    """
    return np.sum(returns * weights)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calculate portfolio volatility

    Args:
        weights: weight array
        cov_matrix: covariance matrix

    Returns:
        Portfolio volatility
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def portfolio_sharpe_ratio(
    weights: np.ndarray,
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate portfolio Sharpe ratio

    Args:
        weights: weight array
        returns: expected return array for each asset
        cov_matrix: covariance matrix
        risk_free_rate: risk-free rate

    Returns:
        Portfolio Sharpe ratio
    """
    p_ret = portfolio_return(weights, returns)
    p_vol = portfolio_volatility(weights, cov_matrix)

    # Check numerical stability
    if p_vol == 0 or np.isclose(p_vol, 0, atol=1e-10):
        return 0.0

    if not np.isfinite(p_ret) or not np.isfinite(p_vol):
        return 0.0

    # Calculate Sharpe ratio
    sharpe = (p_ret - risk_free_rate) / p_vol

    # Ensure return finite value
    if not np.isfinite(sharpe):
        return 0.0

    return sharpe


def negative_sharpe_ratio(
    weights: np.ndarray,
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate negative Sharpe ratio (for minimization)

    Args:
        weights: Weight array
        returns: Expected return array for each asset
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate

    Returns:
        Negative value of portfolio Sharpe ratio
    """
    return -portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate)


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    target_return: Optional[float] = None,
    target_risk: Optional[float] = None,
    objective: str = "sharpe",
) -> Dict[str, Union[pd.Series, float]]:
    """
    Portfolio optimization function

    Args:
        expected_returns: Expected returns for each asset
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate
        target_return: Target return (if specified)
        target_risk: Target risk (if specified)
        objective: Optimization objective, options: 'sharpe', 'min_risk', 'max_return'

    Returns:
        Optimization result dictionary containing weights, return, risk and Sharpe ratio
    """
    # Ensure asset consistency
    assets = expected_returns.index
    if not all(asset in cov_matrix.index for asset in assets):
        raise ValueError(
            "Expected returns and covariance matrix must have the same assets"
        )

    # Prepare data
    n_assets = len(assets)
    returns_array = expected_returns.values
    cov_array = cov_matrix.loc[assets, assets].values

    # Initial weights (equal allocation)
    init_weights = np.ones(n_assets) / n_assets

    # Weight constraints (sum to 1, all positive)
    bounds = tuple((0, 1) for _ in range(n_assets))
    weights_sum_to_1 = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    # Set different optimization functions based on objective
    if objective == "sharpe":
        # Maximize Sharpe ratio
        opt_function = lambda x: negative_sharpe_ratio(
            x, returns_array, cov_array, risk_free_rate
        )
        constraints = [weights_sum_to_1]
    elif objective == "min_risk":
        # Minimize risk
        opt_function = lambda x: portfolio_volatility(x, cov_array)
        constraints = [weights_sum_to_1]

        # If target return is specified, add return constraint
        if target_return is not None:
            return_constraint = {
                "type": "eq",
                "fun": lambda x: portfolio_return(x, returns_array) - target_return,
            }
            constraints.append(return_constraint)
    elif objective == "max_return":
        # Maximize return
        opt_function = lambda x: -portfolio_return(x, returns_array)
        constraints = [weights_sum_to_1]

        # If target risk is specified, add risk constraint
        if target_risk is not None:
            risk_constraint = {
                "type": "eq",
                "fun": lambda x: portfolio_volatility(x, cov_array) - target_risk,
            }
            constraints.append(risk_constraint)
    else:
        raise ValueError(f"Unsupported objective: {objective}")

    # Optimize
    opt_result = sco.minimize(
        opt_function,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not opt_result["success"]:
        raise RuntimeError(f"Optimization failed: {opt_result['message']}")

    # Get optimal weights
    optimal_weights = opt_result["x"]

    # Calculate portfolio return, risk and Sharpe ratio
    opt_return = portfolio_return(optimal_weights, returns_array)
    opt_volatility = portfolio_volatility(optimal_weights, cov_array)
    opt_sharpe = portfolio_sharpe_ratio(
        optimal_weights, returns_array, cov_array, risk_free_rate
    )

    # Create weights Series
    weights_series = pd.Series(optimal_weights, index=assets)

    # Return results
    return {
        "weights": weights_series,
        "return": opt_return,
        "risk": opt_volatility,
        "sharpe_ratio": opt_sharpe,
    }


def efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    points: int = 50,
) -> pd.DataFrame:
    """
    Calculate efficient frontier

    Args:
        expected_returns: Expected returns for each asset
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate
        points: Number of points on efficient frontier

    Returns:
        Efficient frontier DataFrame containing return, risk and Sharpe ratio
    """
    # Find minimum risk portfolio
    min_risk_port = optimize_portfolio(
        expected_returns, cov_matrix, risk_free_rate, objective="min_risk"
    )
    min_return = min_risk_port["return"]
    min_risk = min_risk_port["risk"]

    # Find maximum return portfolio
    max_return_port = optimize_portfolio(
        expected_returns, cov_matrix, risk_free_rate, objective="max_return"
    )
    max_return = max_return_port["return"]

    # Generate target return sequence
    target_returns = np.linspace(min_return, max_return, points)

    # Calculate each point on efficient frontier
    efficient_portfolios = []
    for target_return in target_returns:
        try:
            port = optimize_portfolio(
                expected_returns,
                cov_matrix,
                risk_free_rate,
                target_return=target_return,
                objective="min_risk",
            )
            efficient_portfolios.append(
                {
                    "return": port["return"],
                    "risk": port["risk"],
                    "sharpe_ratio": port["sharpe_ratio"],
                    "weights": port["weights"],
                }
            )
        except Exception as e:
            print(f"Optimization failed for target return {target_return}: {e}")
            continue

    # Create efficient frontier DataFrame
    ef_df = pd.DataFrame(efficient_portfolios)

    return ef_df
