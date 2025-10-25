from typing import Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.rolling import RollingOLS


def estimate_capm(
    returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Estimate CAPM model parameters

    CAPM equation: R_i - R_f = α + β(R_m - R_f) + ε

    Args:
        returns: Asset return series
        market_returns: Market return series
        risk_free_rate: Risk-free rate series, if None then assumed to be 0

    Returns:
        CAPM model parameters dictionary
    """
    # Prepare data
    if risk_free_rate is None:
        # Assume risk-free rate is 0
        excess_returns = returns
        excess_market = market_returns
    else:
        # Calculate excess returns
        excess_returns = returns - risk_free_rate
        excess_market = market_returns - risk_free_rate

    # Create DataFrame for regression
    df = pd.DataFrame(
        {"excess_returns": excess_returns, "excess_market": excess_market}
    ).dropna()

    # Add constant term
    X = sm.add_constant(df["excess_market"])

    # Perform OLS regression
    model = sm.OLS(df["excess_returns"], X)
    results = model.fit()

    # Extract parameters
    alpha = results.params.iloc[0]
    beta = results.params.iloc[1]

    # Calculate predicted values and residuals
    df["predicted"] = alpha + beta * df["excess_market"]
    df["residuals"] = df["excess_returns"] - df["predicted"]

    # Calculate R-squared
    r_squared = results.rsquared

    # Calculate information ratio
    information_ratio = alpha / np.std(df["residuals"]) * np.sqrt(252)  # Annualized

    # Calculate Treynor ratio
    treynor_ratio = (df["excess_returns"].mean() * 252) / beta  # Annualized

    # Return results
    return {
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_squared,
        "p_value_alpha": results.pvalues.iloc[0],
        "p_value_beta": results.pvalues.iloc[1],
        "information_ratio": information_ratio,
        "treynor_ratio": treynor_ratio,
        "residual_std": np.std(df["residuals"]),
        "annualized_alpha": alpha * 252,  # Annualized alpha
        "observations": len(df),
    }


def estimate_fama_french(
    returns: pd.Series,
    market_returns: pd.Series,
    smb: pd.Series,
    hml: pd.Series,
    risk_free_rate: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Estimate Fama-French three-factor model parameters

    FF three-factor equation: R_i - R_f = α + β1(R_m - R_f) + β2(SMB) + β3(HML) + ε

    Args:
        returns: Asset return series
        market_returns: Market return series
        smb: Small Minus Big factor return series
        hml: High Minus Low factor return series
        risk_free_rate: Risk-free rate series, if None then assumed to be 0

    Returns:
        Fama-French three-factor model parameters dictionary
    """
    # Prepare data
    if risk_free_rate is None:
        # Assume risk-free rate is 0
        excess_returns = returns
        excess_market = market_returns
    else:
        # Calculate excess returns
        excess_returns = returns - risk_free_rate
        excess_market = market_returns - risk_free_rate

    # Create DataFrame for regression
    df = pd.DataFrame(
        {
            "excess_returns": excess_returns,
            "excess_market": excess_market,
            "smb": smb,
            "hml": hml,
        }
    ).dropna()

    # Add constant term
    X = sm.add_constant(df[["excess_market", "smb", "hml"]])

    # Perform OLS regression
    model = sm.OLS(df["excess_returns"], X)
    results = model.fit()

    # Extract parameters
    alpha = results.params.iloc[0]
    beta_market = results.params.iloc[1]
    beta_smb = results.params.iloc[2]
    beta_hml = results.params.iloc[3]

    # Calculate predicted values and residuals
    df["predicted"] = (
        alpha
        + beta_market * df["excess_market"]
        + beta_smb * df["smb"]
        + beta_hml * df["hml"]
    )
    df["residuals"] = df["excess_returns"] - df["predicted"]

    # Return results
    return {
        "alpha": alpha,
        "beta_market": beta_market,
        "beta_smb": beta_smb,
        "beta_hml": beta_hml,
        "r_squared": results.rsquared,
        "p_value_alpha": results.pvalues.iloc[0],
        "p_value_market": results.pvalues.iloc[1],
        "p_value_smb": results.pvalues.iloc[2],
        "p_value_hml": results.pvalues.iloc[3],
        "residual_std": np.std(df["residuals"]),
        "annualized_alpha": alpha * 252,  # Annualized alpha
        "observations": len(df),
    }


def time_series_test(
    returns: pd.Series, factors: pd.DataFrame, window: int = 60
) -> pd.DataFrame:
    """
    Perform time series test, calculate rolling factor model parameters

    Args:
        returns: Asset return series
        factors: Factor return DataFrame
        window: Rolling window size

    Returns:
        Rolling coefficients DataFrame
    """
    # Prepare data
    data = pd.concat([returns, factors], axis=1).dropna()
    y = data.iloc[:, 0]  # First column is asset returns
    X = sm.add_constant(data.iloc[:, 1:])  # Remaining columns are factors

    # Use RollingOLS for rolling regression
    rolling_reg = RollingOLS(y, X, window=window)
    rolling_results = rolling_reg.fit()

    # Extract rolling coefficients
    params = rolling_results.params

    # Calculate rolling R-squared
    r2 = pd.Series(index=params.index)
    tvalues = pd.DataFrame(index=params.index, columns=params.columns)

    # Perform OLS regression in each window, calculate R-squared and t-values
    for i in range(window, len(data)):
        y_window = y.iloc[i - window : i]
        X_window = X.iloc[i - window : i]

        model = sm.OLS(y_window, X_window)
        res = model.fit()

        r2.iloc[i - window] = res.rsquared
        for col in params.columns:
            tvalues.loc[params.index[i - window], col] = res.tvalues[col]

    # Combine results
    results = pd.concat([params, r2, tvalues], axis=1)
    results.columns = (
        list(params.columns) + ["R2"] + [f"t_{col}" for col in params.columns]
    )

    return results


def cross_sectional_test(
    returns: pd.DataFrame, factors: pd.DataFrame, frequency: str = "M"
) -> Dict[str, pd.DataFrame]:
    """
    Perform cross-sectional test using Fama-MacBeth regression method

    Args:
        returns: Asset returns DataFrame, each column is an asset
        factors: Asset factor exposure DataFrame, same index as returns, columns are different factors
        frequency: Cross-sectional regression frequency, default is monthly ('M')

    Returns:
        Fama-MacBeth regression results
    """
    # Resample data to specified frequency
    if frequency:
        returns_resampled = returns.resample(frequency).mean()
        factors_resampled = factors.resample(frequency).mean()
    else:
        returns_resampled = returns
        factors_resampled = factors

    # Perform cross-sectional regression for each time point
    time_points = returns_resampled.index
    cross_section_results = []

    for t in time_points:
        if t in factors_resampled.index:
            # Get returns and factors for current time point
            ret_t = returns_resampled.loc[t]
            fact_t = factors_resampled.loc[t]

            # Combine data
            data_t = pd.concat([ret_t, fact_t], axis=1).dropna()

            if (
                len(data_t) > len(factors_resampled.columns) + 2
            ):  # Ensure sufficient observations
                # Perform OLS regression
                y = data_t.iloc[:, 0]
                X = sm.add_constant(data_t.iloc[:, 1:])

                model = sm.OLS(y, X)
                results = model.fit()

                # Save results
                params = results.params
                params.name = t
                cross_section_results.append(params)

    # Aggregate regression coefficients from all time points
    if cross_section_results:
        all_params = pd.concat(cross_section_results, axis=1).T

        # Perform t-test for each factor coefficient
        mean_params = all_params.mean()
        std_params = all_params.std() / np.sqrt(len(all_params))
        t_stats = mean_params / std_params
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(all_params) - 1))

        # Create summary results
        summary = pd.DataFrame(
            {
                "Coefficient": mean_params,
                "Std Error": std_params,
                "t-statistic": t_stats,
                "p-value": p_values,
            }
        )

        return {"time_series_coefficients": all_params, "summary": summary}
    else:
        return {"time_series_coefficients": pd.DataFrame(), "summary": pd.DataFrame()}
