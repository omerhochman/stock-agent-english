import pandas as pd
import numpy as np
from typing import Dict
import statsmodels.api as sm

def analyze_macro_factor_exposures(returns_df: pd.DataFrame, 
                                  macro_factors_df: pd.DataFrame) -> Dict:
    """
    Analyze asset sensitivity to macroeconomic factors
    
    Args:
        returns_df: DataFrame containing multiple asset returns
        macro_factors_df: DataFrame containing macroeconomic factors
        
    Returns:
        Dict: Contains macro factor sensitivity analysis results
    """
    # Align date indices
    common_dates = returns_df.index.intersection(macro_factors_df.index)
    if len(common_dates) < 30:  # Need sufficient data points
        return {
            'success': False,
            'error': f'Insufficient data points, only {len(common_dates)} common dates',
            'exposures': {}
        }
    
    # Align data
    aligned_returns = returns_df.loc[common_dates]
    aligned_factors = macro_factors_df.loc[common_dates]
    
    # Results dictionary
    results = {
        'success': True,
        'exposures': {},
        'sensitivity_summary': []
    }
    
    # Calculate sensitivity to macro factors for each asset
    for asset in returns_df.columns:
        asset_returns = aligned_returns[asset]
        
        # Add constant term
        X = sm.add_constant(aligned_factors)
        
        try:
            # Perform linear regression
            model = sm.OLS(asset_returns, X)
            fit_result = model.fit()
            
            # Extract coefficients
            coefficients = fit_result.params.to_dict()
            pvalues = fit_result.pvalues.to_dict()
            
            # Remove constant term
            if 'const' in coefficients:
                del coefficients['const']
            if 'const' in pvalues:
                del pvalues['const']
            
            # Find significant sensitivities (p-value < 0.05)
            significant_factors = {
                factor: {
                    'coefficient': float(coef),
                    'p_value': float(pvalues[factor])
                }
                for factor, coef in coefficients.items()
                if pvalues[factor] < 0.05
            }
            
            # Store results
            exposures = {
                'all_coefficients': {factor: float(coef) for factor, coef in coefficients.items()},
                'significant_factors': significant_factors,
                'r_squared': float(fit_result.rsquared),
                'adjusted_r_squared': float(fit_result.rsquared_adj)
            }
            
            results['exposures'][asset] = exposures
            
            # Add sensitivity summary
            if significant_factors:
                for factor, data in significant_factors.items():
                    sensitivity = "high" if abs(data['coefficient']) > 0.5 else "medium" if abs(data['coefficient']) > 0.2 else "low"
                    direction = "positive" if data['coefficient'] > 0 else "negative"
                    results['sensitivity_summary'].append(
                        f"{asset} has {direction} {sensitivity} sensitivity to {factor} (coefficient: {data['coefficient']:.4f})"
                    )
            
        except Exception as e:
            results['exposures'][asset] = {
                'error': str(e),
                'r_squared': 0.0,
                'all_coefficients': {},
                'significant_factors': {}
            }
    
    return results

def create_macro_factor_portfolio(returns_df: pd.DataFrame, 
                                 macro_factors_df: pd.DataFrame,
                                 target_factor: str,
                                 exposure_direction: str = 'positive') -> Dict:
    """
    Create portfolio with target exposure to specific macro factor
    
    Args:
        returns_df: DataFrame containing multiple asset returns
        macro_factors_df: DataFrame containing macroeconomic factors
        target_factor: Target macro factor
        exposure_direction: Exposure direction, 'positive' for positive exposure or 'negative' for negative exposure
        
    Returns:
        Dict: Contains portfolio weights and factor exposure
    """
    # First analyze sensitivity of all assets to macro factors
    exposures_analysis = analyze_macro_factor_exposures(returns_df, macro_factors_df)
    
    if not exposures_analysis['success']:
        return {
            'success': False,
            'error': exposures_analysis.get('error', 'Macro factor analysis failed'),
            'weights': {col: 1.0/len(returns_df.columns) for col in returns_df.columns}  # Default equal weights
        }
    
    # Check if target factor exists
    if target_factor not in macro_factors_df.columns:
        return {
            'success': False,
            'error': f'Target factor {target_factor} not in provided macro factors',
            'weights': {col: 1.0/len(returns_df.columns) for col in returns_df.columns}  # Default equal weights
        }
    
    # Extract sensitivity of each asset to target factor
    factor_sensitivities = {}
    for asset, exposure_data in exposures_analysis['exposures'].items():
        if 'all_coefficients' in exposure_data and target_factor in exposure_data['all_coefficients']:
            factor_sensitivities[asset] = exposure_data['all_coefficients'][target_factor]
    
    # If no sensitivity found, return error
    if not factor_sensitivities:
        return {
            'success': False,
            'error': f'No assets have significant sensitivity to factor {target_factor}',
            'weights': {col: 1.0/len(returns_df.columns) for col in returns_df.columns}  # Default equal weights
        }
    
    # Filter assets based on sensitivity direction
    if exposure_direction == 'positive':
        selected_assets = {asset: sens for asset, sens in factor_sensitivities.items() if sens > 0}
    else:  # 'negative'
        selected_assets = {asset: sens for asset, sens in factor_sensitivities.items() if sens < 0}
    
    # If no assets meet criteria, return error
    if not selected_assets:
        return {
            'success': False,
            'error': f'No assets have {exposure_direction} sensitivity to factor {target_factor}',
            'weights': {col: 1.0/len(returns_df.columns) for col in returns_df.columns}  # Default equal weights
        }
    
    # Calculate weights - simple method based on proportion of absolute sensitivity
    total_sensitivity = sum(abs(sens) for sens in selected_assets.values())
    
    if total_sensitivity > 0:
        weights = {asset: abs(sens)/total_sensitivity for asset, sens in selected_assets.items()}
    else:
        # If all sensitivities are 0, use equal weights
        weights = {asset: 1.0/len(selected_assets) for asset in selected_assets}
    
    # Calculate total portfolio exposure to target factor
    portfolio_exposure = sum(sens * weights[asset] for asset, sens in selected_assets.items())
    
    return {
        'success': True,
        'weights': weights,
        'portfolio_exposure': float(portfolio_exposure),
        'target_factor': target_factor,
        'exposure_direction': exposure_direction,
        'asset_sensitivities': {asset: float(sens) for asset, sens in selected_assets.items()}
    }

def analyze_factor_rotation(returns_df: pd.DataFrame, 
                           macro_factors_df: pd.DataFrame, 
                           window: int = 60) -> Dict:
    """
    Analyze changes in asset sensitivity to macro factors over time
    
    Args:
        returns_df: DataFrame containing multiple asset returns
        macro_factors_df: DataFrame containing macroeconomic factors
        window: Rolling window size
        
    Returns:
        Dict: Contains factor rotation analysis results
    """
    # Align date indices
    common_dates = returns_df.index.intersection(macro_factors_df.index)
    if len(common_dates) < window + 10:  # Need sufficient data points
        return {
            'success': False,
            'error': f'Insufficient data points, need at least {window + 10} data points, but only have {len(common_dates)}',
            'rotation_analysis': {}
        }
    
    # Align data
    aligned_returns = returns_df.loc[common_dates]
    aligned_factors = macro_factors_df.loc[common_dates]
    
    # Results dictionary
    results = {
        'success': True,
        'rotation_analysis': {},
        'trend_summary': []
    }
    
    # Calculate rolling factor sensitivity for each asset
    for asset in returns_df.columns:
        asset_returns = aligned_returns[asset]
        
        # Store sensitivity for each time point
        rolling_sensitivities = {}
        
        # Iterate through each time point
        for i in range(window, len(common_dates)):
            current_date = common_dates[i]
            window_returns = asset_returns.iloc[i-window:i]
            window_factors = aligned_factors.iloc[i-window:i]
            
            # Add constant term
            X = sm.add_constant(window_factors)
            
            try:
                # Perform linear regression
                model = sm.OLS(window_returns, X)
                fit_result = model.fit()
                
                # Extract coefficients
                coefficients = fit_result.params.to_dict()
                
                # Remove constant term
                if 'const' in coefficients:
                    del coefficients['const']
                
                # Store sensitivity for this time point
                date_str = current_date.strftime('%Y-%m-%d')
                rolling_sensitivities[date_str] = {
                    factor: float(coef) for factor, coef in coefficients.items()
                }
                
            except Exception as e:
                # Ignore error, continue to next time point
                continue
        
        # Analyze sensitivity trends
        factor_trends = {}
        for factor in macro_factors_df.columns:
            # Extract sensitivity for this factor at all time points
            factor_sensitivities = [
                sensitivities.get(factor, np.nan) 
                for sensitivities in rolling_sensitivities.values()
            ]
            
            # Remove NaN values
            factor_sensitivities = [s for s in factor_sensitivities if not np.isnan(s)]
            
            if len(factor_sensitivities) >= 3:  # Need at least 3 points to analyze trend
                # Simple linear regression to detect trend
                X = np.arange(len(factor_sensitivities)).reshape(-1, 1)
                y = np.array(factor_sensitivities)
                
                try:
                    model = sm.OLS(y, sm.add_constant(X))
                    fit_result = model.fit()
                    
                    # Extract slope and p-value
                    slope = fit_result.params[1]
                    p_value = fit_result.pvalues[1]
                    
                    # Determine trend
                    if p_value < 0.05:  # Significant trend
                        trend = "increasing" if slope > 0 else "decreasing"
                    else:
                        trend = "stable"
                    
                    factor_trends[factor] = {
                        'slope': float(slope),
                        'p_value': float(p_value),
                        'trend': trend,
                        'start_sensitivity': float(factor_sensitivities[0]),
                        'end_sensitivity': float(factor_sensitivities[-1]),
                        'change': float(factor_sensitivities[-1] - factor_sensitivities[0])
                    }
                    
                    # Add to trend summary
                    if trend != "stable":
                        results['trend_summary'].append(
                            f"{asset} sensitivity to {factor} shows {trend} trend (slope: {slope:.4f})"
                        )
                        
                except Exception as e:
                    # Ignore error, continue to next factor
                    continue
        
        # Store rotation analysis results for this asset
        results['rotation_analysis'][asset] = {
            'rolling_sensitivities': rolling_sensitivities,
            'factor_trends': factor_trends
        }
    
    return results