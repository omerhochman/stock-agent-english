"""
Model Estimation API - Provides estimation functionality for various financial models
"""

import pandas as pd
import numpy as np
import traceback
from typing import Dict, List

from .base import logger
from .market_data_api import get_market_returns, get_stock_returns, get_multi_stock_returns
from .fama_french_api import get_fama_french_factors
from .risk_free_api import get_risk_free_rate

def estimate_capm_for_stock(stock_symbol: str,
                           start_date: str = None,
                           end_date: str = None,
                           freq: str = 'D') -> Dict[str, float]:
    """
    Estimate CAPM model parameters for a single stock
    
    Args:
        stock_symbol: Stock code
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly
        
    Returns:
        CAPM model parameters dictionary
    """
    logger.info(f"Estimating CAPM model parameters for stock {stock_symbol}: {start_date} to {end_date}, frequency: {freq}")
    
    try:
        # 1. Get stock returns
        stock_returns_dict = get_stock_returns(stock_symbol, start_date, end_date, freq)
        if not stock_returns_dict or stock_symbol not in stock_returns_dict:
            logger.warning(f"Unable to get stock {stock_symbol} return data")
            return {}
        
        stock_returns = stock_returns_dict[stock_symbol]
        
        # 2. Get market returns (using CSI 300)
        market_returns = get_market_returns("000300", start_date, end_date, freq)
        
        # 3. Get risk-free rate
        risk_free_rate = get_risk_free_rate(start_date, end_date, freq)
        
        # 4. Ensure all data uses the same date index
        common_index = stock_returns.index.intersection(market_returns.index)
        if not risk_free_rate.empty:
            common_index = common_index.intersection(risk_free_rate.index)
        
        if len(common_index) < 20:
            logger.warning(f"Too few data points to reliably estimate CAPM model: {len(common_index)} records")
            return {}
        
        stock_returns = stock_returns.loc[common_index]
        market_returns = market_returns.loc[common_index]
        
        # 5. Estimate CAPM model
        from src.calc.factor_models import estimate_capm
        
        if risk_free_rate.empty:
            capm_results = estimate_capm(stock_returns, market_returns)
        else:
            risk_free_rate = risk_free_rate.loc[common_index]
            capm_results = estimate_capm(stock_returns, market_returns, risk_free_rate)
        
        logger.info(f"Successfully estimated CAPM model parameters for stock {stock_symbol}")
        return capm_results
        
    except Exception as e:
        logger.error(f"Error estimating CAPM model parameters: {e}")
        logger.error(traceback.format_exc())
        return {}

def estimate_fama_french_for_stock(stock_symbol: str,
                                 start_date: str = None,
                                 end_date: str = None,
                                 freq: str = 'D') -> Dict[str, float]:
    """
    Estimate Fama-French three-factor model parameters for a single stock
    
    Args:
        stock_symbol: Stock code
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly
        
    Returns:
        Fama-French three-factor model parameters dictionary
    """
    logger.info(f"Estimating Fama-French three-factor model parameters for stock {stock_symbol}: {start_date} to {end_date}, frequency: {freq}")
    
    try:
        # 1. Get stock returns
        stock_returns_dict = get_stock_returns(stock_symbol, start_date, end_date, freq)
        if not stock_returns_dict or stock_symbol not in stock_returns_dict:
            logger.warning(f"Unable to get stock {stock_symbol} return data")
            return {}
        
        stock_returns = stock_returns_dict[stock_symbol]
        
        # 2. Get Fama-French three factors
        ff_factors = get_fama_french_factors(start_date, end_date, freq)
        
        if not ff_factors:
            logger.warning("Unable to get Fama-French three-factor data")
            return {}
        
        # 3. Ensure all data uses the same date index
        common_index = stock_returns.index.intersection(ff_factors['market_returns'].index)
        
        if len(common_index) < 20:
            logger.warning(f"Too few data points to reliably estimate Fama-French model: {len(common_index)} records")
            return {}
        
        stock_returns = stock_returns.loc[common_index]
        market_returns = ff_factors['market_returns'].loc[common_index]
        smb = ff_factors['smb'].loc[common_index]
        hml = ff_factors['hml'].loc[common_index]
        risk_free_rate = ff_factors['risk_free_rate'].loc[common_index]
        
        # 4. Estimate Fama-French model
        from src.calc.factor_models import estimate_fama_french
        
        ff_results = estimate_fama_french(
            stock_returns, market_returns, smb, hml, risk_free_rate
        )
        
        logger.info(f"Successfully estimated Fama-French three-factor model parameters for stock {stock_symbol}")
        return ff_results
        
    except Exception as e:
        logger.error(f"Error estimating Fama-French model parameters: {e}")
        logger.error(traceback.format_exc())
        return {}

def estimate_beta_for_stocks(symbols: List[str],
                            start_date: str = None,
                            end_date: str = None,
                            freq: str = 'D') -> pd.DataFrame:
    """
    Estimate beta coefficients for multiple stocks
    
    Args:
        symbols: List of stock codes
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly
        
    Returns:
        DataFrame containing beta coefficients
    """
    logger.info(f"Estimating beta coefficients for multiple stocks: {start_date} to {end_date}, stock count: {len(symbols)}, frequency: {freq}")
    
    # Get stock returns data
    returns_df = get_multi_stock_returns(symbols, start_date, end_date, freq)
    
    if returns_df.empty:
        logger.warning("Unable to get stock return data")
        return pd.DataFrame()
    
    # Get market returns
    market_returns = get_market_returns("000300", start_date, end_date, freq)
    
    if market_returns.empty:
        logger.warning("Unable to get market return data")
        return pd.DataFrame()
    
    # Ensure returns and market returns use the same dates
    common_index = returns_df.index.intersection(market_returns.index)
    returns_df = returns_df.loc[common_index]
    market_returns = market_returns.loc[common_index]
    
    # Calculate beta coefficients
    betas = {}
    r_squareds = {}
    
    for symbol in returns_df.columns:
        # Use linear regression to calculate beta coefficient
        import statsmodels.api as sm
        
        X = sm.add_constant(market_returns)
        y = returns_df[symbol]
        
        try:
            model = sm.OLS(y, X).fit()
            betas[symbol] = model.params.iloc[1]  # Beta coefficient
            r_squareds[symbol] = model.rsquared  # R-squared
        except Exception as e:
            logger.warning(f"Error calculating beta coefficient for stock {symbol}: {e}")
            betas[symbol] = np.nan
            r_squareds[symbol] = np.nan
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'symbol': list(betas.keys()),
        'beta': list(betas.values()),
        'r_squared': list(r_squareds.values())
    })
    
    logger.info(f"Successfully estimated beta coefficients for {len(result_df)} stocks")
    return result_df

def calculate_rolling_beta(stock_symbol: str,
                          window: int = 60,
                          start_date: str = None,
                          end_date: str = None,
                          freq: str = 'D') -> pd.Series:
    """
    Calculate rolling beta coefficient for a stock
    
    Args:
        stock_symbol: Stock code
        window: Rolling window size (number of trading days)
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly
        
    Returns:
        Rolling beta coefficient Series
    """
    logger.info(f"Calculating rolling beta coefficient for stock {stock_symbol}: {start_date} to {end_date}, window size: {window}")
    
    # Get stock returns data
    stock_returns_dict = get_stock_returns(stock_symbol, start_date, end_date, freq)
    
    if not stock_returns_dict or stock_symbol not in stock_returns_dict:
        logger.warning(f"Unable to get stock {stock_symbol} return data")
        return pd.Series()
    
    stock_returns = stock_returns_dict[stock_symbol]
    
    # Get market returns
    market_returns = get_market_returns("000300", start_date, end_date, freq)
    
    if market_returns.empty:
        logger.warning("Unable to get market return data")
        return pd.Series()
    
    # Ensure returns and market returns use the same dates
    common_index = stock_returns.index.intersection(market_returns.index)
    stock_returns = stock_returns.loc[common_index]
    market_returns = market_returns.loc[common_index]
    
    # Ensure data length is sufficient
    if len(stock_returns) < window + 10:
        logger.warning(f"Insufficient data length, cannot calculate rolling beta: {len(stock_returns)} < {window + 10}")
        return pd.Series()
    
    # Calculate rolling beta coefficients
    rolling_betas = pd.Series(index=stock_returns.index[window-1:], dtype=float)
    
    for i in range(window-1, len(stock_returns)):
        # Get data within the window
        stock_window = stock_returns.iloc[i-window+1:i+1]
        market_window = market_returns.iloc[i-window+1:i+1]
        
        # Use linear regression to calculate beta coefficient
        import statsmodels.api as sm
        
        X = sm.add_constant(market_window)
        y = stock_window
        
        try:
            model = sm.OLS(y, X).fit()
            rolling_betas.loc[stock_returns.index[i]] = model.params.iloc[1]  # Beta coefficient
        except Exception as e:
            logger.debug(f"Error calculating beta coefficient for {stock_returns.index[i]}: {e}")
            rolling_betas.loc[stock_returns.index[i]] = np.nan
    
    # Remove missing values
    rolling_betas = rolling_betas.dropna()
    
    logger.info(f"Successfully calculated rolling beta coefficient for stock {stock_symbol}: {len(rolling_betas)} records")
    return rolling_betas