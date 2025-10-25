"""
Risk-Free Rate API - Provides risk-free rate data acquisition functionality
"""

import os
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
from .base import logger

def get_risk_free_rate(start_date: str = None, end_date: str = None, 
                      freq: str = 'D', use_cache: bool = True) -> pd.Series:
    """
    Get risk-free rate data (using interbank lending rate or treasury bond yield)
    
    Args:
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly
        use_cache: Whether to use cached data
        
    Returns:
        Series containing risk-free rate, indexed by date
    """
    logger.info(f"Getting risk-free rate data: {start_date} to {end_date}, frequency: {freq}")
    
    # Cache file path
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"risk_free_rate_{freq}.csv")
    
    # Check cache
    if use_cache and os.path.exists(cache_file):
        try:
            cache_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logger.info(f"Successfully loaded risk-free rate data from cache: {len(cache_data)} records")
            
            # Filter date range
            if start_date:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                cache_data = cache_data[cache_data.index >= start_date]
            if end_date:
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                cache_data = cache_data[cache_data.index <= end_date]
                
            if not cache_data.empty:
                rf_series = cache_data["risk_free_rate"]
                return rf_series
        except Exception as e:
            logger.warning(f"Failed to load risk-free rate data from cache: {e}")
    
    try:
        # Standardize date format
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        elif isinstance(end_date, pd.Timestamp):
            end_date = end_date.strftime("%Y-%m-%d")
            
        if not start_date:
            # Default to get one year of data
            if isinstance(end_date, str):
                start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
            else:
                start_date = (end_date - timedelta(days=365)).strftime("%Y-%m-%d")
        elif isinstance(start_date, pd.Timestamp):
            start_date = start_date.strftime("%Y-%m-%d")
        
        # Try using AKShare to get data
        try:
            import akshare as ak
            logger.info("Using AKShare to get government bond yield")
            
            try:
                # Try using China-US government bond yield interface
                bond_data = ak.bond_zh_us_rate(start_date=start_date.replace('-', ''))
                
                # Rename columns to match expected format
                bond_data = bond_data.rename(columns={"date": "date", "China 10-year Treasury Yield": "risk_free_rate"})
                bond_data["date"] = pd.to_datetime(bond_data["date"])
                
                # Select needed columns and filter date range
                bond_data = bond_data[["date", "risk_free_rate"]]
                bond_data = bond_data[(bond_data['date'] >= pd.to_datetime(start_date)) & 
                                    (bond_data['date'] <= pd.to_datetime(end_date))]
                
                # Convert percentage to decimal
                bond_data["risk_free_rate"] = pd.to_numeric(bond_data["risk_free_rate"], errors="coerce") / 100
                
                # Set index and sort
                rf_data = bond_data.set_index("date").sort_index()
                
                # Resample to specified frequency
                if freq == 'W':
                    rf_data = rf_data.resample('W-FRI').last().fillna(method='ffill')
                    # Convert annualized rate to weekly rate
                    rf_data = rf_data / 52
                elif freq == 'M':
                    rf_data = rf_data.resample('M').last().fillna(method='ffill')
                    # Convert annualized rate to monthly rate
                    rf_data = rf_data / 12
                else:  # Daily frequency
                    # Convert annualized rate to daily rate
                    rf_data = rf_data / 252
                
                # Save to cache
                try:
                    rf_data.to_csv(cache_file)
                    logger.info(f"Risk-free rate data saved to cache: {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to save risk-free rate data to cache: {e}")
                
                return rf_data["risk_free_rate"]
                
            except Exception as e1:
                logger.warning(f"Failed to get treasury bond yield using AKShare: {e1}")
                raise Exception("Failed to get treasury bond yield data from AKShare")
                
        except Exception as e:
            logger.warning(f"Failed to get risk-free rate using AKShare: {e}")
            logger.warning("Trying to get data using TuShare")
        
        # Try using TuShare to get data
        try:
            import tushare as ts
            token = os.environ.get('TUSHARE_TOKEN', '')
            if not token:
                logger.warning("TUSHARE_TOKEN environment variable not found")
                raise ValueError("TuShare token not available")
                
            ts.set_token(token)
            pro = ts.pro_api()
            
            # Standardize date format to YYYYMMDD (TuShare format)
            if isinstance(start_date, str) and len(start_date) == 10 and '-' in start_date:
                ts_start_date = start_date.replace('-', '')
            else:
                ts_start_date = start_date
                
            if isinstance(end_date, str) and len(end_date) == 10 and '-' in end_date:
                ts_end_date = end_date.replace('-', '')
            else:
                ts_end_date = end_date
            
            # Use shibor interface to get Shibor data
            try:
                shibor_data = pro.shibor(start_date=ts_start_date, end_date=ts_end_date)
                
                if not shibor_data.empty:
                    # Use 1-year shibor (1y column)
                    shibor_data = shibor_data[['date', '1y']]
                    shibor_data = shibor_data.rename(columns={'1y': 'risk_free_rate'})
                    
                    # Convert format
                    shibor_data['date'] = pd.to_datetime(shibor_data['date'])
                    shibor_data['risk_free_rate'] = pd.to_numeric(shibor_data['risk_free_rate'], errors='coerce') / 100  # Convert to decimal
                    
                    # Set index and sort
                    shibor_data = shibor_data.set_index('date').sort_index()
                    
                    # Resample to specified frequency
                    if freq == 'W':
                        shibor_data = shibor_data.resample('W-FRI').last().fillna(method='ffill')
                        # Convert annualized rate to weekly rate
                        shibor_data = shibor_data / 52
                    elif freq == 'M':
                        shibor_data = shibor_data.resample('M').last().fillna(method='ffill')
                        # Convert annualized rate to monthly rate
                        shibor_data = shibor_data / 12
                    else:  # Daily frequency
                        # Convert annualized rate to daily rate
                        shibor_data = shibor_data / 252
                    
                    # Save to cache
                    try:
                        shibor_data.to_csv(cache_file)
                        logger.info(f"Risk-free rate data saved to cache: {cache_file}")
                    except Exception as e:
                        logger.warning(f"Failed to save risk-free rate data to cache: {e}")
                    
                    return shibor_data["risk_free_rate"]
                else:
                    logger.warning("TuShare Shibor data is empty, using mock data")
                    return _generate_mock_risk_free_rate(start_date, end_date, freq)
            except Exception as e:
                logger.warning(f"Failed to get TuShare Shibor data: {e}")
                logger.warning("Will use mock data")
                return _generate_mock_risk_free_rate(start_date, end_date, freq)
        except Exception as e:
            logger.warning(f"Failed to get risk-free rate using TuShare: {e}")
            logger.warning("Will use mock data")
    
    except Exception as e:
        logger.error(f"Error getting risk-free rate: {e}")
        logger.error(traceback.format_exc())
    
    # If all attempts fail, use mock data
    return _generate_mock_risk_free_rate(start_date, end_date, freq)

def _generate_mock_risk_free_rate(start_date: str, end_date: str, freq: str = 'D') -> pd.Series:
    """
    Generate mock risk-free rate data
    
    Args:
        start_date: Start date
        end_date: End date
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly
        
    Returns:
        Mock risk-free rate Series
    """
    logger.info(f"Generating mock risk-free rate data, frequency: {freq}")
    
    # Handle date parameters
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        # Default to generate one year of data
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Ensure consistent date format
    if isinstance(start_date, str) and len(start_date) == 8:
        start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    if isinstance(end_date, str) and len(end_date) == 8:
        end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    
    # Generate date range based on frequency
    if freq == 'W':
        date_range = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
        # Annualized rate around 2.5%, convert to weekly rate
        base_rate = 0.025 / 52
    elif freq == 'M':
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        # Annualized rate around 2.5%, convert to monthly rate
        base_rate = 0.025 / 12
    else:  # Daily frequency
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        # Annualized rate around 2.5%, convert to daily rate
        base_rate = 0.025 / 252
    
    # Add some random fluctuation
    np.random.seed(42)  # Set random seed for reproducibility
    rates = np.random.normal(base_rate, base_rate * 0.05, len(date_range))
    
    # Create Series
    rf_series = pd.Series(rates, index=date_range)
    
    return rf_series