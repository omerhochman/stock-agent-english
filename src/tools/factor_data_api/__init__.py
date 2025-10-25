"""
Factor Data Acquisition Module - Provides CAPM and Fama-French factor data acquisition functionality

This module contains functions for obtaining market risk premium, size factor and value factor data,
used for estimating CAPM and Fama-French three-factor models.
Supports both TuShare and AKShare data interfaces.

Main Features:
1. Get risk-free rate data
2. Get market and stock return data
3. Get and calculate Fama-French three-factor data
4. Get industry and style index data
5. Get macroeconomic data
6. Estimate CAPM and Fama-French model parameters for stocks

Usage Example:
```python
# Get market return data
from src.tools.factor_data_api import get_market_returns
market_ret = get_market_returns(start_date="2022-01-01", end_date="2022-12-31", freq="D")

# Get stock return data
from src.tools.factor_data_api import get_stock_returns
stock_ret = get_stock_returns(["000001", "600000"], start_date="2022-01-01", end_date="2022-12-31")

# Get Fama-French three-factor data
from src.tools.factor_data_api import get_fama_french_factors
ff_factors = get_fama_french_factors(start_date="2022-01-01", end_date="2022-12-31")

# Estimate CAPM model parameters for stock
from src.tools.factor_data_api import estimate_capm_for_stock
capm_results = estimate_capm_for_stock("000001", start_date="2022-01-01", end_date="2022-12-31")
```
"""

from .base import setup_logger
from .risk_free_api import (
    get_risk_free_rate,
    _generate_mock_risk_free_rate
)
from .market_data_api import (
    get_market_returns,
    _generate_mock_market_returns,
    get_stock_returns,
    get_multi_stock_returns,
    get_stock_covariance_matrix,
    get_index_data,
    get_multiple_index_data
)
from .fama_french_api import (
    calculate_fama_french_factors_tushare,
    get_fama_french_factors,
    _generate_mock_fama_french_factors
)
from .industry_api import (
    get_industry_index_returns,
    get_industry_rotation_factors,
    get_sector_index_returns,
    get_style_index_returns
)
from .macro_api import (
    get_macro_economic_data,
    _generate_mock_macro_data
)
from .model_estimator_api import (
    estimate_capm_for_stock,
    estimate_fama_french_for_stock,
    estimate_beta_for_stocks,
    calculate_rolling_beta
)

__all__ = [
    'get_risk_free_rate',
    'get_market_returns',
    'get_stock_returns',
    'get_multi_stock_returns',
    'get_stock_covariance_matrix',
    'get_index_data',
    'get_multiple_index_data',
    'calculate_fama_french_factors_tushare',
    'get_fama_french_factors',
    'get_industry_index_returns',
    'get_industry_rotation_factors',
    'get_sector_index_returns',
    'get_style_index_returns',
    'get_macro_economic_data',
    'estimate_capm_for_stock',
    'estimate_fama_french_for_stock',
    'estimate_beta_for_stocks',
    'calculate_rolling_beta'
]

#----------------------------------------------------------------------------#
# Risk-free Rate API Function Documentation
#----------------------------------------------------------------------------#

# def get_risk_free_rate(start_date=None, end_date=None, freq='D', use_cache=True):
#     """
#     Get risk-free rate data (interbank lending rate or treasury bond yield)
    
#     Parameters:
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'D'
#         use_cache (bool, optional): Whether to use cache, default is True
        
#     Returns:
#         pd.Series: Series containing risk-free rates, indexed by date
        
#     Example:
#         >>> rf = get_risk_free_rate(start_date="2022-01-01", end_date="2022-12-31", freq="D")
#         >>> print(rf.head())
#         2022-01-01    0.000099
#         2022-01-02    0.000099
#         2022-01-03    0.000099
#         2022-01-04    0.000098
#         2022-01-05    0.000098
#         Name: risk_free_rate, dtype: float64
#     """
#     # Function implementation is in risk_free_api.py
#     pass

# #----------------------------------------------------------------------------#
# # Market Data API Function Documentation
# #----------------------------------------------------------------------------#

# def get_market_returns(index_code="000300", start_date=None, end_date=None, freq='D'):
#     """
#     Get market return data, using CSI 300 index by default
    
#     Parameters:
#         index_code (str, optional): Index code, default is CSI 300("000300")
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'D'
        
#     Returns:
#         pd.Series: Series containing market returns, indexed by date
        
#     Example:
#         >>> market_ret = get_market_returns(start_date="2022-01-01", end_date="2022-12-31", freq="D")
#         >>> print(market_ret.head())
#         2022-01-04    0.018242
#         2022-01-05   -0.016501
#         2022-01-06    0.008458
#         2022-01-07   -0.000097
#         2022-01-10   -0.013861
#         dtype: float64
#     """
#     # Function implementation is in market_data_api.py
#     pass

# def get_stock_returns(symbols, start_date=None, end_date=None, freq='D'):
#     """
#     Get return data for one or more stocks
    
#     Parameters:
#         symbols (str or list): Single stock code or list of stock codes
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'D'
        
#     Returns:
#         dict: Dictionary containing stock returns, keys are stock codes, values are return Series
        
#     Example:
#         >>> stock_ret = get_stock_returns(["000001", "600000"], start_date="2022-01-01", end_date="2022-01-10")
#         >>> print(stock_ret["000001"].head())
#         2022-01-04    0.0125
#         2022-01-05   -0.0080
#         2022-01-06    0.0045
#         2022-01-07   -0.0020
#         2022-01-10   -0.0075
#         Name: 000001, dtype: float64
#     """
#     # Function implementation is in market_data_api.py
#     pass

# def get_multi_stock_returns(symbols, start_date=None, end_date=None, freq='D'):
#     """
#     Get return data for multiple stocks and return as DataFrame
    
#     Parameters:
#         symbols (list): List of stock codes
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'D'
        
#     Returns:
#         pd.DataFrame: DataFrame containing multiple stock returns, columns are stock codes, indexed by date
        
#     Example:
#         >>> returns_df = get_multi_stock_returns(["000001", "600000", "601398"], 
#                                                start_date="2022-01-01", end_date="2022-01-10")
#         >>> print(returns_df.head())
#                      000001    600000    601398
#         2022-01-04   0.0125    0.0205    0.0098
#         2022-01-05  -0.0080   -0.0150   -0.0054
#         2022-01-06   0.0045    0.0089    0.0032
#         2022-01-07  -0.0020   -0.0010   -0.0015
#         2022-01-10  -0.0075   -0.0120   -0.0065
#     """
#     # Function implementation is in market_data_api.py
#     pass

# def get_stock_covariance_matrix(symbols, start_date=None, end_date=None, method="sample", freq='D'):
#     """
#     Calculate covariance matrix and average returns for multiple stock returns
    
#     Parameters:
#         symbols (list): List of stock codes
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         method (str, optional): Covariance matrix estimation method, options "sample" or "ewma", default is "sample"
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'D'
        
#     Returns:
#         tuple: (Covariance matrix DataFrame, Average returns Series)
        
#     Example:
#         >>> cov_matrix, expected_returns = get_stock_covariance_matrix(
#                 ["000001", "600000", "601398"], 
#                 start_date="2022-01-01", 
#                 end_date="2022-12-31"
#             )
#         >>> print(cov_matrix)
#                    000001     600000     601398
#         000001   0.000358   0.000125   0.000098
#         600000   0.000125   0.000402   0.000105
#         601398   0.000098   0.000105   0.000278
#         >>> print(expected_returns)
#         000001    0.0825
#         600000    0.0514
#         601398    0.0327
#         dtype: float64
#     """
#     # Function implementation is in market_data_api.py
#     pass

# def get_index_data(index_symbol="000300", fields=None, start_date=None, end_date=None, freq='D'):
#     """
#     Get index data
    
#     Parameters:
#         index_symbol (str, optional): Index code, default is CSI 300("000300")
#         fields (list, optional): List of fields to retrieve, if None then get all available fields
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'D'
        
#     Returns:
#         pd.DataFrame: DataFrame containing index data
        
#     Example:
#         >>> index_data = get_index_data("000300", fields=["date", "open", "close"], 
#                                        start_date="2022-01-01", end_date="2022-01-10")
#         >>> print(index_data.head())
#                 date      open     close
#         0  2022-01-04  4921.23   5005.89
#         1  2022-01-05  5010.45   4920.23
#         2  2022-01-06  4915.67   4961.81
#         3  2022-01-07  4963.45   4961.33
#         4  2022-01-10  4960.12   4898.54
#     """
#     # Function implementation is in market_data_api.py
#     pass

# def get_multiple_index_data(index_symbols, start_date=None, end_date=None, freq='D'):
#     """
#     Get data for multiple indices
    
#     Parameters:
#         index_symbols (list): List of index codes
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'D'
        
#     Returns:
#         dict: Dictionary containing multiple index data, keys are index codes, values are index data DataFrames
        
#     Example:
#         >>> index_dict = get_multiple_index_data(["000300", "000016", "000905"], 
#                                                start_date="2022-01-01", end_date="2022-01-10")
#         >>> print(index_dict["000300"].head())
#                 date      open     high      low     close      volume       amount
#         0  2022-01-04  4921.23  5010.42  4899.21   5005.89  23491782.0  287541987.0
#         1  2022-01-05  5010.45  5015.74  4915.95   4920.23  26485732.0  298457612.0
#         2  2022-01-06  4915.67  4978.56  4902.45   4961.81  22154578.0  256412378.0
#         ...
#     """
#     # Function implementation is in market_data_api.py
#     pass

#----------------------------------------------------------------------------#
# Fama-French Three-Factor API Function Documentation
#----------------------------------------------------------------------------#

# def get_fama_french_factors(start_date=None, end_date=None, freq='D', use_cache=True):
#     """
#     Get Fama-French three-factor model factor data
    
#     Parameters:
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'D'
#         use_cache (bool, optional): Whether to use cache, default is True
        
#     Returns:
#         dict: Dictionary containing the following factors:
#             - 'market_returns': Market returns (pd.Series)
#             - 'market_excess_returns': Market excess returns (pd.Series)
#             - 'smb': Size factor (pd.Series)
#             - 'hml': Value factor (pd.Series)
#             - 'risk_free_rate': Risk-free rate (pd.Series)
        
#     Example:
#         >>> ff_factors = get_fama_french_factors(start_date="2022-01-01", end_date="2022-12-31", freq="D")
#         >>> print(ff_factors["market_returns"].head())
#         2022-01-04    0.018242
#         2022-01-05   -0.016501
#         2022-01-06    0.008458
#         2022-01-07   -0.000097
#         2022-01-10   -0.013861
#         Name: market_returns, dtype: float64
#         >>> print(ff_factors["smb"].head())
#         2022-01-04    0.001245
#         2022-01-05   -0.000854
#         2022-01-06    0.000987
#         2022-01-07    0.001532
#         2022-01-10   -0.002145
#         Name: smb, dtype: float64
#     """
#     # Function implementation is in fama_french_api.py
#     pass

# def calculate_fama_french_factors_tushare(start_date, end_date, freq='W'):
#     """
#     Calculate Fama-French three-factor data using TuShare
    
#     Parameters:
#         start_date (str): Start date, format: YYYY-MM-DD or YYYYMMDD
#         end_date (str): End date, format: YYYY-MM-DD or YYYYMMDD
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'W'
        
#     Returns:
#         dict: Dictionary containing the following factors:
#             - 'market_returns': Market returns (pd.Series)
#             - 'market_excess_returns': Market excess returns (pd.Series)
#             - 'smb': Size factor (pd.Series)
#             - 'hml': Value factor (pd.Series)
#             - 'risk_free_rate': Risk-free rate (pd.Series)
    
#     Note:
#         This function directly calls TuShare API to calculate factors, requires valid TuShare token and tushare library installation
#     """
#     # Function implementation is in fama_french_api.py
#     pass

#----------------------------------------------------------------------------#
# Industry Data API Function Documentation
#----------------------------------------------------------------------------#

# def get_industry_index_returns(industry_codes, start_date=None, end_date=None, freq='D'):
#     """
#     Get industry index return data
    
#     Parameters:
#         industry_codes (str or list): Industry index code or list of codes, e.g., "801780"(banking)
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'D'
        
#     Returns:
#         pd.DataFrame: DataFrame containing multiple industry index returns, columns are industry names, indexed by date
        
#     Example:
#         >>> industry_returns = get_industry_index_returns(["801120", "801780"], 
#                                                        start_date="2022-01-01", end_date="2022-01-10")
#         >>> print(industry_returns.head())
#                      Food & Beverage  Banking
#         2022-01-04   0.0145    0.0082
#         2022-01-05  -0.0075   -0.0045
#         2022-01-06   0.0058    0.0025
#         2022-01-07  -0.0035   -0.0018
#         2022-01-10  -0.0089   -0.0062
#     """
#     # Function implementation is in industry_api.py
#     pass

# def get_industry_rotation_factors(start_date=None, end_date=None, freq='W'):
#     """
#     Get industry rotation factor data (including industry momentum, valuation, growth, etc.)
    
#     Parameters:
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         freq (str, optional): Data frequency, recommended 'W' for weekly or 'M' for monthly, default is 'W'
        
#     Returns:
#         pd.DataFrame: DataFrame containing industry rotation factors
        
#     Example:
#         >>> rotation_factors = get_industry_rotation_factors(start_date="2022-01-01", end_date="2022-12-31")
#         >>> print(rotation_factors.columns)
#         Index(['Food_Beverage_MOM_1M', 'Food_Beverage_MOM_3M', 'Food_Beverage_MOM_6M', 'Banking_MOM_1M', ...])
#         >>> print(rotation_factors.head())
#                      Food_Beverage_MOM_1M  Food_Beverage_MOM_3M  Food_Beverage_MOM_6M  Banking_MOM_1M  ...
#         2022-01-28       0.027546       0.056821       0.125478   0.017892  ...
#         ...
#     """
#     # Function implementation is in industry_api.py
#     pass

# def get_sector_index_returns(start_date=None, end_date=None, freq='D'):
#     """
#     Get major sector index return data (Shanghai Composite, Shenzhen Component, ChiNext, SME Board, etc.)
    
#     Parameters:
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'D'
        
#     Returns:
#         pd.DataFrame: DataFrame containing major sector index returns, columns are sector names, indexed by date
        
#     Example:
#         >>> sector_returns = get_sector_index_returns(start_date="2022-01-01", end_date="2022-01-10")
#         >>> print(sector_returns.head())
#                      Shanghai_Composite  Shenzhen_Component  ChiNext  CSI_300  ...
#         2022-01-04   0.0182   0.0215   0.0298   0.0192  ...
#         2022-01-05  -0.0165  -0.0195  -0.0254  -0.0170  ...
#         ...
#     """
#     # Function implementation is in industry_api.py
#     pass

# def get_style_index_returns(start_date=None, end_date=None, freq='D'):
#     """
#     Get style index return data (large-cap growth, small-cap value, etc.)
    
#     Parameters:
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'D'
        
#     Returns:
#         pd.DataFrame: DataFrame containing style index returns, columns are style index names, indexed by date
        
#     Example:
#         >>> style_returns = get_style_index_returns(start_date="2022-01-01", end_date="2022-01-10")
#         >>> print(style_returns.head())
#                      300_Value  300_Growth  CSI_Dividend  Fundamental_50  ...
#         2022-01-04   0.0141   0.0235   0.0125   0.0172  ...
#         2022-01-05  -0.0165  -0.0215  -0.0074  -0.0145  ...
#         ...
#     """
#     # Function implementation is in industry_api.py
#     pass

#----------------------------------------------------------------------------#
# Macroeconomic Data API Function Documentation
#----------------------------------------------------------------------------#

# def get_macro_economic_data(indicator_type="gdp", start_date=None, end_date=None):
#     """
#     Get macroeconomic indicator data
    
#     Parameters:
#         indicator_type (str, optional): Indicator type, options:
#                                   "gdp": Gross Domestic Product
#                                   "cpi": Consumer Price Index
#                                   "interest_rate": Interest Rate
#                                   "m2": Money Supply
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
        
#     Returns:
#         pd.DataFrame: DataFrame containing macroeconomic data
        
#     Example:
#         >>> gdp_data = get_macro_economic_data("gdp", start_date="2018-01-01", end_date="2022-12-31")
#         >>> print(gdp_data.head())
#                 date  GDP_Quarterly  GDP_Cumulative  GDP_Year_Over_Year_Growth
#         0  2018-03-31     213456.78     213456.78           6.8
#         1  2018-06-30     225678.90     439135.68           6.7
#         ...
        
#         >>> cpi_data = get_macro_economic_data("cpi", start_date="2022-01-01", end_date="2022-12-31")
#         >>> print(cpi_data.head())
#                 date  National_Year_Over_Year  National_Month_Over_Month  Urban_Year_Over_Year  Rural_Year_Over_Year
#         0  2022-01-31     1.5    0.40      1.5      1.4
#         1  2022-02-28     0.9   -0.30      0.9      0.8
#         ...
#     """
#     # Function implementation is in macro_api.py
#     pass

#----------------------------------------------------------------------------#
# Model Estimation API Function Documentation
#----------------------------------------------------------------------------#

# def estimate_capm_for_stock(stock_symbol, start_date=None, end_date=None, freq='D'):
#     """
#     Estimate CAPM model parameters for a single stock
    
#     Parameters:
#         stock_symbol (str): Stock code
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'D'
        
#     Returns:
#         dict: CAPM model parameter dictionary, including:
#              - 'alpha': Alpha value, representing excess return
#              - 'beta': Beta coefficient, representing systematic risk
#              - 'r_squared': R-squared goodness of fit
#              - 'p_value': Significance P-value
#              - 't_stat': T-statistic
#              - 'std_error': Standard error
        
#     Example:
#         >>> capm_results = estimate_capm_for_stock("000001", start_date="2022-01-01", end_date="2022-12-31")
#         >>> print(capm_results)
#         {'alpha': 0.000125, 'beta': 1.234, 'r_squared': 0.785, 'p_value': 0.0002, 't_stat': 8.976, 'std_error': 0.137}
#     """
#     # Function implementation is in model_estimator_api.py
#     pass

# def estimate_fama_french_for_stock(stock_symbol, start_date=None, end_date=None, freq='D'):
#     """
#     Estimate Fama-French three-factor model parameters for a single stock
    
#     Parameters:
#         stock_symbol (str): Stock code
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'D'
        
#     Returns:
#         dict: Fama-French three-factor model parameter dictionary, including:
#              - 'alpha': Alpha value, representing excess return
#              - 'beta': Market factor coefficient
#              - 'smb': Size factor coefficient
#              - 'hml': Value factor coefficient
#              - 'r_squared': R-squared goodness of fit
#              - 'p_value_beta': Market factor significance P-value
#              - 'p_value_smb': Size factor significance P-value
#              - 'p_value_hml': Value factor significance P-value
        
#     Example:
#         >>> ff_results = estimate_fama_french_for_stock("000001", start_date="2022-01-01", end_date="2022-12-31")
#         >>> print(ff_results)
#         {'alpha': 0.000087, 'beta': 1.156, 'smb': 0.435, 'hml': -0.267, 'r_squared': 0.812, 
#          'p_value_beta': 0.0001, 'p_value_smb': 0.0024, 'p_value_hml': 0.0375}
#     """
#     # Function implementation is in model_estimator_api.py
#     pass

# def estimate_beta_for_stocks(symbols, start_date=None, end_date=None, freq='D'):
#     """
#     Estimate beta coefficients for multiple stocks
    
#     Parameters:
#         symbols (list): List of stock codes
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'D'
        
#     Returns:
#         pd.DataFrame: DataFrame containing beta coefficients, columns include:
#                      - 'symbol': Stock code
#                      - 'beta': Beta coefficient
#                      - 'r_squared': R-squared goodness of fit
        
#     Example:
#         >>> beta_df = estimate_beta_for_stocks(["000001", "600000", "601398"], 
#                                              start_date="2022-01-01", end_date="2022-12-31")
#         >>> print(beta_df)
#             symbol     beta  r_squared
#         0   000001    1.234      0.785
#         1   600000    0.865      0.693
#         2   601398    0.912      0.722
#     """
#     # Function implementation is in model_estimator_api.py
#     pass

# def calculate_rolling_beta(stock_symbol, window=60, start_date=None, end_date=None, freq='D'):
#     """
#     Calculate rolling beta coefficient for a stock
    
#     Parameters:
#         stock_symbol (str): Stock code
#         window (int, optional): Rolling window size (number of trading days), default is 60
#         start_date (str, optional): Start date, format: YYYY-MM-DD
#         end_date (str, optional): End date, format: YYYY-MM-DD
#         freq (str, optional): Data frequency, 'D' daily, 'W' weekly, 'M' monthly, default is 'D'
        
#     Returns:
#         pd.Series: Rolling beta coefficient Series, indexed by date
        
#     Example:
#         >>> rolling_beta = calculate_rolling_beta("000001", window=60, 
#                                                start_date="2022-01-01", end_date="2022-12-31")
#         >>> print(rolling_beta.head())
#         2022-04-01    1.125
#         2022-04-04    1.137
#         2022-04-05    1.142
#         2022-04-06    1.129
#         2022-04-07    1.118
#         dtype: float64
#     """
#     # Function implementation is in model_estimator_api.py
#     pass


#----------------------------------------------------------------------------#
# Complete Usage Examples
#----------------------------------------------------------------------------#

"""
The following are complete usage examples showing how to combine various functions from factor_data_api:

1. Basic Data Retrieval and Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.factor_data_api import get_stock_returns, get_market_returns

# Get stock and market returns
start_date = "2022-01-01"
end_date = "2022-12-31"
stock_returns = get_stock_returns(["000001", "600000"], start_date, end_date)
market_returns = get_market_returns(start_date=start_date, end_date=end_date)

# Visualize returns
plt.figure(figsize=(12, 6))
plt.plot(market_returns.index, market_returns.values, label="CSI 300", linewidth=2)
plt.plot(stock_returns["000001"].index, stock_returns["000001"].values, label="000001", alpha=0.7)
plt.plot(stock_returns["600000"].index, stock_returns["600000"].values, label="600000", alpha=0.7)
plt.title("Stock vs Market Returns Comparison")
plt.xlabel("Date")
plt.ylabel("Daily Returns")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

2. Fama-French Three-Factor Retrieval and Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.factor_data_api import get_fama_french_factors, estimate_fama_french_for_stock

# Get Fama-French three-factor data
start_date = "2021-01-01"
end_date = "2022-12-31"
ff_factors = get_fama_french_factors(start_date=start_date, end_date=end_date, freq="D")

# Visualize three-factor trends
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(ff_factors["market_excess_returns"].index, ff_factors["market_excess_returns"].values)
plt.title("Market Excess Returns (Mkt-RF)")
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(ff_factors["smb"].index, ff_factors["smb"].values)
plt.title("Size Factor (SMB)")
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(ff_factors["hml"].index, ff_factors["hml"].values)
plt.title("Value Factor (HML)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Estimate Fama-French three-factor model parameters for a stock
ff_results = estimate_fama_french_for_stock("000001", start_date=start_date, end_date=end_date)
print("Fama-French Three-Factor Model Parameters:")
for key, value in ff_results.items():
    print(f"{key}: {value:.4f}")
```

3. Industry Rotation Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.tools.factor_data_api import get_industry_index_returns

# Get major industry index returns
start_date = "2022-01-01"
end_date = "2022-12-31"
industry_codes = ["801120", "801080", "801150", "801780", "801750", "801760"]  # Food & Beverage, Electronics, Healthcare, Banking, Computer, Media
industry_returns = get_industry_index_returns(industry_codes, start_date, end_date)

# Calculate cumulative returns
industry_cum_returns = (1 + industry_returns).cumprod() - 1

# Visualize cumulative returns
plt.figure(figsize=(14, 7))
for industry in industry_cum_returns.columns:
    plt.plot(industry_cum_returns.index, industry_cum_returns[industry], label=industry, linewidth=2)
plt.title("Industry Index Cumulative Returns Comparison")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate inter-industry correlations
industry_corr = industry_returns.corr()

# Visualize correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(industry_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
plt.title("Industry Index Returns Correlation")
plt.show()
```

4. Portfolio Beta Coefficient Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.factor_data_api import calculate_rolling_beta, estimate_beta_for_stocks

# Get beta coefficients for multiple stocks
stocks = ["000001", "600000", "601398", "600519", "000651", "002415"]
start_date = "2021-01-01"
end_date = "2022-12-31"
beta_df = estimate_beta_for_stocks(stocks, start_date=start_date, end_date=end_date)
print("Stock Beta Coefficients:")
print(beta_df)

# Calculate rolling beta coefficient for a specific stock
rolling_beta = calculate_rolling_beta("600519", window=60, start_date=start_date, end_date=end_date)

# Visualize rolling beta coefficient
plt.figure(figsize=(12, 6))
plt.plot(rolling_beta.index, rolling_beta.values, linewidth=2)
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
plt.title("600519 Rolling Beta Coefficient (60-day window)")
plt.xlabel("Date")
plt.ylabel("Beta Coefficient")
plt.grid(True, alpha=0.3)
plt.show()
```

5. Macroeconomic Data Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.factor_data_api import get_macro_economic_data

# Get macroeconomic data
gdp_data = get_macro_economic_data("gdp", start_date="2018-01-01")
cpi_data = get_macro_economic_data("cpi", start_date="2020-01-01")

# Visualize GDP growth rate
plt.figure(figsize=(12, 6))
plt.bar(gdp_data["date"], gdp_data["GDP_Year_Over_Year_Growth"], color='steelblue')
plt.title("GDP Year-over-Year Growth Rate")
plt.xlabel("Date")
plt.ylabel("Growth Rate (%)")
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize CPI year-over-year changes
plt.figure(figsize=(12, 6))
plt.plot(cpi_data["date"], cpi_data["National_Year_Over_Year"], linewidth=2, marker='o')
plt.title("CPI Year-over-Year Changes")
plt.xlabel("Date")
plt.ylabel("Change Rate (%)")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
"""