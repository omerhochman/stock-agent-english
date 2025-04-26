"""
因子数据获取模块 - 提供CAPM和Fama-French因子数据的获取功能

此模块包含获取市场风险溢价、规模因子和价值因子的函数，
用于进行CAPM和Fama-French三因子模型的估计。
支持TuShare和AKShare两种数据接口。

主要功能:
1. 获取无风险利率数据
2. 获取市场和股票收益率数据
3. 获取和计算Fama-French三因子数据
4. 获取行业和风格指数数据
5. 获取宏观经济数据
6. 估计股票的CAPM和Fama-French模型参数

使用示例:
```python
# 获取市场收益率数据
from src.tools.factor_data_api import get_market_returns
market_ret = get_market_returns(start_date="2022-01-01", end_date="2022-12-31", freq="D")

# 获取股票收益率数据
from src.tools.factor_data_api import get_stock_returns
stock_ret = get_stock_returns(["000001", "600000"], start_date="2022-01-01", end_date="2022-12-31")

# 获取Fama-French三因子数据
from src.tools.factor_data_api import get_fama_french_factors
ff_factors = get_fama_french_factors(start_date="2022-01-01", end_date="2022-12-31")

# 估计股票的CAPM模型参数
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
# 无风险利率API函数说明
#----------------------------------------------------------------------------#

def get_risk_free_rate(start_date=None, end_date=None, freq='D', use_cache=True):
    """
    获取无风险利率数据（银行间同业拆借利率或国债收益率）
    
    参数:
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'D'
        use_cache (bool, 可选): 是否使用缓存，默认为True
        
    返回:
        pd.Series: 包含无风险利率的Series，索引为日期
        
    示例:
        >>> rf = get_risk_free_rate(start_date="2022-01-01", end_date="2022-12-31", freq="D")
        >>> print(rf.head())
        2022-01-01    0.000099
        2022-01-02    0.000099
        2022-01-03    0.000099
        2022-01-04    0.000098
        2022-01-05    0.000098
        Name: risk_free_rate, dtype: float64
    """
    # 函数实现在risk_free_api.py中
    pass

#----------------------------------------------------------------------------#
# 市场数据API函数说明
#----------------------------------------------------------------------------#

def get_market_returns(index_code="000300", start_date=None, end_date=None, freq='D'):
    """
    获取市场收益率数据，默认使用沪深300指数
    
    参数:
        index_code (str, 可选): 指数代码，默认为沪深300("000300")
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'D'
        
    返回:
        pd.Series: 包含市场收益率的Series，索引为日期
        
    示例:
        >>> market_ret = get_market_returns(start_date="2022-01-01", end_date="2022-12-31", freq="D")
        >>> print(market_ret.head())
        2022-01-04    0.018242
        2022-01-05   -0.016501
        2022-01-06    0.008458
        2022-01-07   -0.000097
        2022-01-10   -0.013861
        dtype: float64
    """
    # 函数实现在market_data_api.py中
    pass

def get_stock_returns(symbols, start_date=None, end_date=None, freq='D'):
    """
    获取一个或多个股票的收益率数据
    
    参数:
        symbols (str 或 list): 单个股票代码或股票代码列表
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'D'
        
    返回:
        dict: 包含股票收益率的字典，键为股票代码，值为收益率Series
        
    示例:
        >>> stock_ret = get_stock_returns(["000001", "600000"], start_date="2022-01-01", end_date="2022-01-10")
        >>> print(stock_ret["000001"].head())
        2022-01-04    0.0125
        2022-01-05   -0.0080
        2022-01-06    0.0045
        2022-01-07   -0.0020
        2022-01-10   -0.0075
        Name: 000001, dtype: float64
    """
    # 函数实现在market_data_api.py中
    pass

def get_multi_stock_returns(symbols, start_date=None, end_date=None, freq='D'):
    """
    获取多个股票的收益率数据并返回为DataFrame
    
    参数:
        symbols (list): 股票代码列表
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'D'
        
    返回:
        pd.DataFrame: 包含多个股票收益率的DataFrame，列为股票代码，索引为日期
        
    示例:
        >>> returns_df = get_multi_stock_returns(["000001", "600000", "601398"], 
                                               start_date="2022-01-01", end_date="2022-01-10")
        >>> print(returns_df.head())
                     000001    600000    601398
        2022-01-04   0.0125    0.0205    0.0098
        2022-01-05  -0.0080   -0.0150   -0.0054
        2022-01-06   0.0045    0.0089    0.0032
        2022-01-07  -0.0020   -0.0010   -0.0015
        2022-01-10  -0.0075   -0.0120   -0.0065
    """
    # 函数实现在market_data_api.py中
    pass

def get_stock_covariance_matrix(symbols, start_date=None, end_date=None, method="sample", freq='D'):
    """
    计算多个股票收益率的协方差矩阵和平均收益率
    
    参数:
        symbols (list): 股票代码列表
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        method (str, 可选): 协方差矩阵估计方法，可选"sample"或"ewma"，默认为"sample"
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'D'
        
    返回:
        tuple: (协方差矩阵DataFrame, 平均收益率Series)
        
    示例:
        >>> cov_matrix, expected_returns = get_stock_covariance_matrix(
                ["000001", "600000", "601398"], 
                start_date="2022-01-01", 
                end_date="2022-12-31"
            )
        >>> print(cov_matrix)
                   000001     600000     601398
        000001   0.000358   0.000125   0.000098
        600000   0.000125   0.000402   0.000105
        601398   0.000098   0.000105   0.000278
        >>> print(expected_returns)
        000001    0.0825
        600000    0.0514
        601398    0.0327
        dtype: float64
    """
    # 函数实现在market_data_api.py中
    pass

def get_index_data(index_symbol="000300", fields=None, start_date=None, end_date=None, freq='D'):
    """
    获取指数数据
    
    参数:
        index_symbol (str, 可选): 指数代码，默认为沪深300("000300")
        fields (list, 可选): 要获取的字段列表，为None则获取所有可用字段
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'D'
        
    返回:
        pd.DataFrame: 包含指数数据的DataFrame
        
    示例:
        >>> index_data = get_index_data("000300", fields=["date", "open", "close"], 
                                       start_date="2022-01-01", end_date="2022-01-10")
        >>> print(index_data.head())
                date      open     close
        0  2022-01-04  4921.23   5005.89
        1  2022-01-05  5010.45   4920.23
        2  2022-01-06  4915.67   4961.81
        3  2022-01-07  4963.45   4961.33
        4  2022-01-10  4960.12   4898.54
    """
    # 函数实现在market_data_api.py中
    pass

def get_multiple_index_data(index_symbols, start_date=None, end_date=None, freq='D'):
    """
    获取多个指数的数据
    
    参数:
        index_symbols (list): 指数代码列表
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'D'
        
    返回:
        dict: 包含多个指数数据的字典，键为指数代码，值为指数数据DataFrame
        
    示例:
        >>> index_dict = get_multiple_index_data(["000300", "000016", "000905"], 
                                               start_date="2022-01-01", end_date="2022-01-10")
        >>> print(index_dict["000300"].head())
                date      open     high      low     close      volume       amount
        0  2022-01-04  4921.23  5010.42  4899.21   5005.89  23491782.0  287541987.0
        1  2022-01-05  5010.45  5015.74  4915.95   4920.23  26485732.0  298457612.0
        2  2022-01-06  4915.67  4978.56  4902.45   4961.81  22154578.0  256412378.0
        ...
    """
    # 函数实现在market_data_api.py中
    pass

#----------------------------------------------------------------------------#
# Fama-French三因子API函数说明
#----------------------------------------------------------------------------#

def get_fama_french_factors(start_date=None, end_date=None, freq='D', use_cache=True):
    """
    获取Fama-French三因子模型的因子数据
    
    参数:
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'D'
        use_cache (bool, 可选): 是否使用缓存，默认为True
        
    返回:
        dict: 包含以下因子的字典:
            - 'market_returns': 市场收益率 (pd.Series)
            - 'market_excess_returns': 市场超额收益率 (pd.Series)
            - 'smb': 规模因子 (pd.Series)
            - 'hml': 价值因子 (pd.Series)
            - 'risk_free_rate': 无风险利率 (pd.Series)
        
    示例:
        >>> ff_factors = get_fama_french_factors(start_date="2022-01-01", end_date="2022-12-31", freq="D")
        >>> print(ff_factors["market_returns"].head())
        2022-01-04    0.018242
        2022-01-05   -0.016501
        2022-01-06    0.008458
        2022-01-07   -0.000097
        2022-01-10   -0.013861
        Name: market_returns, dtype: float64
        >>> print(ff_factors["smb"].head())
        2022-01-04    0.001245
        2022-01-05   -0.000854
        2022-01-06    0.000987
        2022-01-07    0.001532
        2022-01-10   -0.002145
        Name: smb, dtype: float64
    """
    # 函数实现在fama_french_api.py中
    pass

def calculate_fama_french_factors_tushare(start_date, end_date, freq='W'):
    """
    使用TuShare计算Fama-French三因子数据
    
    参数:
        start_date (str): 开始日期，格式：YYYY-MM-DD或YYYYMMDD
        end_date (str): 结束日期，格式：YYYY-MM-DD或YYYYMMDD
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'W'
        
    返回:
        dict: 包含以下因子的字典:
            - 'market_returns': 市场收益率 (pd.Series)
            - 'market_excess_returns': 市场超额收益率 (pd.Series)
            - 'smb': 规模因子 (pd.Series)
            - 'hml': 价值因子 (pd.Series)
            - 'risk_free_rate': 无风险利率 (pd.Series)
    
    注意:
        此函数直接调用TuShare API计算因子，需要有效的TuShare token并安装tushare库
    """
    # 函数实现在fama_french_api.py中
    pass

#----------------------------------------------------------------------------#
# 行业数据API函数说明
#----------------------------------------------------------------------------#

def get_industry_index_returns(industry_codes, start_date=None, end_date=None, freq='D'):
    """
    获取行业指数收益率数据
    
    参数:
        industry_codes (str 或 list): 行业指数代码或代码列表，例如"801780"(银行)
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'D'
        
    返回:
        pd.DataFrame: 包含多个行业指数收益率的DataFrame，列为行业名称，索引为日期
        
    示例:
        >>> industry_returns = get_industry_index_returns(["801120", "801780"], 
                                                       start_date="2022-01-01", end_date="2022-01-10")
        >>> print(industry_returns.head())
                     食品饮料       银行
        2022-01-04   0.0145    0.0082
        2022-01-05  -0.0075   -0.0045
        2022-01-06   0.0058    0.0025
        2022-01-07  -0.0035   -0.0018
        2022-01-10  -0.0089   -0.0062
    """
    # 函数实现在industry_api.py中
    pass

def get_industry_rotation_factors(start_date=None, end_date=None, freq='W'):
    """
    获取行业轮动因子数据 (包括行业动量、估值、成长性等)
    
    参数:
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        freq (str, 可选): 数据频率，推荐'W'为周度或'M'为月度，默认为'W'
        
    返回:
        pd.DataFrame: 包含行业轮动因子的DataFrame
        
    示例:
        >>> rotation_factors = get_industry_rotation_factors(start_date="2022-01-01", end_date="2022-12-31")
        >>> print(rotation_factors.columns)
        Index(['食品饮料_MOM_1M', '食品饮料_MOM_3M', '食品饮料_MOM_6M', '银行_MOM_1M', ...])
        >>> print(rotation_factors.head())
                     食品饮料_MOM_1M  食品饮料_MOM_3M  食品饮料_MOM_6M  银行_MOM_1M  ...
        2022-01-28       0.027546       0.056821       0.125478   0.017892  ...
        ...
    """
    # 函数实现在industry_api.py中
    pass

def get_sector_index_returns(start_date=None, end_date=None, freq='D'):
    """
    获取主要板块指数收益率数据 (上证、深证、创业板、中小板等)
    
    参数:
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'D'
        
    返回:
        pd.DataFrame: 包含主要板块指数收益率的DataFrame，列为板块名称，索引为日期
        
    示例:
        >>> sector_returns = get_sector_index_returns(start_date="2022-01-01", end_date="2022-01-10")
        >>> print(sector_returns.head())
                     上证指数  深证成指  创业板指  沪深300  ...
        2022-01-04   0.0182   0.0215   0.0298   0.0192  ...
        2022-01-05  -0.0165  -0.0195  -0.0254  -0.0170  ...
        ...
    """
    # 函数实现在industry_api.py中
    pass

def get_style_index_returns(start_date=None, end_date=None, freq='D'):
    """
    获取风格指数收益率数据 (大盘成长、小盘价值等)
    
    参数:
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'D'
        
    返回:
        pd.DataFrame: 包含风格指数收益率的DataFrame，列为风格指数名称，索引为日期
        
    示例:
        >>> style_returns = get_style_index_returns(start_date="2022-01-01", end_date="2022-01-10")
        >>> print(style_returns.head())
                     300价值  300成长  中证红利  基本面50  ...
        2022-01-04   0.0141   0.0235   0.0125   0.0172  ...
        2022-01-05  -0.0165  -0.0215  -0.0074  -0.0145  ...
        ...
    """
    # 函数实现在industry_api.py中
    pass

#----------------------------------------------------------------------------#
# 宏观经济数据API函数说明
#----------------------------------------------------------------------------#

def get_macro_economic_data(indicator_type="gdp", start_date=None, end_date=None):
    """
    获取宏观经济指标数据
    
    参数:
        indicator_type (str, 可选): 指标类型，可选值:
                                  "gdp": 国内生产总值
                                  "cpi": 消费者物价指数
                                  "interest_rate": 利率
                                  "m2": 货币供应量
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        
    返回:
        pd.DataFrame: 包含宏观经济数据的DataFrame
        
    示例:
        >>> gdp_data = get_macro_economic_data("gdp", start_date="2018-01-01", end_date="2022-12-31")
        >>> print(gdp_data.head())
                date  国内生产总值_当季值  国内生产总值_累计值  国内生产总值_同比增长
        0  2018-03-31     213456.78     213456.78           6.8
        1  2018-06-30     225678.90     439135.68           6.7
        ...
        
        >>> cpi_data = get_macro_economic_data("cpi", start_date="2022-01-01", end_date="2022-12-31")
        >>> print(cpi_data.head())
                date  全国_同比  全国_环比  城市_同比  农村_同比
        0  2022-01-31     1.5    0.40      1.5      1.4
        1  2022-02-28     0.9   -0.30      0.9      0.8
        ...
    """
    # 函数实现在macro_api.py中
    pass

#----------------------------------------------------------------------------#
# 模型估计API函数说明
#----------------------------------------------------------------------------#

def estimate_capm_for_stock(stock_symbol, start_date=None, end_date=None, freq='D'):
    """
    为单个股票估计CAPM模型参数
    
    参数:
        stock_symbol (str): 股票代码
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'D'
        
    返回:
        dict: CAPM模型参数字典，包括:
             - 'alpha': 阿尔法值，表示超额收益
             - 'beta': 贝塔系数，表示系统性风险
             - 'r_squared': 拟合优度
             - 'p_value': 显著性P值
             - 't_stat': T统计量
             - 'std_error': 标准误差
        
    示例:
        >>> capm_results = estimate_capm_for_stock("000001", start_date="2022-01-01", end_date="2022-12-31")
        >>> print(capm_results)
        {'alpha': 0.000125, 'beta': 1.234, 'r_squared': 0.785, 'p_value': 0.0002, 't_stat': 8.976, 'std_error': 0.137}
    """
    # 函数实现在model_estimator_api.py中
    pass

def estimate_fama_french_for_stock(stock_symbol, start_date=None, end_date=None, freq='D'):
    """
    为单个股票估计Fama-French三因子模型参数
    
    参数:
        stock_symbol (str): 股票代码
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'D'
        
    返回:
        dict: Fama-French三因子模型参数字典，包括:
             - 'alpha': 阿尔法值，表示超额收益
             - 'beta': 市场因子系数
             - 'smb': 规模因子系数
             - 'hml': 价值因子系数
             - 'r_squared': 拟合优度
             - 'p_value_beta': 市场因子显著性P值
             - 'p_value_smb': 规模因子显著性P值
             - 'p_value_hml': 价值因子显著性P值
        
    示例:
        >>> ff_results = estimate_fama_french_for_stock("000001", start_date="2022-01-01", end_date="2022-12-31")
        >>> print(ff_results)
        {'alpha': 0.000087, 'beta': 1.156, 'smb': 0.435, 'hml': -0.267, 'r_squared': 0.812, 
         'p_value_beta': 0.0001, 'p_value_smb': 0.0024, 'p_value_hml': 0.0375}
    """
    # 函数实现在model_estimator_api.py中
    pass

def estimate_beta_for_stocks(symbols, start_date=None, end_date=None, freq='D'):
    """
    估计多个股票的贝塔系数
    
    参数:
        symbols (list): 股票代码列表
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'D'
        
    返回:
        pd.DataFrame: 包含贝塔系数的DataFrame，列包括:
                     - 'symbol': 股票代码
                     - 'beta': 贝塔系数
                     - 'r_squared': 拟合优度
        
    示例:
        >>> beta_df = estimate_beta_for_stocks(["000001", "600000", "601398"], 
                                             start_date="2022-01-01", end_date="2022-12-31")
        >>> print(beta_df)
            symbol     beta  r_squared
        0   000001    1.234      0.785
        1   600000    0.865      0.693
        2   601398    0.912      0.722
    """
    # 函数实现在model_estimator_api.py中
    pass

def calculate_rolling_beta(stock_symbol, window=60, start_date=None, end_date=None, freq='D'):
    """
    计算股票的滚动贝塔系数
    
    参数:
        stock_symbol (str): 股票代码
        window (int, 可选): 滚动窗口大小（交易日数量），默认为60
        start_date (str, 可选): 开始日期，格式：YYYY-MM-DD
        end_date (str, 可选): 结束日期，格式：YYYY-MM-DD
        freq (str, 可选): 数据频率，'D'日度，'W'周度，'M'月度，默认为'D'
        
    返回:
        pd.Series: 滚动贝塔系数Series，索引为日期
        
    示例:
        >>> rolling_beta = calculate_rolling_beta("000001", window=60, 
                                               start_date="2022-01-01", end_date="2022-12-31")
        >>> print(rolling_beta.head())
        2022-04-01    1.125
        2022-04-04    1.137
        2022-04-05    1.142
        2022-04-06    1.129
        2022-04-07    1.118
        dtype: float64
    """
    # 函数实现在model_estimator_api.py中
    pass


#----------------------------------------------------------------------------#
# 完整使用示例
#----------------------------------------------------------------------------#

"""
以下是完整的使用示例，展示如何组合使用factor_data_api的各个函数:

1. 基本数据获取和可视化

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.factor_data_api import get_stock_returns, get_market_returns

# 获取股票和市场收益率
start_date = "2022-01-01"
end_date = "2022-12-31"
stock_returns = get_stock_returns(["000001", "600000"], start_date, end_date)
market_returns = get_market_returns(start_date=start_date, end_date=end_date)

# 可视化收益率
plt.figure(figsize=(12, 6))
plt.plot(market_returns.index, market_returns.values, label="沪深300", linewidth=2)
plt.plot(stock_returns["000001"].index, stock_returns["000001"].values, label="000001", alpha=0.7)
plt.plot(stock_returns["600000"].index, stock_returns["600000"].values, label="600000", alpha=0.7)
plt.title("股票与市场收益率比较")
plt.xlabel("日期")
plt.ylabel("日收益率")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

2. Fama-French三因子获取和分析

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.factor_data_api import get_fama_french_factors, estimate_fama_french_for_stock

# 获取Fama-French三因子数据
start_date = "2021-01-01"
end_date = "2022-12-31"
ff_factors = get_fama_french_factors(start_date=start_date, end_date=end_date, freq="D")

# 可视化三因子走势
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(ff_factors["market_excess_returns"].index, ff_factors["market_excess_returns"].values)
plt.title("市场超额收益率(Mkt-RF)")
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(ff_factors["smb"].index, ff_factors["smb"].values)
plt.title("规模因子(SMB)")
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(ff_factors["hml"].index, ff_factors["hml"].values)
plt.title("价值因子(HML)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 估计股票的Fama-French三因子模型参数
ff_results = estimate_fama_french_for_stock("000001", start_date=start_date, end_date=end_date)
print("Fama-French三因子模型参数:")
for key, value in ff_results.items():
    print(f"{key}: {value:.4f}")
```

3. 行业轮动分析

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.tools.factor_data_api import get_industry_index_returns

# 获取主要行业指数收益率
start_date = "2022-01-01"
end_date = "2022-12-31"
industry_codes = ["801120", "801080", "801150", "801780", "801750", "801760"]  # 食品饮料、电子、医药、银行、计算机、传媒
industry_returns = get_industry_index_returns(industry_codes, start_date, end_date)

# 计算累积收益率
industry_cum_returns = (1 + industry_returns).cumprod() - 1

# 可视化累积收益率
plt.figure(figsize=(14, 7))
for industry in industry_cum_returns.columns:
    plt.plot(industry_cum_returns.index, industry_cum_returns[industry], label=industry, linewidth=2)
plt.title("行业指数累积收益率比较")
plt.xlabel("日期")
plt.ylabel("累积收益率")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 计算行业间相关性
industry_corr = industry_returns.corr()

# 可视化相关性热力图
plt.figure(figsize=(10, 8))
sns.heatmap(industry_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
plt.title("行业指数收益率相关性")
plt.show()
```

4. 投资组合贝塔系数分析

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.factor_data_api import calculate_rolling_beta, estimate_beta_for_stocks

# 获取多只股票的贝塔系数
stocks = ["000001", "600000", "601398", "600519", "000651", "002415"]
start_date = "2021-01-01"
end_date = "2022-12-31"
beta_df = estimate_beta_for_stocks(stocks, start_date=start_date, end_date=end_date)
print("股票贝塔系数:")
print(beta_df)

# 计算某只股票的滚动贝塔系数
rolling_beta = calculate_rolling_beta("600519", window=60, start_date=start_date, end_date=end_date)

# 可视化滚动贝塔系数
plt.figure(figsize=(12, 6))
plt.plot(rolling_beta.index, rolling_beta.values, linewidth=2)
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
plt.title("600519 滚动贝塔系数 (60日窗口)")
plt.xlabel("日期")
plt.ylabel("贝塔系数")
plt.grid(True, alpha=0.3)
plt.show()
```

5. 宏观经济数据分析

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.factor_data_api import get_macro_economic_data

# 获取宏观经济数据
gdp_data = get_macro_economic_data("gdp", start_date="2018-01-01")
cpi_data = get_macro_economic_data("cpi", start_date="2020-01-01")

# 可视化GDP增长率
plt.figure(figsize=(12, 6))
plt.bar(gdp_data["date"], gdp_data["国内生产总值_同比增长"], color='steelblue')
plt.title("GDP同比增长率")
plt.xlabel("日期")
plt.ylabel("增长率(%)")
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 可视化CPI同比变动
plt.figure(figsize=(12, 6))
plt.plot(cpi_data["date"], cpi_data["全国_同比"], linewidth=2, marker='o')
plt.title("CPI同比变动")
plt.xlabel("日期")
plt.ylabel("变动率(%)")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
"""