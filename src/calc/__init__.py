"""
金融计算模块，提供了一系列金融计算工具，包括资产定价模型估计、风险测量、投资组合优化和波动率建模等功能。

主要功能
-------
1. 资产定价模型
   - CAPM模型
   - Fama-French三因子模型
   - 时间序列和横截面检验

2. Beta计算
   - 使用市场数据计算股票Beta值

3. 风险测量
   - VaR (Value at Risk)
   - CVaR (Conditional Value at Risk)
   - VaR回测

4. 投资组合优化
   - 最大化夏普比率
   - 最小化风险
   - 最大化收益
   - 有效前沿计算

5. 波动率模型
   - GARCH(1,1)模型
   - 已实现波动率计算

6. 协方差估计
   - EWMA方法估计协方差矩阵
"""

# 从各子模块导入主要函数
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

# 公开API
__all__ = [
    # 资产定价模型
    'estimate_capm',
    'estimate_fama_french',
    'time_series_test',
    'cross_sectional_test',
    
    # Beta计算
    'calculate_beta',
    
    # 风险测量
    'calculate_historical_var',
    'calculate_conditional_var',
    'calculate_parametric_var',
    'backtesting_var',
    
    # 投资组合优化
    'optimize_portfolio',
    'efficient_frontier',
    'portfolio_return',
    'portfolio_volatility',
    'portfolio_sharpe_ratio',
    
    # 波动率模型
    'fit_garch',
    'forecast_garch_volatility',
    'calculate_realized_volatility',
    
    # 协方差估计
    'estimate_covariance_ewma',

    # 相关性分析
    'analyze_asset_correlations',
    'calculate_optimal_weights_for_correlation',
    'cluster_assets'
]

"""
使用示例
-------

1. 计算股票Beta值

```python
from src.calc import calculate_beta

# 计算某股票相对于沪深300的Beta值
# 参数:
#   ticker: 股票代码，如 "600519"
#   market_index: 市场指数代码，默认为 "000300"（沪深300）
#   start_date: 开始日期，格式为 "YYYY-MM-DD"
#   end_date: 结束日期，格式为 "YYYY-MM-DD"
# 返回值:
#   float: 计算出的Beta值，范围通常在0.2至3.0之间
#   若数据不足，返回默认值1.0

beta = calculate_beta(
    ticker="600519",              # 股票代码
    market_index="000300",        # 市场指数代码
    start_date="2022-01-01",      # 开始日期
    end_date="2022-12-31"         # 结束日期
)
print(f"计算得到的Beta值: {beta:.2f}")
```

2. 使用CAPM模型进行估计

```python
import pandas as pd
from src.calc import estimate_capm

# 准备数据
stock_returns = pd.Series([0.01, -0.02, 0.015, 0.008, -0.01])  # 资产收益率序列
market_returns = pd.Series([0.005, -0.01, 0.012, 0.006, -0.008])  # 市场收益率序列
risk_free_rate = pd.Series([0.001, 0.001, 0.001, 0.001, 0.001])  # 无风险利率序列

# 估计CAPM模型参数
# 参数:
#   returns: 资产收益率序列 (pd.Series)
#   market_returns: 市场收益率序列 (pd.Series)
#   risk_free_rate: 无风险利率序列 (pd.Series, 可选)
# 返回值:
#   Dict[str, float]: 包含以下键值对的字典
#     - alpha: Jensen's Alpha
#     - beta: 系统性风险Beta
#     - r_squared: 确定系数
#     - p_value_alpha, p_value_beta: alpha和beta的p值
#     - information_ratio: 信息比率
#     - treynor_ratio: 特雷诺比率
#     - residual_std: 残差标准差
#     - annualized_alpha: 年化alpha
#     - observations: 观测数量

capm_results = estimate_capm(
    returns=stock_returns,
    market_returns=market_returns,
    risk_free_rate=risk_free_rate
)

print(f"Alpha: {capm_results['alpha']:.4f}")
print(f"Beta: {capm_results['beta']:.4f}")
print(f"R方: {capm_results['r_squared']:.4f}")
print(f"信息比率: {capm_results['information_ratio']:.4f}")
```

3. 使用Fama-French三因子模型

```python
from src.calc import estimate_fama_french

# 准备数据 (同上，并添加SMB和HML因子)
smb = pd.Series([0.003, -0.005, 0.008, 0.002, -0.004])  # 小市值减大市值因子收益率
hml = pd.Series([0.002, 0.003, -0.001, 0.004, 0.001])   # 高价值减低价值因子收益率

# 估计Fama-French三因子模型参数
# 参数:
#   returns: 资产收益率序列 (pd.Series)
#   market_returns: 市场收益率序列 (pd.Series)
#   smb: 小市值减大市值因子收益率序列 (pd.Series)
#   hml: 高价值减低价值因子收益率序列 (pd.Series)
#   risk_free_rate: 无风险利率序列 (pd.Series, 可选)
# 返回值:
#   Dict[str, float]: 包含以下键值对的字典
#     - alpha: Jensen's Alpha
#     - beta_market, beta_smb, beta_hml: 三个因子的Beta值
#     - r_squared: 确定系数
#     - p_value_alpha, p_value_market, p_value_smb, p_value_hml: 各系数的p值
#     - residual_std: 残差标准差
#     - annualized_alpha: 年化alpha
#     - observations: 观测数量

ff3_results = estimate_fama_french(
    returns=stock_returns,
    market_returns=market_returns,
    smb=smb,
    hml=hml,
    risk_free_rate=risk_free_rate
)

print(f"Alpha: {ff3_results['alpha']:.4f}")
print(f"市场Beta: {ff3_results['beta_market']:.4f}")
print(f"SMB Beta: {ff3_results['beta_smb']:.4f}")
print(f"HML Beta: {ff3_results['beta_hml']:.4f}")
```

4. 计算风险价值(VaR)

```python
import numpy as np
from src.calc import calculate_historical_var, calculate_conditional_var

# 创建模拟的日收益率数据
returns = pd.Series(np.random.normal(0.0005, 0.01, 1000))

# 计算历史VaR
# 参数:
#   returns: 收益率序列 (pd.Series)
#   confidence_level: 置信水平，默认0.95 (95%)
#   window: 如果指定，则只使用最近window个样本
# 返回值:
#   float: 在给定置信水平下的VaR值（正值）

hist_var = calculate_historical_var(
    returns=returns,
    confidence_level=0.95,
    window=252  # 使用最近一年的数据
)

# 计算条件VaR (CVaR/Expected Shortfall)
# 参数:
#   returns: 收益率序列 (pd.Series)
#   confidence_level: 置信水平，默认0.95 (95%)
#   window: 如果指定，则只使用最近window个样本
# 返回值:
#   float: CVaR值（正值）

cvar = calculate_conditional_var(
    returns=returns,
    confidence_level=0.95,
    window=252
)

print(f"95%置信水平下的历史VaR: {hist_var:.4f} (相当于{hist_var*100:.2f}%的损失)")
print(f"95%置信水平下的CVaR: {cvar:.4f} (相当于{cvar*100:.2f}%的损失)")
```

5. 投资组合优化

```python
import numpy as np
import pandas as pd
from src.calc import optimize_portfolio, efficient_frontier

# 模拟多个资产的预期收益率和协方差矩阵
assets = ['资产A', '资产B', '资产C', '资产D']
expected_returns = pd.Series([0.08, 0.12, 0.10, 0.07], index=assets)
cov_matrix = pd.DataFrame([
    [0.04, 0.02, 0.01, 0.02],
    [0.02, 0.09, 0.03, 0.01],
    [0.01, 0.03, 0.06, 0.02],
    [0.02, 0.01, 0.02, 0.05]
], index=assets, columns=assets)

# 优化投资组合
# 参数:
#   expected_returns: 各资产预期收益率 (pd.Series)
#   cov_matrix: 协方差矩阵 (pd.DataFrame)
#   risk_free_rate: 无风险利率，默认0.0
#   target_return: 目标收益率（如果指定）
#   target_risk: 目标风险（如果指定）
#   objective: 优化目标, 可选 'sharpe'(最大化夏普比率), 'min_risk'(最小化风险), 'max_return'(最大化收益)
# 返回值:
#   Dict[str, Union[pd.Series, float]]: 包含以下键值对的字典
#     - weights: 最优权重Series
#     - return: 投资组合预期收益率
#     - risk: 投资组合风险（波动率）
#     - sharpe_ratio: 夏普比率

# 最大化夏普比率的组合
max_sharpe_port = optimize_portfolio(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    risk_free_rate=0.03,
    objective='sharpe'
)

print("最大夏普比率组合:")
print(f"最优权重: \n{max_sharpe_port['weights']}")
print(f"预期收益率: {max_sharpe_port['return']:.4f}")
print(f"风险(标准差): {max_sharpe_port['risk']:.4f}")
print(f"夏普比率: {max_sharpe_port['sharpe_ratio']:.4f}")

# 计算有效前沿
# 参数:
#   expected_returns: 各资产预期收益率 (pd.Series)
#   cov_matrix: 协方差矩阵 (pd.DataFrame)
#   risk_free_rate: 无风险利率，默认0.0
#   points: 有效前沿上的点数，默认50
# 返回值:
#   pd.DataFrame: 包含收益率、风险和夏普比率的数据框

ef = efficient_frontier(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    risk_free_rate=0.03,
    points=20
)

print("\n有效前沿上的几个代表点:")
print(ef[['return', 'risk', 'sharpe_ratio']].head())
```

6. GARCH模型拟合与波动率预测

```python
import numpy as np
from src.calc import fit_garch, forecast_garch_volatility

# 模拟收益率数据
np.random.seed(42)
returns = np.random.normal(0, 0.01, 1000)
for i in range(1, len(returns)):
    # 加入波动率聚集效应
    if i > 100 and i < 200:
        returns[i] *= 2  # 创造一段高波动期

# 拟合GARCH(1,1)模型
# 参数:
#   returns: 收益率序列 (np.ndarray)
#   initial_guess: 初始参数猜测 [omega, alpha, beta]，默认为None
# 返回值:
#   Tuple[Dict[str, float], float]: 
#     - Dict: 模型参数，包含 omega, alpha, beta, long_run_variance, persistence
#     - float: 对数似然值

garch_params, loglik = fit_garch(returns)

print("GARCH(1,1)模型参数:")
print(f"omega: {garch_params['omega']:.6f}")
print(f"alpha: {garch_params['alpha']:.4f}")
print(f"beta: {garch_params['beta']:.4f}")
print(f"持续性(alpha+beta): {garch_params['persistence']:.4f}")
print(f"长期方差: {garch_params['long_run_variance']:.6f}")
print(f"对数似然值: {loglik:.2f}")

# 预测未来波动率
# 参数:
#   returns: 历史收益率序列 (np.ndarray)
#   params: GARCH模型参数 (Dict[str, float])
#   forecast_horizon: 预测期数，默认为10
# 返回值:
#   np.ndarray: 预测的波动率序列（标准差）

forecast_vol = forecast_garch_volatility(
    returns=returns,
    params=garch_params,
    forecast_horizon=5
)

print("\n未来5天的波动率预测:")
for i, vol in enumerate(forecast_vol):
    print(f"第{i+1}天: {vol:.6f}")
```

7. 使用EWMA方法估计协方差矩阵

```python
import pandas as pd
import numpy as np
from src.calc import estimate_covariance_ewma

# 创建模拟的多资产收益率数据
np.random.seed(42)
dates = pd.date_range('2022-01-01', periods=500)
assets = ['资产A', '资产B', '资产C']
returns_data = {}

for asset in assets:
    returns_data[asset] = np.random.normal(0, 0.01, 500)
    
returns_df = pd.DataFrame(returns_data, index=dates)

# 使用EWMA方法估计协方差矩阵
# 参数:
#   returns: 收益率DataFrame，每列为一个资产，每行为一个时间点
#   lambda_param: 衰减因子，通常在0.9到0.99之间，值越大表示历史数据权重越高
#   min_periods: 计算需要的最小样本数
# 返回值:
#   pd.DataFrame: 协方差矩阵DataFrame

ewma_cov = estimate_covariance_ewma(
    returns=returns_df,
    lambda_param=0.94,
    min_periods=50
)

print("EWMA协方差矩阵:")
print(ewma_cov)

# 计算相关系数矩阵
std_dev = np.sqrt(np.diag(ewma_cov))
corr_matrix = ewma_cov.copy()
for i in range(len(assets)):
    for j in range(len(assets)):
        corr_matrix.iloc[i, j] = ewma_cov.iloc[i, j] / (std_dev[i] * std_dev[j])

print("\nEWMA相关系数矩阵:")
print(corr_matrix)
```
"""