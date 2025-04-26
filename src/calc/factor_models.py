import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from scipy import stats
from typing import Dict, Optional

def estimate_capm(returns: pd.Series, 
                 market_returns: pd.Series, 
                 risk_free_rate: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    估计CAPM模型参数
    
    CAPM方程: R_i - R_f = α + β(R_m - R_f) + ε
    
    Args:
        returns: 资产收益率序列
        market_returns: 市场收益率序列
        risk_free_rate: 无风险利率序列，如果为None则假设为0
        
    Returns:
        CAPM模型参数字典
    """
    # 准备数据
    if risk_free_rate is None:
        # 假设无风险利率为0
        excess_returns = returns
        excess_market = market_returns
    else:
        # 计算超额收益
        excess_returns = returns - risk_free_rate
        excess_market = market_returns - risk_free_rate
    
    # 创建DataFrame用于回归
    df = pd.DataFrame({
        'excess_returns': excess_returns,
        'excess_market': excess_market
    }).dropna()
    
    # 添加常数项
    X = sm.add_constant(df['excess_market'])
    
    # 进行OLS回归
    model = sm.OLS(df['excess_returns'], X)
    results = model.fit()
    
    # 提取参数
    alpha = results.params[0]
    beta = results.params[1]
    
    # 计算预测值和残差
    df['predicted'] = alpha + beta * df['excess_market']
    df['residuals'] = df['excess_returns'] - df['predicted']
    
    # 计算R方
    r_squared = results.rsquared
    
    # 计算信息比率
    information_ratio = alpha / np.std(df['residuals']) * np.sqrt(252)  # 年化
    
    # 计算特雷诺比率
    treynor_ratio = (df['excess_returns'].mean() * 252) / beta  # 年化
    
    # 返回结果
    return {
        'alpha': alpha,
        'beta': beta,
        'r_squared': r_squared,
        'p_value_alpha': results.pvalues[0],
        'p_value_beta': results.pvalues[1],
        'information_ratio': information_ratio,
        'treynor_ratio': treynor_ratio,
        'residual_std': np.std(df['residuals']),
        'annualized_alpha': alpha * 252,  # 年化alpha
        'observations': len(df)
    }

def estimate_fama_french(returns: pd.Series, 
                        market_returns: pd.Series, 
                        smb: pd.Series,
                        hml: pd.Series,
                        risk_free_rate: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    估计Fama-French三因子模型参数
    
    FF三因子方程: R_i - R_f = α + β1(R_m - R_f) + β2(SMB) + β3(HML) + ε
    
    Args:
        returns: 资产收益率序列
        market_returns: 市场收益率序列
        smb: 小市值减大市值(Small Minus Big)因子收益率序列
        hml: 高价值减低价值(High Minus Low)因子收益率序列
        risk_free_rate: 无风险利率序列，如果为None则假设为0
        
    Returns:
        Fama-French三因子模型参数字典
    """
    # 准备数据
    if risk_free_rate is None:
        # 假设无风险利率为0
        excess_returns = returns
        excess_market = market_returns
    else:
        # 计算超额收益
        excess_returns = returns - risk_free_rate
        excess_market = market_returns - risk_free_rate
    
    # 创建DataFrame用于回归
    df = pd.DataFrame({
        'excess_returns': excess_returns,
        'excess_market': excess_market,
        'smb': smb,
        'hml': hml
    }).dropna()
    
    # 添加常数项
    X = sm.add_constant(df[['excess_market', 'smb', 'hml']])
    
    # 进行OLS回归
    model = sm.OLS(df['excess_returns'], X)
    results = model.fit()
    
    # 提取参数
    alpha = results.params[0]
    beta_market = results.params[1]
    beta_smb = results.params[2]
    beta_hml = results.params[3]
    
    # 计算预测值和残差
    df['predicted'] = (alpha + 
                      beta_market * df['excess_market'] + 
                      beta_smb * df['smb'] + 
                      beta_hml * df['hml'])
    df['residuals'] = df['excess_returns'] - df['predicted']
    
    # 返回结果
    return {
        'alpha': alpha,
        'beta_market': beta_market,
        'beta_smb': beta_smb,
        'beta_hml': beta_hml,
        'r_squared': results.rsquared,
        'p_value_alpha': results.pvalues[0],
        'p_value_market': results.pvalues[1],
        'p_value_smb': results.pvalues[2],
        'p_value_hml': results.pvalues[3],
        'residual_std': np.std(df['residuals']),
        'annualized_alpha': alpha * 252,  # 年化alpha
        'observations': len(df)
    }

def time_series_test(returns: pd.Series, 
                    factors: pd.DataFrame, 
                    window: int = 60) -> pd.DataFrame:
    """
    进行时间序列检验，计算滚动因子模型参数
    
    Args:
        returns: 资产收益率序列
        factors: 因子收益率DataFrame
        window: 滚动窗口大小
        
    Returns:
        滚动系数DataFrame
    """
    # 准备数据
    data = pd.concat([returns, factors], axis=1).dropna()
    y = data.iloc[:, 0]  # 第一列为资产收益率
    X = sm.add_constant(data.iloc[:, 1:])  # 其余列为因子
    
    # 使用RollingOLS进行滚动回归
    rolling_reg = RollingOLS(y, X, window=window)
    rolling_results = rolling_reg.fit()
    
    # 提取滚动系数
    params = rolling_results.params
    
    # 计算滚动R方
    r2 = pd.Series(index=params.index)
    tvalues = pd.DataFrame(index=params.index, columns=params.columns)
    
    # 在每个窗口进行OLS回归，计算R方和t值
    for i in range(window, len(data)):
        y_window = y.iloc[i-window:i]
        X_window = X.iloc[i-window:i]
        
        model = sm.OLS(y_window, X_window)
        res = model.fit()
        
        r2.iloc[i-window] = res.rsquared
        for col in params.columns:
            tvalues.loc[params.index[i-window], col] = res.tvalues[col]
    
    # 合并结果
    results = pd.concat([params, r2, tvalues], axis=1)
    results.columns = list(params.columns) + ['R2'] + [f't_{col}' for col in params.columns]
    
    return results

def cross_sectional_test(returns: pd.DataFrame,
                        factors: pd.DataFrame,
                        frequency: str = 'M') -> Dict[str, pd.DataFrame]:
    """
    进行横截面检验，使用Fama-MacBeth回归方法
    
    Args:
        returns: 资产收益率DataFrame，每列为一个资产
        factors: 资产因子暴露DataFrame，索引与收益率相同，列为不同因子
        frequency: 横截面回归频率，默认为月度('M')
        
    Returns:
        Fama-MacBeth回归结果
    """
    # 将数据重采样到指定频率
    if frequency:
        returns_resampled = returns.resample(frequency).mean()
        factors_resampled = factors.resample(frequency).mean()
    else:
        returns_resampled = returns
        factors_resampled = factors
    
    # 对每个时间点进行横截面回归
    time_points = returns_resampled.index
    cross_section_results = []
    
    for t in time_points:
        if t in factors_resampled.index:
            # 获取当前时间点的收益率和因子
            ret_t = returns_resampled.loc[t]
            fact_t = factors_resampled.loc[t]
            
            # 合并数据
            data_t = pd.concat([ret_t, fact_t], axis=1).dropna()
            
            if len(data_t) > len(factors_resampled.columns) + 2:  # 确保有足够的观测值
                # 进行OLS回归
                y = data_t.iloc[:, 0]
                X = sm.add_constant(data_t.iloc[:, 1:])
                
                model = sm.OLS(y, X)
                results = model.fit()
                
                # 保存结果
                params = results.params
                params.name = t
                cross_section_results.append(params)
    
    # 汇总所有时间点的回归系数
    if cross_section_results:
        all_params = pd.concat(cross_section_results, axis=1).T
        
        # 对每个因子系数进行t检验
        mean_params = all_params.mean()
        std_params = all_params.std() / np.sqrt(len(all_params))
        t_stats = mean_params / std_params
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(all_params) - 1))
        
        # 创建汇总结果
        summary = pd.DataFrame({
            'Coefficient': mean_params,
            'Std Error': std_params,
            't-statistic': t_stats,
            'p-value': p_values
        })
        
        return {
            'time_series_coefficients': all_params,
            'summary': summary
        }
    else:
        return {
            'time_series_coefficients': pd.DataFrame(),
            'summary': pd.DataFrame()
        }