import pandas as pd
import numpy as np
from typing import Dict
import statsmodels.api as sm

def analyze_macro_factor_exposures(returns_df: pd.DataFrame, 
                                  macro_factors_df: pd.DataFrame) -> Dict:
    """
    分析资产对宏观经济因子的敏感性
    
    Args:
        returns_df: 包含多个资产收益率的DataFrame
        macro_factors_df: 包含宏观经济因子的DataFrame
        
    Returns:
        Dict: 包含宏观因子敏感性分析结果
    """
    # 对齐日期索引
    common_dates = returns_df.index.intersection(macro_factors_df.index)
    if len(common_dates) < 30:  # 需要足够的数据点
        return {
            'success': False,
            'error': f'数据点不足，仅有{len(common_dates)}个共同日期',
            'exposures': {}
        }
    
    # 对齐数据
    aligned_returns = returns_df.loc[common_dates]
    aligned_factors = macro_factors_df.loc[common_dates]
    
    # 结果字典
    results = {
        'success': True,
        'exposures': {},
        'sensitivity_summary': []
    }
    
    # 为每个资产计算对宏观因子的敏感性
    for asset in returns_df.columns:
        asset_returns = aligned_returns[asset]
        
        # 添加常数项
        X = sm.add_constant(aligned_factors)
        
        try:
            # 执行线性回归
            model = sm.OLS(asset_returns, X)
            fit_result = model.fit()
            
            # 提取系数
            coefficients = fit_result.params.to_dict()
            pvalues = fit_result.pvalues.to_dict()
            
            # 删除常数项
            if 'const' in coefficients:
                del coefficients['const']
            if 'const' in pvalues:
                del pvalues['const']
            
            # 找出显著的敏感性（p值小于0.05）
            significant_factors = {
                factor: {
                    'coefficient': float(coef),
                    'p_value': float(pvalues[factor])
                }
                for factor, coef in coefficients.items()
                if pvalues[factor] < 0.05
            }
            
            # 存储结果
            exposures = {
                'all_coefficients': {factor: float(coef) for factor, coef in coefficients.items()},
                'significant_factors': significant_factors,
                'r_squared': float(fit_result.rsquared),
                'adjusted_r_squared': float(fit_result.rsquared_adj)
            }
            
            results['exposures'][asset] = exposures
            
            # 添加敏感性总结
            if significant_factors:
                for factor, data in significant_factors.items():
                    sensitivity = "高" if abs(data['coefficient']) > 0.5 else "中" if abs(data['coefficient']) > 0.2 else "低"
                    direction = "正" if data['coefficient'] > 0 else "负"
                    results['sensitivity_summary'].append(
                        f"{asset}对{factor}有{direction}向{sensitivity}敏感性（系数: {data['coefficient']:.4f}）"
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
    创建对特定宏观因子有目标敞口的投资组合
    
    Args:
        returns_df: 包含多个资产收益率的DataFrame
        macro_factors_df: 包含宏观经济因子的DataFrame
        target_factor: 目标宏观因子
        exposure_direction: 敞口方向，'positive'正向敞口或'negative'负向敞口
        
    Returns:
        Dict: 包含投资组合权重和因子敞口
    """
    # 首先分析所有资产对宏观因子的敏感性
    exposures_analysis = analyze_macro_factor_exposures(returns_df, macro_factors_df)
    
    if not exposures_analysis['success']:
        return {
            'success': False,
            'error': exposures_analysis.get('error', '宏观因子分析失败'),
            'weights': {col: 1.0/len(returns_df.columns) for col in returns_df.columns}  # 默认等权重
        }
    
    # 检查目标因子是否存在
    if target_factor not in macro_factors_df.columns:
        return {
            'success': False,
            'error': f'目标因子 {target_factor} 不在提供的宏观因子中',
            'weights': {col: 1.0/len(returns_df.columns) for col in returns_df.columns}  # 默认等权重
        }
    
    # 提取每个资产对目标因子的敏感性
    factor_sensitivities = {}
    for asset, exposure_data in exposures_analysis['exposures'].items():
        if 'all_coefficients' in exposure_data and target_factor in exposure_data['all_coefficients']:
            factor_sensitivities[asset] = exposure_data['all_coefficients'][target_factor]
    
    # 如果没有找到敏感性，返回错误
    if not factor_sensitivities:
        return {
            'success': False,
            'error': f'没有资产对因子 {target_factor} 有显著敏感性',
            'weights': {col: 1.0/len(returns_df.columns) for col in returns_df.columns}  # 默认等权重
        }
    
    # 根据敏感性方向筛选资产
    if exposure_direction == 'positive':
        selected_assets = {asset: sens for asset, sens in factor_sensitivities.items() if sens > 0}
    else:  # 'negative'
        selected_assets = {asset: sens for asset, sens in factor_sensitivities.items() if sens < 0}
    
    # 如果没有满足条件的资产，返回错误
    if not selected_assets:
        return {
            'success': False,
            'error': f'没有资产对因子 {target_factor} 有{exposure_direction}敏感性',
            'weights': {col: 1.0/len(returns_df.columns) for col in returns_df.columns}  # 默认等权重
        }
    
    # 计算权重 - 简单方法是基于敏感性绝对值的比例
    total_sensitivity = sum(abs(sens) for sens in selected_assets.values())
    
    if total_sensitivity > 0:
        weights = {asset: abs(sens)/total_sensitivity for asset, sens in selected_assets.items()}
    else:
        # 如果敏感性都是0，使用等权重
        weights = {asset: 1.0/len(selected_assets) for asset in selected_assets}
    
    # 计算投资组合对目标因子的总敞口
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
    分析资产对宏观因子敏感性随时间的变化
    
    Args:
        returns_df: 包含多个资产收益率的DataFrame
        macro_factors_df: 包含宏观经济因子的DataFrame
        window: 滚动窗口大小
        
    Returns:
        Dict: 包含因子轮动分析结果
    """
    # 对齐日期索引
    common_dates = returns_df.index.intersection(macro_factors_df.index)
    if len(common_dates) < window + 10:  # 需要足够的数据点
        return {
            'success': False,
            'error': f'数据点不足，需要至少 {window + 10} 个数据点，但只有 {len(common_dates)} 个',
            'rotation_analysis': {}
        }
    
    # 对齐数据
    aligned_returns = returns_df.loc[common_dates]
    aligned_factors = macro_factors_df.loc[common_dates]
    
    # 结果字典
    results = {
        'success': True,
        'rotation_analysis': {},
        'trend_summary': []
    }
    
    # 为每个资产计算滚动因子敏感性
    for asset in returns_df.columns:
        asset_returns = aligned_returns[asset]
        
        # 存储每个时间点的敏感性
        rolling_sensitivities = {}
        
        # 遍历每个时间点
        for i in range(window, len(common_dates)):
            current_date = common_dates[i]
            window_returns = asset_returns.iloc[i-window:i]
            window_factors = aligned_factors.iloc[i-window:i]
            
            # 添加常数项
            X = sm.add_constant(window_factors)
            
            try:
                # 执行线性回归
                model = sm.OLS(window_returns, X)
                fit_result = model.fit()
                
                # 提取系数
                coefficients = fit_result.params.to_dict()
                
                # 删除常数项
                if 'const' in coefficients:
                    del coefficients['const']
                
                # 存储该时间点的敏感性
                date_str = current_date.strftime('%Y-%m-%d')
                rolling_sensitivities[date_str] = {
                    factor: float(coef) for factor, coef in coefficients.items()
                }
                
            except Exception as e:
                # 忽略错误，继续下一个时间点
                continue
        
        # 分析敏感性趋势
        factor_trends = {}
        for factor in macro_factors_df.columns:
            # 提取该因子的所有时间点敏感性
            factor_sensitivities = [
                sensitivities.get(factor, np.nan) 
                for sensitivities in rolling_sensitivities.values()
            ]
            
            # 移除NaN值
            factor_sensitivities = [s for s in factor_sensitivities if not np.isnan(s)]
            
            if len(factor_sensitivities) >= 3:  # 至少需要3个点才能分析趋势
                # 简单线性回归来检测趋势
                X = np.arange(len(factor_sensitivities)).reshape(-1, 1)
                y = np.array(factor_sensitivities)
                
                try:
                    model = sm.OLS(y, sm.add_constant(X))
                    fit_result = model.fit()
                    
                    # 提取斜率和p值
                    slope = fit_result.params[1]
                    p_value = fit_result.pvalues[1]
                    
                    # 判断趋势
                    if p_value < 0.05:  # 显著趋势
                        trend = "上升" if slope > 0 else "下降"
                    else:
                        trend = "稳定"
                    
                    factor_trends[factor] = {
                        'slope': float(slope),
                        'p_value': float(p_value),
                        'trend': trend,
                        'start_sensitivity': float(factor_sensitivities[0]),
                        'end_sensitivity': float(factor_sensitivities[-1]),
                        'change': float(factor_sensitivities[-1] - factor_sensitivities[0])
                    }
                    
                    # 添加到趋势总结
                    if trend != "稳定":
                        results['trend_summary'].append(
                            f"{asset}对{factor}的敏感性呈{trend}趋势（斜率: {slope:.4f}）"
                        )
                        
                except Exception as e:
                    # 忽略错误，继续下一个因子
                    continue
        
        # 存储该资产的轮动分析结果
        results['rotation_analysis'][asset] = {
            'rolling_sensitivities': rolling_sensitivities,
            'factor_trends': factor_trends
        }
    
    return results