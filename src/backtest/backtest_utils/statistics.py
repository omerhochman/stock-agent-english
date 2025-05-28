import scipy.stats as stats
import numpy as np
from typing import Dict, Any

class StatisticalAnalyzer:
    """
    统计分析器
    提供各种统计检验和分析功能
    """
    
    def __init__(self):
        pass
    
    def normality_test(self, data: np.ndarray) -> Dict[str, Any]:
        """
        正态性检验
        
        Args:
            data: 数据序列
            
        Returns:
            Dict: 检验结果
        """
        # Shapiro-Wilk检验
        shapiro_stat, shapiro_p = stats.shapiro(data)
        
        # Jarque-Bera检验
        jb_stat, jb_p = stats.jarque_bera(data)
        
        # Kolmogorov-Smirnov检验
        ks_stat, ks_p = stats.kstest(data, 'norm')
        
        return {
            'shapiro_wilk': {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            },
            'jarque_bera': {
                'statistic': jb_stat,
                'p_value': jb_p,
                'is_normal': jb_p > 0.05
            },
            'kolmogorov_smirnov': {
                'statistic': ks_stat,
                'p_value': ks_p,
                'is_normal': ks_p > 0.05
            },
            'overall_conclusion': all([shapiro_p > 0.05, jb_p > 0.05, ks_p > 0.05])
        }
    
    def autocorrelation_test(self, data: np.ndarray, lags: int = 10) -> Dict[str, Any]:
        """
        自相关检验
        
        Args:
            data: 数据序列
            lags: 滞后期数
            
        Returns:
            Dict: 检验结果
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        # Ljung-Box检验
        lb_result = acorr_ljungbox(data, lags=lags, return_df=True)
        
        return {
            'ljung_box_results': lb_result.to_dict(),
            'has_autocorrelation': any(lb_result['lb_pvalue'] < 0.05),
            'significant_lags': lb_result[lb_result['lb_pvalue'] < 0.05].index.tolist()
        }
    
    def stationarity_test(self, data: np.ndarray) -> Dict[str, Any]:
        """
        平稳性检验
        
        Args:
            data: 数据序列
            
        Returns:
            Dict: 检验结果
        """
        from statsmodels.tsa.stattools import adfuller, kpss
        
        # ADF检验
        adf_result = adfuller(data)
        
        # KPSS检验
        kpss_result = kpss(data)
        
        return {
            'adf_test': {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            },
            'kpss_test': {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05
            }
        }
    
    def outlier_detection(self, data: np.ndarray, method: str = 'iqr') -> Dict[str, Any]:
        """
        异常值检测
        
        Args:
            data: 数据序列
            method: 检测方法 ('iqr', 'zscore', 'modified_zscore')
            
        Returns:
            Dict: 检测结果
        """
        outliers_mask = np.zeros(len(data), dtype=bool)
        
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_mask = (data < lower_bound) | (data > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers_mask = z_scores > 3
            
        elif method == 'modified_zscore':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers_mask = np.abs(modified_z_scores) > 3.5
        
        return {
            'method': method,
            'outliers_mask': outliers_mask,
            'outliers_count': np.sum(outliers_mask),
            'outliers_percentage': np.sum(outliers_mask) / len(data) * 100,
            'outliers_indices': np.where(outliers_mask)[0].tolist()
        }