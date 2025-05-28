import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats
from scipy.stats import normaltest

class SignificanceTester:
    """
    金融时间序列显著性检验类
    实现多种统计检验方法
    """
    
    def __init__(self, confidence_level: float = 0.05):
        self.confidence_level = confidence_level
        self.test_results = {}
        
    def diebold_mariano_test(self, errors1: np.ndarray, errors2: np.ndarray, 
                           h: int = 1, power: int = 2) -> Dict[str, Any]:
        """
        Diebold-Mariano检验
        比较两个预测方法的预测精度
        
        Args:
            errors1: 方法1的预测误差
            errors2: 方法2的预测误差
            h: 预测期间
            power: 损失函数的幂次（1=MAE, 2=MSE）
            
        Returns:
            Dict: 检验结果
        """
        # 计算损失差异
        loss1 = np.abs(errors1) ** power
        loss2 = np.abs(errors2) ** power
        loss_diff = loss1 - loss2
        
        # 计算样本统计量
        n = len(loss_diff)
        mean_diff = np.mean(loss_diff)
        
        # 计算长期方差
        # 使用Newey-West方法估计长期方差
        gamma_0 = np.var(loss_diff, ddof=1)
        
        # 计算自协方差
        lags = min(int(np.floor(4 * (n/100)**(2/9))), n-1)
        gamma_sum = 0
        
        for k in range(1, lags + 1):
            if n - k > 0:
                gamma_k = np.cov(loss_diff[:-k], loss_diff[k:])[0, 1]
                weight = 1 - k / (lags + 1)  # Bartlett权重
                gamma_sum += 2 * weight * gamma_k
                
        long_run_var = gamma_0 + gamma_sum
        
        # DM统计量
        if long_run_var <= 0:
            dm_stat = 0
            p_value = 1.0
        else:
            dm_stat = mean_diff / np.sqrt(long_run_var / n)
            
            # Harvey, Leybourne and Newbold (1997) 修正
            dm_stat_corrected = dm_stat * np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
            
            # 计算p值（双侧检验）
            p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=n-1))
        
        return {
            'statistic': dm_stat,
            'p_value': p_value,
            'critical_value': stats.t.ppf(1 - self.confidence_level/2, df=n-1),
            'mean_loss_diff': mean_diff,
            'significant': p_value < self.confidence_level,
            'interpretation': self._interpret_dm_test(mean_diff, p_value)
        }
    
    def _interpret_dm_test(self, mean_diff: float, p_value: float) -> str:
        """解释DM检验结果"""
        if p_value >= self.confidence_level:
            return "两种方法的预测精度无显著差异"
        elif mean_diff < 0:
            return "方法1显著优于方法2"
        else:
            return "方法2显著优于方法1"
    
    def paired_t_test(self, returns1: np.ndarray, returns2: np.ndarray) -> Dict[str, Any]:
        """
        配对t检验
        比较两个策略的平均收益率
        """
        diff = returns1 - returns2
        
        # 正态性检验
        _, normality_p = normaltest(diff)
        
        if normality_p < 0.05:
            # 使用Wilcoxon符号秩检验（非参数）
            statistic, p_value = stats.wilcoxon(returns1, returns2, alternative='two-sided')
            test_type = "Wilcoxon Signed-Rank"
        else:
            # 使用配对t检验
            statistic, p_value = stats.ttest_rel(returns1, returns2)
            test_type = "Paired t-test"
        
        return {
            'test_type': test_type,
            'statistic': statistic,
            'p_value': p_value,
            'mean_diff': np.mean(diff),
            'std_diff': np.std(diff, ddof=1),
            'significant': p_value < self.confidence_level,
            'normality_p': normality_p,
            'interpretation': self._interpret_paired_test(np.mean(diff), p_value)
        }
    
    def _interpret_paired_test(self, mean_diff: float, p_value: float) -> str:
        """解释配对检验结果"""
        if p_value >= self.confidence_level:
            return "两种策略的平均收益率无显著差异"
        elif mean_diff > 0:
            return "策略1的平均收益率显著高于策略2"
        else:
            return "策略2的平均收益率显著高于策略1"
    
    def variance_ratio_test(self, returns: np.ndarray, lags: List[int] = [2, 4, 8, 16]) -> Dict[str, Any]:
        """
        方差比检验
        检验随机游走假设
        Lo and MacKinlay (1988)
        """
        results = {}
        n = len(returns)
        
        for lag in lags:
            # 计算1期收益率方差
            var_1 = np.var(returns, ddof=1)
            
            # Handle division by zero for var_1
            if var_1 == 0 or np.isclose(var_1, 0, atol=1e-10):
                results[f'lag_{lag}'] = {
                    'variance_ratio': 1.0,
                    'statistic_homo': 0.0,
                    'statistic_hetero': 0.0,
                    'p_value_homo': 1.0,
                    'p_value_hetero': 1.0,
                    'significant_homo': False,
                    'significant_hetero': False
                }
                continue
            
            # 计算lag期收益率
            lag_returns = []
            for i in range(0, n - lag + 1, lag):
                lag_return = np.sum(returns[i:i+lag])
                lag_returns.append(lag_return)
            
            var_lag = np.var(lag_returns, ddof=1)
            vr = var_lag / (lag * var_1)
            
            # 计算统计量
            # 同方差假设下的统计量
            vr_stat_homo = (vr - 1) * np.sqrt(n * lag / (2 * (lag - 1)))
            
            # 异方差假设下的统计量（更稳健）
            # 计算异方差调整项
            delta = 0
            for j in range(1, lag):
                sum_term = 0
                for i in range(j, n):
                    sum_term += (returns[i] * returns[i-j]) ** 2
                delta += (2 * (lag - j) / lag) ** 2 * sum_term
            
            theta = (sum(returns**4) - (np.var(returns, ddof=1))**2) / n
            phi = delta / ((n * var_1)**2)
            
            vr_var_hetero = (2 * (2*lag - 1) * (lag - 1)) / (3 * lag * n) + theta + phi
            vr_stat_hetero = (vr - 1) / np.sqrt(vr_var_hetero) if vr_var_hetero > 0 else 0
            
            # 计算p值
            p_value_homo = 2 * (1 - stats.norm.cdf(np.abs(vr_stat_homo)))
            p_value_hetero = 2 * (1 - stats.norm.cdf(np.abs(vr_stat_hetero)))
            
            results[f'lag_{lag}'] = {
                'variance_ratio': vr,
                'statistic_homo': vr_stat_homo,
                'statistic_hetero': vr_stat_hetero,
                'p_value_homo': p_value_homo,
                'p_value_hetero': p_value_hetero,
                'significant_homo': p_value_homo < self.confidence_level,
                'significant_hetero': p_value_hetero < self.confidence_level
            }
        
        return results
    
    def sharpe_ratio_test(self, returns1: np.ndarray, returns2: np.ndarray, 
                         risk_free_rate: float = 0.0) -> Dict[str, Any]:
        """
        夏普比率差异的显著性检验
        Jobson and Korkie (1981), Memmel (2003)
        """
        # 计算夏普比率
        excess_ret1 = returns1 - risk_free_rate
        excess_ret2 = returns2 - risk_free_rate
        
        # 安全计算夏普比率
        def safe_sharpe(excess_returns):
            std_ret = np.std(excess_returns, ddof=1)
            if std_ret == 0 or np.isclose(std_ret, 0, atol=1e-10) or not np.isfinite(std_ret):
                return 0.0
            mean_ret = np.mean(excess_returns)
            if not np.isfinite(mean_ret):
                return 0.0
            sharpe = mean_ret / std_ret
            return 0.0 if not np.isfinite(sharpe) else sharpe
        
        sharpe1 = safe_sharpe(excess_ret1)
        sharpe2 = safe_sharpe(excess_ret2)
        
        n = len(returns1)
        
        # 计算协方差矩阵
        cov_matrix = np.cov(excess_ret1, excess_ret2, ddof=1)
        
        mu1, mu2 = np.mean(excess_ret1), np.mean(excess_ret2)
        sigma1, sigma2 = np.std(excess_ret1, ddof=1), np.std(excess_ret2, ddof=1)
        sigma12 = cov_matrix[0, 1]
        
        if sigma1 == 0 or sigma2 == 0:
            return {
                'sharpe1': sharpe1,
                'sharpe2': sharpe2,
                'difference': sharpe1 - sharpe2,
                'statistic': 0,
                'p_value': 1.0,
                'significant': False,
                'interpretation': "无法计算（标准差为零）"
            }
        
        # Jobson-Korkie统计量
        theta = (sigma1**2 * sigma2**2 + 
                sigma1**2 * mu2**2 + 
                sigma2**2 * mu1**2 - 
                2 * sigma1 * sigma2 * mu1 * mu2 * sigma12/(sigma1 * sigma2))
        
        jk_stat = (sharpe1 - sharpe2) * np.sqrt(n) / np.sqrt(theta/(sigma1**2 * sigma2**2))
        
        # Memmel修正
        memmel_correction = (sharpe1 * sharpe2 - 0.5 * sigma12/(sigma1 * sigma2)) / n
        memmel_stat = jk_stat - memmel_correction * np.sqrt(n)
        
        # 计算p值
        p_value = 2 * (1 - stats.norm.cdf(np.abs(memmel_stat)))
        
        return {
            'sharpe1': sharpe1,
            'sharpe2': sharpe2,
            'difference': sharpe1 - sharpe2,
            'statistic': memmel_stat,
            'p_value': p_value,
            'significant': p_value < self.confidence_level,
            'interpretation': self._interpret_sharpe_test(sharpe1, sharpe2, p_value)
        }
    
    def _interpret_sharpe_test(self, sharpe1: float, sharpe2: float, p_value: float) -> str:
        """解释夏普比率检验结果"""
        if p_value >= self.confidence_level:
            return "两种策略的夏普比率无显著差异"
        elif sharpe1 > sharpe2:
            return "策略1的夏普比率显著高于策略2"
        else:
            return "策略2的夏普比率显著高于策略1"
    
    def alpha_significance_test(self, strategy_returns: np.ndarray, 
                              benchmark_returns: np.ndarray) -> Dict[str, Any]:
        """
        Alpha显著性检验
        使用CAPM模型检验超额收益的显著性
        """
        from sklearn.linear_model import LinearRegression
        from scipy.stats import t
        
        # 准备数据
        X = benchmark_returns.reshape(-1, 1)
        y = strategy_returns
        
        # 回归分析
        reg = LinearRegression().fit(X, y)
        
        # 预测值和残差
        y_pred = reg.predict(X)
        residuals = y - y_pred
        
        # 计算统计量
        n = len(y)
        mse = np.sum(residuals**2) / (n - 2)
        
        # Alpha（截距）的标准误
        x_mean = np.mean(benchmark_returns)
        x_var = np.var(benchmark_returns, ddof=1)
        se_alpha = np.sqrt(mse * (1/n + x_mean**2/(n * x_var)))
        
        # t统计量
        alpha = reg.intercept_
        t_stat = alpha / se_alpha if se_alpha > 0 else 0
        
        # p值
        p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=n-2))
        
        return {
            'alpha': alpha,
            'beta': reg.coef_[0],
            't_statistic': t_stat,
            'p_value': p_value,
            'r_squared': reg.score(X, y),
            'significant': p_value < self.confidence_level,
            'interpretation': self._interpret_alpha_test(alpha, p_value)
        }
    
    def _interpret_alpha_test(self, alpha: float, p_value: float) -> str:
        """解释Alpha检验结果"""
        if p_value >= self.confidence_level:
            return "策略未产生显著的超额收益（Alpha不显著）"
        elif alpha > 0:
            return "策略产生显著的正超额收益"
        else:
            return "策略产生显著的负超额收益"
    
    def bootstrap_confidence_interval(self, data: np.ndarray, statistic_func, 
                                    n_bootstrap: int = 1000, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Bootstrap置信区间
        用于非参数统计推断
        """
        n = len(data)
        bootstrap_stats = []
        
        np.random.seed(42)  # 为了结果可重复
        
        for _ in range(n_bootstrap):
            # 有放回抽样
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # 计算置信区间
        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        original_stat = statistic_func(data)
        
        return {
            'original_statistic': original_stat,
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats, ddof=1),
            'confidence_interval': (ci_lower, ci_upper),
            'bias': np.mean(bootstrap_stats) - original_stat,
            'contains_zero': ci_lower <= 0 <= ci_upper
        }
    
    def comprehensive_comparison(self, strategy1_returns: np.ndarray, 
                               strategy2_returns: np.ndarray,
                               benchmark_returns: Optional[np.ndarray] = None,
                               strategy1_name: str = "Strategy 1",
                               strategy2_name: str = "Strategy 2") -> Dict[str, Any]:
        """
        综合比较分析
        集成多种统计检验方法
        """
        results = {
            'strategy_names': (strategy1_name, strategy2_name),
            'sample_size': len(strategy1_returns)
        }
        
        # 1. 配对t检验
        results['paired_test'] = self.paired_t_test(strategy1_returns, strategy2_returns)
        
        # 2. DM检验（以收益率为基础）
        # 将收益率转换为"误差"（相对于零收益的偏差）
        errors1 = strategy1_returns  # 可以使用收益率本身
        errors2 = strategy2_returns
        results['diebold_mariano'] = self.diebold_mariano_test(errors1, errors2)
        
        # 3. 夏普比率检验
        results['sharpe_test'] = self.sharpe_ratio_test(strategy1_returns, strategy2_returns)
        
        # 4. 如果有基准，进行Alpha检验
        if benchmark_returns is not None:
            results['alpha_test_1'] = self.alpha_significance_test(strategy1_returns, benchmark_returns)
            results['alpha_test_2'] = self.alpha_significance_test(strategy2_returns, benchmark_returns)
        
        # 5. 方差比检验（检验随机游走）
        results['variance_ratio_1'] = self.variance_ratio_test(strategy1_returns)
        results['variance_ratio_2'] = self.variance_ratio_test(strategy2_returns)
        
        # 6. Bootstrap置信区间
        results['bootstrap_mean_1'] = self.bootstrap_confidence_interval(
            strategy1_returns, np.mean)
        results['bootstrap_mean_2'] = self.bootstrap_confidence_interval(
            strategy2_returns, np.mean)
        
        results['bootstrap_sharpe_1'] = self.bootstrap_confidence_interval(
            strategy1_returns, lambda x: np.mean(x) / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 0)
        results['bootstrap_sharpe_2'] = self.bootstrap_confidence_interval(
            strategy2_returns, lambda x: np.mean(x) / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 0)
        
        # 7. 生成综合结论
        results['summary'] = self._generate_comparison_summary(results)
        
        return results
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合比较结论"""
        strategy1_name, strategy2_name = results['strategy_names']
        
        # 统计显著性结果
        significant_tests = []
        
        if results['paired_test']['significant']:
            winner = strategy1_name if results['paired_test']['mean_diff'] > 0 else strategy2_name
            significant_tests.append(f"配对检验：{winner}显著优于另一策略")
        
        if results['diebold_mariano']['significant']:
            significant_tests.append(f"DM检验：{results['diebold_mariano']['interpretation']}")
        
        if results['sharpe_test']['significant']:
            significant_tests.append(f"夏普比率检验：{results['sharpe_test']['interpretation']}")
        
        # Alpha检验结果
        alpha_results = []
        if 'alpha_test_1' in results:
            if results['alpha_test_1']['significant']:
                alpha_results.append(f"{strategy1_name}: {results['alpha_test_1']['interpretation']}")
        if 'alpha_test_2' in results:
            if results['alpha_test_2']['significant']:
                alpha_results.append(f"{strategy2_name}: {results['alpha_test_2']['interpretation']}")
        
        return {
            'significant_differences': significant_tests,
            'alpha_results': alpha_results,
            'overall_conclusion': self._determine_overall_winner(results),
            'statistical_power': len(significant_tests) / 3  # 3个主要比较检验
        }
    
    def _determine_overall_winner(self, results: Dict[str, Any]) -> str:
        """确定总体优胜策略"""
        strategy1_name, strategy2_name = results['strategy_names']
        
        # 计分系统
        score1, score2 = 0, 0
        
        # 配对检验
        if results['paired_test']['significant']:
            if results['paired_test']['mean_diff'] > 0:
                score1 += 1
            else:
                score2 += 1
        
        # DM检验
        if results['diebold_mariano']['significant']:
            if results['diebold_mariano']['mean_loss_diff'] < 0:
                score1 += 1
            else:
                score2 += 1
        
        # 夏普比率检验
        if results['sharpe_test']['significant']:
            if results['sharpe_test']['difference'] > 0:
                score1 += 1
            else:
                score2 += 1
        
        if score1 > score2:
            return f"{strategy1_name}在统计上显著优于{strategy2_name}"
        elif score2 > score1:
            return f"{strategy2_name}在统计上显著优于{strategy1_name}"
        else:
            return "两种策略在统计上无显著差异"