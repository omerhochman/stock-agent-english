from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats import normaltest


class SignificanceTester:
    """
    Financial time series significance testing class
    Implements various statistical testing methods
    """

    def __init__(self, confidence_level: float = 0.05):
        self.confidence_level = confidence_level
        self.test_results = {}

    def diebold_mariano_test(
        self, errors1: np.ndarray, errors2: np.ndarray, h: int = 1, power: int = 2
    ) -> Dict[str, Any]:
        """
        Diebold-Mariano test
        Compare prediction accuracy of two forecasting methods

        Args:
            errors1: Prediction errors of method 1
            errors2: Prediction errors of method 2
            h: Forecast horizon
            power: Power of loss function (1=MAE, 2=MSE)

        Returns:
            Dict: Test results
        """
        # Calculate loss difference
        loss1 = np.abs(errors1) ** power
        loss2 = np.abs(errors2) ** power
        loss_diff = loss1 - loss2

        # Calculate sample statistics
        n = len(loss_diff)
        mean_diff = np.mean(loss_diff)

        # Calculate long-run variance
        # Use Newey-West method to estimate long-run variance
        gamma_0 = np.var(loss_diff, ddof=1)

        # Calculate autocovariance
        lags = min(int(np.floor(4 * (n / 100) ** (2 / 9))), n - 1)
        gamma_sum = 0

        for k in range(1, lags + 1):
            if n - k > 0:
                gamma_k = np.cov(loss_diff[:-k], loss_diff[k:])[0, 1]
                weight = 1 - k / (lags + 1)  # Bartlett weight
                gamma_sum += 2 * weight * gamma_k

        long_run_var = gamma_0 + gamma_sum

        # DM statistic
        if long_run_var <= 0:
            dm_stat = 0
            p_value = 1.0
        else:
            dm_stat = mean_diff / np.sqrt(long_run_var / n)

            # Harvey, Leybourne and Newbold (1997) correction
            dm_stat_corrected = dm_stat * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)

            # Calculate p-value (two-tailed test)
            p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=n - 1))

        return {
            "statistic": dm_stat,
            "p_value": p_value,
            "critical_value": stats.t.ppf(1 - self.confidence_level / 2, df=n - 1),
            "mean_loss_diff": mean_diff,
            "significant": p_value < self.confidence_level,
            "interpretation": self._interpret_dm_test(mean_diff, p_value),
        }

    def _interpret_dm_test(self, mean_diff: float, p_value: float) -> str:
        """Interpret DM test results"""
        if p_value >= self.confidence_level:
            return "No significant difference in prediction accuracy between the two methods"
        elif mean_diff < 0:
            return "Method 1 significantly outperforms method 2"
        else:
            return "Method 2 significantly outperforms method 1"

    def _align_returns(
        self, returns1: np.ndarray, returns2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two return arrays to ensure they have the same length
        Use the length of the shorter array, truncate the longer array from the end
        """
        len1, len2 = len(returns1), len(returns2)

        if len1 == len2:
            return returns1, returns2

        # Use the shorter length
        min_len = min(len1, len2)

        # Truncate from the end, keeping the most recent data
        aligned_returns1 = returns1[-min_len:] if len1 > min_len else returns1
        aligned_returns2 = returns2[-min_len:] if len2 > min_len else returns2

        return aligned_returns1, aligned_returns2

    def paired_t_test(
        self, returns1: np.ndarray, returns2: np.ndarray
    ) -> Dict[str, Any]:
        """
        Paired t-test
        Compare average returns of two strategies
        """
        # Align return arrays
        aligned_returns1, aligned_returns2 = self._align_returns(returns1, returns2)

        diff = aligned_returns1 - aligned_returns2

        # Normality test
        _, normality_p = normaltest(diff)

        if normality_p < 0.05:
            # Use Wilcoxon signed-rank test (non-parametric)
            statistic, p_value = stats.wilcoxon(
                aligned_returns1, aligned_returns2, alternative="two-sided"
            )
            test_type = "Wilcoxon Signed-Rank"
        else:
            # Use paired t-test
            statistic, p_value = stats.ttest_rel(aligned_returns1, aligned_returns2)
            test_type = "Paired t-test"

        return {
            "test_type": test_type,
            "statistic": statistic,
            "p_value": p_value,
            "mean_diff": np.mean(diff),
            "std_diff": np.std(diff, ddof=1),
            "significant": p_value < self.confidence_level,
            "normality_p": normality_p,
            "interpretation": self._interpret_paired_test(np.mean(diff), p_value),
        }

    def _interpret_paired_test(self, mean_diff: float, p_value: float) -> str:
        """Interpret paired test results"""
        if p_value >= self.confidence_level:
            return "No significant difference in average returns between the two strategies"
        elif mean_diff > 0:
            return "Strategy 1's average return is significantly higher than strategy 2"
        else:
            return "Strategy 2's average return is significantly higher than strategy 1"

    def variance_ratio_test(
        self, returns: np.ndarray, lags: List[int] = [2, 4, 8, 16]
    ) -> Dict[str, Any]:
        """
        Variance ratio test
        Test random walk hypothesis
        Lo and MacKinlay (1988)
        """
        results = {}
        n = len(returns)

        for lag in lags:
            # Calculate 1-period return variance
            var_1 = np.var(returns, ddof=1)

            # Handle division by zero for var_1
            if var_1 == 0 or np.isclose(var_1, 0, atol=1e-10):
                results[f"lag_{lag}"] = {
                    "variance_ratio": 1.0,
                    "statistic_homo": 0.0,
                    "statistic_hetero": 0.0,
                    "p_value_homo": 1.0,
                    "p_value_hetero": 1.0,
                    "significant_homo": False,
                    "significant_hetero": False,
                }
                continue

            # Calculate lag-period returns
            lag_returns = []
            for i in range(0, n - lag + 1, lag):
                lag_return = np.sum(returns[i : i + lag])
                lag_returns.append(lag_return)

            var_lag = np.var(lag_returns, ddof=1)
            vr = var_lag / (lag * var_1)

            # Calculate statistics
            # Statistic under homoscedasticity assumption
            vr_stat_homo = (vr - 1) * np.sqrt(n * lag / (2 * (lag - 1)))

            # Statistic under heteroscedasticity assumption (more robust)
            # Calculate heteroscedasticity adjustment term
            delta = 0
            for j in range(1, lag):
                sum_term = 0
                for i in range(j, n):
                    sum_term += (returns[i] * returns[i - j]) ** 2
                delta += (2 * (lag - j) / lag) ** 2 * sum_term

            theta = (sum(returns**4) - (np.var(returns, ddof=1)) ** 2) / n
            phi = delta / ((n * var_1) ** 2)

            vr_var_hetero = (
                (2 * (2 * lag - 1) * (lag - 1)) / (3 * lag * n) + theta + phi
            )
            vr_stat_hetero = (
                (vr - 1) / np.sqrt(vr_var_hetero) if vr_var_hetero > 0 else 0
            )

            # Calculate p-values
            p_value_homo = 2 * (1 - stats.norm.cdf(np.abs(vr_stat_homo)))
            p_value_hetero = 2 * (1 - stats.norm.cdf(np.abs(vr_stat_hetero)))

            results[f"lag_{lag}"] = {
                "variance_ratio": vr,
                "statistic_homo": vr_stat_homo,
                "statistic_hetero": vr_stat_hetero,
                "p_value_homo": p_value_homo,
                "p_value_hetero": p_value_hetero,
                "significant_homo": p_value_homo < self.confidence_level,
                "significant_hetero": p_value_hetero < self.confidence_level,
            }

        return results

    def sharpe_ratio_test(
        self, returns1: np.ndarray, returns2: np.ndarray, risk_free_rate: float = 0.0
    ) -> Dict[str, Any]:
        """
        Significance test for Sharpe ratio difference
        Jobson and Korkie (1981), Memmel (2003)
        """
        # Align return arrays
        aligned_returns1, aligned_returns2 = self._align_returns(returns1, returns2)

        # Calculate Sharpe ratios
        excess_ret1 = aligned_returns1 - risk_free_rate
        excess_ret2 = aligned_returns2 - risk_free_rate

        # Safely calculate Sharpe ratio
        def safe_sharpe(excess_returns):
            std_ret = np.std(excess_returns, ddof=1)
            if (
                std_ret == 0
                or np.isclose(std_ret, 0, atol=1e-10)
                or not np.isfinite(std_ret)
            ):
                return 0.0
            mean_ret = np.mean(excess_returns)
            if not np.isfinite(mean_ret):
                return 0.0
            sharpe = mean_ret / std_ret
            return 0.0 if not np.isfinite(sharpe) else sharpe

        sharpe1 = safe_sharpe(excess_ret1)
        sharpe2 = safe_sharpe(excess_ret2)

        n = len(aligned_returns1)

        # Calculate covariance matrix
        cov_matrix = np.cov(excess_ret1, excess_ret2, ddof=1)

        mu1, mu2 = np.mean(excess_ret1), np.mean(excess_ret2)
        sigma1, sigma2 = np.std(excess_ret1, ddof=1), np.std(excess_ret2, ddof=1)
        sigma12 = cov_matrix[0, 1]

        if sigma1 == 0 or sigma2 == 0:
            return {
                "sharpe1": sharpe1,
                "sharpe2": sharpe2,
                "difference": sharpe1 - sharpe2,
                "statistic": 0,
                "p_value": 1.0,
                "significant": False,
                "interpretation": "Cannot calculate (standard deviation is zero)",
            }

        # Jobson-Korkie statistic
        theta = (
            sigma1**2 * sigma2**2
            + sigma1**2 * mu2**2
            + sigma2**2 * mu1**2
            - 2 * sigma1 * sigma2 * mu1 * mu2 * sigma12 / (sigma1 * sigma2)
        )

        jk_stat = (
            (sharpe1 - sharpe2) * np.sqrt(n) / np.sqrt(theta / (sigma1**2 * sigma2**2))
        )

        # Memmel correction
        memmel_correction = (sharpe1 * sharpe2 - 0.5 * sigma12 / (sigma1 * sigma2)) / n
        memmel_stat = jk_stat - memmel_correction * np.sqrt(n)

        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(np.abs(memmel_stat)))

        return {
            "sharpe1": sharpe1,
            "sharpe2": sharpe2,
            "difference": sharpe1 - sharpe2,
            "statistic": memmel_stat,
            "p_value": p_value,
            "significant": p_value < self.confidence_level,
            "interpretation": self._interpret_sharpe_test(sharpe1, sharpe2, p_value),
        }

    def _interpret_sharpe_test(
        self, sharpe1: float, sharpe2: float, p_value: float
    ) -> str:
        """Interpret Sharpe ratio test results"""
        if p_value >= self.confidence_level:
            return (
                "No significant difference in Sharpe ratios between the two strategies"
            )
        elif sharpe1 > sharpe2:
            return "Strategy 1's Sharpe ratio is significantly higher than strategy 2"
        else:
            return "Strategy 2's Sharpe ratio is significantly higher than strategy 1"

    def alpha_significance_test(
        self, strategy_returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Alpha significance test
        Test significance of excess returns using CAPM model
        """
        from scipy.stats import t
        from sklearn.linear_model import LinearRegression

        # Align return arrays
        aligned_strategy_returns, aligned_benchmark_returns = self._align_returns(
            strategy_returns, benchmark_returns
        )

        # Prepare data
        X = aligned_benchmark_returns.reshape(-1, 1)
        y = aligned_strategy_returns

        # Regression analysis
        reg = LinearRegression().fit(X, y)

        # Predicted values and residuals
        y_pred = reg.predict(X)
        residuals = y - y_pred

        # Calculate statistics
        n = len(y)
        mse = np.sum(residuals**2) / (n - 2)

        # Standard error of Alpha (intercept)
        x_mean = np.mean(aligned_benchmark_returns)
        x_var = np.var(aligned_benchmark_returns, ddof=1)
        se_alpha = np.sqrt(mse * (1 / n + x_mean**2 / (n * x_var)))

        # t-statistic
        alpha = reg.intercept_
        t_stat = alpha / se_alpha if se_alpha > 0 else 0

        # p-value
        p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=n - 2))

        return {
            "alpha": alpha,
            "beta": reg.coef_[0],
            "t_statistic": t_stat,
            "p_value": p_value,
            "r_squared": reg.score(X, y),
            "significant": p_value < self.confidence_level,
            "interpretation": self._interpret_alpha_test(alpha, p_value),
        }

    def _interpret_alpha_test(self, alpha: float, p_value: float) -> str:
        """Interpret Alpha test results"""
        if p_value >= self.confidence_level:
            return "Strategy did not generate significant excess returns (Alpha not significant)"
        elif alpha > 0:
            return "Strategy generates significant positive excess returns"
        else:
            return "Strategy generates significant negative excess returns"

    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Bootstrap confidence interval
        For non-parametric statistical inference
        """
        n = len(data)
        bootstrap_stats = []

        np.random.seed(42)  # For reproducible results

        for _ in range(n_bootstrap):
            # Sampling with replacement
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)

        bootstrap_stats = np.array(bootstrap_stats)

        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)

        original_stat = statistic_func(data)

        return {
            "original_statistic": original_stat,
            "bootstrap_mean": np.mean(bootstrap_stats),
            "bootstrap_std": np.std(bootstrap_stats, ddof=1),
            "confidence_interval": (ci_lower, ci_upper),
            "bias": np.mean(bootstrap_stats) - original_stat,
            "contains_zero": ci_lower <= 0 <= ci_upper,
        }

    def comprehensive_comparison(
        self,
        strategy1_returns: np.ndarray,
        strategy2_returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        strategy1_name: str = "Strategy 1",
        strategy2_name: str = "Strategy 2",
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison analysis
        Integrates multiple statistical testing methods
        """
        # Align return arrays to ensure consistency
        aligned_returns1, aligned_returns2 = self._align_returns(
            strategy1_returns, strategy2_returns
        )

        # Record alignment information
        original_len1, original_len2 = len(strategy1_returns), len(strategy2_returns)
        aligned_len = len(aligned_returns1)

        results = {
            "strategy_names": (strategy1_name, strategy2_name),
            "sample_size": aligned_len,
            "alignment_info": {
                "original_lengths": (original_len1, original_len2),
                "aligned_length": aligned_len,
                "data_trimmed": original_len1 != aligned_len
                or original_len2 != aligned_len,
            },
        }

        # 1. Paired t-test
        results["paired_test"] = self.paired_t_test(aligned_returns1, aligned_returns2)

        # 2. DM test (based on returns)
        # Convert returns to "errors" (deviation from zero return)
        errors1 = aligned_returns1  # Can use returns themselves
        errors2 = aligned_returns2
        results["diebold_mariano"] = self.diebold_mariano_test(errors1, errors2)

        # 3. Sharpe ratio test
        results["sharpe_test"] = self.sharpe_ratio_test(
            aligned_returns1, aligned_returns2
        )

        # 4. If benchmark exists, perform Alpha test
        if benchmark_returns is not None:
            results["alpha_test_1"] = self.alpha_significance_test(
                aligned_returns1, benchmark_returns
            )
            results["alpha_test_2"] = self.alpha_significance_test(
                aligned_returns2, benchmark_returns
            )

        # 5. Variance ratio test (test random walk)
        results["variance_ratio_1"] = self.variance_ratio_test(aligned_returns1)
        results["variance_ratio_2"] = self.variance_ratio_test(aligned_returns2)

        # 6. Bootstrap confidence intervals
        results["bootstrap_mean_1"] = self.bootstrap_confidence_interval(
            aligned_returns1, np.mean
        )
        results["bootstrap_mean_2"] = self.bootstrap_confidence_interval(
            aligned_returns2, np.mean
        )

        results["bootstrap_sharpe_1"] = self.bootstrap_confidence_interval(
            aligned_returns1,
            lambda x: np.mean(x) / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 0,
        )
        results["bootstrap_sharpe_2"] = self.bootstrap_confidence_interval(
            aligned_returns2,
            lambda x: np.mean(x) / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 0,
        )

        # 7. Generate comprehensive conclusions
        results["summary"] = self._generate_comparison_summary(results)

        return results

    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive comparison conclusions"""
        strategy1_name, strategy2_name = results["strategy_names"]

        # Statistical significance results
        significant_tests = []

        if results["paired_test"]["significant"]:
            winner = (
                strategy1_name
                if results["paired_test"]["mean_diff"] > 0
                else strategy2_name
            )
            significant_tests.append(
                f"Paired test: {winner} significantly outperforms the other strategy"
            )

        if results["diebold_mariano"]["significant"]:
            significant_tests.append(
                f"DM test: {results['diebold_mariano']['interpretation']}"
            )

        if results["sharpe_test"]["significant"]:
            significant_tests.append(
                f"Sharpe ratio test: {results['sharpe_test']['interpretation']}"
            )

        # Alpha test results
        alpha_results = []
        if "alpha_test_1" in results:
            if results["alpha_test_1"]["significant"]:
                alpha_results.append(
                    f"{strategy1_name}: {results['alpha_test_1']['interpretation']}"
                )
        if "alpha_test_2" in results:
            if results["alpha_test_2"]["significant"]:
                alpha_results.append(
                    f"{strategy2_name}: {results['alpha_test_2']['interpretation']}"
                )

        return {
            "significant_differences": significant_tests,
            "alpha_results": alpha_results,
            "overall_conclusion": self._determine_overall_winner(results),
            "statistical_power": len(significant_tests) / 3,  # 3 main comparison tests
        }

    def _determine_overall_winner(self, results: Dict[str, Any]) -> str:
        """Determine overall winning strategy"""
        strategy1_name, strategy2_name = results["strategy_names"]

        # Scoring system
        score1, score2 = 0, 0

        # Paired test
        if results["paired_test"]["significant"]:
            if results["paired_test"]["mean_diff"] > 0:
                score1 += 1
            else:
                score2 += 1

        # DM test
        if results["diebold_mariano"]["significant"]:
            if results["diebold_mariano"]["mean_loss_diff"] < 0:
                score1 += 1
            else:
                score2 += 1

        # Sharpe ratio test
        if results["sharpe_test"]["significant"]:
            if results["sharpe_test"]["difference"] > 0:
                score1 += 1
            else:
                score2 += 1

        if score1 > score2:
            return f"{strategy1_name} is statistically significantly better than {strategy2_name}"
        elif score2 > score1:
            return f"{strategy2_name} is statistically significantly better than {strategy1_name}"
        else:
            return "No significant statistical difference between the two strategies"
