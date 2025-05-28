import numpy as np
import pandas as pd
from typing import Dict

class PerformanceAnalyzer:
    """
    性能分析器
    提供投资组合性能分析功能
    """
    
    def __init__(self):
        pass
    
    def calculate_rolling_metrics(self, returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """
        计算滚动性能指标
        
        Args:
            returns: 收益率序列
            window: 滚动窗口大小
            
        Returns:
            pd.DataFrame: 滚动指标
        """
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # 滚动收益率
        rolling_metrics['rolling_return'] = returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1, raw=True
        )
        
        # 滚动波动率
        rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        # 滚动夏普比率
        def safe_rolling_sharpe(x):
            if len(x) == 0:
                return 0.0
            mean_ret = x.mean()
            std_ret = x.std()
            if std_ret == 0 or np.isclose(std_ret, 0, atol=1e-10) or not np.isfinite(std_ret):
                return 0.0
            if not np.isfinite(mean_ret):
                return 0.0
            sharpe = mean_ret / std_ret * np.sqrt(252)
            return 0.0 if not np.isfinite(sharpe) else sharpe
        
        rolling_metrics['rolling_sharpe'] = returns.rolling(window).apply(
            safe_rolling_sharpe, raw=True
        )
        
        # 滚动最大回撤
        rolling_metrics['rolling_max_drawdown'] = returns.rolling(window).apply(
            self._calculate_max_drawdown, raw=True
        )
        
        return rolling_metrics.dropna()
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative / running_max) - 1
        return np.min(drawdown)
    
    def performance_attribution(self, portfolio_returns: pd.Series, 
                              benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        性能归因分析
        
        Args:
            portfolio_returns: 投资组合收益率
            benchmark_returns: 基准收益率
            
        Returns:
            Dict: 归因分析结果
        """
        # 对齐数据
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        portfolio_ret = aligned_data['portfolio']
        benchmark_ret = aligned_data['benchmark']
        
        # 计算各项指标
        excess_returns = portfolio_ret - benchmark_ret
        
        attribution = {
            'total_return_portfolio': (1 + portfolio_ret).prod() - 1,
            'total_return_benchmark': (1 + benchmark_ret).prod() - 1,
            'excess_return': (1 + excess_returns).prod() - 1,
            'tracking_error': excess_returns.std() * np.sqrt(252),
            'information_ratio': excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0,
            'beta': np.cov(portfolio_ret, benchmark_ret)[0, 1] / np.var(benchmark_ret) if np.var(benchmark_ret) != 0 else 1,
            'correlation': portfolio_ret.corr(benchmark_ret),
            'up_capture': self._calculate_capture_ratio(portfolio_ret, benchmark_ret, 'up'),
            'down_capture': self._calculate_capture_ratio(portfolio_ret, benchmark_ret, 'down')
        }
        
        return attribution
    
    def _calculate_capture_ratio(self, portfolio_returns: pd.Series, 
                                benchmark_returns: pd.Series, direction: str) -> float:
        """计算上行/下行捕获比率"""
        if direction == 'up':
            mask = benchmark_returns > 0
        else:
            mask = benchmark_returns < 0
        
        if not mask.any():
            return 0
        
        portfolio_avg = portfolio_returns[mask].mean()
        benchmark_avg = benchmark_returns[mask].mean()
        
        return portfolio_avg / benchmark_avg if benchmark_avg != 0 else 0