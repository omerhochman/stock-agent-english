import numpy as np
from typing import Dict, Any, Optional
from scipy import stats

class PerformanceMetrics:
    """
    投资组合性能指标计算类
    实现完整的投资组合分析指标
    """
    
    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate
        
    def calculate_all_metrics(self, returns: np.ndarray, 
                            risk_free_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        计算所有性能指标
        
        Args:
            returns: 日收益率序列
            risk_free_rate: 无风险利率
            
        Returns:
            Dict: 所有性能指标
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        # 检查输入数据的有效性
        if len(returns) == 0:
            return self._get_zero_metrics()
        
        # 过滤无效值
        valid_returns = returns[np.isfinite(returns)]
        if len(valid_returns) == 0:
            return self._get_zero_metrics()
        
        # 如果所有收益率都是0，返回特殊处理的指标
        if np.all(valid_returns == 0):
            return self._get_zero_return_metrics(len(valid_returns))
            
        daily_rf_rate = risk_free_rate / 252  # 日度无风险利率
        excess_returns = valid_returns - daily_rf_rate
        
        # 基础统计指标
        metrics = {
            # 收益指标
            'total_return': self.total_return(valid_returns),
            'annual_return': self.annual_return(valid_returns),
            'cumulative_return': self.total_return(valid_returns),  # 使用total_return作为cumulative_return的最终值
            
            # 风险指标
            'volatility': self.volatility(valid_returns),
            'downside_volatility': self.downside_volatility(valid_returns),
            'max_drawdown': self.max_drawdown(valid_returns),
            'var_95': self.value_at_risk(valid_returns, 0.95),
            'cvar_95': self.conditional_var(valid_returns, 0.95),
            
            # 风险调整收益指标
            'sharpe_ratio': self.sharpe_ratio(valid_returns, risk_free_rate),
            'sortino_ratio': self.sortino_ratio(valid_returns, risk_free_rate),
            'calmar_ratio': self.calmar_ratio(valid_returns),
            'information_ratio': self.information_ratio(valid_returns, risk_free_rate),
            
            # 交易统计
            'win_rate': self.win_rate(valid_returns),
            'profit_loss_ratio': self.profit_loss_ratio(valid_returns),
            'avg_win': self.average_win(valid_returns),
            'avg_loss': self.average_loss(valid_returns),
            
            # 分布特征
            'skewness': self.skewness(valid_returns),
            'kurtosis': self.kurtosis(valid_returns),
            'tail_ratio': self.tail_ratio(valid_returns),
            
            # 稳定性指标
            'stability': self.stability_ratio(valid_returns),
            'ulcer_index': self.ulcer_index(valid_returns),
            'recovery_factor': self.recovery_factor(valid_returns),
            
            # 市场时机指标
            'up_capture': self.up_capture_ratio(valid_returns),
            'down_capture': self.down_capture_ratio(valid_returns),
            
            # 基本统计
            'total_days': len(valid_returns),
            'positive_days': np.sum(valid_returns > 0),
            'negative_days': np.sum(valid_returns < 0),
            'zero_days': np.sum(valid_returns == 0)
        }
        
        return metrics
    
    def _get_zero_metrics(self) -> Dict[str, Any]:
        """返回空数据时的默认指标"""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'cumulative_return': 0.0,
            'volatility': 0.0,
            'downside_volatility': 0.0,
            'max_drawdown': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'information_ratio': 0.0,
            'win_rate': 0.0,
            'profit_loss_ratio': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'tail_ratio': 0.0,
            'stability': 0.0,
            'ulcer_index': 0.0,
            'recovery_factor': 0.0,
            'up_capture': 0.0,
            'down_capture': 0.0,
            'total_days': 0,
            'positive_days': 0,
            'negative_days': 0,
            'zero_days': 0
        }
    
    def _get_zero_return_metrics(self, days: int) -> Dict[str, Any]:
        """返回收益率全为0时的指标"""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'cumulative_return': 0.0,
            'volatility': 0.0,
            'downside_volatility': 0.0,
            'max_drawdown': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'information_ratio': 0.0,
            'win_rate': 0.0,
            'profit_loss_ratio': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'tail_ratio': 0.0,
            'stability': 0.0,
            'ulcer_index': 0.0,
            'recovery_factor': 0.0,
            'up_capture': 0.0,
            'down_capture': 0.0,
            'total_days': days,
            'positive_days': 0,
            'negative_days': 0,
            'zero_days': days
        }
    
    def total_return(self, returns: np.ndarray) -> float:
        """总收益率"""
        return np.prod(1 + returns) - 1
    
    def annual_return(self, returns: np.ndarray) -> float:
        """年化收益率"""
        total_ret = self.total_return(returns)
        days = len(returns)
        if days == 0:
            return 0
        return (1 + total_ret) ** (252 / days) - 1
    
    def cumulative_return(self, returns: np.ndarray) -> np.ndarray:
        """累计收益率序列"""
        return np.cumprod(1 + returns) - 1
    
    def volatility(self, returns: np.ndarray) -> float:
        """年化波动率"""
        return np.std(returns, ddof=1) * np.sqrt(252)
    
    def downside_volatility(self, returns: np.ndarray, target: float = 0) -> float:
        """下行波动率"""
        downside_returns = returns[returns < target]
        if len(downside_returns) == 0:
            return 0
        return np.std(downside_returns, ddof=1) * np.sqrt(252)
    
    def max_drawdown(self, returns: np.ndarray) -> float:
        """最大回撤"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative / running_max) - 1
        return np.min(drawdown)
    
    def value_at_risk(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """历史VaR"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def conditional_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """条件VaR (Expected Shortfall)"""
        var = self.value_at_risk(returns, confidence)
        return np.mean(returns[returns <= var])
    
    def sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float) -> float:
        """夏普比率"""
        if len(returns) == 0:
            return 0.0
            
        excess_returns = returns - risk_free_rate / 252
        
        # 计算标准差
        std_excess = np.std(excess_returns, ddof=1)
        
        # 检查数值稳定性
        if std_excess == 0 or np.isclose(std_excess, 0, atol=1e-10):
            return 0.0
        
        if not np.isfinite(std_excess):
            return 0.0
        
        # 计算平均超额收益
        mean_excess = np.mean(excess_returns)
        
        if not np.isfinite(mean_excess):
            return 0.0
        
        # 计算夏普比率
        sharpe = mean_excess / std_excess * np.sqrt(252)
        
        # 确保返回有限值
        if not np.isfinite(sharpe):
            return 0.0
            
        return sharpe
    
    def sortino_ratio(self, returns: np.ndarray, risk_free_rate: float, target: float = 0) -> float:
        """索提诺比率"""
        if len(returns) == 0:
            return 0.0
            
        excess_returns = returns - risk_free_rate / 252
        downside_std = self.downside_volatility(returns - risk_free_rate / 252, target) / np.sqrt(252)
        
        # 检查数值稳定性
        if downside_std == 0 or np.isclose(downside_std, 0, atol=1e-10):
            return 0.0
        
        if not np.isfinite(downside_std):
            return 0.0
        
        mean_excess = np.mean(excess_returns)
        
        if not np.isfinite(mean_excess):
            return 0.0
        
        sortino = mean_excess * 252 / (downside_std * np.sqrt(252))
        
        # 确保返回有限值
        if not np.isfinite(sortino):
            return 0.0
            
        return sortino
    
    def calmar_ratio(self, returns: np.ndarray) -> float:
        """卡玛比率"""
        if len(returns) == 0:
            return 0.0
            
        annual_ret = self.annual_return(returns)
        max_dd = abs(self.max_drawdown(returns))
        
        # 检查数值稳定性
        if max_dd == 0 or np.isclose(max_dd, 0, atol=1e-10):
            if annual_ret > 0:
                return float('inf')  # 正收益且无回撤
            else:
                return 0.0
        
        if not np.isfinite(annual_ret) or not np.isfinite(max_dd):
            return 0.0
        
        calmar = annual_ret / max_dd
        
        # 确保返回有限值
        if not np.isfinite(calmar):
            return 0.0
            
        return calmar
    
    def information_ratio(self, returns: np.ndarray, benchmark_rate: float) -> float:
        """信息比率"""
        if len(returns) == 0:
            return 0.0
            
        excess_returns = returns - benchmark_rate / 252
        tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(252)
        
        # 检查数值稳定性
        if tracking_error == 0 or np.isclose(tracking_error, 0, atol=1e-10):
            return 0.0
        
        if not np.isfinite(tracking_error):
            return 0.0
        
        mean_excess = np.mean(excess_returns)
        
        if not np.isfinite(mean_excess):
            return 0.0
        
        info_ratio = mean_excess * 252 / tracking_error
        
        # 确保返回有限值
        if not np.isfinite(info_ratio):
            return 0.0
            
        return info_ratio
    
    def win_rate(self, returns: np.ndarray) -> float:
        """胜率"""
        positive_returns = np.sum(returns > 0)
        total_returns = len(returns[returns != 0])
        return positive_returns / total_returns if total_returns > 0 else 0
    
    def profit_loss_ratio(self, returns: np.ndarray) -> float:
        """盈亏比"""
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0
        
        avg_win = np.mean(wins)
        avg_loss = np.mean(np.abs(losses))
        
        return avg_win / avg_loss if avg_loss != 0 else 0
    
    def average_win(self, returns: np.ndarray) -> float:
        """平均盈利"""
        wins = returns[returns > 0]
        return np.mean(wins) if len(wins) > 0 else 0
    
    def average_loss(self, returns: np.ndarray) -> float:
        """平均亏损"""
        losses = returns[returns < 0]
        return np.mean(losses) if len(losses) > 0 else 0
    
    def skewness(self, returns: np.ndarray) -> float:
        """偏度"""
        return stats.skew(returns)
    
    def kurtosis(self, returns: np.ndarray) -> float:
        """峰度"""
        return stats.kurtosis(returns)
    
    def tail_ratio(self, returns: np.ndarray) -> float:
        """尾部比率"""
        p95 = abs(np.percentile(returns, 95))
        p5 = abs(np.percentile(returns, 5))
        
        # Handle division by zero
        if p5 == 0 or np.isclose(p5, 0, atol=1e-10):
            return np.inf if p95 > 0 else 0
        
        return p95 / p5
    
    def stability_ratio(self, returns: np.ndarray) -> float:
        """稳定性比率"""
        if len(returns) < 2:
            return 0
        cumulative = self.cumulative_return(returns)
        if len(cumulative) == 0:
            return 0
        # 计算累计收益的回归线的R²
        x = np.arange(len(cumulative))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, cumulative)
        return r_value ** 2
    
    def ulcer_index(self, returns: np.ndarray) -> float:
        """溃疡指数"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative / running_max - 1) * 100
        return np.sqrt(np.mean(drawdown ** 2))
    
    def recovery_factor(self, returns: np.ndarray) -> float:
        """恢复因子"""
        total_ret = self.total_return(returns)
        max_dd = abs(self.max_drawdown(returns))
        if max_dd == 0:
            return np.inf if total_ret > 0 else 0
        return total_ret / max_dd
    
    def up_capture_ratio(self, returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None) -> float:
        """上行捕获比率"""
        if benchmark_returns is None:
            # 使用零收益作为基准
            up_periods = returns > 0
            if np.sum(up_periods) == 0:
                return 0
            return np.mean(returns[up_periods]) / 0.001  # 假设基准上涨0.1%
        
        up_periods = benchmark_returns > 0
        if np.sum(up_periods) == 0:
            return 0
        
        portfolio_up = np.mean(returns[up_periods])
        benchmark_up = np.mean(benchmark_returns[up_periods])
        
        return portfolio_up / benchmark_up if benchmark_up != 0 else 0
    
    def down_capture_ratio(self, returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None) -> float:
        """下行捕获比率"""
        if benchmark_returns is None:
            # 使用零收益作为基准
            down_periods = returns < 0
            if np.sum(down_periods) == 0:
                return 0
            return np.mean(returns[down_periods]) / (-0.001)  # 假设基准下跌0.1%
        
        down_periods = benchmark_returns < 0
        if np.sum(down_periods) == 0:
            return 0
        
        portfolio_down = np.mean(returns[down_periods])
        benchmark_down = np.mean(benchmark_returns[down_periods])
        
        return portfolio_down / benchmark_down if benchmark_down != 0 else 0
