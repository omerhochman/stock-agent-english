# 投资组合绩效分析实验
# 此模块实现了投资组合绩效评估的学术方法，包括多种统计显著性测试

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from datetime import datetime, timedelta
import warnings

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import run_hedge_fund
from src.backtester import Backtester
from src.tools.api import get_price_history, prices_to_df
from src.tools.factor_data_api import get_market_returns, get_risk_free_rate
from src.calc.portfolio_optimization import optimize_portfolio
from src.calc.tail_risk_measures import calculate_historical_var, calculate_conditional_var
from src.calc.volatility_models import fit_garch, forecast_garch_volatility

# 忽略警告
warnings.filterwarnings("ignore")


class PortfolioPerformanceTest:
    """投资组合绩效分析测试类"""
    
    def __init__(self, ticker, start_date, end_date, initial_capital=100000):
        """
        初始化投资组合绩效测试
        
        Args:
            ticker (str): 测试股票代码
            start_date (str): 开始日期，格式为 YYYY-MM-DD
            end_date (str): 结束日期，格式为 YYYY-MM-DD
            initial_capital (float): 初始资金
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        # 初始化回测期间的市场数据
        self.market_returns = get_market_returns(
            start_date=start_date, 
            end_date=end_date
        )
        self.risk_free_returns = get_risk_free_rate(
            start_date=start_date, 
            end_date=end_date
        )
        
        # 初始化结果存储
        self.strategy_returns = None
        self.cumulative_returns = None
        self.performance_metrics = {}
        self.hypothesis_tests = {}
        
        # 日志记录设置
        print(f"\n{'='*80}")
        print(f"开始投资组合绩效测试：{ticker} ({start_date} 至 {end_date})")
        print(f"{'='*80}")
    
    def run_backtest(self):
        """运行回测，获取策略收益率数据"""
        print(f"\n{'-'*40}")
        print(f"运行回测...")
        print(f"{'-'*40}")
        
        # 创建回测器实例
        backtester = Backtester(
            agent=run_hedge_fund,
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            num_of_news=5
        )
        
        # 运行回测
        backtester.run_backtest()
        
        # 获取结果
        self.strategy_returns = backtester.returns
        self.backtester = backtester
        self.performance_metrics = backtester.metrics
        
        # 计算累积收益率
        self.cumulative_returns = (1 + self.strategy_returns).cumprod() - 1
        
        # 打印基本回测结果
        print(f"\n基本回测结果:")
        print(f"总收益率: {self.performance_metrics.get('总收益率', 0):.2%}")
        print(f"年化收益率: {self.performance_metrics.get('年化收益率', 0):.2%}")
        print(f"最大回撤: {self.performance_metrics.get('最大回撤', 0):.2%}")
        print(f"夏普比率: {self.performance_metrics.get('夏普比率', 0):.4f}")
        
        return backtester.performance_df
    
    def calculate_academic_metrics(self):
        """计算学术绩效指标"""
        print(f"\n{'-'*40}")
        print(f"计算学术绩效指标...")
        print(f"{'-'*40}")
        
        if self.strategy_returns is None:
            self.run_backtest()
        
        # 对齐数据 - 策略收益率、市场收益率和无风险利率
        merged_returns = pd.concat([
            self.strategy_returns, 
            self.market_returns, 
            self.risk_free_returns
        ], axis=1).dropna()
        
        # 重命名列
        merged_returns.columns = ['strategy_returns', 'market_returns', 'risk_free_rate']
        
        # 计算超额收益
        merged_returns['excess_strategy'] = merged_returns['strategy_returns'] - merged_returns['risk_free_rate']
        merged_returns['excess_market'] = merged_returns['market_returns'] - merged_returns['risk_free_rate']
        
        # 1. CAPM模型回归分析
        X = sm.add_constant(merged_returns['excess_market'])
        capm_model = sm.OLS(merged_returns['excess_strategy'], X).fit()
        
        # 提取参数
        alpha = capm_model.params[0]
        beta = capm_model.params[1]
        alpha_pvalue = capm_model.pvalues[0]
        beta_pvalue = capm_model.pvalues[1]
        r_squared = capm_model.rsquared
        
        # 2. 计算信息比率
        active_returns = merged_returns['strategy_returns'] - merged_returns['market_returns']
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252)
        
        # 3. 计算特雷诺比率
        treynor_ratio = merged_returns['excess_strategy'].mean() / beta * 252 if beta > 0 else np.nan
        
        # 4. 计算下行风险指标
        downside_returns = merged_returns['strategy_returns'][merged_returns['strategy_returns'] < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = merged_returns['excess_strategy'].mean() * 252 / downside_deviation if downside_deviation > 0 else np.nan
        
        # 5. 计算最大回撤
        cum_returns = (1 + merged_returns['strategy_returns']).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max - 1)
        max_drawdown = drawdown.min()
        
        # 6. 计算Omega比率
        threshold = merged_returns['risk_free_rate'].mean()  # 以无风险利率为阈值
        omega_ratio = self._calculate_omega_ratio(merged_returns['strategy_returns'], threshold)
        
        # 7. 计算Calmar比率
        calmar_ratio = merged_returns['strategy_returns'].mean() * 252 / abs(max_drawdown) if max_drawdown != 0 else np.nan
        
        # 8. 计算捕获比率
        up_market = merged_returns['market_returns'][merged_returns['market_returns'] > 0]
        down_market = merged_returns['market_returns'][merged_returns['market_returns'] < 0]
        
        up_capture = merged_returns['strategy_returns'][up_market.index].mean() / up_market.mean() if len(up_market) > 0 else np.nan
        down_capture = merged_returns['strategy_returns'][down_market.index].mean() / down_market.mean() if len(down_market) > 0 else np.nan
        
        # 9. 使用GARCH模型分析波动率聚类特性
        garch_results = {}
        try:
            returns_array = merged_returns['strategy_returns'].values
            if len(returns_array) >= 100:  # 确保有足够的数据
                params, log_likelihood = fit_garch(returns_array)
                garch_results = {
                    'omega': float(params['omega']),
                    'alpha': float(params['alpha']),
                    'beta': float(params['beta']),
                    'persistence': float(params['persistence']),
                    'log_likelihood': float(log_likelihood)
                }
        except Exception as e:
            print(f"GARCH模型拟合失败: {e}")
        
        # 10. 计算尾部风险指标
        var_95 = calculate_historical_var(merged_returns['strategy_returns'], confidence_level=0.95)
        cvar_95 = calculate_conditional_var(merged_returns['strategy_returns'], confidence_level=0.95)
        
        # 11. 计算交易指标
        turnover_rate = getattr(self.backtester, 'turnover_rate', np.nan)  # 换手率
        winning_rate = self.performance_metrics.get('胜率', np.nan)  # 胜率
        profit_loss_ratio = self.performance_metrics.get('盈亏比', np.nan)  # 盈亏比
        
        # 存储结果
        metrics = {
            'alpha': alpha,
            'beta': beta,
            'alpha_pvalue': alpha_pvalue,
            'beta_pvalue': beta_pvalue,
            'r_squared': r_squared,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'omega_ratio': omega_ratio,
            'calmar_ratio': calmar_ratio,
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'turnover_rate': turnover_rate,
            'winning_rate': winning_rate,
            'profit_loss_ratio': profit_loss_ratio,
        }
        
        if garch_results:
            metrics['garch_persistence'] = garch_results['persistence']
        
        self.academic_metrics = metrics
        
        # 打印主要学术指标
        self._print_academic_metrics()
        
        return metrics
    
    def _print_academic_metrics(self):
        """打印主要学术指标"""
        metrics = self.academic_metrics
        
        print("\n学术绩效指标:")
        print(f"CAPM Alpha: {metrics['alpha']*252:.4%} (p值: {metrics['alpha_pvalue']:.4f})")
        alpha_significant = "是" if metrics['alpha_pvalue'] < 0.05 else "否"
        print(f"Alpha显著性: {alpha_significant} (5%水平)")
        
        print(f"Beta: {metrics['beta']:.4f} (p值: {metrics['beta_pvalue']:.4f})")
        print(f"R平方: {metrics['r_squared']:.4f}")
        
        print(f"信息比率: {metrics['information_ratio']:.4f}")
        print(f"特雷诺比率: {metrics['treynor_ratio']:.4f}")
        print(f"索提诺比率: {metrics['sortino_ratio']:.4f}")
        print(f"欧米茄比率: {metrics['omega_ratio']:.4f}")
        print(f"卡尔玛比率: {metrics['calmar_ratio']:.4f}")
        
        print(f"上行捕获率: {metrics['up_capture_ratio']:.4f}")
        print(f"下行捕获率: {metrics['down_capture_ratio']:.4f}")
        
        print(f"95% VaR: {metrics['var_95']:.4%}")
        print(f"95% CVaR: {metrics['cvar_95']:.4%}")
        
        if 'garch_persistence' in metrics:
            print(f"GARCH持续性: {metrics['garch_persistence']:.4f}")
        
        print(f"胜率: {metrics['winning_rate']:.4f}")
        print(f"盈亏比: {metrics['profit_loss_ratio']:.4f}")
    
    def run_hypothesis_tests(self):
        """执行假设检验"""
        print(f"\n{'-'*40}")
        print(f"执行假设检验...")
        print(f"{'-'*40}")
        
        if self.strategy_returns is None:
            self.run_backtest()
        
        if not hasattr(self, 'academic_metrics'):
            self.calculate_academic_metrics()
        
        # 准备数据
        returns = self.strategy_returns
        
        # 1. 正态性检验 - 使用Jarque-Bera检验
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        jb_result = {
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'reject_null': jb_pvalue < 0.05,
            'interpretation': "拒绝收益率服从正态分布假设" if jb_pvalue < 0.05 else "不能拒绝收益率服从正态分布假设"
        }
        
        # 2. 自相关检验 - 使用Ljung-Box检验
        lb_stat, lb_pvalue = acorr_ljungbox(returns, lags=[10], return_df=False)
        lb_result = {
            'statistic': lb_stat[0],
            'p_value': lb_pvalue[0],
            'reject_null': lb_pvalue[0] < 0.05,
            'interpretation': "拒绝收益率序列独立性假设，存在自相关" if lb_pvalue[0] < 0.05 else "不能拒绝收益率序列独立性假设"
        }
        
        # 3. 平稳性检验 - 使用增广Dickey-Fuller检验
        adf_result = sm.tsa.stattools.adfuller(returns)
        adf = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'reject_null': adf_result[1] < 0.05,
            'interpretation': "拒绝单位根假设，序列平稳" if adf_result[1] < 0.05 else "不能拒绝单位根假设，序列非平稳"
        }
        
        # 4. ARCH效应检验 - 使用Engle的ARCH检验
        arch_result = sm.stats.diagnostic.het_arch(returns, nlags=5)
        arch = {
            'statistic': arch_result[0],
            'p_value': arch_result[1],
            'reject_null': arch_result[1] < 0.05,
            'interpretation': "拒绝同方差假设，存在ARCH效应" if arch_result[1] < 0.05 else "不能拒绝同方差假设，不存在ARCH效应"
        }
        
        # 5. 超额收益显著性检验
        market_returns = self.market_returns
        risk_free_returns = self.risk_free_returns
        
        # 对齐数据
        aligned_data = pd.concat([returns, market_returns, risk_free_returns], axis=1).dropna()
        aligned_data.columns = ['strategy', 'market', 'risk_free']
        
        # 计算超额收益
        excess_returns = aligned_data['strategy'] - aligned_data['risk_free']
        
        # 单样本t检验 - 超额收益显著大于0?
        t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
        t_test_result = {
            'statistic': t_stat,
            'p_value': p_value,
            'reject_null': p_value < 0.05 and t_stat > 0,
            'interpretation': "超额收益显著大于零" if p_value < 0.05 and t_stat > 0 else "超额收益不显著不同于零"
        }
        
        # 6. 策略收益率与市场收益率对比检验
        # 计算策略超额收益与市场超额收益
        strategy_excess = aligned_data['strategy'] - aligned_data['risk_free']
        market_excess = aligned_data['market'] - aligned_data['risk_free']
        
        # 配对样本t检验 - 策略超额收益显著优于市场超额收益?
        diff = strategy_excess - market_excess
        paired_t_stat, paired_p_value = stats.ttest_1samp(diff, 0)
        paired_t_test_result = {
            'statistic': paired_t_stat,
            'p_value': paired_p_value,
            'reject_null': paired_p_value < 0.05 and paired_t_stat > 0,
            'interpretation': "策略超额收益显著优于市场超额收益" if paired_p_value < 0.05 and paired_t_stat > 0 else "策略超额收益不显著优于市场超额收益"
        }
        
        # 7. 计算交易胜率的置信区间
        total_trades = self.performance_metrics.get('交易次数', 0)
        winning_trades = int(total_trades * self.performance_metrics.get('胜率', 0))
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            # 使用Wilson评分区间计算胜率的95%置信区间
            from statsmodels.stats.proportion import proportion_confint
            lower, upper = proportion_confint(winning_trades, total_trades, alpha=0.05, method='wilson')
            
            win_rate_ci = {
                'win_rate': win_rate,
                'lower_bound': lower,
                'upper_bound': upper,
                'n_trades': total_trades,
                'interpretation': f"交易胜率为{win_rate:.2%}，95%置信区间为[{lower:.2%}, {upper:.2%}]"
            }
        else:
            win_rate_ci = {
                'win_rate': np.nan,
                'lower_bound': np.nan,
                'upper_bound': np.nan,
                'n_trades': 0,
                'interpretation': "没有足够的交易数据来计算胜率置信区间"
            }
        
        # 存储所有检验结果
        self.hypothesis_tests = {
            'jarque_bera_test': jb_result,
            'ljung_box_test': lb_result,
            'adf_test': adf,
            'arch_test': arch,
            't_test_excess_returns': t_test_result,
            'paired_t_test_vs_market': paired_t_test_result,
            'win_rate_confidence_interval': win_rate_ci
        }
        
        # 打印假设检验结果
        self._print_hypothesis_tests()
        
        return self.hypothesis_tests
    
    def _print_hypothesis_tests(self):
        """打印假设检验结果"""
        tests = self.hypothesis_tests
        
        print("\n假设检验结果:")
        
        print("\n1. 收益率分布检验 (Jarque-Bera正态性检验):")
        print(f"  检验统计量: {tests['jarque_bera_test']['statistic']:.4f}")
        print(f"  p值: {tests['jarque_bera_test']['p_value']:.4f}")
        print(f"  结论: {tests['jarque_bera_test']['interpretation']}")
        
        print("\n2. 收益率自相关检验 (Ljung-Box检验):")
        print(f"  检验统计量: {tests['ljung_box_test']['statistic']:.4f}")
        print(f"  p值: {tests['ljung_box_test']['p_value']:.4f}")
        print(f"  结论: {tests['ljung_box_test']['interpretation']}")
        
        print("\n3. 收益率平稳性检验 (ADF检验):")
        print(f"  检验统计量: {tests['adf_test']['statistic']:.4f}")
        print(f"  p值: {tests['adf_test']['p_value']:.4f}")
        print(f"  结论: {tests['adf_test']['interpretation']}")
        
        print("\n4. 波动率聚类检验 (ARCH效应检验):")
        print(f"  检验统计量: {tests['arch_test']['statistic']:.4f}")
        print(f"  p值: {tests['arch_test']['p_value']:.4f}")
        print(f"  结论: {tests['arch_test']['interpretation']}")
        
        print("\n5. 超额收益显著性检验 (单样本t检验):")
        print(f"  检验统计量: {tests['t_test_excess_returns']['statistic']:.4f}")
        print(f"  p值: {tests['t_test_excess_returns']['p_value']:.4f}")
        print(f"  结论: {tests['t_test_excess_returns']['interpretation']}")
        
        print("\n6. 策略与市场收益对比检验 (配对样本t检验):")
        print(f"  检验统计量: {tests['paired_t_test_vs_market']['statistic']:.4f}")
        print(f"  p值: {tests['paired_t_test_vs_market']['p_value']:.4f}")
        print(f"  结论: {tests['paired_t_test_vs_market']['interpretation']}")
        
        print("\n7. 交易胜率置信区间:")
        print(f"  {tests['win_rate_confidence_interval']['interpretation']}")
    
    def plot_performance_charts(self, save_path=None):
        """绘制绩效分析图表"""
        print(f"\n{'-'*40}")
        print(f"绘制绩效分析图表...")
        print(f"{'-'*40}")
        
        if self.strategy_returns is None:
            self.run_backtest()
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("pastel")
        
        # 创建一个2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.ticker} 投资组合绩效分析', fontsize=16)
        
        # 1. 累积收益率对比图 (左上)
        ax1 = axes[0, 0]
        
        # 对齐数据
        market_returns = self.market_returns
        risk_free_returns = self.risk_free_returns
        aligned_data = pd.concat([
            self.strategy_returns, 
            market_returns, 
            risk_free_returns
        ], axis=1).dropna()
        aligned_data.columns = ['strategy', 'market', 'risk_free']
        
        # 计算累积收益率
        strategy_cum_ret = (1 + aligned_data['strategy']).cumprod() - 1
        market_cum_ret = (1 + aligned_data['market']).cumprod() - 1
        risk_free_cum_ret = (1 + aligned_data['risk_free']).cumprod() - 1
        
        # 绘制累积收益率曲线
        ax1.plot(strategy_cum_ret.index, strategy_cum_ret * 100, label='策略收益', linewidth=2)
        ax1.plot(market_cum_ret.index, market_cum_ret * 100, label='市场收益', linewidth=2, alpha=0.7)
        ax1.plot(risk_free_cum_ret.index, risk_free_cum_ret * 100, label='无风险收益', linewidth=1, linestyle='--', alpha=0.5)
        
        ax1.set_title('累积收益率对比')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('累积收益率 (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 回撤分析图 (右上)
        ax2 = axes[0, 1]
        
        # 计算回撤
        strategy_cum_ret = (1 + self.strategy_returns).cumprod()
        strategy_drawdown = (strategy_cum_ret / strategy_cum_ret.cummax() - 1) * 100
        
        market_cum_ret = (1 + market_returns).cumprod()
        market_drawdown = (market_cum_ret / market_cum_ret.cummax() - 1) * 100
        
        # 对齐数据
        drawdown_data = pd.concat([strategy_drawdown, market_drawdown], axis=1).dropna()
        drawdown_data.columns = ['strategy', 'market']
        
        # 绘制回撤曲线
        ax2.fill_between(drawdown_data.index, 0, drawdown_data['strategy'], color='red', alpha=0.3, label='策略回撤')
        ax2.plot(drawdown_data.index, drawdown_data['market'], color='blue', linestyle='--', alpha=0.7, label='市场回撤')
        
        ax2.set_title('回撤分析')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('回撤 (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 收益率分布图 (左下)
        ax3 = axes[1, 0]
        
        # 计算收益率的均值和标准差
        strategy_mean = self.strategy_returns.mean()
        strategy_std = self.strategy_returns.std()
        market_mean = market_returns.mean()
        market_std = market_returns.std()
        
        # 绘制收益率分布直方图
        sns.histplot(self.strategy_returns, kde=True, ax=ax3, color='blue', alpha=0.6, label='策略收益率')
        sns.histplot(market_returns, kde=True, ax=ax3, color='green', alpha=0.4, label='市场收益率')
        
        # 添加标记线表示均值
        ax3.axvline(strategy_mean, color='blue', linestyle='--', alpha=0.8, label=f'策略均值: {strategy_mean:.4%}')
        ax3.axvline(market_mean, color='green', linestyle='--', alpha=0.8, label=f'市场均值: {market_mean:.4%}')
        
        ax3.set_title('收益率分布对比')
        ax3.set_xlabel('日收益率')
        ax3.set_ylabel('频率')
        ax3.legend()
        
        # 4. 风险-收益散点图 (右下)
        ax4 = axes[1, 1]
        
        # 数据点
        points = [
            {'name': '策略', 'return': self.strategy_returns.mean() * 252, 'risk': self.strategy_returns.std() * np.sqrt(252)},
            {'name': '市场', 'return': market_returns.mean() * 252, 'risk': market_returns.std() * np.sqrt(252)},
            {'name': '无风险', 'return': risk_free_returns.mean() * 252, 'risk': 0}
        ]
        
        # 转换为DataFrame
        points_df = pd.DataFrame(points)
        
        # 绘制散点图
        ax4.scatter(points_df['risk'] * 100, points_df['return'] * 100, s=100)
        
        # 添加标签
        for i, row in points_df.iterrows():
            ax4.annotate(
                row['name'], 
                xy=(row['risk'] * 100, row['return'] * 100),
                xytext=(7, 7),
                textcoords='offset points'
            )
        
        # 绘制资本市场线
        risk_range = np.linspace(0, max(points_df['risk']) * 1.5, 100)
        cml = risk_free_returns.mean() * 252 + (points_df.loc[0, 'return'] - risk_free_returns.mean() * 252) / points_df.loc[0, 'risk'] * risk_range
        ax4.plot(risk_range * 100, cml * 100, 'r--', alpha=0.6, label='资本市场线')
        
        ax4.set_title('风险-收益分析')
        ax4.set_xlabel('风险 (年化波动率 %)')
        ax4.set_ylabel('收益 (年化收益率 %)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存或显示图表
        if save_path:
            plt.savefig(save_path)
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def _calculate_omega_ratio(self, returns, threshold=0):
        """计算欧米茄比率
        
        Args:
            returns: 收益率序列
            threshold: 阈值收益率，默认为0
            
        Returns:
            float: 欧米茄比率
        """
        # 区分收益率大于阈值和小于阈值的部分
        returns_above = returns[returns > threshold]
        returns_below = returns[returns <= threshold]
        
        # 如果没有低于阈值的收益率，返回无穷大
        if len(returns_below) == 0:
            return float('inf')
        
        # 计算阈值以上收益的总和与阈值以下损失的总和之比
        positive_sum = (returns_above - threshold).sum()
        negative_sum = abs((returns_below - threshold).sum())
        
        return positive_sum / negative_sum if negative_sum != 0 else float('inf')
    
    def compare_with_baseline(self, baseline_ticker="000300"):
        """与基准策略比较
        
        Args:
            baseline_ticker: 基准股票代码，默认为沪深300
            
        Returns:
            dict: 比较结果
        """
        print(f"\n{'-'*40}")
        print(f"与基准策略对比...")
        print(f"{'-'*40}")
        
        # 获取基准收益率数据
        baseline_returns = self.market_returns
        
        # 确保有策略收益率数据
        if self.strategy_returns is None:
            self.run_backtest()
        
        # 对齐数据
        aligned_data = pd.concat([self.strategy_returns, baseline_returns], axis=1).dropna()
        aligned_data.columns = ['strategy', 'baseline']
        
        # 计算累积收益率
        strategy_cum_ret = (1 + aligned_data['strategy']).cumprod() - 1
        baseline_cum_ret = (1 + aligned_data['baseline']).cumprod() - 1
        
        # 计算年化收益率
        days = len(aligned_data)
        strategy_annual_return = (1 + strategy_cum_ret.iloc[-1]) ** (252 / days) - 1
        baseline_annual_return = (1 + baseline_cum_ret.iloc[-1]) ** (252 / days) - 1
        
        # 计算年化波动率
        strategy_annual_vol = aligned_data['strategy'].std() * np.sqrt(252)
        baseline_annual_vol = aligned_data['baseline'].std() * np.sqrt(252)
        
        # 计算夏普比率
        risk_free_rate = self.risk_free_returns.mean() * 252
        strategy_sharpe = (strategy_annual_return - risk_free_rate) / strategy_annual_vol
        baseline_sharpe = (baseline_annual_return - risk_free_rate) / baseline_annual_vol
        
        # 计算最大回撤
        strategy_dd = self._calculate_max_drawdown(aligned_data['strategy'])
        baseline_dd = self._calculate_max_drawdown(aligned_data['baseline'])
        
        # 计算相关系数
        correlation = aligned_data['strategy'].corr(aligned_data['baseline'])
        
        # 存储比较结果
        comparison = {
            "总收益率": {
                "策略": strategy_cum_ret.iloc[-1],
                "基准": baseline_cum_ret.iloc[-1],
                "差值": strategy_cum_ret.iloc[-1] - baseline_cum_ret.iloc[-1],
                "相对表现": (strategy_cum_ret.iloc[-1] / baseline_cum_ret.iloc[-1] - 1) if baseline_cum_ret.iloc[-1] > 0 else float('inf')
            },
            "年化收益率": {
                "策略": strategy_annual_return,
                "基准": baseline_annual_return,
                "差值": strategy_annual_return - baseline_annual_return,
                "相对表现": (strategy_annual_return / baseline_annual_return - 1) if baseline_annual_return > 0 else float('inf')
            },
            "年化波动率": {
                "策略": strategy_annual_vol,
                "基准": baseline_annual_vol,
                "差值": strategy_annual_vol - baseline_annual_vol,
                "相对表现": strategy_annual_vol / baseline_annual_vol - 1
            },
            "夏普比率": {
                "策略": strategy_sharpe,
                "基准": baseline_sharpe,
                "差值": strategy_sharpe - baseline_sharpe,
                "相对表现": (strategy_sharpe / baseline_sharpe - 1) if baseline_sharpe > 0 else float('inf')
            },
            "最大回撤": {
                "策略": strategy_dd,
                "基准": baseline_dd,
                "差值": strategy_dd - baseline_dd,
                "相对表现": strategy_dd / baseline_dd - 1
            },
            "相关系数": correlation
        }
        
        # 打印比较结果
        self._print_comparison(comparison)
        
        # 绘制累积收益率对比图
        plt.figure(figsize=(12, 6))
        plt.plot(strategy_cum_ret.index, strategy_cum_ret * 100, label='策略收益', linewidth=2)
        plt.plot(baseline_cum_ret.index, baseline_cum_ret * 100, label='基准收益', linewidth=2, alpha=0.7)
        plt.title('策略与基准累积收益率对比')
        plt.xlabel('日期')
        plt.ylabel('累积收益率 (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return comparison
    
    def _print_comparison(self, comparison):
        """打印比较结果"""
        print("\n策略与基准比较结果:")
        print(f"相关系数: {comparison['相关系数']:.4f}")
        
        print("\n项目\t\t策略\t\t基准\t\t差值\t\t相对表现")
        print("-" * 80)
        
        for metric in ["总收益率", "年化收益率", "年化波动率", "夏普比率", "最大回撤"]:
            print(f"{metric}\t\t{comparison[metric]['策略']:.4f}\t\t{comparison[metric]['基准']:.4f}\t\t{comparison[metric]['差值']:.4f}\t\t{comparison[metric]['相对表现']:.4f}")
    
    def _calculate_max_drawdown(self, returns):
        """计算最大回撤
        
        Args:
            returns: 收益率序列
            
        Returns:
            float: 最大回撤（正值表示）
        """
        # 计算累积收益
        cum_returns = (1 + returns).cumprod()
        # 计算回撤
        max_returns = cum_returns.cummax()
        drawdowns = (cum_returns / max_returns - 1)
        # 返回最大回撤的绝对值
        return abs(drawdowns.min())
    
    def run_ablation_study(self):
        """进行消融实验，分析各个代理的贡献"""
        print(f"\n{'-'*40}")
        print(f"进行消融实验...")
        print(f"{'-'*40}")
        
        print("这部分需要集成到回测框架中，当前只显示所有代理共同作用的结果")
        print("完整实验需要逐个禁用不同代理，分析其对整体性能的影响")
        
        # 打印所有代理共同作用的结果
        if not hasattr(self, 'academic_metrics'):
            self.calculate_academic_metrics()
        
        print("\n所有代理共同作用时的主要指标:")
        print(f"年化收益率: {self.performance_metrics.get('年化收益率', 0):.4f}")
        print(f"夏普比率: {self.academic_metrics.get('information_ratio', 0):.4f}")
        print(f"最大回撤: {self.academic_metrics.get('max_drawdown', 0):.4f}")
        
        return {
            "message": "消融实验需要在回测框架中实现代理禁用功能"
        }


# 运行示例
if __name__ == '__main__':
    # 测试参数
    ticker = "600519"  # 贵州茅台
    start_date = "2023-01-01"
    end_date = "2023-02-28"
    initial_capital = 100000
    
    # 创建测试实例
    test = PortfolioPerformanceTest(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    
    # 运行测试流程
    test.run_backtest()
    test.calculate_academic_metrics()
    test.run_hypothesis_tests()
    test.plot_performance_charts()
    test.compare_with_baseline()
    test.run_ablation_study()
    
    print("\n测试完成！")