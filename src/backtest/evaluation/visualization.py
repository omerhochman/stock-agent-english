import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BacktestVisualizer:
    """
    回测可视化类
    生成专业的回测分析图表
    """
    
    def __init__(self):
        self.style_config = {
            'figure_size': (12, 8),
            'dpi': 300,
            'style': 'seaborn-v0_8',
            'color_palette': 'Set2'
        }
        self._setup_style()
    
    def _setup_style(self):
        """设置图表样式"""
        try:
            plt.style.use(self.style_config['style'])
        except:
            plt.style.use('default')
        
        sns.set_palette(self.style_config['color_palette'])
    
    def create_comparison_charts(self, results: Dict[str, Any], 
                               comparison_results: Dict[str, Any],
                               save_dir: str) -> Dict[str, str]:
        """
        创建完整的比较图表集
        
        Args:
            results: 策略结果
            comparison_results: 比较分析结果
            save_dir: 保存目录
            
        Returns:
            Dict: 生成的图表文件路径
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        chart_paths = {}
        
        # 1. 综合性能对比图
        chart_paths['performance_comparison'] = self._create_performance_comparison(
            results, os.path.join(save_dir, 'performance_comparison.png')
        )
        
        # 2. 累计收益对比图
        chart_paths['cumulative_returns'] = self._create_cumulative_returns_chart(
            results, os.path.join(save_dir, 'cumulative_returns.png')
        )
        
        # 3. 风险收益散点图
        chart_paths['risk_return_scatter'] = self._create_risk_return_scatter(
            results, os.path.join(save_dir, 'risk_return_scatter.png')
        )
        
        # 4. 回撤分析图
        chart_paths['drawdown_analysis'] = self._create_drawdown_analysis(
            results, os.path.join(save_dir, 'drawdown_analysis.png')
        )
        
        # 5. 策略雷达图
        if 'strategy_ranking' in comparison_results:
            chart_paths['radar_chart'] = self._create_strategy_radar_chart(
                comparison_results['strategy_ranking'], 
                os.path.join(save_dir, 'strategy_radar.png')
            )
        
        # 6. 滚动指标图
        chart_paths['rolling_metrics'] = self._create_rolling_metrics_chart(
            results, os.path.join(save_dir, 'rolling_metrics.png')
        )
        
        # 7. 月度收益热图
        chart_paths['monthly_returns'] = self._create_monthly_returns_heatmap(
            results, os.path.join(save_dir, 'monthly_returns.png')
        )
        
        return chart_paths
    
    def _create_performance_comparison(self, results: Dict[str, Any], filepath: str) -> str:
        """创建综合性能对比图"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 提取数据
            strategy_names = list(results.keys())
            total_returns = [results[name].performance_metrics.get('total_return', 0) * 100 
                            for name in strategy_names]
            sharpe_ratios = [results[name].performance_metrics.get('sharpe_ratio', 0) 
                            for name in strategy_names]
            max_drawdowns = [abs(results[name].performance_metrics.get('max_drawdown', 0)) * 100 
                            for name in strategy_names]
            volatilities = [results[name].performance_metrics.get('volatility', 0) * 100 
                        for name in strategy_names]
            
            # 1. 总收益率对比
            colors = sns.color_palette("Set2", len(strategy_names))
            bars1 = ax1.bar(strategy_names, total_returns, color=colors)
            ax1.set_title('总收益率对比', fontsize=14, fontweight='bold')
            ax1.set_ylabel('收益率 (%)', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars1, total_returns):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}%', ha='center', va='bottom')
            
            # 2. 夏普比率对比
            bars2 = ax2.bar(strategy_names, sharpe_ratios, color=colors)
            ax2.set_title('夏普比率对比', fontsize=14, fontweight='bold')
            ax2.set_ylabel('夏普比率', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='优秀水平')
            ax2.legend()
            
            for bar, value in zip(bars2, sharpe_ratios):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # 3. 最大回撤对比
            bars3 = ax3.bar(strategy_names, max_drawdowns, color=colors)
            ax3.set_title('最大回撤对比', fontsize=14, fontweight='bold')
            ax3.set_ylabel('最大回撤 (%)', fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            for bar, value in zip(bars3, max_drawdowns):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}%', ha='center', va='bottom')
            
            # 4. 波动率对比
            bars4 = ax4.bar(strategy_names, volatilities, color=colors)
            ax4.set_title('年化波动率对比', fontsize=14, fontweight='bold')
            ax4.set_ylabel('波动率 (%)', fontsize=12)
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            for bar, value in zip(bars4, volatilities):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            return filepath
            
        except Exception as e:
            print(f"创建性能对比图失败: {e}")
            return ""

    def _create_cumulative_returns_chart(self, results: Dict[str, Any], filepath: str) -> str:
        """创建累计收益对比图"""
        try:
            fig, ax = plt.subplots(figsize=self.style_config['figure_size'])
            
            colors = sns.color_palette("Set2", len(results))
            
            for i, (name, result) in enumerate(results.items()):
                if result.daily_returns is not None and len(result.daily_returns) > 0:
                    # 计算累计收益
                    cumulative_returns = np.cumprod(1 + result.daily_returns) - 1
                    
                    # 创建日期索引
                    dates = pd.date_range(
                        start=pd.Timestamp.now() - pd.Timedelta(days=len(cumulative_returns)),
                        periods=len(cumulative_returns),
                        freq='D'
                    )
                    
                    ax.plot(dates, cumulative_returns * 100, 
                        label=name, color=colors[i], linewidth=2)
            
            ax.set_title('累计收益率对比', fontsize=16, fontweight='bold')
            ax.set_xlabel('日期', fontsize=12)
            ax.set_ylabel('累计收益率 (%)', fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # 设置日期格式
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            return filepath
            
        except Exception as e:
            print(f"创建累计收益图失败: {e}")
            return ""

    def _create_risk_return_scatter(self, results: Dict[str, Any], filepath: str) -> str:
        """创建风险收益散点图"""
        try:
            fig, ax = plt.subplots(figsize=self.style_config['figure_size'])
            
            strategy_names = []
            annual_returns = []
            volatilities = []
            sharpe_ratios = []
            
            for name, result in results.items():
                metrics = result.performance_metrics
                strategy_names.append(name)
                annual_returns.append(metrics.get('annual_return', 0) * 100)
                volatilities.append(metrics.get('volatility', 0) * 100)
                sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
            
            # 创建散点图，点的大小表示夏普比率
            sizes = [max(100, abs(sr) * 200) for sr in sharpe_ratios]
            colors = sns.color_palette("viridis", len(strategy_names))
            
            scatter = ax.scatter(volatilities, annual_returns, 
                            s=sizes, c=colors, alpha=0.7, edgecolors='black')
            
            # 添加策略名称标签
            for i, name in enumerate(strategy_names):
                ax.annotate(name, (volatilities[i], annual_returns[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, ha='left')
            
            # 添加有效前沿参考线
            if len(volatilities) > 1:
                # 绘制风险等级线
                max_vol = max(volatilities)
                vol_range = np.linspace(0, max_vol * 1.1, 100)
                
                # 假设无风险利率3%
                risk_free_rate = 3.0
                
                # 绘制不同夏普比率的等值线
                for sr in [0.5, 1.0, 1.5, 2.0]:
                    expected_returns = risk_free_rate + sr * vol_range
                    ax.plot(vol_range, expected_returns, '--', alpha=0.5, 
                        label=f'夏普比率={sr}')
            
            ax.set_title('风险收益散点图', fontsize=16, fontweight='bold')
            ax.set_xlabel('年化波动率 (%)', fontsize=12)
            ax.set_ylabel('年化收益率 (%)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 添加说明文字
            ax.text(0.02, 0.98, '点的大小表示夏普比率', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            return filepath
            
        except Exception as e:
            print(f"创建风险收益散点图失败: {e}")
            return ""

    def _create_drawdown_analysis(self, results: Dict[str, Any], filepath: str) -> str:
        """创建回撤分析图"""
        try:
            fig, axes = plt.subplots(len(results), 1, 
                                    figsize=(self.style_config['figure_size'][0], 
                                            len(results) * 4))
            
            if len(results) == 1:
                axes = [axes]
            
            colors = sns.color_palette("Set2", len(results))
            
            for i, (name, result) in enumerate(results.items()):
                if result.daily_returns is not None and len(result.daily_returns) > 0:
                    # 计算累计收益和回撤
                    cumulative = np.cumprod(1 + result.daily_returns)
                    running_max = np.maximum.accumulate(cumulative)
                    drawdown = (cumulative / running_max - 1) * 100
                    
                    # 创建日期索引
                    dates = pd.date_range(
                        start=pd.Timestamp.now() - pd.Timedelta(days=len(drawdown)),
                        periods=len(drawdown),
                        freq='D'
                    )
                    
                    # 绘制回撤
                    axes[i].fill_between(dates, 0, drawdown, 
                                    color=colors[i], alpha=0.3, label='回撤')
                    axes[i].plot(dates, drawdown, color=colors[i], linewidth=1)
                    
                    # 标记最大回撤点
                    max_dd_idx = np.argmin(drawdown)
                    max_dd_value = drawdown[max_dd_idx]
                    axes[i].scatter(dates[max_dd_idx], max_dd_value, 
                                color='red', s=100, zorder=5)
                    axes[i].annotate(f'最大回撤: {max_dd_value:.2f}%',
                                xy=(dates[max_dd_idx], max_dd_value),
                                xytext=(10, -10), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='yellow', alpha=0.7))
                    
                    axes[i].set_title(f'{name} - 回撤分析', fontsize=14, fontweight='bold')
                    axes[i].set_ylabel('回撤 (%)', fontsize=12)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].axhline(y=-5, color='orange', linestyle='--', 
                                alpha=0.7, label='5%回撤线')
                    axes[i].axhline(y=-10, color='red', linestyle='--', 
                                alpha=0.7, label='10%回撤线')
                    axes[i].legend()
                    
                    # 设置日期格式
                    import matplotlib.dates as mdates
                    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            return filepath
            
        except Exception as e:
            print(f"创建回撤分析图失败: {e}")
            return ""

    def _create_strategy_radar_chart(self, ranking_data: Dict[str, Any], filepath: str) -> str:
        """创建策略雷达图"""
        try:
            rankings = ranking_data.get('rankings', [])
            if not rankings:
                return ""
            
            # 取前5个策略
            top_strategies = rankings[:5]
            
            # 设置雷达图
            dimensions = ['收益', '风险', '稳定性', '效率', '稳健性']
            angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            colors = sns.color_palette("Set2", len(top_strategies))
            
            for i, strategy in enumerate(top_strategies):
                scores = strategy.scores
                values = [
                    scores.get('return', 0),
                    scores.get('risk', 0),
                    scores.get('stability', 0),
                    scores.get('efficiency', 0),
                    scores.get('robustness', 0)
                ]
                values += values[:1]  # 闭合图形
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                    label=strategy.name, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(dimensions, fontsize=12)
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
            ax.grid(True)
            
            # 添加标题和图例
            ax.set_title('策略五维度雷达图', fontsize=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            return filepath
            
        except Exception as e:
            print(f"创建策略雷达图失败: {e}")
            return ""

    def _create_rolling_metrics_chart(self, results: Dict[str, Any], filepath: str) -> str:
        """创建滚动指标图"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            colors = sns.color_palette("Set2", len(results))
            
            for i, (name, result) in enumerate(results.items()):
                if result.daily_returns is not None and len(result.daily_returns) > 60:
                    returns = result.daily_returns
                    
                    # 计算滚动指标
                    window = 60  # 60天滚动窗口
                    
                    # 滚动收益率
                    rolling_returns = pd.Series(returns).rolling(window).apply(
                        lambda x: (1 + x).prod() - 1, raw=True
                    ) * 100
                    
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
                    
                    rolling_sharpe = pd.Series(returns).rolling(window).apply(
                        safe_rolling_sharpe, raw=True
                    )
                    
                    # 滚动波动率
                    rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252) * 100
                    
                    # 滚动最大回撤
                    rolling_dd = pd.Series(returns).rolling(window).apply(
                        self._calculate_rolling_max_drawdown, raw=True
                    ) * 100
                    
                    # 创建日期索引
                    dates = pd.date_range(
                        start=pd.Timestamp.now() - pd.Timedelta(days=len(returns)),
                        periods=len(returns),
                        freq='D'
                    )
                    
                    # 绘制滚动指标
                    ax1.plot(dates, rolling_returns, label=name, color=colors[i])
                    ax2.plot(dates, rolling_sharpe, label=name, color=colors[i])
                    ax3.plot(dates, rolling_vol, label=name, color=colors[i])
                    ax4.plot(dates, rolling_dd, label=name, color=colors[i])
            
            # 设置图表
            ax1.set_title('60天滚动收益率', fontsize=14, fontweight='bold')
            ax1.set_ylabel('收益率 (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.set_title('60天滚动夏普比率', fontsize=14, fontweight='bold')
            ax2.set_ylabel('夏普比率')
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            ax3.set_title('60天滚动波动率', fontsize=14, fontweight='bold')
            ax3.set_ylabel('波动率 (%)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            ax4.set_title('60天滚动最大回撤', fontsize=14, fontweight='bold')
            ax4.set_ylabel('最大回撤 (%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 设置日期格式
            import matplotlib.dates as mdates
            for ax in [ax1, ax2, ax3, ax4]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            return filepath
            
        except Exception as e:
            print(f"创建滚动指标图失败: {e}")
            return ""

    def _calculate_rolling_max_drawdown(self, returns: np.ndarray) -> float:
        """计算滚动最大回撤"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative / running_max) - 1
        return np.min(drawdown)

    def _create_monthly_returns_heatmap(self, results: Dict[str, Any], filepath: str) -> str:
        """创建月度收益热图"""
        try:
            # 选择第一个策略作为示例（或可以选择AI Agent）
            if 'AI Agent' in results:
                selected_result = results['AI Agent']
                strategy_name = 'AI Agent'
            else:
                selected_result = list(results.values())[0]
                strategy_name = list(results.keys())[0]
            
            if selected_result.daily_returns is None or len(selected_result.daily_returns) < 30:
                print("数据不足，无法生成月度收益热图")
                return ""
            
            # 创建日期索引和收益率序列
            returns = selected_result.daily_returns
            dates = pd.date_range(
                start=pd.Timestamp.now() - pd.Timedelta(days=len(returns)),
                periods=len(returns),
                freq='D'
            )
            
            returns_series = pd.Series(returns, index=dates)
            
            # 计算月度收益
            monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            # 创建年月矩阵
            monthly_data = []
            for date, ret in monthly_returns.items():
                monthly_data.append({
                    'Year': date.year,
                    'Month': date.month,
                    'Return': ret * 100
                })
            
            if not monthly_data:
                print("无法计算月度收益")
                return ""
            
            df_monthly = pd.DataFrame(monthly_data)
            pivot_table = df_monthly.pivot(index='Year', columns='Month', values='Return')
            
            # 创建热图
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 使用发散颜色映射
            sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, cbar_kws={'label': '月收益率 (%)'}, ax=ax)
            
            # 设置月份标签
            month_labels = ['1月', '2月', '3月', '4月', '5月', '6月',
                        '7月', '8月', '9月', '10月', '11月', '12月']
            ax.set_xticklabels([month_labels[i-1] for i in pivot_table.columns])
            
            ax.set_title(f'{strategy_name} - 月度收益热图', fontsize=16, fontweight='bold')
            ax.set_xlabel('月份', fontsize=12)
            ax.set_ylabel('年份', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            return filepath
            
        except Exception as e:
            print(f"创建月度收益热图失败: {e}")
            return ""