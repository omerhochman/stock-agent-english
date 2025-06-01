import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any
import os
import platform
import matplotlib.font_manager as fm
from src.utils.logging_config import setup_logger

# 设置日志记录器
logger = setup_logger('backtest_visualizer')

# 更robust的中文字体设置
def setup_chinese_fonts():
    """设置中文字体支持"""
    try:
        # Windows系统优先使用Microsoft YaHei
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
        
        # 检查可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        selected_font = None
        
        for font in chinese_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        # 设置字体配置为Windows优化
        if selected_font:
            plt.rcParams['font.sans-serif'] = [selected_font]
        else:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置默认字体大小和样式
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none'
        })
    except Exception as e:
        logger.warning(f"字体设置失败，使用默认字体: {e}")

# 初始化字体设置
setup_chinese_fonts()

class BacktestVisualizer:
    """
    回测可视化类
    生成专业的回测分析图表
    """
    
    def __init__(self):
        self.style_config = {
            'figure_size': (12, 8),
            'dpi': 150,  # 降低DPI避免内存问题
            'color_palette': 'Set2'
        }
        self._setup_style()
    
    def _setup_style(self):
        """设置图表样式"""
        try:
            # 确保字体设置生效
            setup_chinese_fonts()
            
            # 使用更稳定的样式设置
            try:
                # 尝试使用seaborn样式
                available_styles = plt.style.available
                if 'seaborn-v0_8' in available_styles:
                    plt.style.use('seaborn-v0_8')
                elif any('seaborn' in style for style in available_styles):
                    # 使用任何可用的seaborn样式
                    seaborn_styles = [s for s in available_styles if 'seaborn' in s]
                    plt.style.use(seaborn_styles[0])
                else:
                    # 如果seaborn不可用，使用默认样式并手动设置
                    plt.style.use('default')
                    
            except Exception as e:
                logger.warning(f"样式设置失败，使用默认样式: {e}")
                plt.style.use('default')
                
            # 手动设置样式参数以确保一致性
            plt.rcParams.update({
                'figure.figsize': self.style_config['figure_size'],
                'figure.dpi': 100,  # 使用较低的DPI
                'axes.grid': True,
                'grid.alpha': 0.3,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'legend.frameon': True,
                'legend.framealpha': 0.8
            })
            
            # 设置颜色调色板
            try:
                sns.set_palette(self.style_config['color_palette'])
            except Exception as e:
                logger.warning(f"颜色调色板设置失败: {e}")
                
        except Exception as e:
            logger.error(f"样式设置失败: {e}")
    
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
        try:
            # 创建保存目录
            os.makedirs(save_dir, exist_ok=True)
            
            chart_paths = {}
            
            # 检查是否有有效的结果数据
            if not results or len(results) == 0:
                logger.warning("没有有效的回测结果数据")
                return chart_paths
            
            # 1. 综合性能对比图
            try:
                chart_paths['performance_comparison'] = self._create_performance_comparison(
                    results, os.path.join(save_dir, 'performance_comparison.png')
                )
            except Exception as e:
                logger.error(f"创建性能对比图失败: {e}")
            
            # 2. 累计收益对比图
            try:
                chart_paths['cumulative_returns'] = self._create_cumulative_returns_chart(
                    results, os.path.join(save_dir, 'cumulative_returns.png')
                )
            except Exception as e:
                logger.error(f"创建累计收益图失败: {e}")
            
            # 3. 风险收益散点图
            try:
                chart_paths['risk_return_scatter'] = self._create_risk_return_scatter(
                    results, os.path.join(save_dir, 'risk_return_scatter.png')
                )
            except Exception as e:
                logger.error(f"创建风险收益散点图失败: {e}")
            
            # 4. 回撤分析图
            try:
                chart_paths['drawdown_analysis'] = self._create_drawdown_analysis(
                    results, os.path.join(save_dir, 'drawdown_analysis.png')
                )
            except Exception as e:
                logger.error(f"创建回撤分析图失败: {e}")
            
            # 5. 策略雷达图
            if comparison_results and 'strategy_ranking' in comparison_results:
                try:
                    chart_paths['radar_chart'] = self._create_strategy_radar_chart(
                        comparison_results['strategy_ranking'], 
                        os.path.join(save_dir, 'strategy_radar.png')
                    )
                except Exception as e:
                    logger.error(f"创建策略雷达图失败: {e}")
            
            # 6. 滚动指标图
            try:
                chart_paths['rolling_metrics'] = self._create_rolling_metrics_chart(
                    results, os.path.join(save_dir, 'rolling_metrics.png')
                )
            except Exception as e:
                logger.error(f"创建滚动指标图失败: {e}")
            
            # 7. 月度收益热图
            try:
                chart_paths['monthly_returns'] = self._create_monthly_returns_heatmap(
                    results, os.path.join(save_dir, 'monthly_returns.png')
                )
            except Exception as e:
                logger.error(f"创建月度收益热图失败: {e}")
            
            logger.info(f"成功创建 {len(chart_paths)} 个图表")
            return chart_paths
            
        except Exception as e:
            logger.error(f"创建图表集失败: {e}")
            return {}
    
    def _create_performance_comparison(self, results: Dict[str, Any], filepath: str) -> str:
        """创建综合性能对比图"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 提取数据
            strategy_names = list(results.keys())
            if not strategy_names:
                logger.warning("没有策略数据")
                return ""
                
            total_returns = []
            sharpe_ratios = []
            max_drawdowns = []
            volatilities = []
            
            for name in strategy_names:
                result = results[name]
                metrics = getattr(result, 'performance_metrics', {})
                
                total_returns.append(metrics.get('total_return', 0) * 100)
                sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                max_drawdowns.append(abs(metrics.get('max_drawdown', 0)) * 100)
                volatilities.append(metrics.get('volatility', 0) * 100)
            
            # 1. 总收益率对比
            try:
                colors = sns.color_palette(self.style_config['color_palette'], len(strategy_names))
            except:
                colors = plt.cm.Set2(np.linspace(0, 1, len(strategy_names)))
                
            bars1 = ax1.bar(strategy_names, total_returns, color=colors)
            ax1.set_title('总收益率对比', fontsize=14, fontweight='bold')
            ax1.set_ylabel('收益率 (%)', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars1, total_returns):
                height = bar.get_height()
                if np.isfinite(height) and np.isfinite(value):
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
                if np.isfinite(height) and np.isfinite(value):
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
                if np.isfinite(height) and np.isfinite(value):
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
                if np.isfinite(height) and np.isfinite(value):
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.2f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"成功创建性能对比图: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"创建性能对比图失败: {e}")
            return ""

    def _create_cumulative_returns_chart(self, results: Dict[str, Any], filepath: str) -> str:
        """创建累计收益对比图"""
        try:
            fig, ax = plt.subplots(figsize=self.style_config['figure_size'])
            
            try:
                colors = sns.color_palette(self.style_config['color_palette'], len(results))
            except:
                colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
            
            valid_data_count = 0
            for i, (name, result) in enumerate(results.items()):
                try:
                    daily_returns = getattr(result, 'daily_returns', None)
                    if daily_returns is not None and len(daily_returns) > 0:
                        # 确保数据是有效的
                        daily_returns = np.array(daily_returns)
                        daily_returns = daily_returns[np.isfinite(daily_returns)]
                        
                        if len(daily_returns) > 0:
                            # 计算累计收益
                            cumulative_returns = np.cumprod(1 + daily_returns) - 1
                            
                            # 创建日期索引
                            dates = pd.date_range(
                                start=pd.Timestamp.now() - pd.Timedelta(days=len(cumulative_returns)),
                                periods=len(cumulative_returns),
                                freq='D'
                            )
                            
                            ax.plot(dates, cumulative_returns * 100, 
                                label=name, color=colors[i], linewidth=2)
                            valid_data_count += 1
                except Exception as e:
                    logger.warning(f"处理策略 {name} 的数据时出错: {e}")
                    continue
            
            if valid_data_count == 0:
                logger.warning("没有有效的收益数据")
                plt.close(fig)
                return ""
            
            ax.set_title('累计收益率对比', fontsize=16, fontweight='bold')
            ax.set_xlabel('日期', fontsize=12)
            ax.set_ylabel('累计收益率 (%)', fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # 设置日期格式
            try:
                import matplotlib.dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.xticks(rotation=45)
            except Exception as e:
                logger.warning(f"设置日期格式失败: {e}")
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"成功创建累计收益图: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"创建累计收益图失败: {e}")
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
                try:
                    metrics = getattr(result, 'performance_metrics', {})
                    
                    annual_return = metrics.get('annual_return', 0) * 100
                    volatility = metrics.get('volatility', 0) * 100
                    sharpe_ratio = metrics.get('sharpe_ratio', 0)
                    
                    # 验证数据有效性
                    if np.isfinite(annual_return) and np.isfinite(volatility) and np.isfinite(sharpe_ratio):
                        strategy_names.append(name)
                        annual_returns.append(annual_return)
                        volatilities.append(volatility)
                        sharpe_ratios.append(sharpe_ratio)
                except Exception as e:
                    logger.warning(f"处理策略 {name} 的风险收益数据时出错: {e}")
                    continue
            
            if len(strategy_names) == 0:
                logger.warning("没有有效的风险收益数据")
                plt.close(fig)
                return ""
            
            # 创建散点图，点的大小表示夏普比率
            sizes = [max(100, abs(sr) * 200) for sr in sharpe_ratios]
            try:
                colors = sns.color_palette("viridis", len(strategy_names))
            except:
                colors = plt.cm.viridis(np.linspace(0, 1, len(strategy_names)))
            
            scatter = ax.scatter(volatilities, annual_returns, 
                            s=sizes, c=colors, alpha=0.7, edgecolors='black')
            
            # 添加策略名称标签
            for i, name in enumerate(strategy_names):
                ax.annotate(name, (volatilities[i], annual_returns[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, ha='left')
            
            # 添加有效前沿参考线
            if len(volatilities) > 1:
                try:
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
                except Exception as e:
                    logger.warning(f"绘制参考线失败: {e}")
            
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
            
            logger.info(f"成功创建风险收益散点图: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"创建风险收益散点图失败: {e}")
            return ""

    def _create_drawdown_analysis(self, results: Dict[str, Any], filepath: str) -> str:
        """创建回撤分析图"""
        try:
            valid_results = []
            for name, result in results.items():
                daily_returns = getattr(result, 'daily_returns', None)
                if daily_returns is not None and len(daily_returns) > 0:
                    valid_results.append((name, result))
            
            if not valid_results:
                logger.warning("没有有效的回撤数据")
                return ""
            
            fig, axes = plt.subplots(len(valid_results), 1, 
                                    figsize=(self.style_config['figure_size'][0], 
                                            len(valid_results) * 4))
            
            if len(valid_results) == 1:
                axes = [axes]
            
            try:
                colors = sns.color_palette(self.style_config['color_palette'], len(valid_results))
            except:
                colors = plt.cm.Set2(np.linspace(0, 1, len(valid_results)))
            
            for i, (name, result) in enumerate(valid_results):
                try:
                    daily_returns = np.array(result.daily_returns)
                    daily_returns = daily_returns[np.isfinite(daily_returns)]
                    
                    if len(daily_returns) > 0:
                        # 计算累计收益和回撤
                        cumulative = np.cumprod(1 + daily_returns)
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
                        try:
                            import matplotlib.dates as mdates
                            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                            axes[i].tick_params(axis='x', rotation=45)
                        except Exception as e:
                            logger.warning(f"设置日期格式失败: {e}")
                            
                except Exception as e:
                    logger.warning(f"处理策略 {name} 的回撤数据时出错: {e}")
                    continue
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"成功创建回撤分析图: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"创建回撤分析图失败: {e}")
            return ""

    def _create_strategy_radar_chart(self, ranking_data: Dict[str, Any], filepath: str) -> str:
        """创建策略雷达图"""
        try:
            rankings = ranking_data.get('rankings', [])
            if not rankings:
                logger.warning("没有策略排名数据")
                return ""
            
            # 取前5个策略
            top_strategies = rankings[:5]
            
            if not top_strategies:
                logger.warning("没有有效的策略数据")
                return ""
            
            # 设置雷达图
            dimensions = ['收益', '风险', '稳定性', '效率', '稳健性']
            angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            try:
                colors = sns.color_palette(self.style_config['color_palette'], len(top_strategies))
            except:
                colors = plt.cm.Set2(np.linspace(0, 1, len(top_strategies)))
            
            for i, strategy in enumerate(top_strategies):
                try:
                    # 检查strategy是否有scores属性
                    if hasattr(strategy, 'scores'):
                        scores = strategy.scores
                    elif isinstance(strategy, dict) and 'scores' in strategy:
                        scores = strategy['scores']
                    else:
                        logger.warning(f"策略 {getattr(strategy, 'name', 'Unknown')} 没有scores数据")
                        continue
                    
                    # 获取策略名称
                    if hasattr(strategy, 'name'):
                        strategy_name = strategy.name
                    elif isinstance(strategy, dict) and 'name' in strategy:
                        strategy_name = strategy['name']
                    else:
                        strategy_name = f"策略{i+1}"
                    
                    values = [
                        scores.get('return', 0),
                        scores.get('risk', 0),
                        scores.get('stability', 0),
                        scores.get('efficiency', 0),
                        scores.get('robustness', 0)
                    ]
                    
                    # 验证数值有效性
                    values = [v if np.isfinite(v) else 0 for v in values]
                    values += values[:1]  # 闭合图形
                    
                    ax.plot(angles, values, 'o-', linewidth=2, 
                        label=strategy_name, color=colors[i])
                    ax.fill(angles, values, alpha=0.25, color=colors[i])
                    
                except Exception as e:
                    logger.warning(f"处理策略雷达图数据时出错: {e}")
                    continue
            
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
            
            logger.info(f"成功创建策略雷达图: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"创建策略雷达图失败: {e}")
            return ""

    def _create_rolling_metrics_chart(self, results: Dict[str, Any], filepath: str) -> str:
        """创建滚动指标图"""
        try:
            # 筛选有效数据
            valid_results = []
            for name, result in results.items():
                daily_returns = getattr(result, 'daily_returns', None)
                if daily_returns is not None and len(daily_returns) > 60:
                    valid_results.append((name, result))
            
            if not valid_results:
                logger.warning("没有足够的数据创建滚动指标图（需要至少60天数据）")
                return ""
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            try:
                colors = sns.color_palette(self.style_config['color_palette'], len(valid_results))
            except:
                colors = plt.cm.Set2(np.linspace(0, 1, len(valid_results)))
            
            for i, (name, result) in enumerate(valid_results):
                try:
                    returns = np.array(result.daily_returns)
                    returns = returns[np.isfinite(returns)]
                    
                    if len(returns) <= 60:
                        continue
                    
                    # 计算滚动指标
                    window = 60  # 60天滚动窗口
                    
                    # 滚动收益率
                    returns_series = pd.Series(returns)
                    rolling_returns = returns_series.rolling(window).apply(
                        lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0, raw=True
                    ) * 100
                    
                    # 滚动夏普比率
                    def safe_rolling_sharpe(x):
                        try:
                            if len(x) == 0:
                                return 0.0
                            mean_ret = np.mean(x)
                            std_ret = np.std(x, ddof=1)
                            if std_ret == 0 or not np.isfinite(std_ret) or not np.isfinite(mean_ret):
                                return 0.0
                            sharpe = mean_ret / std_ret * np.sqrt(252)
                            return sharpe if np.isfinite(sharpe) else 0.0
                        except:
                            return 0.0
                    
                    rolling_sharpe = returns_series.rolling(window).apply(
                        safe_rolling_sharpe, raw=True
                    )
                    
                    # 滚动波动率
                    rolling_vol = returns_series.rolling(window).std() * np.sqrt(252) * 100
                    
                    # 滚动最大回撤
                    def safe_rolling_max_drawdown(x):
                        try:
                            if len(x) == 0:
                                return 0.0
                            cumulative = np.cumprod(1 + x)
                            running_max = np.maximum.accumulate(cumulative)
                            drawdown = (cumulative / running_max) - 1
                            return np.min(drawdown)
                        except:
                            return 0.0
                    
                    rolling_dd = returns_series.rolling(window).apply(
                        safe_rolling_max_drawdown, raw=True
                    ) * 100
                    
                    # 创建日期索引
                    dates = pd.date_range(
                        start=pd.Timestamp.now() - pd.Timedelta(days=len(returns)),
                        periods=len(returns),
                        freq='D'
                    )
                    
                    # 绘制滚动指标（只绘制有效数据）
                    valid_mask = np.isfinite(rolling_returns) & np.isfinite(rolling_sharpe) & np.isfinite(rolling_vol) & np.isfinite(rolling_dd)
                    
                    if np.any(valid_mask):
                        ax1.plot(dates[valid_mask], rolling_returns[valid_mask], label=name, color=colors[i])
                        ax2.plot(dates[valid_mask], rolling_sharpe[valid_mask], label=name, color=colors[i])
                        ax3.plot(dates[valid_mask], rolling_vol[valid_mask], label=name, color=colors[i])
                        ax4.plot(dates[valid_mask], rolling_dd[valid_mask], label=name, color=colors[i])
                        
                except Exception as e:
                    logger.warning(f"处理策略 {name} 的滚动指标时出错: {e}")
                    continue
            
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
            try:
                import matplotlib.dates as mdates
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    ax.tick_params(axis='x', rotation=45)
            except Exception as e:
                logger.warning(f"设置日期格式失败: {e}")
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"成功创建滚动指标图: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"创建滚动指标图失败: {e}")
            return ""

    def _calculate_rolling_max_drawdown(self, returns: np.ndarray) -> float:
        """计算滚动最大回撤"""
        try:
            if len(returns) == 0:
                return 0.0
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative / running_max) - 1
            return np.min(drawdown)
        except:
            return 0.0

    def _create_monthly_returns_heatmap(self, results: Dict[str, Any], filepath: str) -> str:
        """创建月度收益热图"""
        try:
            # 选择第一个策略作为示例（或可以选择AI Agent）
            selected_result = None
            strategy_name = ""
            
            if 'AI Agent' in results:
                selected_result = results['AI Agent']
                strategy_name = 'AI Agent'
            else:
                # 选择第一个有有效数据的策略
                for name, result in results.items():
                    daily_returns = getattr(result, 'daily_returns', None)
                    if daily_returns is not None and len(daily_returns) >= 30:
                        selected_result = result
                        strategy_name = name
                        break
            
            if selected_result is None:
                logger.warning("没有足够的数据生成月度收益热图（需要至少30天数据）")
                return ""
            
            daily_returns = getattr(selected_result, 'daily_returns', None)
            if daily_returns is None or len(daily_returns) < 30:
                logger.warning("数据不足，无法生成月度收益热图")
                return ""
            
            # 创建日期索引和收益率序列
            returns = np.array(daily_returns)
            returns = returns[np.isfinite(returns)]
            
            if len(returns) < 30:
                logger.warning("有效数据不足，无法生成月度收益热图")
                return ""
            
            dates = pd.date_range(
                start=pd.Timestamp.now() - pd.Timedelta(days=len(returns)),
                periods=len(returns),
                freq='D'
            )
            
            returns_series = pd.Series(returns, index=dates)
            
            # 计算月度收益
            try:
                monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0)
            except Exception as e:
                logger.warning(f"计算月度收益失败: {e}")
                return ""
            
            # 创建年月矩阵
            monthly_data = []
            for date, ret in monthly_returns.items():
                if np.isfinite(ret):
                    monthly_data.append({
                        'Year': date.year,
                        'Month': date.month,
                        'Return': ret * 100
                    })
            
            if not monthly_data:
                logger.warning("无法计算有效的月度收益")
                return ""
            
            df_monthly = pd.DataFrame(monthly_data)
            
            try:
                pivot_table = df_monthly.pivot(index='Year', columns='Month', values='Return')
            except Exception as e:
                logger.warning(f"创建数据透视表失败: {e}")
                return ""
            
            # 创建热图
            fig, ax = plt.subplots(figsize=(12, 8))
            
            try:
                # 使用发散颜色映射
                sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn',
                        center=0, cbar_kws={'label': '月收益率 (%)'}, ax=ax)
                
                # 设置月份标签
                month_labels = ['1月', '2月', '3月', '4月', '5月', '6月',
                            '7月', '8月', '9月', '10月', '11月', '12月']
                
                # 只设置存在的月份标签
                existing_months = pivot_table.columns
                month_labels_filtered = [month_labels[i-1] for i in existing_months if 1 <= i <= 12]
                ax.set_xticklabels(month_labels_filtered)
                
                ax.set_title(f'{strategy_name} - 月度收益热图', fontsize=16, fontweight='bold')
                ax.set_xlabel('月份', fontsize=12)
                ax.set_ylabel('年份', fontsize=12)
                
            except Exception as e:
                logger.warning(f"创建热图失败: {e}")
                plt.close(fig)
                return ""
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"成功创建月度收益热图: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"创建月度收益热图失败: {e}")
            return ""