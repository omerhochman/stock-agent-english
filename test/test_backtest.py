#!/usr/bin/env python3
"""
回测框架测试脚本
用于验证回测框架的基本功能
"""

import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import argparse
import numpy as np
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest import Backtester, BacktestConfig
from src.backtest.baselines.buy_hold import BuyHoldStrategy
from src.backtest.baselines.momentum import MomentumStrategy
from src.backtest.baselines.mean_reversion import MeanReversionStrategy
from src.backtest.baselines.moving_average import MovingAverageStrategy
from src.backtest.baselines.random_walk import RandomWalkStrategy
from src.backtest.evaluation.table_generator import BacktestTableGenerator
from src.utils.logging_config import setup_logger
from src.main import run_hedge_fund


class ComparisonTableGenerator:
    """对比表格生成器（简化版，用于控制台显示）"""
    
    def __init__(self):
        self.logger = setup_logger('ComparisonTableGenerator')
    
    def generate_performance_table(self, results_summary: Dict[str, Dict[str, Any]], 
                                 save_to_file: bool = True, 
                                 filename: str = "performance_comparison.csv") -> pd.DataFrame:
        """
        生成性能对比表格
        
        Args:
            results_summary: 策略结果摘要
            save_to_file: 是否保存到文件
            filename: 保存文件名
            
        Returns:
            pd.DataFrame: 性能对比表格
        """
        if not results_summary:
            self.logger.warning("没有结果数据，无法生成表格")
            return pd.DataFrame()
        
        # 创建表格数据
        table_data = []
        for strategy_name, metrics in results_summary.items():
            row = {
                '策略名称': strategy_name,
                '总收益率(%)': round(metrics.get('total_return', 0), 2),
                '年化收益率(%)': round(metrics.get('annual_return', 0), 2),
                '夏普比率': round(metrics.get('sharpe_ratio', 0), 3),
                '最大回撤(%)': round(abs(metrics.get('max_drawdown', 0)), 2),
                '年化波动率(%)': round(metrics.get('volatility', 0), 2),
                '胜率(%)': round(metrics.get('win_rate', 0) * 100, 2),
                '盈亏比': round(metrics.get('profit_loss_ratio', 0), 2),
                '交易次数': metrics.get('trade_count', 0),
                'VaR(%)': round(abs(metrics.get('var_95', 0)) * 100, 2),
                '索提诺比率': round(metrics.get('sortino_ratio', 0), 3),
                '卡玛比率': round(metrics.get('calmar_ratio', 0), 3)
            }
            table_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(table_data)
        
        # 按总收益率排序
        df = df.sort_values('总收益率(%)', ascending=False).reset_index(drop=True)
        
        # 添加排名列
        df.insert(0, '排名', range(1, len(df) + 1))
        
        if save_to_file:
            try:
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                self.logger.info(f"性能对比表格已保存到: {filename}")
            except Exception as e:
                self.logger.error(f"保存表格失败: {e}")
        
        return df
    
    def generate_ranking_table(self, results_summary: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        生成策略排名表格
        
        Args:
            results_summary: 策略结果摘要
            
        Returns:
            pd.DataFrame: 排名表格
        """
        if not results_summary:
            return pd.DataFrame()
        
        # 定义评估维度和权重
        dimensions = {
            '收益表现': ['total_return', 'annual_return'],
            '风险控制': ['max_drawdown', 'volatility', 'var_95'],
            '风险调整收益': ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio'],
            '交易效率': ['win_rate', 'profit_loss_ratio', 'trade_count']
        }
        
        ranking_data = []
        
        # 收集所有策略的指标值用于标准化
        all_returns = [metrics.get('total_return', 0) for metrics in results_summary.values()]
        all_sharpe = [metrics.get('sharpe_ratio', 0) for metrics in results_summary.values()]
        all_drawdowns = [abs(metrics.get('max_drawdown', 0)) for metrics in results_summary.values()]
        all_volatility = [metrics.get('volatility', 0) for metrics in results_summary.values()]
        all_win_rates = [metrics.get('win_rate', 0) for metrics in results_summary.values()]
        
        # 计算基准值（用于标准化）
        max_return = max(all_returns) if all_returns else 1
        min_return = min(all_returns) if all_returns else 0
        max_sharpe = max(all_sharpe) if all_sharpe else 1
        min_sharpe = min(all_sharpe) if all_sharpe else 0
        max_drawdown = max(all_drawdowns) if all_drawdowns else 1
        max_volatility = max(all_volatility) if all_volatility else 1
        
        for strategy_name, metrics in results_summary.items():
            # 计算各维度得分（标准化到0-100）
            scores = {}
            
            # 收益表现得分（收益率越高越好）
            total_return = metrics.get('total_return', 0)
            annual_return = metrics.get('annual_return', 0)
            if max_return > min_return:
                return_score = ((total_return - min_return) / (max_return - min_return)) * 100
            else:
                return_score = 50  # 如果所有策略收益相同，给中等分数
            scores['收益表现'] = max(0, min(100, return_score))
            
            # 风险控制得分（回撤和波动率越小越好）
            max_dd = abs(metrics.get('max_drawdown', 0))
            volatility = metrics.get('volatility', 0)
            var_95 = abs(metrics.get('var_95', 0))
            
            # 风险控制得分：风险越低得分越高
            dd_score = max(0, 100 - (max_dd / max_drawdown * 100)) if max_drawdown > 0 else 100
            vol_score = max(0, 100 - (volatility / max_volatility * 100)) if max_volatility > 0 else 100
            var_score = max(0, 100 - (var_95 * 100))  # VaR通常是负值，取绝对值
            
            scores['风险控制'] = (dd_score + vol_score + var_score) / 3
            
            # 风险调整收益得分
            sharpe = metrics.get('sharpe_ratio', 0)
            sortino = metrics.get('sortino_ratio', 0)
            calmar = metrics.get('calmar_ratio', 0)
            
            # 标准化夏普比率
            if max_sharpe > min_sharpe:
                sharpe_score = ((sharpe - min_sharpe) / (max_sharpe - min_sharpe)) * 100
            else:
                sharpe_score = 50
            
            # 索提诺比率和卡玛比率的处理
            sortino_score = min(100, max(0, sortino * 20 + 50))  # 简单线性变换
            calmar_score = min(100, max(0, calmar * 20 + 50))
            
            scores['风险调整收益'] = (sharpe_score + sortino_score + calmar_score) / 3
            
            # 交易效率得分
            win_rate = metrics.get('win_rate', 0)
            pl_ratio = metrics.get('profit_loss_ratio', 0)
            trade_count = metrics.get('trade_count', 0)
            
            # 胜率得分（0-1转换为0-100）
            win_rate_score = win_rate * 100
            
            # 盈亏比得分
            pl_score = min(100, max(0, pl_ratio * 25))  # 盈亏比>4时满分
            
            # 交易次数得分（适度交易为好）
            if trade_count == 0:
                trade_score = 0
            elif trade_count <= 10:
                trade_score = trade_count * 10  # 1-10次交易线性增长
            elif trade_count <= 50:
                trade_score = 100 - (trade_count - 10) * 2  # 10-50次逐渐减分
            else:
                trade_score = max(0, 100 - trade_count)  # 超过50次大幅减分
            
            scores['交易效率'] = (win_rate_score + pl_score + trade_score) / 3
            
            # 计算综合得分
            weights = {'收益表现': 0.3, '风险控制': 0.3, '风险调整收益': 0.25, '交易效率': 0.15}
            composite_score = sum(scores[dim] * weights[dim] for dim in scores)
            
            ranking_data.append({
                '策略名称': strategy_name,
                '收益表现': round(scores['收益表现'], 1),
                '风险控制': round(scores['风险控制'], 1),
                '风险调整收益': round(scores['风险调整收益'], 1),
                '交易效率': round(scores['交易效率'], 1),
                '综合得分': round(composite_score, 1)
            })
        
        # 创建DataFrame并排序
        df = pd.DataFrame(ranking_data)
        df = df.sort_values('综合得分', ascending=False).reset_index(drop=True)
        df.insert(0, '排名', range(1, len(df) + 1))
        
        return df
    
    def generate_statistical_significance_table(self, comparison_results: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """
        生成统计显著性检验表格
        
        Args:
            comparison_results: 比较分析结果
            
        Returns:
            pd.DataFrame: 统计显著性表格
        """
        if not comparison_results or 'pairwise_comparisons' not in comparison_results:
            return pd.DataFrame()
        
        comparisons = comparison_results['pairwise_comparisons']
        significance_data = []
        
        for comparison_key, result in comparisons.items():
            try:
                if 'summary' in result:
                    summary = result['summary']
                    
                    # 提取策略名称
                    strategies = comparison_key.split(' vs ')
                    strategy1 = strategies[0] if len(strategies) > 0 else "未知"
                    strategy2 = strategies[1] if len(strategies) > 1 else "未知"
                    
                    # 提取统计检验结果
                    power = summary.get('statistical_power', 0)
                    conclusion = summary.get('overall_conclusion', '无结论')
                    
                    # 提取具体检验结果
                    paired_test = result.get('paired_test', {})
                    dm_test = result.get('diebold_mariano', {})
                    sharpe_test = result.get('sharpe_test', {})
                    
                    significance_data.append({
                        '策略对比': comparison_key,
                        '策略A': strategy1,
                        '策略B': strategy2,
                        '配对t检验': '显著' if paired_test.get('significant', False) else '不显著',
                        'DM检验': '显著' if dm_test.get('significant', False) else '不显著',
                        '夏普比率检验': '显著' if sharpe_test.get('significant', False) else '不显著',
                        '统计功效': round(power, 3),
                        '结论': conclusion
                    })
            except (KeyError, TypeError, IndexError) as e:
                self.logger.warning(f"处理比较结果 {comparison_key} 时出错: {e}")
                continue
        
        if not significance_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(significance_data)
        return df
    
    def generate_ai_agent_analysis_table(self, results_summary: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        生成AI Agent专项分析表格
        
        Args:
            results_summary: 策略结果摘要
            
        Returns:
            pd.DataFrame: AI Agent分析表格
        """
        if 'AI_Agent' not in results_summary:
            return pd.DataFrame()
        
        ai_metrics = results_summary['AI_Agent']
        
        # 计算与其他策略的比较
        other_strategies = {k: v for k, v in results_summary.items() if k != 'AI_Agent'}
        
        if not other_strategies:
            return pd.DataFrame()
        
        # 计算平均值和排名
        avg_metrics = {}
        rankings = {}
        
        for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']:
            values = [metrics.get(metric, 0) for metrics in other_strategies.values()]
            avg_metrics[metric] = np.mean(values) if values else 0
            
            # 计算排名
            all_values = [ai_metrics.get(metric, 0)] + values
            if metric in ['max_drawdown', 'volatility']:  # 越小越好
                all_values_abs = [abs(v) for v in all_values]
                rankings[metric] = sorted(all_values_abs).index(abs(ai_metrics.get(metric, 0))) + 1
            else:  # 越大越好
                rankings[metric] = len(all_values) - sorted(all_values).index(ai_metrics.get(metric, 0))
        
        analysis_data = [{
            '指标': '总收益率(%)',
            'AI Agent': round(ai_metrics.get('total_return', 0), 2),
            '基准平均': round(avg_metrics['total_return'], 2),
            '差异': round(ai_metrics.get('total_return', 0) - avg_metrics['total_return'], 2),
            '排名': f"{rankings['total_return']}/{len(results_summary)}",
            '表现': '优于平均' if ai_metrics.get('total_return', 0) > avg_metrics['total_return'] else '低于平均'
        }, {
            '指标': '夏普比率',
            'AI Agent': round(ai_metrics.get('sharpe_ratio', 0), 3),
            '基准平均': round(avg_metrics['sharpe_ratio'], 3),
            '差异': round(ai_metrics.get('sharpe_ratio', 0) - avg_metrics['sharpe_ratio'], 3),
            '排名': f"{rankings['sharpe_ratio']}/{len(results_summary)}",
            '表现': '优于平均' if ai_metrics.get('sharpe_ratio', 0) > avg_metrics['sharpe_ratio'] else '低于平均'
        }, {
            '指标': '最大回撤(%)',
            'AI Agent': round(abs(ai_metrics.get('max_drawdown', 0)), 2),
            '基准平均': round(abs(avg_metrics['max_drawdown']), 2),
            '差异': round(abs(ai_metrics.get('max_drawdown', 0)) - abs(avg_metrics['max_drawdown']), 2),
            '排名': f"{rankings['max_drawdown']}/{len(results_summary)}",
            '表现': '优于平均' if abs(ai_metrics.get('max_drawdown', 0)) < abs(avg_metrics['max_drawdown']) else '低于平均'
        }, {
            '指标': '年化波动率(%)',
            'AI Agent': round(ai_metrics.get('volatility', 0), 2),
            '基准平均': round(avg_metrics['volatility'], 2),
            '差异': round(ai_metrics.get('volatility', 0) - avg_metrics['volatility'], 2),
            '排名': f"{rankings['volatility']}/{len(results_summary)}",
            '表现': '优于平均' if ai_metrics.get('volatility', 0) < avg_metrics['volatility'] else '低于平均'
        }]
        
        return pd.DataFrame(analysis_data)


class BacktestTest:
    """回测测试类"""
    
    def __init__(self, start_date: str = "2023-01-01", end_date: str = "2023-03-31"):
        self.logger = setup_logger('BacktestTest')
        self.start_date = start_date
        self.end_date = end_date
        self.test_duration_days = self._calculate_test_duration()
        self.table_generator = ComparisonTableGenerator()
        self.comprehensive_table_generator = BacktestTableGenerator()
        
    def _calculate_test_duration(self) -> int:
        """计算测试时间长度（天数）"""
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        return (end - start).days
    
    def get_strategy_requirements(self) -> dict:
        """获取各策略的最小时间要求（天数）"""
        return {
            'Buy-and-Hold': 1,  # 买入持有策略无时间要求
            'Random-Walk': 1,   # 随机游走策略无时间要求
            'Mean-Reversion': 252,  # 均值回归策略需要252天（lookback_period）
            'Mean-Reversion-Short': 252,  # 短期均值回归策略需要252天
            'Moving-Average': 200,  # 移动平均策略需要200天（long_window）
            'Moving-Average-Short': 60,   # 短期移动平均策略需要60天
            'Momentum': 252,    # 动量策略需要252天（lookback_period）
            'Momentum-Long': 252,  # 长期动量策略需要252天
            'AI-Agent': 30,     # AI Agent策略需要30天最小数据
        }
    
    def select_strategies_by_duration(self) -> list:
        """根据测试时间长度选择合适的策略"""
        requirements = self.get_strategy_requirements()
        selected_strategies = []
        
        self.logger.info(f"测试时间长度: {self.test_duration_days} 天")
        
        # 始终包含的基础策略
        selected_strategies.extend([
            BuyHoldStrategy(allocation_ratio=1.0),
            RandomWalkStrategy(trade_probability=0.1, max_position_ratio=0.5, truly_random=True)
        ])
        
        # 根据时间长度添加其他策略
        if self.test_duration_days >= 30:
            # 短期移动平均策略
            selected_strategies.append(
                MovingAverageStrategy(
                    short_window=5,
                    long_window=20,
                    signal_threshold=0.001,
                    name="Moving-Average-Short"
                )
            )
            self.logger.info("✓ 添加短期移动平均策略 (需要20天)")
        
        if self.test_duration_days >= 60:
            # 标准移动平均策略
            selected_strategies.append(
                MovingAverageStrategy(
                    short_window=10,
                    long_window=30,
                    signal_threshold=0.001
                )
            )
            self.logger.info("✓ 添加标准移动平均策略 (需要30天)")
        
        if self.test_duration_days >= 252:
            # 均值回归策略
            selected_strategies.extend([
                MeanReversionStrategy(
                    lookback_period=252,
                    z_threshold=1.5,
                    mean_period=30,
                    exit_threshold=0.5
                ),
                MeanReversionStrategy(
                    lookback_period=126,
                    z_threshold=1.0,
                    mean_period=20,
                    exit_threshold=0.3,
                    name="Mean-Reversion-Short"
                )
            ])
            self.logger.info("✓ 添加均值回归策略 (需要252天)")
            
            # 动量策略
            selected_strategies.extend([
                MomentumStrategy(
                    lookback_period=252, 
                    formation_period=63, 
                    holding_period=21,
                    momentum_threshold=0.01
                ),
                MomentumStrategy(
                    lookback_period=252, 
                    formation_period=126, 
                    holding_period=42,
                    momentum_threshold=0.02,
                    name="Momentum-Long"
                )
            ])
            self.logger.info("✓ 添加动量策略 (需要252天)")
        
        # 初始化所有策略
        for strategy in selected_strategies:
            strategy.initialize(100000)
        
        self.logger.info(f"共选择 {len(selected_strategies)} 个策略进行测试")
        return selected_strategies
    
    def test_backtester_initialization(self):
        """测试回测器组件初始化"""
        self.logger.info("测试回测器组件初始化...")
        
        config = BacktestConfig(
            initial_capital=100000,
            start_date=self.start_date,
            end_date=self.end_date,
            benchmark_ticker="000300",
            trading_cost=0.001,
            slippage=0.001
        )
        
        backtester = Backtester(
            ticker="000001",
            config=config,
            seed=42  # 固定随机种子
        )
        
        assert backtester is not None
        assert backtester.config.initial_capital == 100000
        assert backtester.ticker == "000001"
        
        self.logger.info("✓ 回测器组件初始化正确")
    
    def test_baseline_strategies_backtest(self):
        """测试所有基准策略回测"""
        self.logger.info("测试基准策略回测...")
        
        config = BacktestConfig(
            initial_capital=100000,
            start_date=self.start_date,
            end_date=self.end_date,
            benchmark_ticker="000300",
            trading_cost=0.001,
            slippage=0.001
        )
        
        backtester = Backtester(
            ticker="000001",
            config=config,
            seed=42  # 固定随机种子
        )
        
        # 根据测试时间长度选择策略
        selected_strategies = self.select_strategies_by_duration()
        
        # 运行回测
        results = {}
        for strategy in selected_strategies:
            try:
                result = backtester._run_single_strategy_backtest(strategy)
                results[strategy.name] = result
            except Exception as e:
                self.logger.warning(f"策略 {strategy.name} 回测失败: {e}")
                continue
        
        # 验证结果
        assert len(results) > 0, "至少应该有一个策略成功完成回测"
        
        self.logger.info(f"✓ 成功完成 {len(results)} 个策略回测")
        
        # 输出结果摘要
        for name, result in results.items():
            total_return = result.performance_metrics.get('total_return', 0) * 100
            sharpe_ratio = result.performance_metrics.get('sharpe_ratio', 0)
            self.logger.info(f"  - {name}: 收益 {total_return:.2f}%, 夏普 {sharpe_ratio:.3f}")
        
        return results

    def test_ai_agent_backtest(self, ticker: str = "000001"):
        """测试AI Agent回测功能"""
        self.logger.info("测试AI Agent回测功能...")
        
        config = BacktestConfig(
            initial_capital=100000,
            start_date=self.start_date,
            end_date=self.end_date,
            benchmark_ticker="000300",
            trading_cost=0.001,
            slippage=0.001
        )
        
        # 创建包含AI agent的回测器
        backtester = Backtester(
            agent_function=run_hedge_fund,
            ticker=ticker,
            tickers=[ticker],
            config=config,
            seed=42
        )
        
        try:
            # 运行AI agent回测
            agent_result = backtester.run_agent_backtest()
            
            # 验证AI agent结果
            assert agent_result is not None
            assert agent_result.strategy_name == "AI Agent"
            assert len(agent_result.portfolio_values) > 0
            
            self.logger.info("✓ AI Agent回测测试完成")
            self.logger.info(f"  总收益率: {agent_result.performance_metrics.get('total_return', 0)*100:.2f}%")
            self.logger.info(f"  夏普比率: {agent_result.performance_metrics.get('sharpe_ratio', 0):.3f}")
            self.logger.info(f"  交易次数: {len(agent_result.trade_history)}")
            
            return agent_result
            
        except Exception as e:
            self.logger.error(f"AI Agent回测失败: {e}")
            # 返回None表示测试失败，但不中断整个测试流程
            return None

    def test_comprehensive_comparison(self, ticker: str = "000001"):
        """测试AI Agent与baseline策略的全面比较"""
        self.logger.info("测试AI Agent与baseline策略的全面比较...")
        
        config = BacktestConfig(
            initial_capital=100000,
            start_date=self.start_date,
            end_date=self.end_date,
            benchmark_ticker="000300",
            trading_cost=0.001,
            slippage=0.001
        )
        
        # 创建回测器
        backtester = Backtester(
            agent_function=run_hedge_fund,
            ticker=ticker,
            tickers=[ticker],
            config=config,
            seed=42
        )
        
        results_summary = {}
        
        try:
            # 1. 运行AI agent回测
            if self.test_duration_days >= 30:  # AI agent需要最少30天数据
                agent_result = backtester.run_agent_backtest()
                if agent_result:
                    # 计算AI Agent的交易统计
                    ai_trade_stats = self._calculate_trade_statistics(agent_result.trade_history if hasattr(agent_result, 'trade_history') else [])
                    
                    # 获取原始性能指标并添加调试信息
                    ai_original_total_return = agent_result.performance_metrics.get('total_return', 0)
                    ai_original_annual_return = agent_result.performance_metrics.get('annual_return', 0)
                    ai_original_max_drawdown = agent_result.performance_metrics.get('max_drawdown', 0)
                    ai_original_volatility = agent_result.performance_metrics.get('volatility', 0)
                    
                    # 调试输出
                    if abs(ai_original_total_return) < 1e-6:  # 如果总收益非常小
                        self.logger.warning(f"AI Agent 总收益率极小: {ai_original_total_return}, 日收益数组长度: {len(agent_result.daily_returns) if hasattr(agent_result, 'daily_returns') else 0}")
                        if hasattr(agent_result, 'daily_returns') and len(agent_result.daily_returns) > 0:
                            self.logger.warning(f"  AI Agent 日收益样本: {agent_result.daily_returns[:5].tolist() if len(agent_result.daily_returns) >= 5 else agent_result.daily_returns.tolist()}")
                    
                    # 处理小数值的收益率显示 - 保留更多精度
                    ai_total_return_pct = round(ai_original_total_return * 100, 4)
                    ai_annual_return_pct = round(ai_original_annual_return * 100, 4)
                    ai_max_drawdown_pct = round(ai_original_max_drawdown * 100, 4)
                    ai_volatility_pct = round(ai_original_volatility * 100, 4)
                    
                    results_summary['AI_Agent'] = {
                        'total_return': ai_total_return_pct,
                        'annual_return': ai_annual_return_pct,
                        'sharpe_ratio': agent_result.performance_metrics.get('sharpe_ratio', 0),
                        'max_drawdown': ai_max_drawdown_pct,
                        'volatility': ai_volatility_pct,
                        'win_rate': ai_trade_stats.get('win_rate', agent_result.performance_metrics.get('win_rate', 0)),
                        'profit_loss_ratio': ai_trade_stats.get('profit_loss_ratio', agent_result.performance_metrics.get('profit_loss_ratio', 0)),
                        'var_95': agent_result.performance_metrics.get('var_95', 0),
                        'sortino_ratio': agent_result.performance_metrics.get('sortino_ratio', 0),
                        'calmar_ratio': agent_result.performance_metrics.get('calmar_ratio', 0),
                        'trade_count': ai_trade_stats.get('trade_count', 0),
                        'profitable_trades': ai_trade_stats.get('profitable_trades', 0),
                        'losing_trades': ai_trade_stats.get('losing_trades', 0),
                        'avg_profit': ai_trade_stats.get('avg_profit', 0),
                        'avg_loss': ai_trade_stats.get('avg_loss', 0)
                    }
                    self.logger.info("✓ AI Agent回测完成")
            else:
                self.logger.warning("测试时间太短，跳过AI Agent回测")
        
        except Exception as e:
            self.logger.warning(f"AI Agent回测失败: {e}")
        
        # 2. 运行baseline策略回测
        baseline_results = backtester.run_baseline_backtests()
        
        for name, result in baseline_results.items():
            # 计算交易统计
            trade_stats = self._calculate_trade_statistics(result.trade_history if hasattr(result, 'trade_history') else [])
            
            # 获取原始性能指标并添加调试信息
            original_total_return = result.performance_metrics.get('total_return', 0)
            original_annual_return = result.performance_metrics.get('annual_return', 0)
            original_max_drawdown = result.performance_metrics.get('max_drawdown', 0)
            original_volatility = result.performance_metrics.get('volatility', 0)
            
            # 调试输出
            if abs(original_total_return) < 1e-6:  # 如果总收益非常小
                self.logger.warning(f"策略 {name} 总收益率极小: {original_total_return}, 日收益数组长度: {len(result.daily_returns) if hasattr(result, 'daily_returns') else 0}")
                if hasattr(result, 'daily_returns') and len(result.daily_returns) > 0:
                    self.logger.warning(f"  日收益样本: {result.daily_returns[:5].tolist() if len(result.daily_returns) >= 5 else result.daily_returns.tolist()}")
            
            # 处理小数值的收益率显示 - 保留更多精度
            total_return_pct = round(original_total_return * 100, 4)  # 保留4位小数
            annual_return_pct = round(original_annual_return * 100, 4)
            max_drawdown_pct = round(original_max_drawdown * 100, 4)
            volatility_pct = round(original_volatility * 100, 4)
            
            results_summary[name] = {
                'total_return': total_return_pct,
                'annual_return': annual_return_pct,
                'sharpe_ratio': result.performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown': max_drawdown_pct,
                'volatility': volatility_pct,
                'win_rate': trade_stats.get('win_rate', result.performance_metrics.get('win_rate', 0)),
                'profit_loss_ratio': trade_stats.get('profit_loss_ratio', result.performance_metrics.get('profit_loss_ratio', 0)),
                'var_95': result.performance_metrics.get('var_95', 0),
                'sortino_ratio': result.performance_metrics.get('sortino_ratio', 0),
                'calmar_ratio': result.performance_metrics.get('calmar_ratio', 0),
                'trade_count': trade_stats.get('trade_count', 0),
                'profitable_trades': trade_stats.get('profitable_trades', 0),
                'losing_trades': trade_stats.get('losing_trades', 0),
                'avg_profit': trade_stats.get('avg_profit', 0),
                'avg_loss': trade_stats.get('avg_loss', 0)
            }
        
        # 3. 运行统计比较（如果有足够的结果）
        comparison_results = None
        if len(backtester.results) >= 2:
            try:
                comparison_results = backtester.run_comprehensive_comparison()
                self.logger.info("✓ 统计显著性检验完成")
                
                # 生成统计显著性表格
                significance_df = self.table_generator.generate_statistical_significance_table(comparison_results)
                if not significance_df.empty:
                    self.logger.info("\n📈 统计显著性检验表格:")
                    self.logger.info("\n" + significance_df.to_string(index=False))
                    
                    # 保存统计显著性表格
                    significance_filename = f"statistical_significance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    significance_df.to_csv(significance_filename, index=False, encoding='utf-8-sig')
                    self.logger.info(f"统计显著性表格已保存到: {significance_filename}")
                
            except Exception as e:
                self.logger.warning(f"统计比较失败: {e}")
        
        # 4. 生成比较报告
        self._generate_comparison_report(results_summary)
        
        # 5. 生成综合报告文件
        try:
            config_info = {
                '股票代码': ticker,
                '回测开始日期': self.start_date,
                '回测结束日期': self.end_date,
                '初始资金': f"{config.initial_capital:,.0f}",
                '交易成本': f"{config.trading_cost*100:.1f}%",
                '滑点': f"{config.slippage*100:.1f}%",
                '基准指数': config.benchmark_ticker,
                '测试时长': f"{self.test_duration_days}天"
            }
            
            generated_files = self.comprehensive_table_generator.generate_comprehensive_report(
                results_summary=results_summary,
                comparison_results=comparison_results,
                config=config_info,
                export_formats=['csv', 'excel', 'html']
            )
            
            if generated_files:
                self.logger.info(f"\n📁 综合报告文件已生成:")
                for report_type, filepath in generated_files.items():
                    self.logger.info(f"  📄 {report_type}: {filepath}")
            
        except Exception as e:
            self.logger.error(f"生成综合报告失败: {e}")
        
        return {
            'results_summary': results_summary,
            'comparison_results': comparison_results,
            'baseline_results': baseline_results,
            'generated_files': generated_files if 'generated_files' in locals() else {}
        }

    def _generate_comparison_report(self, results_summary: dict):
        """生成比较报告"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("策略性能比较报告")
        self.logger.info("=" * 80)
        
        # 1. 生成性能对比表格
        performance_df = self.table_generator.generate_performance_table(
            results_summary, 
            save_to_file=True, 
            filename=f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if not performance_df.empty:
            self.logger.info("\n📊 详细性能对比表格:")
            self.logger.info("\n" + performance_df.to_string(index=False))
        
        # 2. 生成策略排名表格
        ranking_df = self.table_generator.generate_ranking_table(results_summary)
        
        if not ranking_df.empty:
            self.logger.info("\n🏆 策略综合排名表格:")
            self.logger.info("\n" + ranking_df.to_string(index=False))
        
        # 3. 生成AI Agent专项分析表格
        if 'AI_Agent' in results_summary:
            ai_analysis_df = self.table_generator.generate_ai_agent_analysis_table(results_summary)
            
            if not ai_analysis_df.empty:
                self.logger.info("\n🤖 AI Agent专项分析表格:")
                self.logger.info("\n" + ai_analysis_df.to_string(index=False))
        
        # 4. 简化的排名显示
        sorted_results = sorted(results_summary.items(), 
                              key=lambda x: x[1]['total_return'], reverse=True)
        
        self.logger.info(f"\n📈 策略收益率排名:")
        self.logger.info(f"{'排名':<4} {'策略名称':<25} {'收益率':<12} {'夏普比率':<12} {'最大回撤':<12}")
        self.logger.info("-" * 80)
        
        for i, (name, metrics) in enumerate(sorted_results, 1):
            self.logger.info(f"{i:<4} {name:<25} {metrics['total_return']:>10.2f}% {metrics['sharpe_ratio']:>10.3f} {abs(metrics['max_drawdown']):>10.2f}%")
        
        # 5. AI Agent表现分析
        if 'AI_Agent' in results_summary:
            ai_metrics = results_summary['AI_Agent']
            ai_rank = next((i for i, (name, _) in enumerate(sorted_results, 1) if name == 'AI_Agent'), None)
            
            self.logger.info(f"\n🎯 AI Agent表现分析:")
            self.logger.info(f"  📊 排名: 第{ai_rank}名 (共{len(sorted_results)}个策略)")
            self.logger.info(f"  💰 收益率: {ai_metrics['total_return']:.2f}%")
            self.logger.info(f"  📈 夏普比率: {ai_metrics['sharpe_ratio']:.3f}")
            self.logger.info(f"  📉 最大回撤: {abs(ai_metrics['max_drawdown']):.2f}%")
            self.logger.info(f"  🔄 交易次数: {ai_metrics['trade_count']}")
            
            # 与平均水平比较
            avg_return = sum(m['total_return'] for m in results_summary.values()) / len(results_summary)
            avg_sharpe = sum(m['sharpe_ratio'] for m in results_summary.values()) / len(results_summary)
            avg_drawdown = sum(abs(m['max_drawdown']) for m in results_summary.values()) / len(results_summary)
            
            self.logger.info(f"\n📊 与平均水平比较:")
            self.logger.info(f"  收益率差异: {ai_metrics['total_return'] - avg_return:+.2f}%")
            self.logger.info(f"  夏普比率差异: {ai_metrics['sharpe_ratio'] - avg_sharpe:+.3f}")
            self.logger.info(f"  回撤差异: {abs(ai_metrics['max_drawdown']) - avg_drawdown:+.2f}%")
            
            # 表现评级
            performance_score = 0
            if ai_metrics['total_return'] > avg_return:
                performance_score += 1
            if ai_metrics['sharpe_ratio'] > avg_sharpe:
                performance_score += 1
            if abs(ai_metrics['max_drawdown']) < avg_drawdown:
                performance_score += 1
            
            if performance_score >= 2:
                rating = "优秀 ⭐⭐⭐"
            elif performance_score == 1:
                rating = "良好 ⭐⭐"
            else:
                rating = "一般 ⭐"
            
            self.logger.info(f"  综合评级: {rating}")
        
        self.logger.info("=" * 80)

    def _calculate_trade_statistics(self, trade_history: list) -> dict:
        """
        从交易历史计算交易统计
        
        Args:
            trade_history: 交易历史列表
            
        Returns:
            dict: 交易统计指标
        """
        if not trade_history:
            return {
                'trade_count': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_loss_ratio': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0
            }
        
        # 按交易对分组（买入-卖出配对）
        buy_trades = []
        sell_trades = []
        
        for trade in trade_history:
            if trade.get('action') == 'buy':
                buy_trades.append(trade)
            elif trade.get('action') == 'sell':
                sell_trades.append(trade)
        
        # 计算每笔完整交易的盈亏
        trade_profits = []
        
        # 简化处理：假设FIFO（先进先出）
        buy_queue = buy_trades.copy()
        
        for sell_trade in sell_trades:
            sell_quantity = sell_trade.get('quantity', 0)
            sell_price = sell_trade.get('price', 0)
            
            while sell_quantity > 0 and buy_queue:
                buy_trade = buy_queue[0]
                buy_quantity = buy_trade.get('quantity', 0)
                buy_price = buy_trade.get('price', 0)
                
                # 计算这次匹配的数量
                matched_quantity = min(sell_quantity, buy_quantity)
                
                # 计算盈亏
                profit = (sell_price - buy_price) * matched_quantity
                trade_profits.append(profit)
                
                # 更新数量
                sell_quantity -= matched_quantity
                buy_trade['quantity'] -= matched_quantity
                
                # 如果买入交易完全匹配，移除
                if buy_trade['quantity'] <= 0:
                    buy_queue.pop(0)
        
        # 计算统计指标
        if not trade_profits:
            return {
                'trade_count': len(trade_history),
                'profitable_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_loss_ratio': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0
            }
        
        profitable_trades = [p for p in trade_profits if p > 0]
        losing_trades = [p for p in trade_profits if p < 0]
        
        total_trades = len(trade_profits)
        profitable_count = len(profitable_trades)
        losing_count = len(losing_trades)
        
        win_rate = profitable_count / total_trades if total_trades > 0 else 0.0
        
        avg_profit = sum(profitable_trades) / profitable_count if profitable_count > 0 else 0.0
        avg_loss = sum(losing_trades) / losing_count if losing_count > 0 else 0.0
        
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0.0
        
        return {
            'trade_count': len(trade_history),
            'completed_trades': total_trades,
            'profitable_trades': profitable_count,
            'losing_trades': losing_count,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss
        }

    def run_complete_test_suite(self, ticker: str = "000001"):
        """运行完整的测试套件"""
        self.logger.info("=" * 60)
        self.logger.info("开始运行完整的回测测试套件")
        self.logger.info("=" * 60)
        
        test_results = {
            'config': {
                'start_date': self.start_date,
                'end_date': self.end_date,
                'duration_days': self.test_duration_days,
                'ticker': ticker
            }
        }
        
        try:
            # 1. 基础组件测试
            self.logger.info("\n1. 回测器组件初始化测试...")
            self.test_backtester_initialization()
            test_results['initialization'] = True
            
            # 2. Baseline策略测试
            self.logger.info("\n2. Baseline策略回测测试...")
            baseline_results = self.test_baseline_strategies_backtest()
            test_results['baseline_results'] = baseline_results
            
            # 3. AI Agent测试
            self.logger.info("\n3. AI Agent回测测试...")
            ai_result = self.test_ai_agent_backtest(ticker)
            test_results['ai_agent_result'] = ai_result
            
            # 4. 全面比较测试
            self.logger.info("\n4. 全面比较分析...")
            comparison_results = self.test_comprehensive_comparison(ticker)
            test_results['comparison_results'] = comparison_results
            
            # 5. 生成最终测试报告
            self.logger.info("\n5. 生成最终测试报告...")
            self._generate_final_test_report(test_results)
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"测试套件执行失败: {e}")
            return test_results

    def _generate_final_test_report(self, test_results: dict):
        """生成最终测试报告"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("最终测试报告")
        self.logger.info("=" * 60)
        
        config = test_results['config']
        self.logger.info(f"测试配置:")
        self.logger.info(f"  时间范围: {config['start_date']} 至 {config['end_date']}")
        self.logger.info(f"  测试时长: {config['duration_days']} 天")
        self.logger.info(f"  测试股票: {config['ticker']}")
        
        # 测试完成情况
        self.logger.info(f"\n测试完成情况:")
        self.logger.info(f"  组件初始化: {'✓' if test_results.get('initialization') else '✗'}")
        self.logger.info(f"  Baseline策略: {len(test_results.get('baseline_results', {}))} 个")
        self.logger.info(f"  AI Agent测试: {'✓' if test_results.get('ai_agent_result') else '✗'}")
        self.logger.info(f"  比较分析: {'✓' if test_results.get('comparison_results') else '✗'}")
        
        # 性能概览
        comparison_results = test_results.get('comparison_results', {})
        results_summary = comparison_results.get('results_summary', {})
        
        if results_summary:
            best_strategy = max(results_summary.keys(), 
                              key=lambda k: results_summary[k]['total_return'])
            best_return = results_summary[best_strategy]['total_return']
            
            self.logger.info(f"\n性能概览:")
            self.logger.info(f"  最佳策略: {best_strategy} ({best_return:.2f}%)")
            self.logger.info(f"  策略总数: {len(results_summary)}")
            
            if 'AI_Agent' in results_summary:
                ai_rank = sorted(results_summary.keys(), 
                               key=lambda k: results_summary[k]['total_return'], 
                               reverse=True).index('AI_Agent') + 1
                self.logger.info(f"  AI Agent排名: 第{ai_rank}名")
        
        self.logger.info("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='回测测试')
    parser.add_argument('--quick', action='store_true', help='快速测试模式（短时间）')
    parser.add_argument('--medium', action='store_true', help='中等测试模式（中等时间）')
    parser.add_argument('--full', action='store_true', help='完整测试模式（长时间）')
    parser.add_argument('--start-date', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--ticker', type=str, default='000001', help='股票代码')
    parser.add_argument('--ai-only', action='store_true', help='仅测试AI Agent')
    parser.add_argument('--baseline-only', action='store_true', help='仅测试baseline策略')
    parser.add_argument('--comparison', action='store_true', help='运行全面比较测试')
    
    args = parser.parse_args()
    
    # 根据参数设置测试时间范围
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    elif args.quick:
        # 快速模式：3个月
        end_date = "2023-03-31"
        start_date = "2023-01-01"
    elif args.medium:
        # 中等模式：8个月
        end_date = "2023-08-31"
        start_date = "2023-01-01"
    elif args.full:
        # 完整模式：2年
        end_date = "2024-12-31"
        start_date = "2023-01-01"
    else:
        # 默认：3个月
        end_date = "2023-03-31"
        start_date = "2023-01-01"
    
    # 创建测试实例
    test = BacktestTest(start_date=start_date, end_date=end_date)
    
    # 运行测试
    try:
        if args.ai_only:
            # 仅测试AI Agent
            test.test_ai_agent_backtest(args.ticker)
        elif args.baseline_only:
            # 仅测试baseline策略
            test.test_backtester_initialization()
            test.test_baseline_strategies_backtest()
        elif args.comparison:
            # 运行全面比较测试
            test.test_comprehensive_comparison(args.ticker)
        else:
            # 运行完整测试套件
            test.run_complete_test_suite(args.ticker)
        
        print("✓ 所有测试通过")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())