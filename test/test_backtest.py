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

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest import Backtester, BacktestConfig
from src.backtest.baselines.buy_hold import BuyHoldStrategy
from src.backtest.baselines.momentum import MomentumStrategy
from src.backtest.baselines.mean_reversion import MeanReversionStrategy
from src.backtest.baselines.moving_average import MovingAverageStrategy
from src.backtest.baselines.random_walk import RandomWalkStrategy
from src.utils.logging_config import setup_logger


class BacktestTest:
    """回测测试类"""
    
    def __init__(self, start_date: str = "2023-01-01", end_date: str = "2023-03-31"):
        self.logger = setup_logger('BacktestTest')
        self.start_date = start_date
        self.end_date = end_date
        self.test_duration_days = self._calculate_test_duration()
        
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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='回测测试')
    parser.add_argument('--quick', action='store_true', help='快速测试模式（短时间）')
    parser.add_argument('--medium', action='store_true', help='中等测试模式（中等时间）')
    parser.add_argument('--full', action='store_true', help='完整测试模式（长时间）')
    parser.add_argument('--start-date', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='结束日期 (YYYY-MM-DD)')
    
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
        test.test_backtester_initialization()
        test.test_baseline_strategies_backtest()
        print("✓ 所有测试通过")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())