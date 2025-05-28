import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import time
import uuid
from dataclasses import dataclass, field

from .baselines.base_strategy import BaseStrategy, Signal, Portfolio
from .baselines.buy_hold import BuyHoldStrategy
from .baselines.momentum import MomentumStrategy
from .baselines.mean_reversion import MeanReversionStrategy
from .baselines.moving_average import MovingAverageStrategy
from .baselines.random_walk import RandomWalkStrategy
from .evaluation.significance import SignificanceTester
from .evaluation.metrics import PerformanceMetrics
from .evaluation.comparison import StrategyComparator
from .evaluation.visualization import BacktestVisualizer
from .execution.trade_executor import TradeExecutor
from .execution.cost_model import CostModel
from .backtest_utils.data_utils import DataProcessor
from .backtest_utils.performance import PerformanceAnalyzer
from src.tools.api import get_price_data
from src.utils.logging_config import setup_logger

@dataclass
class BacktestConfig:
    """回测配置类"""
    initial_capital: float = 100000
    start_date: str = ""
    end_date: str = ""
    benchmark_ticker: str = "000300"  # 沪深300作为基准
    trading_cost: float = 0.001  # 0.1%交易成本
    slippage: float = 0.001      # 0.1%滑点
    num_of_news: int = 5
    rebalance_frequency: str = "day"
    risk_free_rate: float = 0.03
    confidence_level: float = 0.05
    
@dataclass 
class BacktestResult:
    """回测结果类"""
    strategy_name: str
    portfolio_values: List[Dict[str, Any]] = field(default_factory=list)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class Backtester:
    """
    回测引擎主类
    支持多策略比较和统计显著性检验
    """
    
    def __init__(self, agent_function: Optional[Callable] = None, 
                 ticker: str = "", tickers: Optional[List[str]] = None,
                 config: Optional[BacktestConfig] = None, seed: int = 42):
        """
        初始化回测器
        
        Args:
            agent_function: 智能体函数
            ticker: 主要股票代码
            tickers: 多个股票代码列表
            config: 回测配置
            seed: 随机种子，确保结果可重现
        """
        # 设置全局随机种子，确保结果可重现
        np.random.seed(seed)
        
        self.agent_function = agent_function
        self.ticker = ticker
        self.tickers = tickers or ([ticker] if ticker else [])
        self.config = config or BacktestConfig()
        
        # 初始化组件
        self.logger = setup_logger('Backtester')
        self.data_processor = DataProcessor()
        self.trade_executor = TradeExecutor()
        self.cost_model = CostModel(
            trading_cost=self.config.trading_cost,
            slippage=self.config.slippage
        )
        self.performance_analyzer = PerformanceAnalyzer()
        self.significance_tester = SignificanceTester(self.config.confidence_level)
        self.strategy_comparator = StrategyComparator()
        self.visualizer = BacktestVisualizer()
        
        # API调用管理
        self._api_call_count = 0
        self._api_window_start = time.time()
        self._last_api_call = 0
        
        # 结果存储
        self.results: Dict[str, BacktestResult] = {}
        self.baseline_strategies: List[BaseStrategy] = []
        self.comparison_results: Dict[str, Any] = {}
        
        # 验证配置
        self._validate_config()
        
        # 初始化基准策略
        self.initialize_baseline_strategies()
        
    def _validate_config(self):
        """验证配置参数"""
        try:
            start = datetime.strptime(self.config.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.config.end_date, "%Y-%m-%d")
            if start >= end:
                raise ValueError("开始日期必须早于结束日期")
            if self.config.initial_capital <= 0:
                raise ValueError("初始资金必须大于0")
            self.logger.info("配置参数验证通过")
        except Exception as e:
            self.logger.error(f"配置参数验证失败: {str(e)}")
            raise
    
    def initialize_baseline_strategies(self) -> List[BaseStrategy]:
        """
        初始化baseline策略
        基于学术研究的经典策略集合
        """
        strategies = [
            # 1. 买入持有策略
            BuyHoldStrategy(allocation_ratio=1.0),
            
            # 2. 动量策略（多参数组合）
            MomentumStrategy(
                lookback_period=252, 
                formation_period=63, 
                holding_period=21,
                momentum_threshold=0.02
            ),
            MomentumStrategy(
                lookback_period=252, 
                formation_period=126, 
                holding_period=42,
                momentum_threshold=0.05,
                name="Momentum-Long"
            ),
            
            # 3. 均值回归策略（多参数组合）
            MeanReversionStrategy(
                lookback_period=252,
                z_threshold=2.0,
                mean_period=50,
                exit_threshold=0.5
            ),
            MeanReversionStrategy(
                lookback_period=252,
                z_threshold=1.5,
                mean_period=20,
                exit_threshold=0.3,
                name="Mean-Reversion-Short"
            ),
            
            # 4. 移动平均策略（多组合）
            MovingAverageStrategy(
                short_window=50,
                long_window=200,
                signal_threshold=0.02
            ),
            MovingAverageStrategy(
                short_window=20,
                long_window=60,
                signal_threshold=0.01,
                name="Moving-Average-Short"
            ),
            
            # 5. 随机游走策略（控制组）- 真正的随机性
            RandomWalkStrategy(
                trade_probability=0.1,
                max_position_ratio=0.5,
                truly_random=True  # 使用真随机，每次运行结果不同
            )
        ]
        
        # 初始化策略
        for strategy in strategies:
            strategy.initialize(self.config.initial_capital)
            
        self.baseline_strategies = strategies
        self.logger.info(f"初始化了{len(strategies)}个baseline策略")
        
        return strategies
    
    def run_agent_backtest(self) -> BacktestResult:
        """运行智能体回测"""
        if not self.agent_function:
            raise ValueError("未提供智能体函数")
            
        self.logger.info("开始运行智能体回测...")
        
        # 初始化投资组合
        portfolio = {
            "cash": self.config.initial_capital,
            "stock": 0
        }
        
        # 获取交易日期
        dates = pd.date_range(
            self.config.start_date, 
            self.config.end_date, 
            freq="B"
        )
        
        portfolio_values = []
        trade_history = []
        daily_returns = []
        
        for current_date in dates:
            try:
                # 获取智能体决策
                decision_data = self._get_agent_decision(
                    current_date.strftime("%Y-%m-%d"),
                    portfolio
                )
                
                # 获取当前价格
                current_price = self._get_current_price(
                    current_date.strftime("%Y-%m-%d")
                )
                
                if current_price is None:
                    continue
                
                # 执行交易
                executed_trades = self.trade_executor.execute_trade(
                    decision_data,
                    portfolio,
                    current_price,
                    current_date.strftime("%Y-%m-%d"),
                    self.cost_model
                )
                
                # 更新投资组合 - 交易执行器已经更新了portfolio
                trade_history.extend(executed_trades)
                
                # 计算投资组合价值
                total_value = portfolio["cash"] + portfolio["stock"] * current_price
                portfolio["portfolio_value"] = total_value
                
                # 计算日收益率
                if len(portfolio_values) > 0:
                    prev_value = portfolio_values[-1]["Portfolio Value"]
                    daily_return = (total_value / prev_value - 1) if prev_value > 0 else 0
                else:
                    daily_return = 0
                
                daily_returns.append(daily_return)
                
                # 记录投资组合价值
                portfolio_values.append({
                    "Date": current_date,
                    "Portfolio Value": total_value,
                    "Daily Return": daily_return,
                    "Cash": portfolio["cash"],
                    "Stock": portfolio["stock"]
                })
                
            except Exception as e:
                self.logger.error(f"处理日期 {current_date} 时出错: {e}")
                continue
        
        # 计算性能指标
        metrics = PerformanceMetrics()
        performance_metrics = metrics.calculate_all_metrics(
            np.array(daily_returns),
            self.config.risk_free_rate
        )
        
        result = BacktestResult(
            strategy_name="AI Agent",
            portfolio_values=portfolio_values,
            trade_history=trade_history,
            daily_returns=np.array(daily_returns),
            performance_metrics=performance_metrics
        )
        
        self.results["AI Agent"] = result
        self.logger.info("智能体回测完成")
        
        return result
    
    def run_baseline_backtests(self) -> Dict[str, BacktestResult]:
        """运行所有baseline策略回测"""
        if not self.baseline_strategies:
            self.initialize_baseline_strategies()
        
        self.logger.info(f"开始运行{len(self.baseline_strategies)}个baseline策略回测...")
        
        baseline_results = {}
        
        for strategy in self.baseline_strategies:
            self.logger.info(f"运行策略: {strategy.name}")
            
            try:
                result = self._run_single_strategy_backtest(strategy)
                baseline_results[strategy.name] = result
                self.results[strategy.name] = result
                
            except Exception as e:
                self.logger.error(f"策略 {strategy.name} 回测失败: {e}")
                continue
        
        self.logger.info(f"完成{len(baseline_results)}个baseline策略回测")
        return baseline_results
    
    def _run_single_strategy_backtest(self, strategy: BaseStrategy) -> BacktestResult:
        """运行单个策略回测"""
        # 重置策略状态
        strategy.reset()
        strategy.initialize(self.config.initial_capital)
        
        # 初始化投资组合
        portfolio_obj = Portfolio(
            cash=self.config.initial_capital,
            stock=0,
            total_value=self.config.initial_capital
        )
        
        # 获取交易日期
        dates = pd.date_range(
            self.config.start_date, 
            self.config.end_date, 
            freq="B"
        )
        
        # 根据策略类型确定所需的历史数据长度
        required_lookback_days = self._get_strategy_lookback_days(strategy)
        
        portfolio_values = []
        trade_history = []
        daily_returns = []
        
        for i, current_date in enumerate(dates):
            try:
                # 动态计算历史数据起始日期
                lookback_start = (current_date - timedelta(days=required_lookback_days)).strftime("%Y-%m-%d")
                current_date_str = current_date.strftime("%Y-%m-%d")
                
                data = self._get_price_data(lookback_start, current_date_str)
                if data is None or data.empty:
                    continue
                
                current_price = data.iloc[-1]['close']
                
                # 生成交易信号
                signal = strategy.generate_signal(
                    data, 
                    portfolio_obj, 
                    current_date_str
                )
                
                # 记录信号
                strategy.record_signal(signal, current_date_str, current_price)
                
                # 执行交易
                executed_quantity = self._execute_strategy_trade(
                    signal, 
                    portfolio_obj, 
                    current_price, 
                    current_date_str,
                    strategy
                )
                
                # 更新投资组合价值
                total_value = portfolio_obj.cash + portfolio_obj.stock * current_price
                portfolio_obj.total_value = total_value
                
                # 计算日收益率
                if len(portfolio_values) > 0:
                    prev_value = portfolio_values[-1]["Portfolio Value"]
                    daily_return = (total_value / prev_value - 1) if prev_value > 0 else 0
                else:
                    daily_return = 0
                
                daily_returns.append(daily_return)
                
                # 记录投资组合价值
                portfolio_values.append({
                    "Date": current_date,
                    "Portfolio Value": total_value,
                    "Daily Return": daily_return,
                    "Cash": portfolio_obj.cash,
                    "Stock": portfolio_obj.stock,
                    "Signal": signal.action,
                    "Confidence": signal.confidence
                })
                
            except Exception as e:
                self.logger.warning(f"策略 {strategy.name} 在日期 {current_date} 处理失败: {e}")
                continue
        
        # 计算性能指标
        metrics = PerformanceMetrics()
        performance_metrics = metrics.calculate_all_metrics(
            np.array(daily_returns),
            self.config.risk_free_rate
        )
        
        result = BacktestResult(
            strategy_name=strategy.name,
            portfolio_values=portfolio_values,
            trade_history=strategy.trade_history,
            daily_returns=np.array(daily_returns),
            performance_metrics=performance_metrics,
            metadata={
                'strategy_type': getattr(strategy, 'strategy_type', 'unknown'),
                'parameters': strategy.parameters,
                'total_signals': len(strategy.signal_history)
            }
        )
        
        # Store result in results dictionary
        self.results[strategy.name] = result
        
        return result
    
    def _execute_strategy_trade(self, signal: Signal, portfolio: Portfolio, 
                              current_price: float, date: str, 
                              strategy: BaseStrategy) -> int:
        """执行策略交易"""
        executed_quantity = 0
        
        if signal.action == "buy" and signal.quantity > 0:
            # 计算交易成本
            cost_without_fees = signal.quantity * current_price
            total_cost = self.cost_model.calculate_total_cost(
                cost_without_fees, 'buy'
            )
            
            if total_cost <= portfolio.cash:
                portfolio.stock += signal.quantity
                portfolio.cash -= total_cost
                executed_quantity = signal.quantity
                
                # 记录交易
                strategy.record_trade(
                    'buy', signal.quantity, current_price, date,
                    cost_without_fees, total_cost - cost_without_fees
                )
            else:
                # 计算最大可买数量
                max_quantity = int(portfolio.cash / (current_price * (1 + self.config.trading_cost + self.config.slippage)))
                if max_quantity > 0:
                    cost_without_fees = max_quantity * current_price
                    total_cost = self.cost_model.calculate_total_cost(
                        cost_without_fees, 'buy'
                    )
                    
                    portfolio.stock += max_quantity
                    portfolio.cash -= total_cost
                    executed_quantity = max_quantity
                    
                    strategy.record_trade(
                        'buy', max_quantity, current_price, date,
                        cost_without_fees, total_cost - cost_without_fees
                    )
        
        elif signal.action == "sell" and signal.quantity > 0:
            quantity = min(signal.quantity, portfolio.stock)
            if quantity > 0:
                value_without_fees = quantity * current_price
                net_value = self.cost_model.calculate_net_proceeds(
                    value_without_fees, 'sell'
                )
                
                portfolio.cash += net_value
                portfolio.stock -= quantity
                executed_quantity = quantity
                
                strategy.record_trade(
                    'sell', quantity, current_price, date,
                    value_without_fees, value_without_fees - net_value
                )
        
        return executed_quantity
    
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """运行综合比较分析"""
        self.logger.info("开始综合比较分析...")
        
        if len(self.results) < 2:
            raise ValueError("需要至少2个策略结果进行比较")
        
        # 获取所有策略的收益率
        strategy_returns = {}
        for name, result in self.results.items():
            strategy_returns[name] = result.daily_returns
        
        # 两两比较所有策略
        pairwise_comparisons = {}
        strategy_names = list(self.results.keys())
        
        for i in range(len(strategy_names)):
            for j in range(i + 1, len(strategy_names)):
                name1, name2 = strategy_names[i], strategy_names[j]
                
                comparison_key = f"{name1}_vs_{name2}"
                
                # 进行统计显著性检验
                comparison_result = self.significance_tester.comprehensive_comparison(
                    strategy_returns[name1],
                    strategy_returns[name2],
                    strategy1_name=name1,
                    strategy2_name=name2
                )
                
                pairwise_comparisons[comparison_key] = comparison_result
        
        # 策略排名分析
        ranking_analysis = self.strategy_comparator.rank_strategies(self.results)
        
        # 生成综合报告
        comprehensive_report = {
            'pairwise_comparisons': pairwise_comparisons,
            'strategy_ranking': ranking_analysis,
            'performance_summary': self._generate_performance_summary(),
            'statistical_summary': self._generate_statistical_summary(pairwise_comparisons),
            'recommendations': self._generate_recommendations(ranking_analysis, pairwise_comparisons)
        }
        
        self.comparison_results = comprehensive_report
        self.logger.info("综合比较分析完成")
        
        return comprehensive_report
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """生成性能摘要"""
        summary = {}
        
        for name, result in self.results.items():
            metrics = result.performance_metrics
            summary[name] = {
                'total_return': metrics.get('total_return', 0),
                'annual_return': metrics.get('annual_return', 0),
                'volatility': metrics.get('volatility', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'total_trades': len(result.trade_history)
            }
        
        return summary
    
    def _generate_statistical_summary(self, pairwise_comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """生成统计摘要"""
        significant_pairs = []
        inconclusive_pairs = []
        
        for pair_name, comparison in pairwise_comparisons.items():
            if comparison['summary']['statistical_power'] > 0.5:
                significant_pairs.append({
                    'pair': pair_name,
                    'conclusion': comparison['summary']['overall_conclusion'],
                    'power': comparison['summary']['statistical_power']
                })
            else:
                inconclusive_pairs.append({
                    'pair': pair_name,
                    'reason': '统计功效不足',
                    'power': comparison['summary']['statistical_power']
                })
        
        return {
            'significant_differences': significant_pairs,
            'inconclusive_results': inconclusive_pairs,
            'total_comparisons': len(pairwise_comparisons),
            'significant_ratio': len(significant_pairs) / len(pairwise_comparisons)
        }
    
    def _generate_recommendations(self, ranking: Dict[str, Any], 
                                comparisons: Dict[str, Any]) -> List[str]:
        """生成投资建议"""
        recommendations = []
        
        # 基于排名的建议
        if 'by_sharpe' in ranking and ranking['by_sharpe']:
            top_strategy = ranking['by_sharpe'][0]
            recommendations.append(
                f"基于夏普比率排名，推荐使用 {top_strategy['strategy']} 策略 "
                f"(夏普比率: {top_strategy['sharpe_ratio']:.3f})"
            )
        elif 'top_performers' in ranking and ranking['top_performers']:
            top_strategy = ranking['top_performers'][0]
            recommendations.append(
                f"基于综合表现，推荐使用 {top_strategy['name']} 策略 "
                f"(综合得分: {top_strategy['composite_score']:.3f})"
            )
        
        # 基于风险调整收益的建议
        if self.results:
            best_sharpe = max(self.results.items(), 
                             key=lambda x: x[1].performance_metrics.get('sharpe_ratio', 0))
            recommendations.append(
                f"基于风险调整收益，{best_sharpe[0]} 策略具有最佳夏普比率 "
                f"({best_sharpe[1].performance_metrics.get('sharpe_ratio', 0):.3f})"
            )
        
        # 基于统计显著性的建议
        significant_winners = []
        for comparison in comparisons.values():
            try:
                if ('summary' in comparison and 
                    comparison['summary'].get('statistical_power', 0) > 0.7):
                    conclusion = comparison['summary'].get('overall_conclusion', '')
                    if '显著优于' in conclusion:
                        winner = conclusion.split('显著优于')[0].strip()
                        significant_winners.append(winner)
            except (KeyError, AttributeError):
                continue
        
        if significant_winners:
            from collections import Counter
            most_common_winner = Counter(significant_winners).most_common(1)[0][0]
            recommendations.append(
                f"基于统计显著性检验，{most_common_winner} 在多项比较中表现优异"
            )
        
        return recommendations
    
    def generate_visualization(self) -> Dict[str, str]:
        """生成可视化图表"""
        return self.visualizer.create_comparison_charts(
            self.results, 
            self.comparison_results,
            save_dir=f"assets/img/backtest_{self.ticker}_{datetime.now().strftime('%Y%m%d')}"
        )
    
    def _get_agent_decision(self, current_date: str, portfolio: dict) -> dict:
        """获取智能体决策"""
        # API限制处理
        self._handle_api_limits()
        
        try:
            # 增加回看窗口以支持区制检测 - 至少需要100天数据
            # 区制检测需要足够的历史数据来计算技术指标和特征
            lookback_days = 100  # 从30天增加到100天
            lookback_start = (pd.to_datetime(current_date) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            result = self.agent_function(
                run_id=str(uuid.uuid4()),
                ticker=self.ticker,
                start_date=lookback_start,
                end_date=current_date,
                portfolio=portfolio,
                num_of_news=self.config.num_of_news,
                tickers=self.tickers
            )
            
            # 解析结果
            if isinstance(result, dict) and "decision" in result:
                return result["decision"]
            elif isinstance(result, str):
                import json
                parsed = json.loads(result.replace('```json\n', '').replace('\n```', '').strip())
                return parsed.get("decision", {"action": "hold", "quantity": 0})
            else:
                return {"action": "hold", "quantity": 0}
                
        except Exception as e:
            self.logger.warning(f"获取智能体决策失败: {e}")
            return {"action": "hold", "quantity": 0}
    
    def _handle_api_limits(self):
        """处理API限制"""
        current_time = time.time()
        
        # 重置时间窗口
        if current_time - self._api_window_start >= 60:
            self._api_call_count = 0
            self._api_window_start = current_time
        
        # 检查API限制
        if self._api_call_count >= 8:
            wait_time = 60 - (current_time - self._api_window_start)
            if wait_time > 0:
                time.sleep(wait_time)
                self._api_call_count = 0
                self._api_window_start = time.time()
        
        # 确保调用间隔
        if self._last_api_call:
            time_since_last = time.time() - self._last_api_call
            if time_since_last < 3:
                time.sleep(3 - time_since_last)
        
        self._last_api_call = time.time()
        self._api_call_count += 1
    
    def _get_current_price(self, date: str) -> Optional[float]:
        """获取当前价格"""
        try:
            data = get_price_data(self.ticker, date, date)
            if data is not None and not data.empty:
                return data.iloc[-1]['close']
        except Exception as e:
            self.logger.warning(f"获取价格数据失败 {date}: {e}")
        return None
    
    def _get_price_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取价格数据"""
        try:
            return get_price_data(self.ticker, start_date, end_date)
        except Exception as e:
            self.logger.warning(f"获取价格数据失败 {start_date}-{end_date}: {e}")
            return None
    
    def save_results(self, filepath: str):
        """保存回测结果"""
        import pickle
        
        results_data = {
            'config': self.config,
            'results': self.results,
            'comparison_results': self.comparison_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results_data, f)
        
        self.logger.info(f"回测结果已保存至: {filepath}")
    
    def load_results(self, filepath: str):
        """加载回测结果"""
        import pickle
        
        with open(filepath, 'rb') as f:
            results_data = pickle.load(f)
        
        self.config = results_data['config']
        self.results = results_data['results']
        self.comparison_results = results_data.get('comparison_results', {})
        
        self.logger.info(f"回测结果已从 {filepath} 加载")
    
    def export_report(self, format: str = 'html') -> str:
        """导出回测报告"""
        if format == 'html':
            return self._generate_html_report()
        elif format == 'pdf':
            return self._generate_pdf_report()
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def _generate_html_report(self) -> str:
        """生成HTML报告"""
        # 使用模板引擎生成详细的HTML报告
        # 包含所有图表、统计结果和分析
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>回测分析报告</title>
            <meta charset="utf-8">
        </head>
        <body>
            <h1>A股投资Agent系统回测分析报告</h1>
            <h2>配置信息</h2>
            <p>股票代码: {self.ticker}</p>
            <p>回测期间: {self.config.start_date} 至 {self.config.end_date}</p>
            <p>初始资金: {self.config.initial_capital:,.2f}</p>
            
            <h2>策略表现摘要</h2>
            {self._format_performance_table()}
            
            <h2>统计显著性检验结果</h2>
            {self._format_significance_results()}
            
            <h2>投资建议</h2>
            {self._format_recommendations()}
        </body>
        </html>
        """
        
        filename = f"backtest_report_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename
    
    def _format_performance_table(self) -> str:
        """格式化性能表格"""
        if not hasattr(self, 'comparison_results') or not self.comparison_results:
            return "<p>暂无性能数据</p>"
        
        performance = self.comparison_results.get('performance_summary', {})
        
        table_html = "<table border='1'><tr>"
        table_html += "<th>策略</th><th>总收益率</th><th>年化收益率</th><th>波动率</th><th>夏普比率</th><th>最大回撤</th></tr>"
        
        for strategy, metrics in performance.items():
            table_html += f"<tr>"
            table_html += f"<td>{strategy}</td>"
            table_html += f"<td>{metrics.get('total_return', 0)*100:.2f}%</td>"
            table_html += f"<td>{metrics.get('annual_return', 0)*100:.2f}%</td>"
            table_html += f"<td>{metrics.get('volatility', 0)*100:.2f}%</td>"
            table_html += f"<td>{metrics.get('sharpe_ratio', 0):.3f}</td>"
            table_html += f"<td>{metrics.get('max_drawdown', 0)*100:.2f}%</td>"
            table_html += f"</tr>"
        
        table_html += "</table>"
        return table_html
    
    def _format_significance_results(self) -> str:
        """格式化显著性检验结果"""
        if not hasattr(self, 'comparison_results') or not self.comparison_results:
            return "<p>暂无统计检验结果</p>"
        
        stats_summary = self.comparison_results.get('statistical_summary', {})
        significant = stats_summary.get('significant_differences', [])
        
        if not significant:
            return "<p>未发现策略间的显著差异</p>"
        
        html = "<ul>"
        for item in significant:
            html += f"<li>{item['conclusion']} (统计功效: {item['power']:.3f})</li>"
        html += "</ul>"
        
        return html
    
    def _format_recommendations(self) -> str:
        """格式化投资建议"""
        if not hasattr(self, 'comparison_results') or not self.comparison_results:
            return "<p>暂无投资建议</p>"
        
        recommendations = self.comparison_results.get('recommendations', [])
        
        if not recommendations:
            return "<p>暂无具体建议</p>"
        
        html = "<ol>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ol>"
        
        return html
    
    def _generate_pdf_report(self) -> str:
        """生成PDF报告"""
        # 暂时返回HTML文件名
        return self._generate_html_report()
    
    def _get_strategy_lookback_days(self, strategy: BaseStrategy) -> int:
        """根据策略类型确定所需的历史数据天数"""
        strategy_name = strategy.name.lower()
        
        # 根据策略类型设置不同的历史数据需求
        if 'momentum' in strategy_name:
            return 400  # 动量策略需要更多历史数据
        elif 'moving-average' in strategy_name:
            # 移动平均策略根据窗口大小确定
            if hasattr(strategy, 'long_window'):
                return max(400, strategy.long_window + 100)  # 长窗口 + 缓冲
            else:
                return 300
        elif 'mean-reversion' in strategy_name:
            return 400  # 均值回归策略需要足够的历史数据
        else:
            return 365  # 默认一年历史数据