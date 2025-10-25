import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.tools.api import get_price_data
from src.utils.logging_config import setup_logger

from .backtest_utils.data_utils import DataProcessor
from .backtest_utils.performance import PerformanceAnalyzer
from .baselines.base_strategy import BaseStrategy, Portfolio, Signal
from .baselines.bollinger_strategy import BollingerStrategy
from .baselines.buy_hold import BuyHoldStrategy
from .baselines.macd_strategy import MACDStrategy
from .baselines.mean_reversion import MeanReversionStrategy
from .baselines.momentum import MomentumStrategy
from .baselines.moving_average import MovingAverageStrategy
from .baselines.random_walk import RandomWalkStrategy
from .baselines.rsi_strategy import RSIStrategy
from .evaluation.comparison import StrategyComparator
from .evaluation.metrics import PerformanceMetrics
from .evaluation.significance import SignificanceTester
from .evaluation.visualization import BacktestVisualizer
from .execution.cost_model import CostModel
from .execution.trade_executor import TradeExecutor


@dataclass
class BacktestConfig:
    """Backtest configuration class"""

    initial_capital: float = 100000
    start_date: str = ""
    end_date: str = ""
    benchmark_ticker: str = "000300"  # CSI 300 as benchmark
    trading_cost: float = 0.001  # 0.1% trading cost
    slippage: float = 0.001  # 0.1% slippage
    num_of_news: int = 5
    rebalance_frequency: str = "day"
    risk_free_rate: float = 0.03
    confidence_level: float = 0.05
    use_cache: bool = True  # Whether to use data cache
    cache_ttl_days: int = 3  # Cache validity period (days)


@dataclass
class BacktestResult:
    """Backtest result class"""

    strategy_name: str
    portfolio_values: List[Dict[str, Any]] = field(default_factory=list)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Backtester:
    """
    Main backtest engine class
    Supports multi-strategy comparison and statistical significance testing
    """

    def __init__(
        self,
        agent_function: Optional[Callable] = None,
        ticker: str = "",
        tickers: Optional[List[str]] = None,
        config: Optional[BacktestConfig] = None,
        seed: int = 42,
    ):
        """
        Initialize backtester

        Args:
            agent_function: Agent function
            ticker: Main stock ticker
            tickers: List of multiple stock tickers
            config: Backtest configuration
            seed: Random seed to ensure reproducible results
        """
        # Set global random seed to ensure reproducible results
        np.random.seed(seed)

        self.agent_function = agent_function
        self.ticker = ticker
        self.tickers = tickers or ([ticker] if ticker else [])
        self.config = config or BacktestConfig()

        # Initialize components
        self.logger = setup_logger("Backtester")
        self.data_processor = DataProcessor()
        self.trade_executor = TradeExecutor()
        self.cost_model = CostModel(
            trading_cost=self.config.trading_cost, slippage=self.config.slippage
        )
        self.performance_analyzer = PerformanceAnalyzer()
        self.significance_tester = SignificanceTester(self.config.confidence_level)
        self.strategy_comparator = StrategyComparator()
        self.visualizer = BacktestVisualizer()

        # API call management
        self._api_call_count = 0
        self._api_window_start = time.time()
        self._last_api_call = 0

        # Result storage
        self.results: Dict[str, BacktestResult] = {}
        self.baseline_strategies: List[BaseStrategy] = []
        self.comparison_results: Dict[str, Any] = {}

        # Validate configuration
        self._validate_config()

        # Initialize baseline strategies
        self.initialize_baseline_strategies()

    def _validate_config(self):
        """Validate configuration parameters"""
        try:
            start = datetime.strptime(self.config.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.config.end_date, "%Y-%m-%d")
            if start >= end:
                raise ValueError("Start date must be earlier than end date")
            if self.config.initial_capital <= 0:
                raise ValueError("Initial capital must be greater than 0")
            self.logger.info("Configuration parameters validated successfully")
        except Exception as e:
            self.logger.error(f"Configuration parameter validation failed: {str(e)}")
            raise

    def initialize_baseline_strategies(self) -> List[BaseStrategy]:
        """
        Initialize baseline strategies
        Classic strategy collection based on academic research
        """
        strategies = [
            # 1. Buy and hold strategy
            BuyHoldStrategy(allocation_ratio=1.0),
            # 2. Momentum strategy (multi-parameter combination)
            MomentumStrategy(
                lookback_period=252,
                formation_period=63,
                holding_period=21,
                momentum_threshold=0.02,
            ),
            MomentumStrategy(
                lookback_period=252,
                formation_period=126,
                holding_period=42,
                momentum_threshold=0.05,
                name="Momentum-Long",
            ),
            # 3. Mean reversion strategy (multi-parameter combination)
            MeanReversionStrategy(
                lookback_period=252, z_threshold=2.0, mean_period=50, exit_threshold=0.5
            ),
            MeanReversionStrategy(
                lookback_period=252,
                z_threshold=1.5,
                mean_period=20,
                exit_threshold=0.3,
                name="Mean-Reversion-Short",
            ),
            # 4. Moving average strategy (optimized parameters)
            MovingAverageStrategy(
                short_window=20, long_window=60, signal_threshold=0.01
            ),
            MovingAverageStrategy(
                short_window=10,
                long_window=30,
                signal_threshold=0.005,
                name="Moving-Average-Short",
            ),
            # 5. RSI strategy (multi-parameter combination)
            RSIStrategy(rsi_period=14, overbought=70, oversold=30, name="RSI-Standard"),
            # 6. Bollinger Bands strategy (mean reversion and breakout modes)
            BollingerStrategy(
                period=20,
                std_dev=2.0,
                strategy_mode="mean_reversion",
                name="Bollinger-MeanReversion",
            ),
            BollingerStrategy(
                period=20,
                std_dev=2.0,
                strategy_mode="breakout",
                name="Bollinger-Breakout",
            ),
            # 7. MACD strategy (multi-parameter combination)
            MACDStrategy(
                fast_period=12, slow_period=26, signal_period=9, name="MACD-Standard"
            ),
            MACDStrategy(
                fast_period=8, slow_period=21, signal_period=5, name="MACD-Fast"
            ),
            # 8. Random walk strategy (control group) - true randomness
            RandomWalkStrategy(
                trade_probability=0.1,
                max_position_ratio=0.5,
                truly_random=True,  # Use true randomness, different results each run
            ),
        ]

        # Initialize strategies
        for strategy in strategies:
            strategy.initialize(self.config.initial_capital)

        self.baseline_strategies = strategies
        self.logger.info(f"Initialized {len(strategies)} baseline strategies")

        return strategies

    def run_agent_backtest(self) -> BacktestResult:
        """Run agent backtest"""
        if not self.agent_function:
            raise ValueError("Agent function not provided")

        self.logger.info("Starting agent backtest...")

        # Initialize portfolio
        portfolio = {"cash": self.config.initial_capital, "stock": 0}

        # Get trading dates
        dates = pd.date_range(self.config.start_date, self.config.end_date, freq="B")

        portfolio_values = []
        trade_history = []
        daily_returns = []

        # Add debug counters
        decision_count = 0
        trade_count = 0

        for current_date in dates:
            try:
                # Get agent decision
                decision_data = self._get_agent_decision(
                    current_date.strftime("%Y-%m-%d"), portfolio
                )

                decision_count += 1

                # Debug output
                if decision_count <= 5:  # Only show first 5 decisions
                    self.logger.info(
                        f"Date {current_date.strftime('%Y-%m-%d')}: Decision = {decision_data}"
                    )

                # Get current price
                current_price = self._get_current_price(
                    current_date.strftime("%Y-%m-%d")
                )

                if current_price is None:
                    self.logger.warning(
                        f"Unable to get price data for {current_date.strftime('%Y-%m-%d')}"
                    )
                    continue

                # Execute trades
                executed_trades = self.trade_executor.execute_trade(
                    decision_data,
                    portfolio,
                    current_price,
                    current_date.strftime("%Y-%m-%d"),
                    self.cost_model,
                )

                # Record trades
                if executed_trades:
                    trade_count += len(executed_trades)
                    trade_history.extend(executed_trades)
                    self.logger.info(f"Executed trades: {executed_trades}")

                # Calculate portfolio value
                total_value = portfolio["cash"] + portfolio["stock"] * current_price
                portfolio["portfolio_value"] = total_value

                # Calculate daily return
                if len(portfolio_values) > 0:
                    prev_value = portfolio_values[-1]["Portfolio Value"]
                    daily_return = (
                        (total_value / prev_value - 1) if prev_value > 0 else 0
                    )
                else:
                    daily_return = 0

                daily_returns.append(daily_return)

                # Record portfolio value
                portfolio_values.append(
                    {
                        "Date": current_date,
                        "Portfolio Value": total_value,
                        "Daily Return": daily_return,
                        "Cash": portfolio["cash"],
                        "Stock": portfolio["stock"],
                    }
                )

            except Exception as e:
                self.logger.error(f"Error processing date {current_date}: {e}")
                continue

        # Summary statistics
        self.logger.info(f"AI Agent backtest completed:")
        self.logger.info(f"  - Decisions processed: {decision_count}")
        self.logger.info(f"  - Trades executed: {trade_count}")
        self.logger.info(f"  - Final cash: {portfolio['cash']:.2f}")
        self.logger.info(f"  - Final stock holdings: {portfolio['stock']}")

        if len(daily_returns) == 0:
            self.logger.warning("No return data generated")
            daily_returns = [0.0]  # Prevent errors from empty array

        # Calculate performance metrics
        metrics = PerformanceMetrics()
        performance_metrics = metrics.calculate_all_metrics(
            np.array(daily_returns), self.config.risk_free_rate
        )

        result = BacktestResult(
            strategy_name="AI Agent",
            portfolio_values=portfolio_values,
            trade_history=trade_history,
            daily_returns=np.array(daily_returns),
            performance_metrics=performance_metrics,
        )

        self.results["AI Agent"] = result
        self.logger.info("Agent backtest completed")

        return result

    def run_baseline_backtests(self) -> Dict[str, BacktestResult]:
        """Run all baseline strategy backtests"""
        if not self.baseline_strategies:
            self.initialize_baseline_strategies()

        self.logger.info(
            f"Starting {len(self.baseline_strategies)} baseline strategy backtests..."
        )

        baseline_results = {}

        for strategy in self.baseline_strategies:
            self.logger.info(f"Running strategy: {strategy.name}")

            try:
                result = self._run_single_strategy_backtest(strategy)
                baseline_results[strategy.name] = result
                self.results[strategy.name] = result

            except Exception as e:
                self.logger.error(f"Strategy {strategy.name} backtest failed: {e}")
                continue

        self.logger.info(
            f"Completed {len(baseline_results)} baseline strategy backtests"
        )
        return baseline_results

    def _run_single_strategy_backtest(self, strategy: BaseStrategy) -> BacktestResult:
        """Run single strategy backtest"""
        # Reset strategy state
        strategy.reset()
        strategy.initialize(self.config.initial_capital)

        # Initialize portfolio
        portfolio_obj = Portfolio(
            cash=self.config.initial_capital,
            stock=0,
            total_value=self.config.initial_capital,
        )

        # Get trading dates
        dates = pd.date_range(self.config.start_date, self.config.end_date, freq="B")

        # Determine required historical data length based on strategy type
        required_lookback_days = self._get_strategy_lookback_days(strategy)

        portfolio_values = []
        trade_history = []
        daily_returns = []

        for i, current_date in enumerate(dates):
            try:
                # Dynamically calculate historical data start date
                lookback_start = (
                    current_date - timedelta(days=required_lookback_days)
                ).strftime("%Y-%m-%d")
                current_date_str = current_date.strftime("%Y-%m-%d")

                data = self._get_price_data(lookback_start, current_date_str)
                if data is None or data.empty:
                    continue

                current_price = data.iloc[-1]["close"]

                # Generate trading signal
                signal = strategy.generate_signal(data, portfolio_obj, current_date_str)

                # Record signal
                strategy.record_signal(signal, current_date_str, current_price)

                # Execute trade
                executed_quantity = self._execute_strategy_trade(
                    signal, portfolio_obj, current_price, current_date_str, strategy
                )

                # Update portfolio value
                total_value = portfolio_obj.cash + portfolio_obj.stock * current_price
                portfolio_obj.total_value = total_value

                # Calculate daily return
                if len(portfolio_values) > 0:
                    prev_value = portfolio_values[-1]["Portfolio Value"]
                    daily_return = (
                        (total_value / prev_value - 1) if prev_value > 0 else 0
                    )
                else:
                    daily_return = 0

                daily_returns.append(daily_return)

                # Record portfolio value
                portfolio_values.append(
                    {
                        "Date": current_date,
                        "Portfolio Value": total_value,
                        "Daily Return": daily_return,
                        "Cash": portfolio_obj.cash,
                        "Stock": portfolio_obj.stock,
                        "Signal": signal.action,
                        "Confidence": signal.confidence,
                    }
                )

            except Exception as e:
                self.logger.warning(
                    f"Strategy {strategy.name} failed to process date {current_date}: {e}"
                )
                continue

        # Calculate performance metrics
        metrics = PerformanceMetrics()
        performance_metrics = metrics.calculate_all_metrics(
            np.array(daily_returns), self.config.risk_free_rate
        )

        result = BacktestResult(
            strategy_name=strategy.name,
            portfolio_values=portfolio_values,
            trade_history=strategy.trade_history,
            daily_returns=np.array(daily_returns),
            performance_metrics=performance_metrics,
            metadata={
                "strategy_type": getattr(strategy, "strategy_type", "unknown"),
                "parameters": strategy.parameters,
                "total_signals": len(strategy.signal_history),
            },
        )

        # Store result in results dictionary
        self.results[strategy.name] = result

        return result

    def _execute_strategy_trade(
        self,
        signal: Signal,
        portfolio: Portfolio,
        current_price: float,
        date: str,
        strategy: BaseStrategy,
    ) -> int:
        """Execute strategy trade"""
        executed_quantity = 0

        if signal.action == "buy" and signal.quantity > 0:
            # Calculate trading cost
            cost_without_fees = signal.quantity * current_price
            total_cost = self.cost_model.calculate_total_cost(cost_without_fees, "buy")

            if total_cost <= portfolio.cash:
                portfolio.stock += signal.quantity
                portfolio.cash -= total_cost
                executed_quantity = signal.quantity

                # Record trade
                strategy.record_trade(
                    "buy",
                    signal.quantity,
                    current_price,
                    date,
                    cost_without_fees,
                    total_cost - cost_without_fees,
                )
            else:
                # Calculate maximum buyable quantity
                max_quantity = int(
                    portfolio.cash
                    / (
                        current_price
                        * (1 + self.config.trading_cost + self.config.slippage)
                    )
                )
                if max_quantity > 0:
                    cost_without_fees = max_quantity * current_price
                    total_cost = self.cost_model.calculate_total_cost(
                        cost_without_fees, "buy"
                    )

                    portfolio.stock += max_quantity
                    portfolio.cash -= total_cost
                    executed_quantity = max_quantity

                    strategy.record_trade(
                        "buy",
                        max_quantity,
                        current_price,
                        date,
                        cost_without_fees,
                        total_cost - cost_without_fees,
                    )

        elif signal.action == "sell" and signal.quantity > 0:
            quantity = min(signal.quantity, portfolio.stock)
            if quantity > 0:
                value_without_fees = quantity * current_price
                net_value = self.cost_model.calculate_net_proceeds(
                    value_without_fees, "sell"
                )

                portfolio.cash += net_value
                portfolio.stock -= quantity
                executed_quantity = quantity

                strategy.record_trade(
                    "sell",
                    quantity,
                    current_price,
                    date,
                    value_without_fees,
                    value_without_fees - net_value,
                )

        return executed_quantity

    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """Run comprehensive comparison analysis"""
        self.logger.info("Starting comprehensive comparison analysis...")

        if len(self.results) < 2:
            raise ValueError("At least 2 strategy results are required for comparison")

        # Get returns for all strategies
        strategy_returns = {}
        for name, result in self.results.items():
            strategy_returns[name] = result.daily_returns

        # Pairwise comparison of all strategies
        pairwise_comparisons = {}
        strategy_names = list(self.results.keys())

        for i in range(len(strategy_names)):
            for j in range(i + 1, len(strategy_names)):
                name1, name2 = strategy_names[i], strategy_names[j]

                comparison_key = f"{name1}_vs_{name2}"

                # Perform statistical significance testing
                comparison_result = self.significance_tester.comprehensive_comparison(
                    strategy_returns[name1],
                    strategy_returns[name2],
                    strategy1_name=name1,
                    strategy2_name=name2,
                )

                pairwise_comparisons[comparison_key] = comparison_result

        # Strategy ranking analysis
        ranking_analysis = self.strategy_comparator.rank_strategies(self.results)

        # Generate comprehensive report
        comprehensive_report = {
            "pairwise_comparisons": pairwise_comparisons,
            "strategy_ranking": ranking_analysis,
            "performance_summary": self._generate_performance_summary(),
            "statistical_summary": self._generate_statistical_summary(
                pairwise_comparisons
            ),
            "recommendations": self._generate_recommendations(
                ranking_analysis, pairwise_comparisons
            ),
        }

        self.comparison_results = comprehensive_report
        self.logger.info("Comprehensive comparison analysis completed")

        return comprehensive_report

    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        summary = {}

        for name, result in self.results.items():
            metrics = result.performance_metrics
            summary[name] = {
                "total_return": metrics.get("total_return", 0),
                "annual_return": metrics.get("annual_return", 0),
                "volatility": metrics.get("volatility", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "win_rate": metrics.get("win_rate", 0),
                "total_trades": len(result.trade_history),
            }

        return summary

    def _generate_statistical_summary(
        self, pairwise_comparisons: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate statistical summary"""
        significant_pairs = []
        inconclusive_pairs = []

        for pair_name, comparison in pairwise_comparisons.items():
            if comparison["summary"]["statistical_power"] > 0.5:
                significant_pairs.append(
                    {
                        "pair": pair_name,
                        "conclusion": comparison["summary"]["overall_conclusion"],
                        "power": comparison["summary"]["statistical_power"],
                    }
                )
            else:
                inconclusive_pairs.append(
                    {
                        "pair": pair_name,
                        "reason": "Insufficient statistical power",
                        "power": comparison["summary"]["statistical_power"],
                    }
                )

        return {
            "significant_differences": significant_pairs,
            "inconclusive_results": inconclusive_pairs,
            "total_comparisons": len(pairwise_comparisons),
            "significant_ratio": len(significant_pairs) / len(pairwise_comparisons),
        }

    def _generate_recommendations(
        self, ranking: Dict[str, Any], comparisons: Dict[str, Any]
    ) -> List[str]:
        """Generate investment recommendations"""
        recommendations = []

        # Recommendations based on ranking
        if "by_sharpe" in ranking and ranking["by_sharpe"]:
            top_strategy = ranking["by_sharpe"][0]
            recommendations.append(
                f"Based on Sharpe ratio ranking, recommend using {top_strategy['strategy']} strategy "
                f"(Sharpe ratio: {top_strategy['sharpe_ratio']:.3f})"
            )
        elif "top_performers" in ranking and ranking["top_performers"]:
            top_strategy = ranking["top_performers"][0]
            recommendations.append(
                f"Based on comprehensive performance, recommend using {top_strategy['name']} strategy "
                f"(Composite score: {top_strategy['composite_score']:.3f})"
            )

        # Recommendations based on risk-adjusted returns
        if self.results:
            best_sharpe = max(
                self.results.items(),
                key=lambda x: x[1].performance_metrics.get("sharpe_ratio", 0),
            )
            recommendations.append(
                f"Based on risk-adjusted returns, {best_sharpe[0]} strategy has the best Sharpe ratio "
                f"({best_sharpe[1].performance_metrics.get('sharpe_ratio', 0):.3f})"
            )

        # Recommendations based on statistical significance
        significant_winners = []
        for comparison in comparisons.values():
            try:
                if (
                    "summary" in comparison
                    and comparison["summary"].get("statistical_power", 0) > 0.7
                ):
                    conclusion = comparison["summary"].get("overall_conclusion", "")
                    if "significantly outperforms" in conclusion:
                        winner = conclusion.split("significantly outperforms")[
                            0
                        ].strip()
                        significant_winners.append(winner)
            except (KeyError, AttributeError):
                continue

        if significant_winners:
            from collections import Counter

            most_common_winner = Counter(significant_winners).most_common(1)[0][0]
            recommendations.append(
                f"Based on statistical significance tests, {most_common_winner} performs excellently in multiple comparisons"
            )

        return recommendations

    def generate_visualization(self) -> Dict[str, str]:
        """Generate visualization charts"""
        return self.visualizer.create_comparison_charts(
            self.results,
            self.comparison_results,
            save_dir=f"assets/img/backtest_{self.ticker}_{datetime.now().strftime('%Y%m%d')}",
        )

    def _get_agent_decision(self, current_date: str, portfolio: dict) -> dict:
        """Get agent decision"""
        # API limit handling
        self._handle_api_limits()

        try:
            # Increase lookback window to support regime detection - at least 100 days of data needed
            # Regime detection requires sufficient historical data to calculate technical indicators and features
            lookback_days = 100  # Increased from 30 days to 100 days
            lookback_start = (
                pd.to_datetime(current_date) - timedelta(days=lookback_days)
            ).strftime("%Y-%m-%d")

            result = self.agent_function(
                run_id=str(uuid.uuid4()),
                ticker=self.ticker,
                start_date=lookback_start,
                end_date=current_date,
                portfolio=portfolio,
                num_of_news=self.config.num_of_news,
                tickers=self.tickers,
            )

            # Parse result - enhanced parsing logic
            if isinstance(result, dict) and "decision" in result:
                return result["decision"]
            elif isinstance(result, str):
                import json

                try:
                    # Try to parse JSON
                    parsed = json.loads(
                        result.replace("```json\n", "").replace("\n```", "").strip()
                    )

                    # If it's portfolio_management_agent return format
                    if isinstance(parsed, dict):
                        # Check if there's nested decision structure
                        if "decision" in parsed:
                            decision = parsed["decision"]
                        else:
                            decision = parsed

                        # Extract action and quantity
                        action = decision.get("action", "hold")
                        quantity = decision.get("quantity", 0)

                        # Ensure quantity is integer
                        if isinstance(quantity, (int, float)):
                            quantity = int(quantity)
                        else:
                            quantity = 0

                        return {"action": action, "quantity": quantity}

                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(
                        f"Failed to parse JSON decision: {e}, using default decision"
                    )
                    return {"action": "hold", "quantity": 0}
            else:
                return {"action": "hold", "quantity": 0}

        except Exception as e:
            self.logger.warning(f"Failed to get agent decision: {e}")
            return {"action": "hold", "quantity": 0}

    def _handle_api_limits(self):
        """Handle API limits"""
        current_time = time.time()

        # Reset time window
        if current_time - self._api_window_start >= 60:
            self._api_call_count = 0
            self._api_window_start = current_time

        # Check API limits
        if self._api_call_count >= 8:
            wait_time = 60 - (current_time - self._api_window_start)
            if wait_time > 0:
                time.sleep(wait_time)
                self._api_call_count = 0
                self._api_window_start = time.time()

        # Ensure call interval
        if self._last_api_call:
            time_since_last = time.time() - self._last_api_call
            if time_since_last < 3:
                time.sleep(3 - time_since_last)

        self._last_api_call = time.time()
        self._api_call_count += 1

    def _get_current_price(self, date: str) -> Optional[float]:
        """Get current price"""
        try:
            data = get_price_data(self.ticker, date, date)
            if data is not None and not data.empty:
                return data.iloc[-1]["close"]
        except Exception as e:
            self.logger.warning(f"Failed to get price data {date}: {e}")
        return None

    def _get_price_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get price data"""
        try:
            return get_price_data(self.ticker, start_date, end_date)
        except Exception as e:
            self.logger.warning(
                f"Failed to get price data {start_date}-{end_date}: {e}"
            )
            return None

    def save_results(self, filepath: str):
        """Save backtest results"""
        import pickle

        results_data = {
            "config": self.config,
            "results": self.results,
            "comparison_results": self.comparison_results,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(results_data, f)

        self.logger.info(f"Backtest results saved to: {filepath}")

    def load_results(self, filepath: str):
        """Load backtest results"""
        import pickle

        with open(filepath, "rb") as f:
            results_data = pickle.load(f)

        self.config = results_data["config"]
        self.results = results_data["results"]
        self.comparison_results = results_data.get("comparison_results", {})

        self.logger.info(f"Backtest results loaded from {filepath}")

    def export_report(self, format: str = "html") -> str:
        """Export backtest report"""
        if format == "html":
            return self._generate_html_report()
        elif format == "pdf":
            return self._generate_pdf_report()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_html_report(self) -> str:
        """Generate HTML report"""
        # Use template engine to generate detailed HTML report
        # Include all charts, statistical results and analysis
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Analysis Report</title>
            <meta charset="utf-8">
        </head>
        <body>
            <h1>A-Share Investment Agent System Backtest Analysis Report</h1>
            <h2>Configuration Information</h2>
            <p>Stock Ticker: {self.ticker}</p>
            <p>Backtest Period: {self.config.start_date} to {self.config.end_date}</p>
            <p>Initial Capital: {self.config.initial_capital:,.2f}</p>
            
            <h2>Strategy Performance Summary</h2>
            {self._format_performance_table()}
            
            <h2>Statistical Significance Test Results</h2>
            {self._format_significance_results()}
            
            <h2>Investment Recommendations</h2>
            {self._format_recommendations()}
        </body>
        </html>
        """

        filename = f"backtest_report_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)

        return filename

    def _format_performance_table(self) -> str:
        """Format performance table"""
        if not hasattr(self, "comparison_results") or not self.comparison_results:
            return "<p>No performance data available</p>"

        performance = self.comparison_results.get("performance_summary", {})

        table_html = "<table border='1'><tr>"
        table_html += "<th>Strategy</th><th>Total Return</th><th>Annual Return</th><th>Volatility</th><th>Sharpe Ratio</th><th>Max Drawdown</th></tr>"

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
        """Format significance test results"""
        if not hasattr(self, "comparison_results") or not self.comparison_results:
            return "<p>No statistical test results available</p>"

        stats_summary = self.comparison_results.get("statistical_summary", {})
        significant = stats_summary.get("significant_differences", [])

        if not significant:
            return "<p>No significant differences found between strategies</p>"

        html = "<ul>"
        for item in significant:
            html += f"<li>{item['conclusion']} (Statistical power: {item['power']:.3f})</li>"
        html += "</ul>"

        return html

    def _format_recommendations(self) -> str:
        """Format investment recommendations"""
        if not hasattr(self, "comparison_results") or not self.comparison_results:
            return "<p>No investment recommendations available</p>"

        recommendations = self.comparison_results.get("recommendations", [])

        if not recommendations:
            return "<p>No specific recommendations available</p>"

        html = "<ol>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ol>"

        return html

    def _generate_pdf_report(self) -> str:
        """Generate PDF report"""
        # Temporarily return HTML filename
        return self._generate_html_report()

    def _get_strategy_lookback_days(self, strategy: BaseStrategy) -> int:
        """Determine required historical data days based on strategy type"""
        strategy_name = strategy.name.lower()

        # Set different historical data requirements based on strategy type
        if "momentum" in strategy_name:
            return 400  # Momentum strategy needs more historical data
        elif "moving-average" in strategy_name:
            # Moving average strategy determined by window size
            if hasattr(strategy, "long_window"):
                return max(400, strategy.long_window + 100)  # Long window + buffer
            else:
                return 300
        elif "mean-reversion" in strategy_name:
            return 400  # Mean reversion strategy needs sufficient historical data
        else:
            return 365  # Default one year of historical data
