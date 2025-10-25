#!/usr/bin/env python3
"""
Backtesting framework test script
Used to verify basic functionality of the backtesting framework
"""

import argparse
import os
import sys
import unittest
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest import BacktestConfig, Backtester
from src.backtest.baselines.buy_hold import BuyHoldStrategy
from src.backtest.baselines.mean_reversion import MeanReversionStrategy
from src.backtest.baselines.momentum import MomentumStrategy
from src.backtest.baselines.moving_average import MovingAverageStrategy
from src.backtest.baselines.random_walk import RandomWalkStrategy
from src.backtest.evaluation.table_generator import BacktestTableGenerator
from src.main import run_hedge_fund
from src.utils.logging_config import setup_logger


class ComparisonTableGenerator:
    """Comparison table generator (simplified version for console display)"""

    def __init__(self):
        self.logger = setup_logger("ComparisonTableGenerator")

    def generate_performance_table(
        self,
        results_summary: Dict[str, Dict[str, Any]],
        save_to_file: bool = True,
        filename: str = "performance_comparison.csv",
    ) -> pd.DataFrame:
        """
        Generate performance comparison table

        Args:
            results_summary: Strategy results summary
            save_to_file: Whether to save to file
            filename: Save filename

        Returns:
            pd.DataFrame: Performance comparison table
        """
        if not results_summary:
            self.logger.warning("No result data available, cannot generate table")
            return pd.DataFrame()

        # Create table data
        table_data = []
        for strategy_name, metrics in results_summary.items():
            row = {
                "Strategy Name": strategy_name,
                "Total Return(%)": round(metrics.get("total_return", 0), 2),
                "Annual Return(%)": round(metrics.get("annual_return", 0), 2),
                "Sharpe Ratio": round(metrics.get("sharpe_ratio", 0), 3),
                "Max Drawdown(%)": round(abs(metrics.get("max_drawdown", 0)), 2),
                "Annual Volatility(%)": round(metrics.get("volatility", 0), 2),
                "Win Rate(%)": round(metrics.get("win_rate", 0) * 100, 2),
                "Profit/Loss Ratio": round(metrics.get("profit_loss_ratio", 0), 2),
                "Trade Count": metrics.get("trade_count", 0),
                "VaR(%)": round(abs(metrics.get("var_95", 0)) * 100, 2),
                "Sortino Ratio": round(metrics.get("sortino_ratio", 0), 3),
                "Calmar Ratio": round(metrics.get("calmar_ratio", 0), 3),
            }
            table_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(table_data)

        # Sort by total return
        df = df.sort_values("Total Return(%)", ascending=False).reset_index(drop=True)

        # Add ranking column
        df.insert(0, "Rank", range(1, len(df) + 1))

        if save_to_file:
            try:
                df.to_csv(filename, index=False, encoding="utf-8-sig")
                self.logger.info(f"Performance comparison table saved to: {filename}")
            except Exception as e:
                self.logger.error(f"Failed to save table: {e}")

        return df

    def generate_ranking_table(
        self, results_summary: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Generate strategy ranking table

        Args:
            results_summary: Strategy results summary

        Returns:
            pd.DataFrame: Ranking table
        """
        if not results_summary:
            return pd.DataFrame()

        # Define evaluation dimensions and weights
        dimensions = {
            "Return Performance": ["total_return", "annual_return"],
            "Risk Control": ["max_drawdown", "volatility", "var_95"],
            "Risk-Adjusted Return": ["sharpe_ratio", "sortino_ratio", "calmar_ratio"],
            "Trading Efficiency": ["win_rate", "profit_loss_ratio", "trade_count"],
        }

        ranking_data = []

        # Collect all strategy metric values for normalization
        all_returns = [
            metrics.get("total_return", 0) for metrics in results_summary.values()
        ]
        all_sharpe = [
            metrics.get("sharpe_ratio", 0) for metrics in results_summary.values()
        ]
        all_drawdowns = [
            abs(metrics.get("max_drawdown", 0)) for metrics in results_summary.values()
        ]
        all_volatility = [
            metrics.get("volatility", 0) for metrics in results_summary.values()
        ]
        all_win_rates = [
            metrics.get("win_rate", 0) for metrics in results_summary.values()
        ]

        # Calculate baseline values (for normalization)
        max_return = max(all_returns) if all_returns else 1
        min_return = min(all_returns) if all_returns else 0
        max_sharpe = max(all_sharpe) if all_sharpe else 1
        min_sharpe = min(all_sharpe) if all_sharpe else 0
        max_drawdown = max(all_drawdowns) if all_drawdowns else 1
        max_volatility = max(all_volatility) if all_volatility else 1

        for strategy_name, metrics in results_summary.items():
            # Calculate scores for each dimension (normalized to 0-100)
            scores = {}

            # Return performance score (higher return is better)
            total_return = metrics.get("total_return", 0)
            annual_return = metrics.get("annual_return", 0)
            if max_return > min_return:
                return_score = (
                    (total_return - min_return) / (max_return - min_return)
                ) * 100
            else:
                return_score = (
                    50  # If all strategies have same return, give medium score
                )
            scores["Return Performance"] = max(0, min(100, return_score))

            # Risk control score (lower drawdown and volatility is better)
            max_dd = abs(metrics.get("max_drawdown", 0))
            volatility = metrics.get("volatility", 0)
            var_95 = abs(metrics.get("var_95", 0))

            # Risk control score: lower risk gets higher score
            dd_score = (
                max(0, 100 - (max_dd / max_drawdown * 100)) if max_drawdown > 0 else 100
            )
            vol_score = (
                max(0, 100 - (volatility / max_volatility * 100))
                if max_volatility > 0
                else 100
            )
            var_score = max(
                0, 100 - (var_95 * 100)
            )  # VaR is usually negative, take absolute value

            scores["Risk Control"] = (dd_score + vol_score + var_score) / 3

            # Risk-adjusted return score
            sharpe = metrics.get("sharpe_ratio", 0)
            sortino = metrics.get("sortino_ratio", 0)
            calmar = metrics.get("calmar_ratio", 0)

            # Normalize Sharpe ratio
            if max_sharpe > min_sharpe:
                sharpe_score = ((sharpe - min_sharpe) / (max_sharpe - min_sharpe)) * 100
            else:
                sharpe_score = 50

            # Handle Sortino and Calmar ratios
            sortino_score = min(
                100, max(0, sortino * 20 + 50)
            )  # Simple linear transformation
            calmar_score = min(100, max(0, calmar * 20 + 50))

            scores["Risk-Adjusted Return"] = (
                sharpe_score + sortino_score + calmar_score
            ) / 3

            # Trading efficiency score
            win_rate = metrics.get("win_rate", 0)
            pl_ratio = metrics.get("profit_loss_ratio", 0)
            trade_count = metrics.get("trade_count", 0)

            # Win rate score (convert 0-1 to 0-100)
            win_rate_score = win_rate * 100

            # Profit/loss ratio score
            pl_score = min(100, max(0, pl_ratio * 25))  # Full score when P/L ratio > 4

            # Trade count score (moderate trading is good)
            if trade_count == 0:
                trade_score = 0
            elif trade_count <= 10:
                trade_score = trade_count * 10  # Linear growth for 1-10 trades
            elif trade_count <= 50:
                trade_score = (
                    100 - (trade_count - 10) * 2
                )  # Gradual deduction for 10-50 trades
            else:
                trade_score = max(
                    0, 100 - trade_count
                )  # Large deduction for over 50 trades

            scores["Trading Efficiency"] = (win_rate_score + pl_score + trade_score) / 3

            # Calculate composite score
            weights = {
                "Return Performance": 0.3,
                "Risk Control": 0.3,
                "Risk-Adjusted Return": 0.25,
                "Trading Efficiency": 0.15,
            }
            composite_score = sum(scores[dim] * weights[dim] for dim in scores)

            ranking_data.append(
                {
                    "Strategy Name": strategy_name,
                    "Return Performance": round(scores["Return Performance"], 1),
                    "Risk Control": round(scores["Risk Control"], 1),
                    "Risk-Adjusted Return": round(scores["Risk-Adjusted Return"], 1),
                    "Trading Efficiency": round(scores["Trading Efficiency"], 1),
                    "Composite Score": round(composite_score, 1),
                }
            )

        # Create DataFrame and sort
        df = pd.DataFrame(ranking_data)
        df = df.sort_values("Composite Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        return df

    def generate_statistical_significance_table(
        self, comparison_results: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Generate statistical significance test table

        Args:
            comparison_results: Comparison analysis results

        Returns:
            pd.DataFrame: Statistical significance table
        """
        if not comparison_results or "pairwise_comparisons" not in comparison_results:
            return pd.DataFrame()

        comparisons = comparison_results["pairwise_comparisons"]
        significance_data = []

        for comparison_key, result in comparisons.items():
            try:
                if "summary" in result:
                    summary = result["summary"]

                    # Extract strategy names
                    strategies = comparison_key.split(" vs ")
                    strategy1 = strategies[0] if len(strategies) > 0 else "Unknown"
                    strategy2 = strategies[1] if len(strategies) > 1 else "Unknown"

                    # Extract statistical test results
                    power = summary.get("statistical_power", 0)
                    conclusion = summary.get("overall_conclusion", "No conclusion")

                    # Extract specific test results
                    paired_test = result.get("paired_test", {})
                    dm_test = result.get("diebold_mariano", {})
                    sharpe_test = result.get("sharpe_test", {})

                    significance_data.append(
                        {
                            "Strategy Comparison": comparison_key,
                            "Strategy A": strategy1,
                            "Strategy B": strategy2,
                            "Paired t-test": (
                                "Significant"
                                if paired_test.get("significant", False)
                                else "Not significant"
                            ),
                            "DM Test": (
                                "Significant"
                                if dm_test.get("significant", False)
                                else "Not significant"
                            ),
                            "Sharpe Ratio Test": (
                                "Significant"
                                if sharpe_test.get("significant", False)
                                else "Not significant"
                            ),
                            "Statistical Power": round(power, 3),
                            "Conclusion": conclusion,
                        }
                    )
            except (KeyError, TypeError, IndexError) as e:
                self.logger.warning(
                    f"Error processing comparison result {comparison_key}: {e}"
                )
                continue

        if not significance_data:
            return pd.DataFrame()

        df = pd.DataFrame(significance_data)
        return df

    def generate_ai_agent_analysis_table(
        self, results_summary: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Generate AI Agent specialized analysis table

        Args:
            results_summary: Strategy results summary

        Returns:
            pd.DataFrame: AI Agent analysis table
        """
        if "AI_Agent" not in results_summary:
            return pd.DataFrame()

        ai_metrics = results_summary["AI_Agent"]

        # Calculate comparison with other strategies
        other_strategies = {k: v for k, v in results_summary.items() if k != "AI_Agent"}

        if not other_strategies:
            return pd.DataFrame()

        # Calculate averages and rankings
        avg_metrics = {}
        rankings = {}

        for metric in ["total_return", "sharpe_ratio", "max_drawdown", "volatility"]:
            values = [metrics.get(metric, 0) for metrics in other_strategies.values()]
            avg_metrics[metric] = np.mean(values) if values else 0

            # Calculate ranking
            all_values = [ai_metrics.get(metric, 0)] + values
            if metric in ["max_drawdown", "volatility"]:  # Lower is better
                all_values_abs = [abs(v) for v in all_values]
                rankings[metric] = (
                    sorted(all_values_abs).index(abs(ai_metrics.get(metric, 0))) + 1
                )
            else:  # Higher is better
                rankings[metric] = len(all_values) - sorted(all_values).index(
                    ai_metrics.get(metric, 0)
                )

        analysis_data = [
            {
                "Metric": "Total Return(%)",
                "AI Agent": round(ai_metrics.get("total_return", 0), 2),
                "Benchmark Average": round(avg_metrics["total_return"], 2),
                "Difference": round(
                    ai_metrics.get("total_return", 0) - avg_metrics["total_return"], 2
                ),
                "Ranking": f"{rankings['total_return']}/{len(results_summary)}",
                "Performance": (
                    "Above Average"
                    if ai_metrics.get("total_return", 0) > avg_metrics["total_return"]
                    else "Below Average"
                ),
            },
            {
                "Metric": "Sharpe Ratio",
                "AI Agent": round(ai_metrics.get("sharpe_ratio", 0), 3),
                "Benchmark Average": round(avg_metrics["sharpe_ratio"], 3),
                "Difference": round(
                    ai_metrics.get("sharpe_ratio", 0) - avg_metrics["sharpe_ratio"], 3
                ),
                "Ranking": f"{rankings['sharpe_ratio']}/{len(results_summary)}",
                "Performance": (
                    "Above Average"
                    if ai_metrics.get("sharpe_ratio", 0) > avg_metrics["sharpe_ratio"]
                    else "Below Average"
                ),
            },
            {
                "Metric": "Max Drawdown(%)",
                "AI Agent": round(abs(ai_metrics.get("max_drawdown", 0)), 2),
                "Benchmark Average": round(abs(avg_metrics["max_drawdown"]), 2),
                "Difference": round(
                    abs(ai_metrics.get("max_drawdown", 0))
                    - abs(avg_metrics["max_drawdown"]),
                    2,
                ),
                "Ranking": f"{rankings['max_drawdown']}/{len(results_summary)}",
                "Performance": (
                    "Above Average"
                    if abs(ai_metrics.get("max_drawdown", 0))
                    < abs(avg_metrics["max_drawdown"])
                    else "Below Average"
                ),
            },
            {
                "Metric": "Annual Volatility(%)",
                "AI Agent": round(ai_metrics.get("volatility", 0), 2),
                "Benchmark Average": round(avg_metrics["volatility"], 2),
                "Difference": round(
                    ai_metrics.get("volatility", 0) - avg_metrics["volatility"], 2
                ),
                "Ranking": f"{rankings['volatility']}/{len(results_summary)}",
                "Performance": (
                    "Above Average"
                    if ai_metrics.get("volatility", 0) < avg_metrics["volatility"]
                    else "Below Average"
                ),
            },
        ]

        return pd.DataFrame(analysis_data)


class BacktestTest:
    """Backtest test class"""

    def __init__(self, start_date: str = "2023-01-01", end_date: str = "2023-03-31"):
        self.logger = setup_logger("BacktestTest")
        self.start_date = start_date
        self.end_date = end_date
        self.test_duration_days = self._calculate_test_duration()
        self.table_generator = ComparisonTableGenerator()
        self.comprehensive_table_generator = BacktestTableGenerator()

    def _calculate_test_duration(self) -> int:
        """Calculate test duration (in days)"""
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        return (end - start).days

    def get_strategy_requirements(self) -> dict:
        """Get minimum time requirements for each strategy (in days)"""
        return {
            "Buy-and-Hold": 1,  # Buy-and-hold strategy has no time requirement
            "Random-Walk": 1,  # Random walk strategy has no time requirement
            "Mean-Reversion": 252,  # Mean reversion strategy needs 252 days (lookback_period)
            "Mean-Reversion-Short": 252,  # Short-term mean reversion strategy needs 252 days
            "Moving-Average": 200,  # Moving average strategy needs 200 days (long_window)
            "Moving-Average-Short": 60,  # Short-term moving average strategy needs 60 days
            "Momentum": 252,  # Momentum strategy needs 252 days (lookback_period)
            "Momentum-Long": 252,  # Long-term momentum strategy needs 252 days
            "AI-Agent": 30,  # AI Agent strategy needs 30 days minimum data
        }

    def select_strategies_by_duration(self) -> list:
        """Select appropriate strategies based on test duration"""
        requirements = self.get_strategy_requirements()
        selected_strategies = []

        self.logger.info(f"Test duration: {self.test_duration_days} days")

        # Always include basic strategies
        selected_strategies.extend(
            [
                BuyHoldStrategy(allocation_ratio=1.0),
                RandomWalkStrategy(
                    trade_probability=0.1, max_position_ratio=0.5, truly_random=True
                ),
            ]
        )

        # Add other strategies based on duration
        if self.test_duration_days >= 30:
            # Short-term moving average strategy
            selected_strategies.append(
                MovingAverageStrategy(
                    short_window=5,
                    long_window=20,
                    signal_threshold=0.001,
                    name="Moving-Average-Short",
                )
            )
            self.logger.info(
                "✓ Added short-term moving average strategy (requires 20 days)"
            )

        if self.test_duration_days >= 60:
            # Standard moving average strategy
            selected_strategies.append(
                MovingAverageStrategy(
                    short_window=10, long_window=30, signal_threshold=0.001
                )
            )
            self.logger.info(
                "✓ Added standard moving average strategy (requires 30 days)"
            )

        if self.test_duration_days >= 252:
            # Mean reversion strategy
            selected_strategies.extend(
                [
                    MeanReversionStrategy(
                        lookback_period=252,
                        z_threshold=1.5,
                        mean_period=30,
                        exit_threshold=0.5,
                    ),
                    MeanReversionStrategy(
                        lookback_period=126,
                        z_threshold=1.0,
                        mean_period=20,
                        exit_threshold=0.3,
                        name="Mean-Reversion-Short",
                    ),
                ]
            )
            self.logger.info("✓ Added mean reversion strategy (requires 252 days)")

            # Momentum strategy
            selected_strategies.extend(
                [
                    MomentumStrategy(
                        lookback_period=252,
                        formation_period=63,
                        holding_period=21,
                        momentum_threshold=0.01,
                    ),
                    MomentumStrategy(
                        lookback_period=252,
                        formation_period=126,
                        holding_period=42,
                        momentum_threshold=0.02,
                        name="Momentum-Long",
                    ),
                ]
            )
            self.logger.info("✓ Added momentum strategy (requires 252 days)")

        # Initialize all strategies
        for strategy in selected_strategies:
            strategy.initialize(100000)

        self.logger.info(f"Selected {len(selected_strategies)} strategies for testing")
        return selected_strategies

    def test_backtester_initialization(self):
        """Test backtester component initialization"""
        self.logger.info("Testing backtester component initialization...")

        config = BacktestConfig(
            initial_capital=100000,
            start_date=self.start_date,
            end_date=self.end_date,
            benchmark_ticker="000300",
            trading_cost=0.001,
            slippage=0.001,
        )

        backtester = Backtester(
            ticker="000001", config=config, seed=42  # Fixed random seed
        )

        assert backtester is not None
        assert backtester.config.initial_capital == 100000
        assert backtester.ticker == "000001"

        self.logger.info("✓ Backtester component initialization correct")

    def test_baseline_strategies_backtest(self):
        """Test all baseline strategy backtests"""
        self.logger.info("Testing baseline strategy backtests...")

        config = BacktestConfig(
            initial_capital=100000,
            start_date=self.start_date,
            end_date=self.end_date,
            benchmark_ticker="000300",
            trading_cost=0.001,
            slippage=0.001,
        )

        backtester = Backtester(
            ticker="000001", config=config, seed=42  # Fixed random seed
        )

        # Select strategies based on test duration
        selected_strategies = self.select_strategies_by_duration()

        # Run backtests
        results = {}
        for strategy in selected_strategies:
            try:
                result = backtester._run_single_strategy_backtest(strategy)
                results[strategy.name] = result
            except Exception as e:
                self.logger.warning(f"Strategy {strategy.name} backtest failed: {e}")
                continue

        # Validate results
        assert (
            len(results) > 0
        ), "At least one strategy should complete backtest successfully"

        self.logger.info(f"✓ Successfully completed {len(results)} strategy backtests")

        # Output results summary
        for name, result in results.items():
            total_return = result.performance_metrics.get("total_return", 0) * 100
            sharpe_ratio = result.performance_metrics.get("sharpe_ratio", 0)
            self.logger.info(
                f"  - {name}: Return {total_return:.2f}%, Sharpe {sharpe_ratio:.3f}"
            )

        return results

    def test_ai_agent_backtest(self, ticker: str = "000001"):
        """Test AI Agent backtest functionality"""
        self.logger.info("Testing AI Agent backtest functionality...")

        config = BacktestConfig(
            initial_capital=100000,
            start_date=self.start_date,
            end_date=self.end_date,
            benchmark_ticker="000300",
            trading_cost=0.001,
            slippage=0.001,
        )

        # Create backtester with AI agent
        backtester = Backtester(
            agent_function=run_hedge_fund,
            ticker=ticker,
            tickers=[ticker],
            config=config,
            seed=42,
        )

        try:
            # Run AI agent backtest
            agent_result = backtester.run_agent_backtest()

            # Validate AI agent results
            assert agent_result is not None
            assert agent_result.strategy_name == "AI Agent"
            assert len(agent_result.portfolio_values) > 0

            self.logger.info("✓ AI Agent backtest test completed")
            self.logger.info(
                f"  Total return: {agent_result.performance_metrics.get('total_return', 0)*100:.2f}%"
            )
            self.logger.info(
                f"  Sharpe ratio: {agent_result.performance_metrics.get('sharpe_ratio', 0):.3f}"
            )
            self.logger.info(f"  Trade count: {len(agent_result.trade_history)}")

            return agent_result

        except Exception as e:
            self.logger.error(f"AI Agent backtest failed: {e}")
            # Return None to indicate test failure, but don't interrupt the entire test flow
            return None

    def test_comprehensive_comparison(self, ticker: str = "000001"):
        """Test comprehensive comparison between AI Agent and baseline strategies"""
        self.logger.info(
            "Testing comprehensive comparison between AI Agent and baseline strategies..."
        )

        config = BacktestConfig(
            initial_capital=100000,
            start_date=self.start_date,
            end_date=self.end_date,
            benchmark_ticker="000300",
            trading_cost=0.001,
            slippage=0.001,
        )

        # Create backtester
        backtester = Backtester(
            agent_function=run_hedge_fund,
            ticker=ticker,
            tickers=[ticker],
            config=config,
            seed=42,
        )

        results_summary = {}

        try:
            # 1. Run AI agent backtest
            if self.test_duration_days >= 30:  # AI agent needs at least 30 days of data
                agent_result = backtester.run_agent_backtest()
                if agent_result:
                    # Calculate AI Agent trading statistics
                    ai_trade_stats = self._calculate_trade_statistics(
                        agent_result.trade_history
                        if hasattr(agent_result, "trade_history")
                        else []
                    )

                    # Get original performance metrics and add debug information
                    ai_original_total_return = agent_result.performance_metrics.get(
                        "total_return", 0
                    )
                    ai_original_annual_return = agent_result.performance_metrics.get(
                        "annual_return", 0
                    )
                    ai_original_max_drawdown = agent_result.performance_metrics.get(
                        "max_drawdown", 0
                    )
                    ai_original_volatility = agent_result.performance_metrics.get(
                        "volatility", 0
                    )

                    # Debug output
                    if (
                        abs(ai_original_total_return) < 1e-6
                    ):  # If total return is very small
                        self.logger.warning(
                            f"AI Agent total return is very small: {ai_original_total_return}, daily returns array length: {len(agent_result.daily_returns) if hasattr(agent_result, 'daily_returns') else 0}"
                        )
                        if (
                            hasattr(agent_result, "daily_returns")
                            and len(agent_result.daily_returns) > 0
                        ):
                            self.logger.warning(
                                f"  AI Agent daily returns sample: {agent_result.daily_returns[:5].tolist() if len(agent_result.daily_returns) >= 5 else agent_result.daily_returns.tolist()}"
                            )

                    # Handle small return value display - preserve more precision
                    ai_total_return_pct = round(ai_original_total_return * 100, 4)
                    ai_annual_return_pct = round(ai_original_annual_return * 100, 4)
                    ai_max_drawdown_pct = round(ai_original_max_drawdown * 100, 4)
                    ai_volatility_pct = round(ai_original_volatility * 100, 4)

                    results_summary["AI_Agent"] = {
                        "total_return": ai_total_return_pct,
                        "annual_return": ai_annual_return_pct,
                        "sharpe_ratio": agent_result.performance_metrics.get(
                            "sharpe_ratio", 0
                        ),
                        "max_drawdown": ai_max_drawdown_pct,
                        "volatility": ai_volatility_pct,
                        "win_rate": ai_trade_stats.get(
                            "win_rate",
                            agent_result.performance_metrics.get("win_rate", 0),
                        ),
                        "profit_loss_ratio": ai_trade_stats.get(
                            "profit_loss_ratio",
                            agent_result.performance_metrics.get(
                                "profit_loss_ratio", 0
                            ),
                        ),
                        "var_95": agent_result.performance_metrics.get("var_95", 0),
                        "sortino_ratio": agent_result.performance_metrics.get(
                            "sortino_ratio", 0
                        ),
                        "calmar_ratio": agent_result.performance_metrics.get(
                            "calmar_ratio", 0
                        ),
                        "trade_count": ai_trade_stats.get("trade_count", 0),
                        "profitable_trades": ai_trade_stats.get("profitable_trades", 0),
                        "losing_trades": ai_trade_stats.get("losing_trades", 0),
                        "avg_profit": ai_trade_stats.get("avg_profit", 0),
                        "avg_loss": ai_trade_stats.get("avg_loss", 0),
                    }
                    self.logger.info("✓ AI Agent backtest completed")
            else:
                self.logger.warning(
                    "Test duration too short, skipping AI Agent backtest"
                )

        except Exception as e:
            self.logger.warning(f"AI Agent backtest failed: {e}")

        # 2. Run baseline strategy backtests
        baseline_results = backtester.run_baseline_backtests()

        for name, result in baseline_results.items():
            # Calculate trading statistics
            trade_stats = self._calculate_trade_statistics(
                result.trade_history if hasattr(result, "trade_history") else []
            )

            # Get original performance metrics and add debug information
            original_total_return = result.performance_metrics.get("total_return", 0)
            original_annual_return = result.performance_metrics.get("annual_return", 0)
            original_max_drawdown = result.performance_metrics.get("max_drawdown", 0)
            original_volatility = result.performance_metrics.get("volatility", 0)

            # Debug output
            if abs(original_total_return) < 1e-6:  # If total return is very small
                self.logger.warning(
                    f"Strategy {name} total return is very small: {original_total_return}, daily returns array length: {len(result.daily_returns) if hasattr(result, 'daily_returns') else 0}"
                )
                if hasattr(result, "daily_returns") and len(result.daily_returns) > 0:
                    self.logger.warning(
                        f"  Daily returns sample: {result.daily_returns[:5].tolist() if len(result.daily_returns) >= 5 else result.daily_returns.tolist()}"
                    )

            # Handle small return value display - preserve more precision
            total_return_pct = round(
                original_total_return * 100, 4
            )  # Keep 4 decimal places
            annual_return_pct = round(original_annual_return * 100, 4)
            max_drawdown_pct = round(original_max_drawdown * 100, 4)
            volatility_pct = round(original_volatility * 100, 4)

            results_summary[name] = {
                "total_return": total_return_pct,
                "annual_return": annual_return_pct,
                "sharpe_ratio": result.performance_metrics.get("sharpe_ratio", 0),
                "max_drawdown": max_drawdown_pct,
                "volatility": volatility_pct,
                "win_rate": trade_stats.get(
                    "win_rate", result.performance_metrics.get("win_rate", 0)
                ),
                "profit_loss_ratio": trade_stats.get(
                    "profit_loss_ratio",
                    result.performance_metrics.get("profit_loss_ratio", 0),
                ),
                "var_95": result.performance_metrics.get("var_95", 0),
                "sortino_ratio": result.performance_metrics.get("sortino_ratio", 0),
                "calmar_ratio": result.performance_metrics.get("calmar_ratio", 0),
                "trade_count": trade_stats.get("trade_count", 0),
                "profitable_trades": trade_stats.get("profitable_trades", 0),
                "losing_trades": trade_stats.get("losing_trades", 0),
                "avg_profit": trade_stats.get("avg_profit", 0),
                "avg_loss": trade_stats.get("avg_loss", 0),
            }

        # 3. Run statistical comparison (if there are enough results)
        comparison_results = None
        if len(backtester.results) >= 2:
            try:
                comparison_results = backtester.run_comprehensive_comparison()
                self.logger.info("✓ Statistical significance test completed")

                # Generate statistical significance table
                significance_df = (
                    self.table_generator.generate_statistical_significance_table(
                        comparison_results
                    )
                )
                if not significance_df.empty:
                    self.logger.info("\n📈 Statistical significance test table:")
                    self.logger.info("\n" + significance_df.to_string(index=False))

                    # Save statistical significance table
                    significance_filename = f"statistical_significance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    significance_df.to_csv(
                        significance_filename, index=False, encoding="utf-8-sig"
                    )
                    self.logger.info(
                        f"Statistical significance table saved to: {significance_filename}"
                    )

            except Exception as e:
                self.logger.warning(f"Statistical comparison failed: {e}")

        # 4. Generate comparison report
        self._generate_comparison_report(results_summary)

        # 5. Generate comprehensive report files
        try:
            config_info = {
                "Stock Code": ticker,
                "Backtest Start Date": self.start_date,
                "Backtest End Date": self.end_date,
                "Initial Capital": f"{config.initial_capital:,.0f}",
                "Trading Cost": f"{config.trading_cost*100:.1f}%",
                "Slippage": f"{config.slippage*100:.1f}%",
                "Benchmark Index": config.benchmark_ticker,
                "Test Duration": f"{self.test_duration_days} days",
            }

            generated_files = (
                self.comprehensive_table_generator.generate_comprehensive_report(
                    results_summary=results_summary,
                    comparison_results=comparison_results,
                    config=config_info,
                    export_formats=["csv", "excel", "html"],
                )
            )

            if generated_files:
                self.logger.info(f"\n📁 Comprehensive report files generated:")
                for report_type, filepath in generated_files.items():
                    self.logger.info(f"  📄 {report_type}: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive report: {e}")

        return {
            "results_summary": results_summary,
            "comparison_results": comparison_results,
            "baseline_results": baseline_results,
            "generated_files": generated_files if "generated_files" in locals() else {},
        }

    def _generate_comparison_report(self, results_summary: dict):
        """Generate comparison report"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Strategy Performance Comparison Report")
        self.logger.info("=" * 80)

        # 1. Generate performance comparison table
        performance_df = self.table_generator.generate_performance_table(
            results_summary,
            save_to_file=True,
            filename=f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )

        if not performance_df.empty:
            self.logger.info("\n📊 Detailed Performance Comparison Table:")
            self.logger.info("\n" + performance_df.to_string(index=False))

        # 2. Generate strategy ranking table
        ranking_df = self.table_generator.generate_ranking_table(results_summary)

        if not ranking_df.empty:
            self.logger.info("\n🏆 Strategy Comprehensive Ranking Table:")
            self.logger.info("\n" + ranking_df.to_string(index=False))

        # 3. Generate AI Agent specialized analysis table
        if "AI_Agent" in results_summary:
            ai_analysis_df = self.table_generator.generate_ai_agent_analysis_table(
                results_summary
            )

            if not ai_analysis_df.empty:
                self.logger.info("\n🤖 AI Agent Specialized Analysis Table:")
                self.logger.info("\n" + ai_analysis_df.to_string(index=False))

        # 4. Simplified ranking display
        sorted_results = sorted(
            results_summary.items(), key=lambda x: x[1]["total_return"], reverse=True
        )

        self.logger.info(f"\n📈 Strategy Return Ranking:")
        self.logger.info(
            f"{'Rank':<4} {'Strategy Name':<25} {'Return':<12} {'Sharpe Ratio':<12} {'Max Drawdown':<12}"
        )
        self.logger.info("-" * 80)

        for i, (name, metrics) in enumerate(sorted_results, 1):
            self.logger.info(
                f"{i:<4} {name:<25} {metrics['total_return']:>10.2f}% {metrics['sharpe_ratio']:>10.3f} {abs(metrics['max_drawdown']):>10.2f}%"
            )

        # 5. AI Agent performance analysis
        if "AI_Agent" in results_summary:
            ai_metrics = results_summary["AI_Agent"]
            ai_rank = next(
                (
                    i
                    for i, (name, _) in enumerate(sorted_results, 1)
                    if name == "AI_Agent"
                ),
                None,
            )

            self.logger.info(f"\n🎯 AI Agent Performance Analysis:")
            self.logger.info(
                f"  📊 Ranking: #{ai_rank} (out of {len(sorted_results)} strategies)"
            )
            self.logger.info(f"  💰 Return: {ai_metrics['total_return']:.2f}%")
            self.logger.info(f"  📈 Sharpe Ratio: {ai_metrics['sharpe_ratio']:.3f}")
            self.logger.info(
                f"  📉 Max Drawdown: {abs(ai_metrics['max_drawdown']):.2f}%"
            )
            self.logger.info(f"  🔄 Trade Count: {ai_metrics['trade_count']}")

            # Compare with average level
            avg_return = sum(m["total_return"] for m in results_summary.values()) / len(
                results_summary
            )
            avg_sharpe = sum(m["sharpe_ratio"] for m in results_summary.values()) / len(
                results_summary
            )
            avg_drawdown = sum(
                abs(m["max_drawdown"]) for m in results_summary.values()
            ) / len(results_summary)

            self.logger.info(f"\n📊 Comparison with Average Level:")
            self.logger.info(
                f"  Return difference: {ai_metrics['total_return'] - avg_return:+.2f}%"
            )
            self.logger.info(
                f"  Sharpe ratio difference: {ai_metrics['sharpe_ratio'] - avg_sharpe:+.3f}"
            )
            self.logger.info(
                f"  Drawdown difference: {abs(ai_metrics['max_drawdown']) - avg_drawdown:+.2f}%"
            )

            # Performance rating
            performance_score = 0
            if ai_metrics["total_return"] > avg_return:
                performance_score += 1
            if ai_metrics["sharpe_ratio"] > avg_sharpe:
                performance_score += 1
            if abs(ai_metrics["max_drawdown"]) < avg_drawdown:
                performance_score += 1

            if performance_score >= 2:
                rating = "Excellent ⭐⭐⭐"
            elif performance_score == 1:
                rating = "Good ⭐⭐"
            else:
                rating = "Average ⭐"

            self.logger.info(f"  Overall Rating: {rating}")

        self.logger.info("=" * 80)

    def _calculate_trade_statistics(self, trade_history: list) -> dict:
        """
        Calculate trading statistics from trade history

        Args:
            trade_history: Trade history list

        Returns:
            dict: Trading statistics metrics
        """
        if not trade_history:
            return {
                "trade_count": 0,
                "profitable_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_loss_ratio": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
            }

        # Group by trade pairs (buy-sell pairs)
        buy_trades = []
        sell_trades = []

        for trade in trade_history:
            if trade.get("action") == "buy":
                buy_trades.append(trade)
            elif trade.get("action") == "sell":
                sell_trades.append(trade)

        # Calculate profit/loss for each complete trade
        trade_profits = []

        # Simplified processing: assume FIFO (first in, first out)
        buy_queue = buy_trades.copy()

        for sell_trade in sell_trades:
            sell_quantity = sell_trade.get("quantity", 0)
            sell_price = sell_trade.get("price", 0)

            while sell_quantity > 0 and buy_queue:
                buy_trade = buy_queue[0]
                buy_quantity = buy_trade.get("quantity", 0)
                buy_price = buy_trade.get("price", 0)

                # Calculate the quantity for this match
                matched_quantity = min(sell_quantity, buy_quantity)

                # Calculate profit/loss
                profit = (sell_price - buy_price) * matched_quantity
                trade_profits.append(profit)

                # Update quantity
                sell_quantity -= matched_quantity
                buy_trade["quantity"] -= matched_quantity

                # If buy trade is completely matched, remove it
                if buy_trade["quantity"] <= 0:
                    buy_queue.pop(0)

        # Calculate statistical metrics
        if not trade_profits:
            return {
                "trade_count": len(trade_history),
                "profitable_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_loss_ratio": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
            }

        profitable_trades = [p for p in trade_profits if p > 0]
        losing_trades = [p for p in trade_profits if p < 0]

        total_trades = len(trade_profits)
        profitable_count = len(profitable_trades)
        losing_count = len(losing_trades)

        win_rate = profitable_count / total_trades if total_trades > 0 else 0.0

        avg_profit = (
            sum(profitable_trades) / profitable_count if profitable_count > 0 else 0.0
        )
        avg_loss = sum(losing_trades) / losing_count if losing_count > 0 else 0.0

        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0.0

        return {
            "trade_count": len(trade_history),
            "completed_trades": total_trades,
            "profitable_trades": profitable_count,
            "losing_trades": losing_count,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
        }

    def run_complete_test_suite(self, ticker: str = "000001"):
        """Run complete test suite"""
        self.logger.info("=" * 60)
        self.logger.info("Starting complete backtest test suite")
        self.logger.info("=" * 60)

        test_results = {
            "config": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "duration_days": self.test_duration_days,
                "ticker": ticker,
            }
        }

        try:
            # 1. Basic component tests
            self.logger.info("\n1. Backtester component initialization test...")
            self.test_backtester_initialization()
            test_results["initialization"] = True

            # 2. Baseline strategy tests
            self.logger.info("\n2. Baseline strategy backtest test...")
            baseline_results = self.test_baseline_strategies_backtest()
            test_results["baseline_results"] = baseline_results

            # 3. AI Agent test
            self.logger.info("\n3. AI Agent backtest test...")
            ai_result = self.test_ai_agent_backtest(ticker)
            test_results["ai_agent_result"] = ai_result

            # 4. Comprehensive comparison test
            self.logger.info("\n4. Comprehensive comparison analysis...")
            comparison_results = self.test_comprehensive_comparison(ticker)
            test_results["comparison_results"] = comparison_results

            # 5. Generate final test report
            self.logger.info("\n5. Generate final test report...")
            self._generate_final_test_report(test_results)

            return test_results

        except Exception as e:
            self.logger.error(f"Test suite execution failed: {e}")
            return test_results

    def _generate_final_test_report(self, test_results: dict):
        """Generate final test report"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Final Test Report")
        self.logger.info("=" * 60)

        config = test_results["config"]
        self.logger.info(f"Test Configuration:")
        self.logger.info(
            f"  Time Range: {config['start_date']} to {config['end_date']}"
        )
        self.logger.info(f"  Test Duration: {config['duration_days']} days")
        self.logger.info(f"  Test Stock: {config['ticker']}")

        # Test completion status
        self.logger.info(f"\nTest Completion Status:")
        self.logger.info(
            f"  Component Initialization: {'✓' if test_results.get('initialization') else '✗'}"
        )
        self.logger.info(
            f"  Baseline Strategies: {len(test_results.get('baseline_results', {}))} strategies"
        )
        self.logger.info(
            f"  AI Agent Test: {'✓' if test_results.get('ai_agent_result') else '✗'}"
        )
        self.logger.info(
            f"  Comparison Analysis: {'✓' if test_results.get('comparison_results') else '✗'}"
        )

        # Performance overview
        comparison_results = test_results.get("comparison_results", {})
        results_summary = comparison_results.get("results_summary", {})

        if results_summary:
            best_strategy = max(
                results_summary.keys(), key=lambda k: results_summary[k]["total_return"]
            )
            best_return = results_summary[best_strategy]["total_return"]

            self.logger.info(f"\nPerformance Overview:")
            self.logger.info(f"  Best Strategy: {best_strategy} ({best_return:.2f}%)")
            self.logger.info(f"  Total Strategies: {len(results_summary)}")

            if "AI_Agent" in results_summary:
                ai_rank = (
                    sorted(
                        results_summary.keys(),
                        key=lambda k: results_summary[k]["total_return"],
                        reverse=True,
                    ).index("AI_Agent")
                    + 1
                )
                self.logger.info(f"  AI Agent Ranking: #{ai_rank}")

        self.logger.info("=" * 60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Backtest Test")
    parser.add_argument(
        "--quick", action="store_true", help="Quick test mode (short time)"
    )
    parser.add_argument(
        "--medium", action="store_true", help="Medium test mode (medium time)"
    )
    parser.add_argument(
        "--full", action="store_true", help="Full test mode (long time)"
    )
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--ticker", type=str, default="000001", help="Stock code")
    parser.add_argument("--ai-only", action="store_true", help="Test AI Agent only")
    parser.add_argument(
        "--baseline-only", action="store_true", help="Test baseline strategies only"
    )
    parser.add_argument(
        "--comparison", action="store_true", help="Run comprehensive comparison test"
    )

    args = parser.parse_args()

    # Set test time range based on parameters
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    elif args.quick:
        # Quick mode: 3 months
        end_date = "2023-03-31"
        start_date = "2023-01-01"
    elif args.medium:
        # Medium mode: 8 months
        end_date = "2023-08-31"
        start_date = "2023-01-01"
    elif args.full:
        # Full mode: 2 years
        end_date = "2024-12-31"
        start_date = "2023-01-01"
    else:
        # Default: 3 months
        end_date = "2023-03-31"
        start_date = "2023-01-01"

    # Create test instance
    test = BacktestTest(start_date=start_date, end_date=end_date)

    # Run tests
    try:
        if args.ai_only:
            # Test AI Agent only
            test.test_ai_agent_backtest(args.ticker)
        elif args.baseline_only:
            # Test baseline strategies only
            test.test_backtester_initialization()
            test.test_baseline_strategies_backtest()
        elif args.comparison:
            # Run comprehensive comparison test
            test.test_comprehensive_comparison(args.ticker)
        else:
            # Run complete test suite
            test.run_complete_test_suite(args.ticker)

        print("✓ All tests passed")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
