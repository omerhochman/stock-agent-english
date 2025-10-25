"""
Backtest result table generator
Provides multi-format comparison table generation and export functionality
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestTableGenerator:
    """Backtest result table generator"""

    def __init__(self, output_dir: str = "backtest_reports"):
        """
        Initialize table generator

        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_comprehensive_report(
        self,
        results_summary: Dict[str, Dict[str, Any]],
        comparison_results: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        export_formats: List[str] = ["csv", "excel", "html"],
    ) -> Dict[str, str]:
        """
        Generate comprehensive backtest report

        Args:
            results_summary: Strategy results summary
            comparison_results: Comparison analysis results
            config: Backtest configuration information
            export_formats: Export format list

        Returns:
            Dict: Generated file paths
        """
        if not results_summary:
            logger.warning("No result data, cannot generate report")
            return {}

        generated_files = {}

        try:
            # 1. Generate performance comparison table
            performance_df = self._create_performance_table(results_summary)

            # 2. Generate strategy ranking table
            ranking_df = self._create_ranking_table(results_summary)

            # 3. Generate risk metrics table
            risk_df = self._create_risk_metrics_table(results_summary)

            # 4. Generate trading statistics table
            trading_df = self._create_trading_statistics_table(results_summary)

            # 5. Generate AI Agent special analysis table
            ai_analysis_df = None
            if "AI_Agent" in results_summary:
                ai_analysis_df = self._create_ai_agent_analysis_table(results_summary)

            # 6. Generate statistical significance table
            significance_df = None
            if comparison_results:
                significance_df = self._create_statistical_significance_table(
                    comparison_results
                )

            # Export different formats
            for format_type in export_formats:
                if format_type.lower() == "csv":
                    csv_files = self._export_csv_tables(
                        performance_df,
                        ranking_df,
                        risk_df,
                        trading_df,
                        ai_analysis_df,
                        significance_df,
                    )
                    generated_files.update(csv_files)

                elif format_type.lower() == "excel":
                    excel_file = self._export_excel_workbook(
                        performance_df,
                        ranking_df,
                        risk_df,
                        trading_df,
                        ai_analysis_df,
                        significance_df,
                        config,
                    )
                    if excel_file:
                        generated_files["excel_report"] = excel_file

                elif format_type.lower() == "html":
                    html_file = self._export_html_report(
                        performance_df,
                        ranking_df,
                        risk_df,
                        trading_df,
                        ai_analysis_df,
                        significance_df,
                        config,
                    )
                    if html_file:
                        generated_files["html_report"] = html_file

            logger.info(f"Successfully generated {len(generated_files)} report files")
            return generated_files

        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            return {}

    def _create_performance_table(
        self, results_summary: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create performance comparison table"""
        table_data = []

        for strategy_name, metrics in results_summary.items():
            row = {
                "Strategy Name": strategy_name,
                "Total Return (%)": round(metrics.get("total_return", 0), 2),
                "Annual Return (%)": round(metrics.get("annual_return", 0), 2),
                "Sharpe Ratio": round(metrics.get("sharpe_ratio", 0), 3),
                "Sortino Ratio": round(metrics.get("sortino_ratio", 0), 3),
                "Calmar Ratio": round(metrics.get("calmar_ratio", 0), 3),
                "Information Ratio": round(metrics.get("information_ratio", 0), 3),
                "Max Drawdown (%)": round(abs(metrics.get("max_drawdown", 0)), 2),
                "Annual Volatility (%)": round(metrics.get("volatility", 0), 2),
                "VaR 95% (%)": round(abs(metrics.get("var_95", 0)) * 100, 2),
                "CVaR 95% (%)": round(abs(metrics.get("cvar_95", 0)) * 100, 2),
                "Win Rate (%)": round(metrics.get("win_rate", 0) * 100, 2),
                "Profit/Loss Ratio": round(metrics.get("profit_loss_ratio", 0), 2),
                "Trade Count": metrics.get("trade_count", 0),
                "Avg Holding Days": round(metrics.get("avg_holding_period", 0), 1),
                "Turnover Rate (%)": round(metrics.get("turnover_rate", 0) * 100, 2),
            }
            table_data.append(row)

        df = pd.DataFrame(table_data)
        df = df.sort_values("Total Return (%)", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        return df

    def _create_ranking_table(
        self, results_summary: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create strategy ranking table"""
        ranking_data = []

        # Define evaluation dimension weights
        weights = {
            "Return Performance": 0.30,
            "Risk Control": 0.25,
            "Risk-Adjusted Return": 0.25,
            "Trading Efficiency": 0.20,
        }

        for strategy_name, metrics in results_summary.items():
            # Calculate dimension scores
            scores = self._calculate_dimension_scores(metrics)

            # Calculate composite score
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

        df = pd.DataFrame(ranking_data)
        df = df.sort_values("Composite Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        return df

    def _create_risk_metrics_table(
        self, results_summary: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create risk metrics table"""
        risk_data = []

        for strategy_name, metrics in results_summary.items():
            row = {
                "Strategy Name": strategy_name,
                "Max Drawdown(%)": round(abs(metrics.get("max_drawdown", 0)), 2),
                "Drawdown Duration (Days)": metrics.get("max_drawdown_duration", 0),
                "Annualized Volatility(%)": round(metrics.get("volatility", 0), 2),
                "Downside Volatility(%)": round(
                    metrics.get("downside_volatility", 0), 2
                ),
                "VaR_95(%)": round(abs(metrics.get("var_95", 0)) * 100, 2),
                "CVaR_95(%)": round(abs(metrics.get("cvar_95", 0)) * 100, 2),
                "Skewness": round(metrics.get("skewness", 0), 3),
                "Kurtosis": round(metrics.get("kurtosis", 0), 3),
                "Beta": round(metrics.get("beta", 0), 3),
                "Tracking Error(%)": round(metrics.get("tracking_error", 0) * 100, 2),
                "Information Ratio": round(metrics.get("information_ratio", 0), 3),
            }
            risk_data.append(row)

        df = pd.DataFrame(risk_data)
        # Sort by maximum drawdown (lower is better)
        df = df.sort_values("Max Drawdown(%)", ascending=True).reset_index(drop=True)
        df.insert(0, "Risk Rank", range(1, len(df) + 1))

        return df

    def _create_trading_statistics_table(
        self, results_summary: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create trading statistics table"""
        trading_data = []

        for strategy_name, metrics in results_summary.items():
            row = {
                "Strategy Name": strategy_name,
                "Total Trades": metrics.get("trade_count", 0),
                "Profitable Trades": metrics.get("profitable_trades", 0),
                "Losing Trades": metrics.get("losing_trades", 0),
                "Win Rate(%)": round(metrics.get("win_rate", 0) * 100, 2),
                "Average Profit(%)": round(metrics.get("avg_profit", 0) * 100, 2),
                "Average Loss(%)": round(metrics.get("avg_loss", 0) * 100, 2),
                "Profit/Loss Ratio": round(metrics.get("profit_loss_ratio", 0), 2),
                "Max Single Profit(%)": round(metrics.get("max_profit", 0) * 100, 2),
                "Max Single Loss(%)": round(metrics.get("max_loss", 0) * 100, 2),
                "Average Holding Days": round(metrics.get("avg_holding_period", 0), 1),
                "Turnover Rate(%)": round(metrics.get("turnover_rate", 0) * 100, 2),
                "Trading Costs(%)": round(metrics.get("total_costs", 0) * 100, 2),
            }
            trading_data.append(row)

        df = pd.DataFrame(trading_data)
        # Sort by win rate
        df = df.sort_values("Win Rate(%)", ascending=False).reset_index(drop=True)
        df.insert(0, "Trading Rank", range(1, len(df) + 1))

        return df

    def _create_ai_agent_analysis_table(
        self, results_summary: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create AI Agent special analysis table"""
        if "AI_Agent" not in results_summary:
            return pd.DataFrame()

        ai_metrics = results_summary["AI_Agent"]
        other_strategies = {k: v for k, v in results_summary.items() if k != "AI_Agent"}

        if not other_strategies:
            return pd.DataFrame()

        analysis_data = []

        # Key metrics comparison
        key_metrics = [
            ("Total Return(%)", "total_return", False),
            ("Annual Return(%)", "annual_return", False),
            ("Sharpe Ratio", "sharpe_ratio", False),
            ("Max Drawdown(%)", "max_drawdown", True),
            ("Annual Volatility(%)", "volatility", True),
            ("Win Rate(%)", "win_rate", False),
            ("Profit/Loss Ratio", "profit_loss_ratio", False),
        ]

        for metric_name, metric_key, lower_is_better in key_metrics:
            ai_value = ai_metrics.get(metric_key, 0)
            if metric_name in [
                "Total Return(%)",
                "Annual Return(%)",
                "Max Drawdown(%)",
                "Annual Volatility(%)",
            ]:
                ai_value = ai_value if metric_key != "max_drawdown" else abs(ai_value)
            elif metric_name == "Win Rate(%)":
                ai_value = ai_value * 100

            # Calculate benchmark statistics
            other_values = [
                metrics.get(metric_key, 0) for metrics in other_strategies.values()
            ]
            if metric_name in [
                "Total Return(%)",
                "Annual Return(%)",
                "Max Drawdown(%)",
                "Annual Volatility(%)",
            ]:
                other_values = [
                    v if metric_key != "max_drawdown" else abs(v) for v in other_values
                ]
            elif metric_name == "Win Rate(%)":
                other_values = [v * 100 for v in other_values]

            avg_value = np.mean(other_values) if other_values else 0
            median_value = np.median(other_values) if other_values else 0
            best_value = (
                min(other_values)
                if lower_is_better and other_values
                else max(other_values) if other_values else 0
            )

            # Calculate ranking
            all_values = [ai_value] + other_values
            if lower_is_better:
                rank = sorted(all_values).index(ai_value) + 1
            else:
                rank = len(all_values) - sorted(all_values).index(ai_value)

            # Judge performance
            if lower_is_better:
                performance = (
                    "Better than Average" if ai_value < avg_value else "Below Average"
                )
                vs_best = "Better than Best" if ai_value < best_value else "Below Best"
            else:
                performance = (
                    "Better than Average" if ai_value > avg_value else "Below Average"
                )
                vs_best = "Better than Best" if ai_value > best_value else "Below Best"

            analysis_data.append(
                {
                    "Metric": metric_name,
                    "AI Agent": round(ai_value, 3),
                    "Benchmark Average": round(avg_value, 3),
                    "Benchmark Median": round(median_value, 3),
                    "Benchmark Best": round(best_value, 3),
                    "Rank": f"{rank}/{len(results_summary)}",
                    "vs Average": performance,
                    "vs Best": vs_best,
                    "Percentile": round(
                        (len(all_values) - rank + 1) / len(all_values) * 100, 1
                    ),
                }
            )

        return pd.DataFrame(analysis_data)

    def _create_statistical_significance_table(
        self, comparison_results: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create statistical significance test table"""
        if "pairwise_comparisons" not in comparison_results:
            return pd.DataFrame()

        comparisons = comparison_results["pairwise_comparisons"]
        significance_data = []

        for comparison_key, result in comparisons.items():
            try:
                if "summary" not in result:
                    continue

                summary = result["summary"]
                strategies = comparison_key.split(" vs ")
                strategy1 = strategies[0] if len(strategies) > 0 else "Unknown"
                strategy2 = strategies[1] if len(strategies) > 1 else "Unknown"

                # Extract test results
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
                            else "Not Significant"
                        ),
                        "t-statistic": round(paired_test.get("statistic", 0), 3),
                        "p-value (Paired)": round(paired_test.get("p_value", 1), 4),
                        "DM Test": (
                            "Significant"
                            if dm_test.get("significant", False)
                            else "Not Significant"
                        ),
                        "DM Statistic": round(dm_test.get("statistic", 0), 3),
                        "p-value (DM)": round(dm_test.get("p_value", 1), 4),
                        "Sharpe Ratio Test": (
                            "Significant"
                            if sharpe_test.get("significant", False)
                            else "Not Significant"
                        ),
                        "Sharpe Difference": round(
                            sharpe_test.get("sharpe_diff", 0), 3
                        ),
                        "p-value (Sharpe)": round(sharpe_test.get("p_value", 1), 4),
                        "Statistical Power": round(
                            summary.get("statistical_power", 0), 3
                        ),
                        "Conclusion": summary.get(
                            "overall_conclusion", "No Conclusion"
                        ),
                    }
                )

            except (KeyError, TypeError, IndexError) as e:
                logger.warning(
                    f"Error processing comparison result {comparison_key}: {e}"
                )
                continue

        if not significance_data:
            return pd.DataFrame()

        return pd.DataFrame(significance_data)

    def _calculate_dimension_scores(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate dimension scores"""
        scores = {}

        # Return performance score (0-100)
        total_return = metrics.get("total_return", 0)
        annual_return = metrics.get("annual_return", 0)
        scores["Return Performance"] = min(
            100, max(0, (total_return + annual_return) * 50)
        )

        # Risk control score (0-100, lower is better)
        max_dd = abs(metrics.get("max_drawdown", 0))
        volatility = metrics.get("volatility", 0)
        var_95 = abs(metrics.get("var_95", 0))
        risk_penalty = (max_dd + volatility + var_95) * 100
        scores["Risk Control"] = max(0, 100 - risk_penalty)

        # Risk-adjusted return score (0-100)
        sharpe = metrics.get("sharpe_ratio", 0)
        sortino = metrics.get("sortino_ratio", 0)
        calmar = metrics.get("calmar_ratio", 0)
        scores["Risk-Adjusted Return"] = min(
            100, max(0, (sharpe + sortino + calmar) * 20)
        )

        # Trading efficiency score (0-100)
        win_rate = metrics.get("win_rate", 0)
        pl_ratio = metrics.get("profit_loss_ratio", 0)
        trade_count = metrics.get("trade_count", 0)
        efficiency_score = win_rate * 50 + min(50, pl_ratio * 25)
        if trade_count > 0:
            efficiency_score += min(20, trade_count / 10)
        scores["Trading Efficiency"] = min(100, efficiency_score)

        return scores

    def _export_csv_tables(self, *tables) -> Dict[str, str]:
        """Export CSV format tables"""
        csv_files = {}
        table_names = [
            "performance_comparison",
            "strategy_ranking",
            "risk_metrics",
            "trading_statistics",
            "ai_agent_analysis",
            "statistical_significance",
        ]

        for i, table in enumerate(tables):
            if table is not None and not table.empty:
                filename = f"{table_names[i]}_{self.timestamp}.csv"
                filepath = self.output_dir / filename

                try:
                    table.to_csv(filepath, index=False, encoding="utf-8-sig")
                    csv_files[table_names[i]] = str(filepath)
                    logger.info(f"CSV table saved: {filepath}")
                except Exception as e:
                    logger.error(f"Failed to save CSV table {filename}: {e}")

        return csv_files

    def _export_excel_workbook(self, *tables, config=None) -> Optional[str]:
        """Export Excel workbook"""
        try:
            filename = f"backtest_comprehensive_report_{self.timestamp}.xlsx"
            filepath = self.output_dir / filename

            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                sheet_names = [
                    "Performance Comparison",
                    "Strategy Ranking",
                    "Risk Metrics",
                    "Trading Statistics",
                    "AI Special Analysis",
                    "Statistical Significance",
                ]

                for i, (table, sheet_name) in enumerate(zip(tables, sheet_names)):
                    if table is not None and not table.empty:
                        table.to_excel(writer, sheet_name=sheet_name, index=False)

                # Add configuration information worksheet
                if config:
                    config_df = pd.DataFrame(
                        list(config.items()), columns=["Configuration Item", "Value"]
                    )
                    config_df.to_excel(
                        writer, sheet_name="Backtest Configuration", index=False
                    )

            logger.info(f"Excel report saved: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save Excel report: {e}")
            return None

    def _export_html_report(self, *tables, config=None) -> Optional[str]:
        """Export HTML report"""
        try:
            filename = f"backtest_report_{self.timestamp}.html"
            filepath = self.output_dir / filename

            html_content = self._generate_html_content(tables, config)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"HTML report saved: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save HTML report: {e}")
            return None

    def _generate_html_content(self, tables, config=None) -> str:
        """Generate HTML report content"""
        html_parts = [
            """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Backtest Analysis Report</title>
                <style>
                    body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                    .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                    h1 { color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                    h2 { color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }
                    table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 12px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
                    th { background-color: #3498db; color: white; font-weight: bold; }
                    tr:nth-child(even) { background-color: #f2f2f2; }
                    tr:hover { background-color: #e8f4fd; }
                    .config-info { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
                    .timestamp { text-align: right; color: #7f8c8d; font-size: 12px; margin-top: 20px; }
                    .highlight { background-color: #f39c12; color: white; font-weight: bold; }
                    .positive { color: #27ae60; font-weight: bold; }
                    .negative { color: #e74c3c; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚀 A-Share Investment Agent Backtest Analysis Report</h1>
            """
        ]

        # Add configuration information
        if config:
            html_parts.append('<div class="config-info">')
            html_parts.append("<h3>📋 Backtest Configuration Information</h3>")
            for key, value in config.items():
                html_parts.append(f"<p><strong>{key}:</strong> {value}</p>")
            html_parts.append("</div>")

        # Add tables
        table_titles = [
            "📊 Strategy Performance Comparison Table",
            "🏆 Strategy Comprehensive Ranking Table",
            "⚠️ Risk Metrics Comparison Table",
            "📈 Trading Statistics Table",
            "🤖 AI Agent Special Analysis",
            "📉 Statistical Significance Test",
        ]

        for i, (table, title) in enumerate(zip(tables, table_titles)):
            if table is not None and not table.empty:
                html_parts.append(f"<h2>{title}</h2>")

                # Add styles to table
                table_html = table.to_html(index=False, escape=False, classes="table")

                # Add color coding
                if "Rank" in table.columns:
                    # Add color for ranking
                    table_html = table_html.replace(
                        "<td>1</td>", '<td class="highlight">1</td>'
                    )

                html_parts.append(table_html)

        # Add footer
        html_parts.append(
            f"""
                    <div class="timestamp">
                        <p>Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                        <p>© A-Share Investment Agent System - Backtest Analysis Report</p>
                    </div>
                </div>
            </body>
            </html>
        """
        )

        return "".join(html_parts)
