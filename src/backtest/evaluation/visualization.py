import os
import platform
from typing import Any, Dict

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging_config import setup_logger

# Setup logger
logger = setup_logger("backtest_visualizer")


# More robust Chinese font setup
def setup_chinese_fonts():
    """Setup Chinese font support"""
    try:
        # Windows system prioritizes Microsoft YaHei
        chinese_fonts = [
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]

        # Check available fonts
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        selected_font = None

        for font in chinese_fonts:
            if font in available_fonts:
                selected_font = font
                break

        # Set font configuration optimized for Windows
        if selected_font:
            plt.rcParams["font.sans-serif"] = [selected_font]
        else:
            plt.rcParams["font.sans-serif"] = [
                "Microsoft YaHei",
                "SimHei",
                "DejaVu Sans",
            ]

        plt.rcParams["axes.unicode_minus"] = False

        # Set default font size and style
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.titlesize": 16,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.edgecolor": "none",
            }
        )
    except Exception as e:
        logger.warning(f"Font setup failed, using default font: {e}")


# Initialize font setup
setup_chinese_fonts()


class BacktestVisualizer:
    """
    Backtest visualization class
    Generates professional backtest analysis charts
    """

    def __init__(self):
        self.style_config = {
            "figure_size": (12, 8),
            "dpi": 150,  # Lower DPI to avoid memory issues
            "color_palette": "Set2",
        }
        self._setup_style()

    def _setup_style(self):
        """Setup chart style"""
        try:
            # Ensure font setup takes effect
            setup_chinese_fonts()

            # Use more stable style settings
            try:
                # Try to use seaborn style
                available_styles = plt.style.available
                if "seaborn-v0_8" in available_styles:
                    plt.style.use("seaborn-v0_8")
                elif any("seaborn" in style for style in available_styles):
                    # Use any available seaborn style
                    seaborn_styles = [s for s in available_styles if "seaborn" in s]
                    plt.style.use(seaborn_styles[0])
                else:
                    # If seaborn not available, use default style and set manually
                    plt.style.use("default")

            except Exception as e:
                logger.warning(f"Style setup failed, using default style: {e}")
                plt.style.use("default")

            # Manually set style parameters to ensure consistency
            plt.rcParams.update(
                {
                    "figure.figsize": self.style_config["figure_size"],
                    "figure.dpi": 100,  # Use lower DPI
                    "axes.grid": True,
                    "grid.alpha": 0.3,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "legend.frameon": True,
                    "legend.framealpha": 0.8,
                }
            )

            # Set color palette
            try:
                sns.set_palette(self.style_config["color_palette"])
            except Exception as e:
                logger.warning(f"Color palette setup failed: {e}")

        except Exception as e:
            logger.error(f"Style setup failed: {e}")

    def create_comparison_charts(
        self, results: Dict[str, Any], comparison_results: Dict[str, Any], save_dir: str
    ) -> Dict[str, str]:
        """
        Create complete comparison chart set

        Args:
            results: Strategy results
            comparison_results: Comparison analysis results
            save_dir: Save directory

        Returns:
            Dict: Generated chart file paths
        """
        try:
            # Create save directory
            os.makedirs(save_dir, exist_ok=True)

            chart_paths = {}

            # Check if there is valid result data
            if not results or len(results) == 0:
                logger.warning("No valid backtest result data")
                return chart_paths

            # 1. Comprehensive performance comparison chart
            try:
                chart_paths["performance_comparison"] = (
                    self._create_performance_comparison(
                        results, os.path.join(save_dir, "performance_comparison.png")
                    )
                )
            except Exception as e:
                logger.error(f"Failed to create performance comparison chart: {e}")

            # 2. Cumulative returns comparison chart
            try:
                chart_paths["cumulative_returns"] = (
                    self._create_cumulative_returns_chart(
                        results, os.path.join(save_dir, "cumulative_returns.png")
                    )
                )
            except Exception as e:
                logger.error(f"Failed to create cumulative returns chart: {e}")

            # 3. Risk-return scatter plot
            try:
                chart_paths["risk_return_scatter"] = self._create_risk_return_scatter(
                    results, os.path.join(save_dir, "risk_return_scatter.png")
                )
            except Exception as e:
                logger.error(f"Failed to create risk-return scatter plot: {e}")

            # 4. Drawdown analysis chart
            try:
                chart_paths["drawdown_analysis"] = self._create_drawdown_analysis(
                    results, os.path.join(save_dir, "drawdown_analysis.png")
                )
            except Exception as e:
                logger.error(f"Failed to create drawdown analysis chart: {e}")

            # 5. Strategy radar chart
            if comparison_results and "strategy_ranking" in comparison_results:
                try:
                    chart_paths["radar_chart"] = self._create_strategy_radar_chart(
                        comparison_results["strategy_ranking"],
                        os.path.join(save_dir, "strategy_radar.png"),
                    )
                except Exception as e:
                    logger.error(f"Failed to create strategy radar chart: {e}")

            # 6. Rolling metrics chart
            try:
                chart_paths["rolling_metrics"] = self._create_rolling_metrics_chart(
                    results, os.path.join(save_dir, "rolling_metrics.png")
                )
            except Exception as e:
                logger.error(f"Failed to create rolling metrics chart: {e}")

            # 7. Monthly returns heatmap
            try:
                chart_paths["monthly_returns"] = self._create_monthly_returns_heatmap(
                    results, os.path.join(save_dir, "monthly_returns.png")
                )
            except Exception as e:
                logger.error(f"Failed to create monthly returns heatmap: {e}")

            logger.info(f"Successfully created {len(chart_paths)} charts")
            return chart_paths

        except Exception as e:
            logger.error(f"Failed to create chart set: {e}")
            return {}

    def _create_performance_comparison(
        self, results: Dict[str, Any], filepath: str
    ) -> str:
        """Create comprehensive performance comparison chart"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # Extract data
            strategy_names = list(results.keys())
            if not strategy_names:
                logger.warning("No strategy data")
                return ""

            total_returns = []
            sharpe_ratios = []
            max_drawdowns = []
            volatilities = []

            for name in strategy_names:
                result = results[name]
                metrics = getattr(result, "performance_metrics", {})

                total_returns.append(metrics.get("total_return", 0) * 100)
                sharpe_ratios.append(metrics.get("sharpe_ratio", 0))
                max_drawdowns.append(abs(metrics.get("max_drawdown", 0)) * 100)
                volatilities.append(metrics.get("volatility", 0) * 100)

            # 1. Total return comparison
            try:
                colors = sns.color_palette(
                    self.style_config["color_palette"], len(strategy_names)
                )
            except:
                colors = plt.cm.Set2(np.linspace(0, 1, len(strategy_names)))

            bars1 = ax1.bar(strategy_names, total_returns, color=colors)
            ax1.set_title("Total Return Comparison", fontsize=14, fontweight="bold")
            ax1.set_ylabel("Return (%)", fontsize=12)
            ax1.tick_params(axis="x", rotation=45)
            ax1.grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars1, total_returns):
                height = bar.get_height()
                if np.isfinite(height) and np.isfinite(value):
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{value:.2f}%",
                        ha="center",
                        va="bottom",
                    )

            # 2. Sharpe ratio comparison
            bars2 = ax2.bar(strategy_names, sharpe_ratios, color=colors)
            ax2.set_title("Sharpe Ratio Comparison", fontsize=14, fontweight="bold")
            ax2.set_ylabel("Sharpe Ratio", fontsize=12)
            ax2.tick_params(axis="x", rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(
                y=1.0, color="red", linestyle="--", alpha=0.7, label="Excellent Level"
            )
            ax2.legend()

            for bar, value in zip(bars2, sharpe_ratios):
                height = bar.get_height()
                if np.isfinite(height) and np.isfinite(value):
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                    )

            # 3. Maximum drawdown comparison
            bars3 = ax3.bar(strategy_names, max_drawdowns, color=colors)
            ax3.set_title("Maximum Drawdown Comparison", fontsize=14, fontweight="bold")
            ax3.set_ylabel("Maximum Drawdown (%)", fontsize=12)
            ax3.tick_params(axis="x", rotation=45)
            ax3.grid(True, alpha=0.3)

            for bar, value in zip(bars3, max_drawdowns):
                height = bar.get_height()
                if np.isfinite(height) and np.isfinite(value):
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{value:.2f}%",
                        ha="center",
                        va="bottom",
                    )

            # 4. Volatility comparison
            bars4 = ax4.bar(strategy_names, volatilities, color=colors)
            ax4.set_title(
                "Annualized Volatility Comparison", fontsize=14, fontweight="bold"
            )
            ax4.set_ylabel("Volatility (%)", fontsize=12)
            ax4.tick_params(axis="x", rotation=45)
            ax4.grid(True, alpha=0.3)

            for bar, value in zip(bars4, volatilities):
                height = bar.get_height()
                if np.isfinite(height) and np.isfinite(value):
                    ax4.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{value:.2f}%",
                        ha="center",
                        va="bottom",
                    )

            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config["dpi"], bbox_inches="tight")
            plt.close(fig)

            logger.info(
                f"Successfully created performance comparison chart: {filepath}"
            )
            return filepath

        except Exception as e:
            logger.error(f"Failed to create performance comparison chart: {e}")
            return ""

    def _create_cumulative_returns_chart(
        self, results: Dict[str, Any], filepath: str
    ) -> str:
        """Create cumulative returns comparison chart"""
        try:
            fig, ax = plt.subplots(figsize=self.style_config["figure_size"])

            try:
                colors = sns.color_palette(
                    self.style_config["color_palette"], len(results)
                )
            except:
                colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

            valid_data_count = 0
            for i, (name, result) in enumerate(results.items()):
                try:
                    daily_returns = getattr(result, "daily_returns", None)
                    if daily_returns is not None and len(daily_returns) > 0:
                        # Ensure data is valid
                        daily_returns = np.array(daily_returns)
                        daily_returns = daily_returns[np.isfinite(daily_returns)]

                        if len(daily_returns) > 0:
                            # Calculate cumulative returns
                            cumulative_returns = np.cumprod(1 + daily_returns) - 1

                            # Create date index
                            dates = pd.date_range(
                                start=pd.Timestamp.now()
                                - pd.Timedelta(days=len(cumulative_returns)),
                                periods=len(cumulative_returns),
                                freq="D",
                            )

                            ax.plot(
                                dates,
                                cumulative_returns * 100,
                                label=name,
                                color=colors[i],
                                linewidth=2,
                            )
                            valid_data_count += 1
                except Exception as e:
                    logger.warning(f"Error processing strategy {name} data: {e}")
                    continue

            if valid_data_count == 0:
                logger.warning("No valid return data")
                plt.close(fig)
                return ""

            ax.set_title("Cumulative Return Comparison", fontsize=16, fontweight="bold")
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Cumulative Return (%)", fontsize=12)
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

            # Set date format
            try:
                import matplotlib.dates as mdates

                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.xticks(rotation=45)
            except Exception as e:
                logger.warning(f"Failed to set date format: {e}")

            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config["dpi"], bbox_inches="tight")
            plt.close(fig)

            logger.info(f"Successfully created cumulative returns chart: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to create cumulative returns chart: {e}")
            return ""

    def _create_risk_return_scatter(
        self, results: Dict[str, Any], filepath: str
    ) -> str:
        """Create risk-return scatter plot"""
        try:
            fig, ax = plt.subplots(figsize=self.style_config["figure_size"])

            strategy_names = []
            annual_returns = []
            volatilities = []
            sharpe_ratios = []

            for name, result in results.items():
                try:
                    metrics = getattr(result, "performance_metrics", {})

                    annual_return = metrics.get("annual_return", 0) * 100
                    volatility = metrics.get("volatility", 0) * 100
                    sharpe_ratio = metrics.get("sharpe_ratio", 0)

                    # Validate data validity
                    if (
                        np.isfinite(annual_return)
                        and np.isfinite(volatility)
                        and np.isfinite(sharpe_ratio)
                    ):
                        strategy_names.append(name)
                        annual_returns.append(annual_return)
                        volatilities.append(volatility)
                        sharpe_ratios.append(sharpe_ratio)
                except Exception as e:
                    logger.warning(
                        f"Error processing strategy {name} risk-return data: {e}"
                    )
                    continue

            if len(strategy_names) == 0:
                logger.warning("No valid risk-return data")
                plt.close(fig)
                return ""

            # Create scatter plot, point size represents Sharpe ratio
            sizes = [max(100, abs(sr) * 200) for sr in sharpe_ratios]
            try:
                colors = sns.color_palette("viridis", len(strategy_names))
            except:
                colors = plt.cm.viridis(np.linspace(0, 1, len(strategy_names)))

            scatter = ax.scatter(
                volatilities,
                annual_returns,
                s=sizes,
                c=colors,
                alpha=0.7,
                edgecolors="black",
            )

            # Add strategy name labels
            for i, name in enumerate(strategy_names):
                ax.annotate(
                    name,
                    (volatilities[i], annual_returns[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=10,
                    ha="left",
                )

            # Add efficient frontier reference lines
            if len(volatilities) > 1:
                try:
                    # Draw risk level lines
                    max_vol = max(volatilities)
                    vol_range = np.linspace(0, max_vol * 1.1, 100)

                    # Assume risk-free rate of 3%
                    risk_free_rate = 3.0

                    # Draw iso-Sharpe ratio lines
                    for sr in [0.5, 1.0, 1.5, 2.0]:
                        expected_returns = risk_free_rate + sr * vol_range
                        ax.plot(
                            vol_range,
                            expected_returns,
                            "--",
                            alpha=0.5,
                            label=f"Sharpe Ratio={sr}",
                        )
                except Exception as e:
                    logger.warning(f"Failed to draw reference lines: {e}")

            ax.set_title("Risk-Return Scatter Plot", fontsize=16, fontweight="bold")
            ax.set_xlabel("Annualized Volatility (%)", fontsize=12)
            ax.set_ylabel("Annualized Return (%)", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add explanatory text
            ax.text(
                0.02,
                0.98,
                "Point size represents Sharpe ratio",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config["dpi"], bbox_inches="tight")
            plt.close(fig)

            logger.info(f"Successfully created risk-return scatter plot: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to create risk-return scatter plot: {e}")
            return ""

    def _create_drawdown_analysis(self, results: Dict[str, Any], filepath: str) -> str:
        """Create drawdown analysis chart"""
        try:
            valid_results = []
            for name, result in results.items():
                daily_returns = getattr(result, "daily_returns", None)
                if daily_returns is not None and len(daily_returns) > 0:
                    valid_results.append((name, result))

            if not valid_results:
                logger.warning("No valid drawdown data")
                return ""

            fig, axes = plt.subplots(
                len(valid_results),
                1,
                figsize=(self.style_config["figure_size"][0], len(valid_results) * 4),
            )

            if len(valid_results) == 1:
                axes = [axes]

            try:
                colors = sns.color_palette(
                    self.style_config["color_palette"], len(valid_results)
                )
            except:
                colors = plt.cm.Set2(np.linspace(0, 1, len(valid_results)))

            for i, (name, result) in enumerate(valid_results):
                try:
                    daily_returns = np.array(result.daily_returns)
                    daily_returns = daily_returns[np.isfinite(daily_returns)]

                    if len(daily_returns) > 0:
                        # Calculate cumulative returns and drawdown
                        cumulative = np.cumprod(1 + daily_returns)
                        running_max = np.maximum.accumulate(cumulative)
                        drawdown = (cumulative / running_max - 1) * 100

                        # Create date index
                        dates = pd.date_range(
                            start=pd.Timestamp.now() - pd.Timedelta(days=len(drawdown)),
                            periods=len(drawdown),
                            freq="D",
                        )

                        # Plot drawdown
                        axes[i].fill_between(
                            dates,
                            0,
                            drawdown,
                            color=colors[i],
                            alpha=0.3,
                            label="Drawdown",
                        )
                        axes[i].plot(dates, drawdown, color=colors[i], linewidth=1)

                        # Mark maximum drawdown point
                        max_dd_idx = np.argmin(drawdown)
                        max_dd_value = drawdown[max_dd_idx]
                        axes[i].scatter(
                            dates[max_dd_idx],
                            max_dd_value,
                            color="red",
                            s=100,
                            zorder=5,
                        )
                        axes[i].annotate(
                            f"Max Drawdown: {max_dd_value:.2f}%",
                            xy=(dates[max_dd_idx], max_dd_value),
                            xytext=(10, -10),
                            textcoords="offset points",
                            bbox=dict(
                                boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7
                            ),
                        )

                        axes[i].set_title(
                            f"{name} - Drawdown Analysis",
                            fontsize=14,
                            fontweight="bold",
                        )
                        axes[i].set_ylabel("Drawdown (%)", fontsize=12)
                        axes[i].grid(True, alpha=0.3)
                        axes[i].axhline(
                            y=-5,
                            color="orange",
                            linestyle="--",
                            alpha=0.7,
                            label="5% Drawdown Line",
                        )
                        axes[i].axhline(
                            y=-10,
                            color="red",
                            linestyle="--",
                            alpha=0.7,
                            label="10% Drawdown Line",
                        )
                        axes[i].legend()

                        # Set date format
                        try:
                            import matplotlib.dates as mdates

                            axes[i].xaxis.set_major_formatter(
                                mdates.DateFormatter("%Y-%m")
                            )
                            axes[i].tick_params(axis="x", rotation=45)
                        except Exception as e:
                            logger.warning(f"Failed to set date format: {e}")

                except Exception as e:
                    logger.warning(
                        f"Error processing strategy {name} drawdown data: {e}"
                    )
                    continue

            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config["dpi"], bbox_inches="tight")
            plt.close(fig)

            logger.info(f"Successfully created drawdown analysis chart: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to create drawdown analysis chart: {e}")
            return ""

    def _create_strategy_radar_chart(
        self, ranking_data: Dict[str, Any], filepath: str
    ) -> str:
        """Create strategy radar chart"""
        try:
            rankings = ranking_data.get("rankings", [])
            if not rankings:
                logger.warning("No strategy ranking data")
                return ""

            # Take top 5 strategies
            top_strategies = rankings[:5]

            if not top_strategies:
                logger.warning("No valid strategy data")
                return ""

            # Set radar chart
            dimensions = ["Return", "Risk", "Stability", "Efficiency", "Robustness"]
            angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
            angles += angles[:1]  # Close the shape

            fig, ax = plt.subplots(
                figsize=(10, 10), subplot_kw=dict(projection="polar")
            )

            try:
                colors = sns.color_palette(
                    self.style_config["color_palette"], len(top_strategies)
                )
            except:
                colors = plt.cm.Set2(np.linspace(0, 1, len(top_strategies)))

            for i, strategy in enumerate(top_strategies):
                try:
                    # Check if strategy has scores attribute
                    if hasattr(strategy, "scores"):
                        scores = strategy.scores
                    elif isinstance(strategy, dict) and "scores" in strategy:
                        scores = strategy["scores"]
                    else:
                        logger.warning(
                            f"Strategy {getattr(strategy, 'name', 'Unknown')} has no scores data"
                        )
                        continue

                    # Get strategy name
                    if hasattr(strategy, "name"):
                        strategy_name = strategy.name
                    elif isinstance(strategy, dict) and "name" in strategy:
                        strategy_name = strategy["name"]
                    else:
                        strategy_name = f"Strategy{i+1}"

                    values = [
                        scores.get("return", 0),
                        scores.get("risk", 0),
                        scores.get("stability", 0),
                        scores.get("efficiency", 0),
                        scores.get("robustness", 0),
                    ]

                    # Validate value validity
                    values = [v if np.isfinite(v) else 0 for v in values]
                    values += values[:1]  # Close the shape

                    ax.plot(
                        angles,
                        values,
                        "o-",
                        linewidth=2,
                        label=strategy_name,
                        color=colors[i],
                    )
                    ax.fill(angles, values, alpha=0.25, color=colors[i])

                except Exception as e:
                    logger.warning(f"Error processing strategy radar chart data: {e}")
                    continue

            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(dimensions, fontsize=12)
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=10)
            ax.grid(True)

            # Add title and legend
            ax.set_title(
                "Strategy Five-Dimension Radar Chart",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config["dpi"], bbox_inches="tight")
            plt.close(fig)

            logger.info(f"Successfully created strategy radar chart: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to create strategy radar chart: {e}")
            return ""

    def _create_rolling_metrics_chart(
        self, results: Dict[str, Any], filepath: str
    ) -> str:
        """Create rolling metrics chart"""
        try:
            # Filter valid data
            valid_results = []
            for name, result in results.items():
                daily_returns = getattr(result, "daily_returns", None)
                if daily_returns is not None and len(daily_returns) > 60:
                    valid_results.append((name, result))

            if not valid_results:
                logger.warning(
                    "Insufficient data to create rolling metrics chart (need at least 60 days of data)"
                )
                return ""

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            try:
                colors = sns.color_palette(
                    self.style_config["color_palette"], len(valid_results)
                )
            except:
                colors = plt.cm.Set2(np.linspace(0, 1, len(valid_results)))

            for i, (name, result) in enumerate(valid_results):
                try:
                    returns = np.array(result.daily_returns)
                    returns = returns[np.isfinite(returns)]

                    if len(returns) <= 60:
                        continue

                    # Calculate rolling metrics
                    window = 60  # 60-day rolling window

                    # Rolling returns
                    returns_series = pd.Series(returns)
                    rolling_returns = (
                        returns_series.rolling(window).apply(
                            lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0, raw=True
                        )
                        * 100
                    )

                    # Rolling Sharpe ratio
                    def safe_rolling_sharpe(x):
                        try:
                            if len(x) == 0:
                                return 0.0
                            mean_ret = np.mean(x)
                            std_ret = np.std(x, ddof=1)
                            if (
                                std_ret == 0
                                or not np.isfinite(std_ret)
                                or not np.isfinite(mean_ret)
                            ):
                                return 0.0
                            sharpe = mean_ret / std_ret * np.sqrt(252)
                            return sharpe if np.isfinite(sharpe) else 0.0
                        except:
                            return 0.0

                    rolling_sharpe = returns_series.rolling(window).apply(
                        safe_rolling_sharpe, raw=True
                    )

                    # Rolling volatility
                    rolling_vol = (
                        returns_series.rolling(window).std() * np.sqrt(252) * 100
                    )

                    # Rolling maximum drawdown
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

                    rolling_dd = (
                        returns_series.rolling(window).apply(
                            safe_rolling_max_drawdown, raw=True
                        )
                        * 100
                    )

                    # Create date index
                    dates = pd.date_range(
                        start=pd.Timestamp.now() - pd.Timedelta(days=len(returns)),
                        periods=len(returns),
                        freq="D",
                    )

                    # Plot rolling metrics (only plot valid data)
                    valid_mask = (
                        np.isfinite(rolling_returns)
                        & np.isfinite(rolling_sharpe)
                        & np.isfinite(rolling_vol)
                        & np.isfinite(rolling_dd)
                    )

                    if np.any(valid_mask):
                        ax1.plot(
                            dates[valid_mask],
                            rolling_returns[valid_mask],
                            label=name,
                            color=colors[i],
                        )
                        ax2.plot(
                            dates[valid_mask],
                            rolling_sharpe[valid_mask],
                            label=name,
                            color=colors[i],
                        )
                        ax3.plot(
                            dates[valid_mask],
                            rolling_vol[valid_mask],
                            label=name,
                            color=colors[i],
                        )
                        ax4.plot(
                            dates[valid_mask],
                            rolling_dd[valid_mask],
                            label=name,
                            color=colors[i],
                        )

                except Exception as e:
                    logger.warning(
                        f"Error processing strategy {name} rolling metrics: {e}"
                    )
                    continue

            # Set chart
            ax1.set_title("60-Day Rolling Returns", fontsize=14, fontweight="bold")
            ax1.set_ylabel("Returns (%)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.set_title("60-Day Rolling Sharpe Ratio", fontsize=14, fontweight="bold")
            ax2.set_ylabel("Sharpe Ratio")
            ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.7)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            ax3.set_title("60-Day Rolling Volatility", fontsize=14, fontweight="bold")
            ax3.set_ylabel("Volatility (%)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            ax4.set_title(
                "60-Day Rolling Maximum Drawdown", fontsize=14, fontweight="bold"
            )
            ax4.set_ylabel("Maximum Drawdown (%)")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # Set date format
            try:
                import matplotlib.dates as mdates

                for ax in [ax1, ax2, ax3, ax4]:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                    ax.tick_params(axis="x", rotation=45)
            except Exception as e:
                logger.warning(f"Failed to set date format: {e}")

            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config["dpi"], bbox_inches="tight")
            plt.close(fig)

            logger.info(f"Successfully created rolling metrics chart: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to create rolling metrics chart: {e}")
            return ""

    def _calculate_rolling_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate rolling maximum drawdown"""
        try:
            if len(returns) == 0:
                return 0.0
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative / running_max) - 1
            return np.min(drawdown)
        except:
            return 0.0

    def _create_monthly_returns_heatmap(
        self, results: Dict[str, Any], filepath: str
    ) -> str:
        """Create monthly returns heatmap"""
        try:
            # Select first strategy as example (or can choose AI Agent)
            selected_result = None
            strategy_name = ""

            if "AI Agent" in results:
                selected_result = results["AI Agent"]
                strategy_name = "AI Agent"
            else:
                # Select first strategy with valid data
                for name, result in results.items():
                    daily_returns = getattr(result, "daily_returns", None)
                    if daily_returns is not None and len(daily_returns) >= 30:
                        selected_result = result
                        strategy_name = name
                        break

            if selected_result is None:
                logger.warning(
                    "Insufficient data to generate monthly returns heatmap (need at least 30 days of data)"
                )
                return ""

            daily_returns = getattr(selected_result, "daily_returns", None)
            if daily_returns is None or len(daily_returns) < 30:
                logger.warning("Insufficient data to generate monthly returns heatmap")
                return ""

            # Create date index and returns series
            returns = np.array(daily_returns)
            returns = returns[np.isfinite(returns)]

            if len(returns) < 30:
                logger.warning(
                    "Insufficient valid data to generate monthly returns heatmap"
                )
                return ""

            dates = pd.date_range(
                start=pd.Timestamp.now() - pd.Timedelta(days=len(returns)),
                periods=len(returns),
                freq="D",
            )

            returns_series = pd.Series(returns, index=dates)

            # Calculate monthly returns
            try:
                monthly_returns = returns_series.resample("M").apply(
                    lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
                )
            except Exception as e:
                logger.warning(f"Failed to calculate monthly returns: {e}")
                return ""

            # Create year-month matrix
            monthly_data = []
            for date, ret in monthly_returns.items():
                if np.isfinite(ret):
                    monthly_data.append(
                        {"Year": date.year, "Month": date.month, "Return": ret * 100}
                    )

            if not monthly_data:
                logger.warning("Unable to calculate valid monthly returns")
                return ""

            df_monthly = pd.DataFrame(monthly_data)

            try:
                pivot_table = df_monthly.pivot(
                    index="Year", columns="Month", values="Return"
                )
            except Exception as e:
                logger.warning(f"Failed to create pivot table: {e}")
                return ""

            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))

            try:
                # Use diverging color mapping
                sns.heatmap(
                    pivot_table,
                    annot=True,
                    fmt=".2f",
                    cmap="RdYlGn",
                    center=0,
                    cbar_kws={"label": "Monthly Returns (%)"},
                    ax=ax,
                )

                # Set month labels
                month_labels = [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ]

                # Only set existing month labels
                existing_months = pivot_table.columns
                month_labels_filtered = [
                    month_labels[i - 1] for i in existing_months if 1 <= i <= 12
                ]
                ax.set_xticklabels(month_labels_filtered)

                ax.set_title(
                    f"{strategy_name} - Monthly Returns Heatmap",
                    fontsize=16,
                    fontweight="bold",
                )
                ax.set_xlabel("Month", fontsize=12)
                ax.set_ylabel("Year", fontsize=12)

            except Exception as e:
                logger.warning(f"Failed to create heatmap: {e}")
                plt.close(fig)
                return ""

            plt.tight_layout()
            plt.savefig(filepath, dpi=self.style_config["dpi"], bbox_inches="tight")
            plt.close(fig)

            logger.info(f"Successfully created monthly returns heatmap: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to create monthly returns heatmap: {e}")
            return ""
