import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src.utils.logging_config import setup_logger

# Setup logging
logger = setup_logger("model_evaluation")


class ModelEvaluator:
    """Model evaluation class, responsible for data splitting, model evaluation and result visualization"""

    def __init__(self, output_dir: str = "models/evaluation"):
        """
        Initialize evaluator

        Args:
            output_dir: Evaluation result output directory
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def split_data(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        shuffle: bool = False,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset according to given ratios

        Args:
            data: Data to be split
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            shuffle: Whether to shuffle data
            random_state: Random seed

        Returns:
            Tuple of training set, validation set, test set
        """
        # Ensure ratios sum to 1
        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10
        ), "Ratios must sum to 1"

        # Copy data to avoid modifying original data
        df = data.copy()

        if shuffle:
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Calculate split points
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        # Split data
        train_data = df.iloc[:train_end].copy()
        val_data = df.iloc[train_end:val_end].copy()
        test_data = df.iloc[val_end:].copy()

        logger.info(
            f"Data split - Training set: {len(train_data)} rows ({train_ratio*100:.1f}%), "
            f"Validation set: {len(val_data)} rows ({val_ratio*100:.1f}%), "
            f"Test set: {len(test_data)} rows ({test_ratio*100:.1f}%)"
        )

        return train_data, val_data, test_data

    def evaluate_regression_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
        dataset_name: str = "test",
    ) -> Dict[str, float]:
        """
        Evaluate regression model performance

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Model name
            dataset_name: Dataset name

        Returns:
            Dictionary containing evaluation metrics
        """
        # Calculate evaluation metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Record results
        metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

        logger.info(
            f"{model_name} regression evaluation results on {dataset_name} set:"
        )
        logger.info(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # Save metrics
        with open(
            f"{self.output_dir}/{model_name}_{dataset_name}_regression_metrics.json",
            "w",
        ) as f:
            json.dump(metrics, f, indent=4)

        return metrics

    def evaluate_classification_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
        dataset_name: str = "test",
    ) -> Dict[str, float]:
        """
        Evaluate classification model performance

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Model name
            dataset_name: Dataset name

        Returns:
            Dictionary containing evaluation metrics
        """
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Record results
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
        }

        logger.info(
            f"{model_name} classification evaluation results on {dataset_name} set:"
        )
        logger.info(
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}"
        )
        logger.info(f"Confusion matrix:\n{cm}")

        # Visualize confusion matrix
        self._plot_confusion_matrix(cm, model_name, dataset_name)

        # Save metrics
        with open(
            f"{self.output_dir}/{model_name}_{dataset_name}_classification_metrics.json",
            "w",
        ) as f:
            json.dump(metrics, f, indent=4)

        return metrics

    def visualize_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        date_index: Optional[pd.DatetimeIndex] = None,
        model_name: str = "model",
        dataset_name: str = "test",
        title: str = "Prediction vs True Value",
    ):
        """
        Visualize regression prediction results

        Args:
            y_true: True values
            y_pred: Predicted values
            date_index: Date index (optional)
            model_name: Model name
            dataset_name: Dataset name
            title: Chart title
        """
        plt.figure(figsize=(12, 6))

        # If there's a date index, use it as x-axis
        if date_index is not None and len(date_index) == len(y_true):
            plt.plot(
                date_index,
                y_true,
                label="True Values",
                marker="o",
                linestyle="-",
                markersize=3,
            )
            plt.plot(
                date_index,
                y_pred,
                label="Predicted Values",
                marker="x",
                linestyle="--",
                markersize=3,
            )
            plt.gcf().autofmt_xdate()
        else:
            plt.plot(
                y_true, label="True Values", marker="o", linestyle="-", markersize=3
            )
            plt.plot(
                y_pred,
                label="Predicted Values",
                marker="x",
                linestyle="--",
                markersize=3,
            )

        plt.title(title)
        plt.xlabel("Time" if date_index is not None else "Samples")
        plt.ylabel("Price")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Calculate errors
        errors = y_pred - y_true

        # Add error plot
        plt.figure(figsize=(12, 3))
        if date_index is not None and len(date_index) == len(errors):
            plt.bar(date_index, errors, color="r", alpha=0.6)
            plt.gcf().autofmt_xdate()
        else:
            plt.bar(range(len(errors)), errors, color="r", alpha=0.6)

        plt.title("Prediction Errors")
        plt.xlabel("Time" if date_index is not None else "Samples")
        plt.ylabel("Error")
        plt.grid(True, alpha=0.3)

        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(
            f"{self.output_dir}/{model_name}_{dataset_name}_prediction_{timestamp}.png"
        )
        plt.close("all")

    def visualize_forecast(
        self,
        history: np.ndarray,
        forecast: np.ndarray,
        date_index: Optional[pd.DatetimeIndex] = None,
        model_name: str = "model",
        title: str = "Price Forecast",
    ):
        """
        Visualize forecast results

        Args:
            history: Historical true values
            forecast: Predicted future values
            date_index: Date index containing history and future (optional)
            model_name: Model name
            title: Chart title
        """
        plt.figure(figsize=(12, 6))

        # Determine date index
        if date_index is not None and len(date_index) == len(history) + len(forecast):
            history_dates = date_index[: len(history)]
            forecast_dates = date_index[len(history) :]

            plt.plot(history_dates, history, label="Historical Data", color="blue")
            plt.plot(
                forecast_dates, forecast, label="Forecast", color="red", linestyle="--"
            )
            plt.axvline(x=forecast_dates[0], color="green", linestyle="-", alpha=0.5)
            plt.gcf().autofmt_xdate()
        else:
            # If no date index, use sample index
            x_history = np.arange(len(history))
            x_forecast = np.arange(len(history), len(history) + len(forecast))

            plt.plot(x_history, history, label="Historical Data", color="blue")
            plt.plot(
                x_forecast, forecast, label="Forecast", color="red", linestyle="--"
            )
            plt.axvline(x=len(history), color="green", linestyle="-", alpha=0.5)

        plt.title(title)
        plt.xlabel("Date" if date_index is not None else "Time Step")
        plt.ylabel("Price")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{self.output_dir}/{model_name}_forecast_{timestamp}.png")
        plt.close()

    def _plot_confusion_matrix(
        self, cm: np.ndarray, model_name: str, dataset_name: str
    ):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{model_name} Confusion Matrix on {dataset_name} Set")
        plt.ylabel("True Labels")
        plt.xlabel("Predicted Labels")

        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(
            f"{self.output_dir}/{model_name}_{dataset_name}_confusion_matrix_{timestamp}.png"
        )
        plt.close()

    def feature_importance_plot(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        model_name: str = "model",
        top_n: int = 20,
    ):
        """
        Plot feature importance chart

        Args:
            feature_names: List of feature names
            importances: Feature importance array
            model_name: Model name
            top_n: Display top N important features
        """
        # Create feature importance DataFrame
        feature_imp = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        # Take top N features
        if top_n < len(feature_imp):
            feature_imp = feature_imp.head(top_n)

        # Plot bar chart
        plt.figure(figsize=(10, 8))
        sns.barplot(x="importance", y="feature", data=feature_imp)
        plt.title(f"{model_name} Feature Importance (Top {len(feature_imp)})")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()

        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(
            f"{self.output_dir}/{model_name}_feature_importance_{timestamp}.png"
        )
        plt.close()

    def summary_statistics(
        self, data: pd.DataFrame, name: str = "dataset"
    ) -> Dict[str, Any]:
        """
        Calculate summary statistics for dataset

        Args:
            data: Data DataFrame
            name: Dataset name

        Returns:
            Statistics dictionary
        """
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])

        # Basic statistics
        stats = numeric_data.describe().T

        # Add other statistics
        stats["skew"] = numeric_data.skew()
        stats["kurtosis"] = numeric_data.kurtosis()

        # Save results
        stats_dict = stats.to_dict()
        with open(f"{self.output_dir}/{name}_statistics.json", "w") as f:
            json.dump(stats_dict, f, indent=4)

        logger.info(f"{name} dataset statistics saved")

        return stats_dict

    def compare_train_test_distributions(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title_prefix: str = "",
    ):
        """
        Compare training and test set distributions

        Args:
            train_data: Training data
            test_data: Test data
            columns: Columns to compare (defaults to all numeric columns)
            title_prefix: Title prefix
        """
        # Select columns to compare
        if columns is None:
            columns = train_data.select_dtypes(include=[np.number]).columns.tolist()

        # Limit number of columns to avoid generating too many charts
        if len(columns) > 10:
            logger.info(f"Too many columns ({len(columns)}), only showing first 10")
            columns = columns[:10]

        # Create histogram comparison for each column
        for col in columns:
            if col in train_data.columns and col in test_data.columns:
                plt.figure(figsize=(10, 6))

                # Plot training set distribution
                sns.histplot(
                    train_data[col].dropna(),
                    color="blue",
                    alpha=0.5,
                    label="Train",
                    kde=True,
                    stat="density",
                )

                # Plot test set distribution
                sns.histplot(
                    test_data[col].dropna(),
                    color="red",
                    alpha=0.5,
                    label="Test",
                    kde=True,
                    stat="density",
                )

                plt.title(f"{title_prefix} {col} Distribution Comparison")
                plt.xlabel(col)
                plt.ylabel("Density")
                plt.legend()

                # Save chart
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plt.savefig(
                    f"{self.output_dir}/distribution_compare_{col}_{timestamp}.png"
                )
                plt.close()

        logger.info(f"Generated {len(columns)} distribution comparison charts")


def create_evaluator(output_dir: str = "models/evaluation") -> ModelEvaluator:
    """
    Create and return evaluator instance factory function

    Args:
        output_dir: Evaluation result output directory

    Returns:
        ModelEvaluator instance
    """
    return ModelEvaluator(output_dir)
