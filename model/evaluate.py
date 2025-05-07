import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
from src.utils.logging_config import setup_logger
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# 设置日志
logger = setup_logger('model_evaluation')

class ModelEvaluator:
    """模型评估类，负责数据划分、模型评估和结果可视化"""
    
    def __init__(self, output_dir: str = 'models/evaluation'):
        """
        初始化评估器
        
        Args:
            output_dir: 评估结果输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def split_data(self, data: pd.DataFrame, train_ratio: float = 0.7, 
                  val_ratio: float = 0.2, test_ratio: float = 0.1,
                  shuffle: bool = False, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按照给定比例划分数据集
        
        Args:
            data: 要划分的数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            shuffle: 是否打乱数据
            random_state: 随机种子
            
        Returns:
            训练集、验证集、测试集的元组
        """
        # 确保比例和为1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "比例之和必须为1"
        
        # 复制数据以避免修改原始数据
        df = data.copy()
        
        if shuffle:
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # 计算分割点
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # 划分数据
        train_data = df.iloc[:train_end].copy()
        val_data = df.iloc[train_end:val_end].copy()
        test_data = df.iloc[val_end:].copy()
        
        logger.info(f"数据已划分 - 训练集: {len(train_data)}行 ({train_ratio*100:.1f}%), "
                   f"验证集: {len(val_data)}行 ({val_ratio*100:.1f}%), "
                   f"测试集: {len(test_data)}行 ({test_ratio*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def evaluate_regression_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 model_name: str = 'model', dataset_name: str = 'test') -> Dict[str, float]:
        """
        评估回归模型性能
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            dataset_name: 数据集名称
            
        Returns:
            包含评估指标的字典
        """
        # 计算评估指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 记录结果
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"{model_name} 在 {dataset_name} 集上的回归评估结果:")
        logger.info(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # 保存指标
        with open(f"{self.output_dir}/{model_name}_{dataset_name}_regression_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics
    
    def evaluate_classification_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     model_name: str = 'model', dataset_name: str = 'test') -> Dict[str, float]:
        """
        评估分类模型性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            model_name: 模型名称
            dataset_name: 数据集名称
            
        Returns:
            包含评估指标的字典
        """
        # 计算评估指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 记录结果
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
        
        logger.info(f"{model_name} 在 {dataset_name} 集上的分类评估结果:")
        logger.info(f"准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
        logger.info(f"混淆矩阵:\n{cm}")
        
        # 可视化混淆矩阵
        self._plot_confusion_matrix(cm, model_name, dataset_name)
        
        # 保存指标
        with open(f"{self.output_dir}/{model_name}_{dataset_name}_classification_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics
    
    def visualize_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             date_index: Optional[pd.DatetimeIndex] = None,
                             model_name: str = 'model', dataset_name: str = 'test', 
                             title: str = 'Prediction vs True Value'):
        """
        可视化回归预测结果
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            date_index: 日期索引（可选）
            model_name: 模型名称
            dataset_name: 数据集名称
            title: 图表标题
        """
        plt.figure(figsize=(12, 6))
        
        # 如果有日期索引，使用它作为x轴
        if date_index is not None and len(date_index) == len(y_true):
            plt.plot(date_index, y_true, label='真实值', marker='o', linestyle='-', markersize=3)
            plt.plot(date_index, y_pred, label='预测值', marker='x', linestyle='--', markersize=3)
            plt.gcf().autofmt_xdate()
        else:
            plt.plot(y_true, label='真实值', marker='o', linestyle='-', markersize=3)
            plt.plot(y_pred, label='预测值', marker='x', linestyle='--', markersize=3)
        
        plt.title(title)
        plt.xlabel('时间' if date_index is not None else '样本')
        plt.ylabel('价格')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 计算误差
        errors = y_pred - y_true
        
        # 添加误差图
        plt.figure(figsize=(12, 3))
        if date_index is not None and len(date_index) == len(errors):
            plt.bar(date_index, errors, color='r', alpha=0.6)
            plt.gcf().autofmt_xdate()
        else:
            plt.bar(range(len(errors)), errors, color='r', alpha=0.6)
        
        plt.title('预测误差')
        plt.xlabel('时间' if date_index is not None else '样本')
        plt.ylabel('误差')
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f"{self.output_dir}/{model_name}_{dataset_name}_prediction_{timestamp}.png")
        plt.close('all')
    
    def visualize_forecast(self, history: np.ndarray, forecast: np.ndarray, 
                          date_index: Optional[pd.DatetimeIndex] = None,
                          model_name: str = 'model', title: str = 'Price Forecast'):
        """
        可视化预测结果
        
        Args:
            history: 历史真实值
            forecast: 预测未来值
            date_index: 包含历史和未来的日期索引（可选）
            model_name: 模型名称
            title: 图表标题
        """
        plt.figure(figsize=(12, 6))
        
        # 确定日期索引
        if date_index is not None and len(date_index) == len(history) + len(forecast):
            history_dates = date_index[:len(history)]
            forecast_dates = date_index[len(history):]
            
            plt.plot(history_dates, history, label='历史数据', color='blue')
            plt.plot(forecast_dates, forecast, label='预测', color='red', linestyle='--')
            plt.axvline(x=forecast_dates[0], color='green', linestyle='-', alpha=0.5)
            plt.gcf().autofmt_xdate()
        else:
            # 如果没有日期索引，使用样本索引
            x_history = np.arange(len(history))
            x_forecast = np.arange(len(history), len(history) + len(forecast))
            
            plt.plot(x_history, history, label='历史数据', color='blue')
            plt.plot(x_forecast, forecast, label='预测', color='red', linestyle='--')
            plt.axvline(x=len(history), color='green', linestyle='-', alpha=0.5)
        
        plt.title(title)
        plt.xlabel('日期' if date_index is not None else '时间步')
        plt.ylabel('价格')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f"{self.output_dir}/{model_name}_forecast_{timestamp}.png")
        plt.close()
    
    def _plot_confusion_matrix(self, cm: np.ndarray, model_name: str, dataset_name: str):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'{model_name} 在 {dataset_name} 集上的混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f"{self.output_dir}/{model_name}_{dataset_name}_confusion_matrix_{timestamp}.png")
        plt.close()
    
    def feature_importance_plot(self, feature_names: List[str], importances: np.ndarray, 
                               model_name: str = 'model', top_n: int = 20):
        """
        绘制特征重要性图表
        
        Args:
            feature_names: 特征名称列表
            importances: 特征重要性数组
            model_name: 模型名称
            top_n: 显示前N个重要特征
        """
        # 创建特征重要性DataFrame
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # 截取前N个特征
        if top_n < len(feature_imp):
            feature_imp = feature_imp.head(top_n)
        
        # 绘制条形图
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_imp)
        plt.title(f'{model_name} 特征重要性 (Top {len(feature_imp)})')
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f"{self.output_dir}/{model_name}_feature_importance_{timestamp}.png")
        plt.close()

    def summary_statistics(self, data: pd.DataFrame, name: str = 'dataset') -> Dict[str, Any]:
        """
        计算数据集的摘要统计信息
        
        Args:
            data: 数据DataFrame
            name: 数据集名称
            
        Returns:
            统计信息字典
        """
        # 选择数值列
        numeric_data = data.select_dtypes(include=[np.number])
        
        # 基本统计信息
        stats = numeric_data.describe().T
        
        # 添加其他统计量
        stats['skew'] = numeric_data.skew()
        stats['kurtosis'] = numeric_data.kurtosis()
        
        # 保存结果
        stats_dict = stats.to_dict()
        with open(f"{self.output_dir}/{name}_statistics.json", 'w') as f:
            json.dump(stats_dict, f, indent=4)
        
        logger.info(f"{name} 数据集统计信息已保存")
        
        return stats_dict

    def compare_train_test_distributions(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                                        columns: Optional[List[str]] = None, 
                                        title_prefix: str = ''):
        """
        比较训练集和测试集的分布
        
        Args:
            train_data: 训练数据
            test_data: 测试数据
            columns: 要比较的列（默认为所有数值列）
            title_prefix: 标题前缀
        """
        # 选择要比较的列
        if columns is None:
            columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 限制列数以避免生成太多图表
        if len(columns) > 10:
            logger.info(f"列数太多 ({len(columns)})，只显示前10列")
            columns = columns[:10]
        
        # 为每列创建直方图比较
        for col in columns:
            if col in train_data.columns and col in test_data.columns:
                plt.figure(figsize=(10, 6))
                
                # 绘制训练集分布
                sns.histplot(train_data[col].dropna(), color='blue', alpha=0.5, 
                            label='Train', kde=True, stat='density')
                
                # 绘制测试集分布
                sns.histplot(test_data[col].dropna(), color='red', alpha=0.5, 
                            label='Test', kde=True, stat='density')
                
                plt.title(f'{title_prefix} {col} 分布比较')
                plt.xlabel(col)
                plt.ylabel('密度')
                plt.legend()
                
                # 保存图表
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plt.savefig(f"{self.output_dir}/distribution_compare_{col}_{timestamp}.png")
                plt.close()
        
        logger.info(f"已生成 {len(columns)} 个分布比较图表")

def create_evaluator(output_dir: str = 'models/evaluation') -> ModelEvaluator:
    """
    创建并返回评估器实例的工厂函数
    
    Args:
        output_dir: 评估结果输出目录
        
    Returns:
        ModelEvaluator实例
    """
    return ModelEvaluator(output_dir) 