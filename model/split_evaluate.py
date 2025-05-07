#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# 添加项目根目录到系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from model.evaluate import ModelEvaluator
from model.dl import MLAgent, preprocess_stock_data
from model.rl import RLTradingAgent
from model.deap_factors import FactorAgent

from src.utils.logging_config import setup_logger
logger = setup_logger('split_evaluate')

def split_and_evaluate(
    ticker: str,
    price_data: pd.DataFrame, 
    model_type: str = 'dl',
    train_ratio: float = 0.7, 
    val_ratio: float = 0.2, 
    test_ratio: float = 0.1,
    shuffle: bool = False,
    params: Optional[Dict] = None,
    save_dir: str = 'models',
    eval_dir: str = 'models/evaluation',
    verbose: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    划分数据，训练模型并进行评估
    
    Args:
        ticker: 股票代码
        price_data: 价格数据
        model_type: 模型类型 ('dl', 'rl', 'factor')
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        shuffle: 是否打乱数据
        params: 模型参数
        save_dir: 模型保存目录
        eval_dir: 评估结果保存目录
        verbose: 是否显示详细信息
        
    Returns:
        训练好的模型和评估结果
    """
    if verbose:
        logger.info(f"开始数据划分和模型评估 - 股票代码: {ticker}")
        logger.info(f"数据划分比例 - 训练集: {train_ratio*100:.1f}%, 验证集: {val_ratio*100:.1f}%, 测试集: {test_ratio*100:.1f}%")
    
    # 初始化评估器
    evaluator = ModelEvaluator(output_dir=eval_dir)
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # 预处理数据
    if model_type == 'dl' or model_type == 'rl':
        processed_data = preprocess_stock_data(price_data)
    else:
        processed_data = price_data.copy()
    
    if verbose:
        logger.info(f"数据预处理完成，形状: {processed_data.shape}")
    
    # 划分数据
    train_data, val_data, test_data = evaluator.split_data(
        data=processed_data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shuffle=shuffle
    )
    
    if verbose:
        logger.info(f"数据划分完成 - 训练集: {train_data.shape}, 验证集: {val_data.shape}, 测试集: {test_data.shape}")
        
    # 保存训练测试数据分布对比图
    evaluator.compare_train_test_distributions(
        train_data, test_data, 
        columns=['close', 'returns', 'volatility_20d'] if all(col in processed_data.columns for col in ['close', 'returns', 'volatility_20d']) else None,
        title_prefix=f'{ticker}'
    )
    
    # 计算并保存统计信息
    evaluator.summary_statistics(train_data, name=f'{ticker}_train')
    evaluator.summary_statistics(test_data, name=f'{ticker}_test')
    
    # 根据模型类型训练模型
    agent = None
    evaluation_results = {}
    
    try:
        if model_type == 'dl':
            agent, eval_results = train_and_evaluate_dl(
                ticker, train_data, val_data, test_data, 
                params=params, evaluator=evaluator, save_dir=save_dir, verbose=verbose
            )
            evaluation_results.update(eval_results)
            
        elif model_type == 'rl':
            agent, eval_results = train_and_evaluate_rl(
                ticker, train_data, val_data, test_data, 
                params=params, evaluator=evaluator, save_dir=save_dir, verbose=verbose
            )
            evaluation_results.update(eval_results)
            
        elif model_type == 'factor':
            agent, eval_results = train_and_evaluate_factor(
                ticker, train_data, val_data, test_data, 
                params=params, evaluator=evaluator, save_dir=save_dir, verbose=verbose
            )
            evaluation_results.update(eval_results)
            
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
    except Exception as e:
        logger.error(f"训练和评估模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
    # 保存评估结果摘要
    with open(f"{eval_dir}/{ticker}_{model_type}_evaluation_summary.json", 'w') as f:
        json.dump(evaluation_results, f, indent=4)
        
    if verbose:
        logger.info(f"模型评估完成，结果已保存到 {eval_dir}")
        
    return agent, evaluation_results


def train_and_evaluate_dl(
    ticker: str,
    train_data: pd.DataFrame, 
    val_data: pd.DataFrame, 
    test_data: pd.DataFrame,
    params: Optional[Dict] = None,
    evaluator: Optional[ModelEvaluator] = None,
    save_dir: str = 'models',
    verbose: bool = True
) -> Tuple[MLAgent, Dict[str, Any]]:
    """
    训练和评估深度学习模型
    
    Args:
        ticker: 股票代码
        train_data: 训练集
        val_data: 验证集
        test_data: 测试集
        params: 模型参数
        evaluator: 评估器实例
        save_dir: 模型保存目录
        verbose: 是否显示详细信息
        
    Returns:
        训练好的MLAgent实例和评估结果
    """
    if evaluator is None:
        evaluator = ModelEvaluator(f"{save_dir}/evaluation")
    
    # 设置默认参数
    default_params = {
        'seq_length': 15,
        'forecast_days': 10,
        'hidden_dim': 128,
        'num_layers': 3,
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.0005
    }
    
    # 合并用户提供的参数
    model_params = default_params.copy()
    if params:
        for key, value in params.items():
            if key in model_params:
                model_params[key] = value
    
    if verbose:
        logger.info("使用以下参数训练深度学习模型:")
        for key, value in model_params.items():
            logger.info(f"  {key}: {value}")
    
    # 初始化MLAgent
    ml_agent = MLAgent(model_dir=save_dir)
    
    # 准备特征列
    feature_cols = ['close', 'ma5', 'ma10', 'ma20', 'rsi', 'macd']
    missing_features = [col for col in feature_cols if col not in train_data.columns]
    if missing_features:
        if verbose:
            logger.warning(f"以下特征在数据中缺失: {missing_features}")
            logger.info("将使用可用的特征列")
        feature_cols = [col for col in feature_cols if col in train_data.columns]
        if not feature_cols:
            feature_cols = ['close']
            if verbose:
                logger.info("只使用close列作为特征")
    
    # 使用训练集训练模型
    ml_agent.train_models(
        train_data,
        seq_length=model_params['seq_length'],
        feature_cols=feature_cols,
        hidden_dim=model_params['hidden_dim'],
        num_layers=model_params['num_layers'],
        forecast_days=model_params['forecast_days'],
        epochs=model_params['epochs'],
        batch_size=model_params['batch_size'],
        learning_rate=model_params['learning_rate']
    )
    
    # 评估结果字典
    evaluation_results = {
        'model_type': 'dl',
        'params': model_params,
        'feature_cols': feature_cols,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'results': {}
    }
    
    # 在测试集上评估LSTM模型
    if ml_agent.dl_module.lstm_model is not None:
        # 获取测试集的真实值和预测值
        try:
            # 在测试集上生成预测
            test_predictions = []
            test_targets = []
            
            # 序列长度
            seq_length = model_params['seq_length']
            
            # 对测试集中的每个窗口进行预测
            for i in range(seq_length, len(test_data)):
                # 获取序列窗口
                test_window = test_data.iloc[i-seq_length:i]
                target = test_data.iloc[i]['close']
                
                # 进行预测
                pred = ml_agent.dl_module.predict_lstm(
                    test_window, 
                    feature_cols=feature_cols,
                    seq_length=seq_length,
                    target_col='close'
                )[0]  # 只取第一个预测值，即下一步预测
                
                test_predictions.append(pred)
                test_targets.append(target)
            
            # 转换为NumPy数组
            test_predictions = np.array(test_predictions)
            test_targets = np.array(test_targets)
            
            # 评估回归模型
            lstm_metrics = evaluator.evaluate_regression_model(
                test_targets, test_predictions, 
                model_name=f'{ticker}_lstm', dataset_name='test'
            )
            
            # 可视化预测结果
            try:
                date_index = None
                if 'date' in test_data.columns:
                    # 从序列长度之后开始的日期
                    date_index = test_data['date'].iloc[seq_length:].values
                elif test_data.index.name == 'date':
                    date_index = test_data.index[seq_length:]
                
                evaluator.visualize_predictions(
                    test_targets, test_predictions, 
                    date_index=date_index,
                    model_name=f'{ticker}_lstm', 
                    dataset_name='test',
                    title=f'{ticker} - LSTM模型测试集预测'
                )
            except Exception as viz_error:
                logger.error(f"可视化预测结果时出错: {str(viz_error)}")
            
            # 生成未来预测
            forecast = ml_agent.dl_module.predict_lstm(
                test_data.iloc[-seq_length:], 
                feature_cols=feature_cols,
                seq_length=seq_length,
                target_col='close'
            )
            
            # 可视化未来预测
            try:
                history = test_data['close'].iloc[-30:].values
                evaluator.visualize_forecast(
                    history, forecast,
                    model_name=f'{ticker}_lstm',
                    title=f'{ticker} - 未来{len(forecast)}天价格预测'
                )
            except Exception as fore_error:
                logger.error(f"可视化未来预测时出错: {str(fore_error)}")
            
            # 添加到评估结果
            evaluation_results['results']['lstm'] = {
                'metrics': lstm_metrics,
                'future_forecast': forecast.tolist()
            }
            
        except Exception as e:
            logger.error(f"评估LSTM模型时出错: {str(e)}")
    
    # 在测试集上评估随机森林分类器
    if ml_agent.dl_module.rf_model is not None:
        try:
            # 准备测试集特征和标签
            test_features = ml_agent.dl_module.prepare_features(test_data)
            test_labels = ml_agent.dl_module.generate_training_labels(test_data, forward_days=model_params['forecast_days'])
            
            # 确保特征和标签具有相同的索引
            common_idx = test_features.index.intersection(test_labels.index)
            if len(common_idx) > 0:
                # 获取预测
                test_features_scaled = ml_agent.dl_module.feature_scaler.transform(test_features.loc[common_idx])
                test_pred = ml_agent.dl_module.rf_model.predict(test_features_scaled)
                test_true = test_labels.loc[common_idx].values
                
                # 评估分类模型
                rf_metrics = evaluator.evaluate_classification_model(
                    test_true, test_pred,
                    model_name=f'{ticker}_rf', dataset_name='test'
                )
                
                # 特征重要性可视化
                try:
                    evaluator.feature_importance_plot(
                        feature_names=test_features.columns.tolist(),
                        importances=ml_agent.dl_module.rf_model.feature_importances_,
                        model_name=f'{ticker}_rf'
                    )
                except Exception as feat_error:
                    logger.error(f"可视化特征重要性时出错: {str(feat_error)}")
                
                # 添加到评估结果
                evaluation_results['results']['random_forest'] = {
                    'metrics': rf_metrics,
                    'feature_importance': dict(zip(
                        test_features.columns.tolist(),
                        ml_agent.dl_module.rf_model.feature_importances_.tolist()
                    ))
                }
        except Exception as e:
            logger.error(f"评估随机森林模型时出错: {str(e)}")
    
    return ml_agent, evaluation_results


def train_and_evaluate_rl(
    ticker: str,
    train_data: pd.DataFrame, 
    val_data: pd.DataFrame, 
    test_data: pd.DataFrame,
    params: Optional[Dict] = None,
    evaluator: Optional[ModelEvaluator] = None,
    save_dir: str = 'models',
    verbose: bool = True
) -> Tuple[RLTradingAgent, Dict[str, Any]]:
    """
    训练和评估强化学习模型
    
    Args:
        ticker: 股票代码
        train_data: 训练集
        val_data: 验证集
        test_data: 测试集
        params: 模型参数
        evaluator: 评估器实例
        save_dir: 模型保存目录
        verbose: 是否显示详细信息
        
    Returns:
        训练好的RLTradingAgent实例和评估结果
    """
    if evaluator is None:
        evaluator = ModelEvaluator(f"{save_dir}/evaluation")
    
    # 实现强化学习模型的训练和评估
    # 由于代码量较大，这里只是一个框架
    # TODO: 实现具体的训练和评估逻辑
    
    # 简单的返回值
    rl_agent = RLTradingAgent()
    evaluation_results = {
        'model_type': 'rl',
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'results': {}
    }
    
    return rl_agent, evaluation_results


def train_and_evaluate_factor(
    ticker: str,
    train_data: pd.DataFrame, 
    val_data: pd.DataFrame, 
    test_data: pd.DataFrame,
    params: Optional[Dict] = None,
    evaluator: Optional[ModelEvaluator] = None,
    save_dir: str = 'factors',
    verbose: bool = True
) -> Tuple[FactorAgent, Dict[str, Any]]:
    """
    训练和评估因子模型
    
    Args:
        ticker: 股票代码
        train_data: 训练集
        val_data: 验证集
        test_data: 测试集
        params: 模型参数
        evaluator: 评估器实例
        save_dir: 模型保存目录
        verbose: 是否显示详细信息
        
    Returns:
        训练好的FactorAgent实例和评估结果
    """
    if evaluator is None:
        evaluator = ModelEvaluator(f"{save_dir}/evaluation")
    
    # 实现因子模型的训练和评估
    # 由于代码量较大，这里只是一个框架
    # TODO: 实现具体的训练和评估逻辑
    
    # 简单的返回值
    factor_agent = FactorAgent()
    evaluation_results = {
        'model_type': 'factor',
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'results': {}
    }
    
    return factor_agent, evaluation_results


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='股票模型数据划分与评估工具')
    
    # 基本参数
    parser.add_argument('--ticker', type=str, required=True,
                        help='股票代码，例如: 600519')
    parser.add_argument('--start-date', type=str,
                        default=(datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d'),
                        help='开始日期，格式：YYYY-MM-DD，默认为两年前')
    parser.add_argument('--end-date', type=str,
                        default=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                        help='结束日期，格式：YYYY-MM-DD，默认为昨天')
    
    # 模型选择
    parser.add_argument('--model', type=str, choices=['dl', 'rl', 'factor', 'all'],
                        default='dl',
                        help='模型类型: dl (深度学习), rl (强化学习), factor (遗传编程因子), all (所有)')
    
    # 数据划分参数
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例，默认0.7')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='验证集比例，默认0.2')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='测试集比例，默认0.1')
    parser.add_argument('--shuffle', action='store_true',
                        help='是否打乱数据')
    
    # 模型参数
    parser.add_argument('--params', type=str, default=None,
                        help='模型参数，JSON格式字符串，例如: \'{"hidden_dim": 128, "epochs": 100}\'')
    
    # 其他选项
    parser.add_argument('--model-dir', type=str, default='models',
                        help='模型保存目录，默认为 "models"')
    parser.add_argument('--eval-dir', type=str, default='models/evaluation',
                        help='评估结果保存目录，默认为 "models/evaluation"')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='显示详细输出')
    
    args = parser.parse_args()
    
    # 解析JSON参数
    params = None
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError:
            logger.error(f"错误: 无法解析参数JSON: {args.params}")
            logger.error("请使用正确的JSON格式，例如: '{\"hidden_dim\": 128, \"epochs\": 100}'")
            sys.exit(1)
    
    # 从train.py导入数据处理函数
    from model.train.train import process_data
    
    # 获取并处理股票数据
    price_data = process_data(args.ticker, args.start_date, args.end_date, args.verbose)
    if price_data is None or (isinstance(price_data, pd.DataFrame) and price_data.empty):
        logger.error("错误: 无法获取股票数据或数据为空")
        sys.exit(1)
    
    # 执行模型训练和评估
    if args.model == 'all':
        models_to_train = ['dl', 'rl', 'factor']
    else:
        models_to_train = [args.model]
    
    for model_type in models_to_train:
        logger.info(f"\n{'='*50}")
        logger.info(f"开始处理 {model_type} 模型")
        logger.info(f"{'='*50}")
        
        save_dir = args.eval_dir if model_type == 'factor' else args.model_dir
        
        agent, results = split_and_evaluate(
            ticker=args.ticker,
            price_data=price_data,
            model_type=model_type,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            shuffle=args.shuffle,
            params=params,
            save_dir=save_dir,
            eval_dir=args.eval_dir,
            verbose=args.verbose
        )
        
        logger.info(f"\n{'='*50}")
        logger.info(f"{model_type} 模型处理完成")
        logger.info(f"{'='*50}") 