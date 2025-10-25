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

# Add project root directory to system path
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
    Split data, train model and evaluate
    
    Args:
        ticker: Stock code
        price_data: Price data
        model_type: Model type ('dl', 'rl', 'factor')
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        shuffle: Whether to shuffle data
        params: Model parameters
        save_dir: Model save directory
        eval_dir: Evaluation results save directory
        verbose: Whether to show detailed information
        
    Returns:
        Trained model and evaluation results
    """
    if verbose:
        logger.info(f"Starting data splitting and model evaluation - Stock code: {ticker}")
        logger.info(f"Data split ratio - Training set: {train_ratio*100:.1f}%, Validation set: {val_ratio*100:.1f}%, Test set: {test_ratio*100:.1f}%")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir=eval_dir)
    
    # Ensure directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Preprocess data
    if model_type == 'dl' or model_type == 'rl':
        processed_data = preprocess_stock_data(price_data)
    else:
        processed_data = price_data.copy()
    
    if verbose:
        logger.info(f"Data preprocessing completed, shape: {processed_data.shape}")
    
    # Split data
    train_data, val_data, test_data = evaluator.split_data(
        data=processed_data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shuffle=shuffle
    )
    
    if verbose:
        logger.info(f"Data splitting completed - Training set: {train_data.shape}, Validation set: {val_data.shape}, Test set: {test_data.shape}")
        
        # Save training-test data distribution comparison chart
        evaluator.compare_train_test_distributions(
            train_data, test_data, 
            columns=['close', 'returns', 'volatility_20d'] if all(col in processed_data.columns for col in ['close', 'returns', 'volatility_20d']) else None,
            title_prefix=f'{ticker}'
        )
        
        # Calculate and save statistical information
        evaluator.summary_statistics(train_data, name=f'{ticker}_train')
        evaluator.summary_statistics(test_data, name=f'{ticker}_test')
    
    # Train model based on model type
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
            raise ValueError(f"Unsupported model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error training and evaluating model: {str(e)}")
        import traceback
        traceback.print_exc()
        
    # Save evaluation results summary
    with open(f"{eval_dir}/{ticker}_{model_type}_evaluation_summary.json", 'w') as f:
        json.dump(evaluation_results, f, indent=4)
        
    if verbose:
        logger.info(f"Model evaluation completed, results saved to {eval_dir}")
        
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
    Train and evaluate deep learning model
    
    Args:
        ticker: Stock code
        train_data: Training set
        val_data: Validation set
        test_data: Test set
        params: Model parameters
        evaluator: Evaluator instance
        save_dir: Model save directory
        verbose: Whether to show detailed information
        
    Returns:
        Trained MLAgent instance and evaluation results
    """
    if evaluator is None:
        evaluator = ModelEvaluator(f"{save_dir}/evaluation")
    
    # Set default parameters
    default_params = {
        'seq_length': 15,
        'forecast_days': 10,
        'hidden_dim': 128,
        'num_layers': 3,
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.0005
    }
    
    # Merge user-provided parameters
    model_params = default_params.copy()
    if params:
        for key, value in params.items():
            if key in model_params:
                model_params[key] = value
    
    if verbose:
        logger.info("Training deep learning model with the following parameters:")
        for key, value in model_params.items():
            logger.info(f"  {key}: {value}")
    
    # Initialize MLAgent
    ml_agent = MLAgent(model_dir=save_dir)
    
    # Prepare feature columns
    feature_cols = ['close', 'ma5', 'ma10', 'ma20', 'rsi', 'macd']
    missing_features = [col for col in feature_cols if col not in train_data.columns]
    if missing_features:
        if verbose:
            logger.warning(f"The following features are missing from the data: {missing_features}")
            logger.info("Will use available feature columns")
        feature_cols = [col for col in feature_cols if col in train_data.columns]
        if not feature_cols:
            feature_cols = ['close']
            if verbose:
                logger.info("Only using close column as feature")
    
    # Train model using training set
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
    
    # Evaluation results dictionary
    evaluation_results = {
        'model_type': 'dl',
        'params': model_params,
        'feature_cols': feature_cols,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'results': {}
    }
    
    # Evaluate LSTM model on test set
    if ml_agent.dl_module.lstm_model is not None:
        # Get true values and predictions for test set
        try:
            # Generate predictions on test set
            test_predictions = []
            test_targets = []
            
            # Sequence length
            seq_length = model_params['seq_length']
            
            # Make predictions for each window in test set
            for i in range(seq_length, len(test_data)):
                # Get sequence window
                test_window = test_data.iloc[i-seq_length:i]
                target = test_data.iloc[i]['close']
                
                # Make prediction
                pred = ml_agent.dl_module.predict_lstm(
                    test_window, 
                    feature_cols=feature_cols,
                    seq_length=seq_length,
                    target_col='close'
                )[0]  # Only take the first prediction value, i.e., next step prediction
                
                test_predictions.append(pred)
                test_targets.append(target)
            
            # Convert to NumPy arrays
            test_predictions = np.array(test_predictions)
            test_targets = np.array(test_targets)
            
            # Evaluate regression model
            lstm_metrics = evaluator.evaluate_regression_model(
                test_targets, test_predictions, 
                model_name=f'{ticker}_lstm', dataset_name='test'
            )
            
            # Visualize prediction results
            try:
                date_index = None
                if 'date' in test_data.columns:
                    # Dates starting from after sequence length
                    date_index = test_data['date'].iloc[seq_length:].values
                elif test_data.index.name == 'date':
                    date_index = test_data.index[seq_length:]
                
                evaluator.visualize_predictions(
                    test_targets, test_predictions, 
                    date_index=date_index,
                    model_name=f'{ticker}_lstm', 
                    dataset_name='test',
                    title=f'{ticker} - LSTM Model Test Set Prediction'
                )
            except Exception as viz_error:
                logger.error(f"Error visualizing prediction results: {str(viz_error)}")
            
            # Generate future predictions
            forecast = ml_agent.dl_module.predict_lstm(
                test_data.iloc[-seq_length:], 
                feature_cols=feature_cols,
                seq_length=seq_length,
                target_col='close'
            )
            
            # Visualize future predictions
            try:
                history = test_data['close'].iloc[-30:].values
                evaluator.visualize_forecast(
                    history, forecast,
                    model_name=f'{ticker}_lstm',
                    title=f'{ticker} - Future {len(forecast)} Day Price Prediction'
                )
            except Exception as fore_error:
                logger.error(f"Error visualizing future predictions: {str(fore_error)}")
            
            # Add to evaluation results
            evaluation_results['results']['lstm'] = {
                'metrics': lstm_metrics,
                'future_forecast': forecast.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating LSTM model: {str(e)}")
    
    # Evaluate random forest classifier on test set
    if ml_agent.dl_module.rf_model is not None:
        try:
            # Prepare test set features and labels
            test_features = ml_agent.dl_module.prepare_features(test_data)
            test_labels = ml_agent.dl_module.generate_training_labels(test_data, forward_days=model_params['forecast_days'])
            
            # Ensure features and labels have the same index
            common_idx = test_features.index.intersection(test_labels.index)
            if len(common_idx) > 0:
                # Get predictions
                test_features_scaled = ml_agent.dl_module.feature_scaler.transform(test_features.loc[common_idx])
                test_pred = ml_agent.dl_module.rf_model.predict(test_features_scaled)
                test_true = test_labels.loc[common_idx].values
                
                # Evaluate classification model
                rf_metrics = evaluator.evaluate_classification_model(
                    test_true, test_pred,
                    model_name=f'{ticker}_rf', dataset_name='test'
                )
                
                # Visualize feature importance
                try:
                    evaluator.feature_importance_plot(
                        feature_names=test_features.columns.tolist(),
                        importances=ml_agent.dl_module.rf_model.feature_importances_,
                        model_name=f'{ticker}_rf'
                    )
                except Exception as feat_error:
                    logger.error(f"Error visualizing feature importance: {str(feat_error)}")
                
                # Add to evaluation results
                evaluation_results['results']['random_forest'] = {
                    'metrics': rf_metrics,
                    'feature_importance': dict(zip(
                        test_features.columns.tolist(),
                        ml_agent.dl_module.rf_model.feature_importances_.tolist()
                    ))
                }
        except Exception as e:
            logger.error(f"Error evaluating random forest model: {str(e)}")
    
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
    Train and evaluate reinforcement learning model
    
    Args:
        ticker: Stock code
        train_data: Training set
        val_data: Validation set
        test_data: Test set
        params: Model parameters
        evaluator: Evaluator instance
        save_dir: Model save directory
        verbose: Whether to show detailed information
        
    Returns:
        Trained RLTradingAgent instance and evaluation results
    """
    if evaluator is None:
        evaluator = ModelEvaluator(f"{save_dir}/evaluation")
    
    # Implement reinforcement learning model training and evaluation
    # Due to large code volume, this is just a framework
    # TODO: Implement specific training and evaluation logic
    
    # Simple return value
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
    Train and evaluate factor model
    
    Args:
        ticker: Stock code
        train_data: Training set
        val_data: Validation set
        test_data: Test set
        params: Model parameters
        evaluator: Evaluator instance
        save_dir: Model save directory
        verbose: Whether to show detailed information
        
    Returns:
        Trained FactorAgent instance and evaluation results
    """
    if evaluator is None:
        evaluator = ModelEvaluator(f"{save_dir}/evaluation")
    
    # Implement factor model training and evaluation
    # Due to large code volume, this is just a framework
    # TODO: Implement specific training and evaluation logic
    
    # Simple return value
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
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Stock model data splitting and evaluation tool')
    
    # Basic parameters
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock code, e.g.: 600519')
    parser.add_argument('--start-date', type=str,
                        default=(datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d'),
                        help='Start date, format: YYYY-MM-DD, default is two years ago')
    parser.add_argument('--end-date', type=str,
                        default=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                        help='End date, format: YYYY-MM-DD, default is yesterday')
    
    # Model selection
    parser.add_argument('--model', type=str, choices=['dl', 'rl', 'factor', 'all'],
                        default='dl',
                        help='Model type: dl (deep learning), rl (reinforcement learning), factor (genetic programming factor), all (all)')
    
    # Data splitting parameters
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio, default 0.7')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation set ratio, default 0.2')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Test set ratio, default 0.1')
    parser.add_argument('--shuffle', action='store_true',
                        help='Whether to shuffle data')
    
    # Model parameters
    parser.add_argument('--params', type=str, default=None,
                        help='Model parameters, JSON format string, e.g.: \'{"hidden_dim": 128, "epochs": 100}\'')
    
    # Other options
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Model save directory, default "models"')
    parser.add_argument('--eval-dir', type=str, default='models/evaluation',
                        help='Evaluation results save directory, default "models/evaluation"')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Show detailed output')
    
    args = parser.parse_args()
    
    # Parse JSON parameters
    params = None
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError:
            logger.error(f"Error: Unable to parse parameter JSON: {args.params}")
            logger.error("Please use correct JSON format, e.g.: '{\"hidden_dim\": 128, \"epochs\": 100}'")
            sys.exit(1)
    
    # Import data processing function from train.py
    from model.train.train import process_data
    
    # Get and process stock data
    price_data = process_data(args.ticker, args.start_date, args.end_date, args.verbose)
    if price_data is None or (isinstance(price_data, pd.DataFrame) and price_data.empty):
        logger.error("Error: Unable to get stock data or data is empty")
        sys.exit(1)
    
    # Execute model training and evaluation
    if args.model == 'all':
        models_to_train = ['dl', 'rl', 'factor']
    else:
        models_to_train = [args.model]
    
    for model_type in models_to_train:
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting processing of {model_type} model")
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
        logger.info(f"{model_type} model processing completed")
        logger.info(f"{'='*50}") 