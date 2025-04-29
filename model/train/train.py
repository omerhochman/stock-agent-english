#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票交易模型训练和测试脚本
用于训练深度学习、强化学习和遗传编程因子模型
"""

import argparse
import pandas as pd
import json
from datetime import datetime, timedelta
from model.dl import MLAgent, preprocess_stock_data
from model.rl import RLTradingAgent, RLTrader
from model.deap_factors import FactorAgent
from src.tools.api import get_price_history


def process_data(ticker, start_date, end_date, verbose=True):
    """获取并处理股票数据"""
    if verbose:
        print(f"正在获取 {ticker} 从 {start_date} 到 {end_date} 的数据...")
    
    try:
        prices_data = get_price_history(ticker, start_date, end_date)        
        if isinstance(prices_data, list):
            if verbose:
                print(f"获取到 {len(prices_data)} 条记录（列表格式），正在转换为DataFrame...")
            prices_df = pd.DataFrame(prices_data)
            if 'date' in prices_df.columns:
                prices_df['date'] = pd.to_datetime(prices_df['date'])
            
            # 确保日期列格式正确
            if 'date' in prices_df.columns:
                prices_df['date'] = pd.to_datetime(prices_df['date'])
        else:
            if verbose:
                print(f"获取到 {len(prices_data)} 条记录...")
            prices_df = prices_data
            
        if verbose:
            print(f"数据处理完成。")
        
        return prices_df
    
    except Exception as e:
        print(f"获取或处理数据时出错: {e}")
        return None


def train_dl_model(prices_df, params=None, save_dir='models', verbose=True):
    """训练深度学习模型"""
    if verbose:
        print("正在准备训练深度学习模型...")
    
    try:
        # 预处理数据
        processed_data = preprocess_stock_data(prices_df)
        
        if verbose:
            print(f"预处理完成，数据形状: {processed_data.shape}")
        
        # 设置默认参数
        default_params = {
            'seq_length': 10,
            'forecast_days': 5,
            'hidden_dim': 64,
            'num_layers': 2,
            'epochs': 50,
            'batch_size': 16,
            'learning_rate': 0.001
        }
        
        # 合并用户提供的参数
        if params:
            for key, value in params.items():
                if key in default_params:
                    default_params[key] = value
        
        if verbose:
            print("使用以下参数训练模型:")
            for key, value in default_params.items():
                print(f"  {key}: {value}")
        
        # 初始化并训练模型
        ml_agent = MLAgent(model_dir=save_dir)
        ml_agent.train_models(
            price_data=processed_data,
            seq_length=default_params['seq_length'],
            forecast_days=default_params['forecast_days'],
            hidden_dim=default_params['hidden_dim'],
            num_layers=default_params['num_layers'],
            epochs=default_params['epochs'],
            batch_size=default_params['batch_size'],
            learning_rate=default_params['learning_rate']
        )
        
        # 生成交易信号
        signals = ml_agent.generate_signals(processed_data)
        
        if verbose:
            print(f"训练完成。生成的交易信号: {signals.get('signal', 'unknown')}, 置信度: {signals.get('confidence', 0)}")
        
        return ml_agent, signals
    
    except Exception as e:
        print(f"训练深度学习模型时出错: {e}")
        return None, None


def train_rl_model(prices_df, params=None, save_dir='models', verbose=True):
    """训练强化学习模型"""
    if verbose:
        print("正在准备训练强化学习模型...")
    
    try:
        # 检查数据量是否足够
        if len(prices_df) < 200:
            print(f"警告: 数据量较少({len(prices_df)}行)，可能不足以训练RL模型。")
            print("建议使用至少1年的数据（约252个交易日）。")
        
        # 设置默认参数
        default_params = {
            'n_episodes': 500,
            'batch_size': 32,
            'reward_scaling': 1.0,
            'initial_balance': 100000,
            'transaction_fee_percent': 0.001,
            'window_size': 20,
            'max_steps': 63
        }
        
        # 合并用户提供的参数
        if params:
            for key, value in params.items():
                if key in default_params:
                    default_params[key] = value
        
        if verbose:
            print("使用以下参数训练模型:")
            for key, value in default_params.items():
                print(f"  {key}: {value}")
        
        # 初始化RL交易系统
        rl_trader = RLTrader(model_dir=save_dir)
        
        # 根据数据长度调整max_steps和window_size参数，避免low >= high错误
        data_len = len(prices_df)
        if default_params['window_size'] + default_params['max_steps'] >= data_len:
            adjusted_max_steps = max(1, data_len - default_params['window_size'] - 1)
            print(f"警告: 数据长度不足，调整max_steps参数从{default_params['max_steps']}到{adjusted_max_steps}")
            default_params['max_steps'] = adjusted_max_steps
        
        # 训练模型
        if verbose:
            print("开始训练RL模型，这可能需要一些时间...")
        
        training_history = rl_trader.train(
            df=prices_df,
            initial_balance=default_params['initial_balance'],
            transaction_fee_percent=default_params['transaction_fee_percent'],
            n_episodes=default_params['n_episodes'],
            batch_size=default_params['batch_size'],
            reward_scaling=default_params['reward_scaling'],
            max_steps=default_params['max_steps']
        )
        
        # 初始化交易代理并生成信号
        rl_agent = RLTradingAgent(model_dir=save_dir)
        rl_agent.load_model("best_model")
        signals = rl_agent.generate_signals(prices_df)
        
        if verbose:
            print(f"训练完成。生成的交易信号: {signals.get('signal', 'unknown')}, 置信度: {signals.get('confidence', 0)}")
        
        return rl_agent, signals
    
    except Exception as e:
        print(f"训练强化学习模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def train_factor_model(prices_df, params=None, save_dir='factors', verbose=True):
    """训练遗传编程因子模型"""
    if verbose:
        print("正在准备训练遗传编程因子模型...")
    
    try:
        # 设置默认参数
        default_params = {
            'n_factors': 5,
            'population_size': 100,
            'n_generations': 20,
            'future_return_periods': 5,
            'min_fitness': 0.05
        }
        
        # 合并用户提供的参数
        if params:
            for key, value in params.items():
                if key in default_params:
                    default_params[key] = value
        
        if verbose:
            print("使用以下参数生成因子:")
            for key, value in default_params.items():
                print(f"  {key}: {value}")
        
        # 初始化因子挖掘模块
        factor_agent = FactorAgent(model_dir=save_dir)
        
        # 生成因子
        if verbose:
            print("开始生成因子，这可能需要较长时间...")
        
        factors = factor_agent.generate_factors(
            price_data=prices_df,
            n_factors=default_params['n_factors'],
            population_size=default_params['population_size'],
            n_generations=default_params['n_generations']
        )
        
        # 生成交易信号
        signals = factor_agent.generate_signals(prices_df)
        
        if verbose:
            print(f"因子生成完成。生成的交易信号: {signals.get('signal', 'unknown')}, 置信度: {signals.get('confidence', 0)}")
            print(f"生成了 {len(factors)} 个因子")
        
        return factor_agent, signals
    
    except Exception as e:
        print(f"训练遗传编程因子模型时出错: {e}")
        return None, None


def load_model(model_type, model_dir, verbose=True):
    """加载已训练的模型"""
    if verbose:
        print(f"正在加载{model_type}模型...")
    
    try:
        if model_type == 'dl':
            agent = MLAgent(model_dir=model_dir)
            agent.load_models()
            if verbose:
                print("深度学习模型加载成功")
            return agent
        
        elif model_type == 'rl':
            agent = RLTradingAgent(model_dir=model_dir)
            success = agent.load_model()
            if success and verbose:
                print("强化学习模型加载成功")
            elif verbose:
                print("强化学习模型加载失败")
            return agent
        
        elif model_type == 'factor':
            agent = FactorAgent(model_dir=model_dir)
            factors = agent.load_factors()
            if factors and verbose:
                print(f"因子模型加载成功，加载了{len(factors)}个因子")
            elif verbose:
                print("因子模型加载失败或没有找到因子")
            return agent
        
        else:
            if verbose:
                print(f"不支持的模型类型: {model_type}")
            return None
    
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None


def generate_signals(agent, model_type, prices_df, verbose=True):
    """使用模型生成交易信号"""
    if verbose:
        print(f"使用{model_type}模型生成交易信号...")
    
    try:
        if agent is None:
            return None
        
        signals = agent.generate_signals(prices_df)
        
        if verbose:
            print(f"生成的交易信号: {signals.get('signal', 'unknown')}, 置信度: {signals.get('confidence', 0)}")
            if 'reasoning' in signals:
                print(f"决策理由: {signals['reasoning']}")
        
        return signals
    
    except Exception as e:
        print(f"生成交易信号时出错: {e}")
        return None


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='股票交易模型训练和测试工具')
    
    # 基本参数
    parser.add_argument('--ticker', type=str, required=True,
                        help='股票代码，例如: 600519')
    parser.add_argument('--start-date', type=str,
                        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                        help='开始日期，格式：YYYY-MM-DD，默认为一年前')
    parser.add_argument('--end-date', type=str,
                        default = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                        help='结束日期，格式：YYYY-MM-DD，默认为昨天')
    
    # 模型选择和操作
    parser.add_argument('--model', type=str, choices=['dl', 'rl', 'factor', 'all'],
                        default='dl',
                        help='模型类型: dl (深度学习), rl (强化学习), factor (遗传编程因子), all (所有)')
    parser.add_argument('--action', type=str, choices=['train', 'test', 'both'],
                        default='both',
                        help='操作类型: train (仅训练), test (仅测试), both (训练并测试)')
    
    # 模型参数
    parser.add_argument('--params', type=str, default=None,
                        help='模型参数，JSON格式字符串，例如: \'{"hidden_dim": 128, "epochs": 100}\'')
    
    # 其他选项
    parser.add_argument('--model-dir', type=str, default='models',
                        help='模型保存目录，默认为 "models"')
    parser.add_argument('--factor-dir', type=str, default='factors',
                        help='因子模型保存目录，默认为 "factors"')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='显示详细输出')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 解析JSON参数
    params = None
    if args.params:
        try:
            params = json.loads(args.params)
            if args.verbose:
                print(f"使用自定义参数: {params}")
        except json.JSONDecodeError:
            print(f"错误: 无法解析参数JSON: {args.params}")
            print("请使用正确的JSON格式，例如: '{\"hidden_dim\": 128, \"epochs\": 100}'")
            return
    
    # 获取并处理股票数据
    prices_df = process_data(args.ticker, args.start_date, args.end_date, args.verbose)
    if prices_df is None or len(prices_df) == 0:
        print("错误: 无法获取股票数据或数据为空")
        return
    
    # 执行模型训练和测试
    models_to_train = ['dl', 'rl', 'factor'] if args.model == 'all' else [args.model]
    results = {}
    
    for model_type in models_to_train:
        if args.verbose:
            print(f"\n{'='*50}")
            print(f"开始处理 {model_type} 模型")
            print(f"{'='*50}")
        
        model_dir = args.factor_dir if model_type == 'factor' else args.model_dir
        agent = None
        signals = None
        
        # 训练模型
        if args.action in ['train', 'both']:
            if args.verbose:
                print(f"\n--- 训练 {model_type} 模型 ---")
            
            if model_type == 'dl':
                agent, signals = train_dl_model(prices_df, params, model_dir, args.verbose)
            elif model_type == 'rl':
                agent, signals = train_rl_model(prices_df, params, model_dir, args.verbose)
            elif model_type == 'factor':
                agent, signals = train_factor_model(prices_df, params, model_dir, args.verbose)
        
        # 测试模型
        if args.action in ['test', 'both']:
            if args.verbose:
                print(f"\n--- 测试 {model_type} 模型 ---")
            
            if args.action == 'test' or agent is None:
                agent = load_model(model_type, model_dir, args.verbose)
            
            if agent is not None:
                signals = generate_signals(agent, model_type, prices_df, args.verbose)
        
        # 保存结果
        if signals:
            results[model_type] = signals
    
    # 打印综合结果
    if args.verbose and len(results) > 0:
        print("\n")
        print("="*50)
        print("综合结果:")
        print("="*50)
        
        for model_type, signals in results.items():
            signal = signals.get('signal', 'unknown')
            confidence = signals.get('confidence', 0)
            print(f"{model_type:6} 模型: {signal:8} (置信度: {confidence:.2f})")
        
        if len(results) > 1:
            signal_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
            weighted_sum = 0
            total_weight = 0
            
            for model_type, signals in results.items():
                signal = signals.get('signal', 'neutral')
                confidence = signals.get('confidence', 0.5)
                
                signal_counts[signal] += 1
                
                # 使用置信度作为权重
                if signal == 'bullish':
                    weighted_sum += confidence
                elif signal == 'bearish':
                    weighted_sum -= confidence
                
                total_weight += confidence
            
            # 计算综合信号
            if total_weight > 0:
                normalized_score = weighted_sum / total_weight
            else:
                normalized_score = 0
            
            # 确定最终信号
            if normalized_score > 0.2:
                final_signal = 'bullish'
            elif normalized_score < -0.2:
                final_signal = 'bearish'
            else:
                final_signal = 'neutral'
            
            print("\n模型投票结果:")
            print(f"看多: {signal_counts['bullish']}，看空: {signal_counts['bearish']}，中性: {signal_counts['neutral']}")
            print(f"加权评分: {normalized_score:.2f} (范围: -1到1)")
            print(f"综合信号: {final_signal}")


if __name__ == '__main__':
    main()