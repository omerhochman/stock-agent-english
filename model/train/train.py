#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# 添加项目根目录到系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from model.dl import MLAgent, preprocess_stock_data
from model.rl import RLTradingAgent, RLTrader, StockTradingEnv
from model.deap_factors import FactorAgent, FactorMiningModule
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
            
            # 确保日期列格式正确
            if 'date' in prices_df.columns:
                prices_df['date'] = pd.to_datetime(prices_df['date'])
        else:
            if verbose:
                if hasattr(prices_data, 'shape'):
                    print(f"获取到 {prices_data.shape[0]} 条记录...")
                else:
                    print(f"获取到数据类型: {type(prices_data)}...")
            prices_df = prices_data
            
        if verbose:
            if isinstance(prices_df, pd.DataFrame):
                print(f"数据处理完成。形状: {prices_df.shape}")
                if not prices_df.empty:
                    print(f"列名: {list(prices_df.columns)}")
            else:
                print(f"数据处理完成。数据类型: {type(prices_df)}")
        
        return prices_df
    
    except Exception as e:
        print(f"获取或处理数据时出错: {e}")
        return None


def add_technical_indicators(df):
    """添加缺失的技术指标"""
    # 确保数据框有必要的列
    if 'close' not in df.columns and 'Close' in df.columns:
        df['close'] = df['Close']
    if 'open' not in df.columns and 'Open' in df.columns:
        df['open'] = df['Open']
    if 'high' not in df.columns and 'High' in df.columns:
        df['high'] = df['High']
    if 'low' not in df.columns and 'Low' in df.columns:
        df['low'] = df['Low']
    if 'volume' not in df.columns and 'Volume' in df.columns:
        df['volume'] = df['Volume']
    
    # 添加必要的技术指标 (如果不存在)
    if 'ma10' not in df.columns:
        df['ma10'] = df['close'].rolling(window=10).mean()
    
    if 'rsi' not in df.columns:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    if 'macd' not in df.columns or 'macd_signal' not in df.columns or 'macd_hist' not in df.columns:
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
    
    if 'volatility_5d' not in df.columns:
        df['volatility_5d'] = df['close'].pct_change().rolling(window=5).std()
    
    if 'volatility_10d' not in df.columns:
        df['volatility_10d'] = df['close'].pct_change().rolling(window=10).std()
    
    if 'volatility_20d' not in df.columns:
        df['volatility_20d'] = df['close'].pct_change().rolling(window=20).std()
    
    # 填充NaN值
    df = df.ffill().bfill().fillna(0)
    
    return df


def train_dl_model(prices_df, params=None, save_dir='models', verbose=True):
    """训练深度学习模型"""
    if verbose:
        print("正在准备训练深度学习模型...")
    
    try:
        # 预处理数据
        processed_data = preprocess_stock_data(prices_df)
        
        if verbose:
            print(f"预处理完成，数据形状: {processed_data.shape}")
        
        # 设置默认参数 (不直接传递给train_models，而是分别设置)
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
        
        # 初始化并训练模型 - 修正：正确调用train_models()方法
        ml_agent = MLAgent(model_dir=save_dir)
        ml_agent.train_models(processed_data)  # 不传递参数，使用默认设置
        
        # 生成交易信号
        signals = ml_agent.generate_signals(processed_data)
        
        if verbose:
            print(f"训练完成。生成的交易信号: {signals.get('signal', 'unknown')}, 置信度: {signals.get('confidence', 0)}")
        
        return ml_agent, signals
    
    except Exception as e:
        print(f"训练深度学习模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def train_rl_model(prices_df, params=None, save_dir='models', verbose=True):
    """训练强化学习模型"""
    if verbose:
        print("正在准备训练强化学习模型...")
    
    try:
        # 检查数据量是否足够
        if len(prices_df) < 100:
            print(f"警告: 数据量较少({len(prices_df)}行)，可能不足以训练RL模型。")
            print("建议使用至少100个交易日的数据。")
        
        # 添加缺失的技术指标
        enhanced_df = add_technical_indicators(prices_df.copy())
        
        # 确认所有必要的技术指标都已添加
        required_indicators = ['ma10', 'rsi', 'macd', 'macd_signal', 'macd_hist', 
                               'volatility_5d', 'volatility_10d', 'volatility_20d']
        missing_indicators = [ind for ind in required_indicators if ind not in enhanced_df.columns]
        if missing_indicators:
            print(f"警告: 仍有缺失的技术指标: {missing_indicators}")
            print("尝试继续训练...")
        
        # 设置默认参数
        window_size = 10
        available_data_points = len(enhanced_df) - window_size
        default_params = {
            'n_episodes': 100,
            'batch_size': 32,
            'reward_scaling': 1.0,
            'initial_balance': 100000,
            'transaction_fee_percent': 0.001,
            'window_size': window_size,
            'max_steps': max(20, available_data_points // 2)  # 使用一半可用数据点，但至少20步
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
        
        # 临时保存并加载数据，以确保格式正确
        temp_csv = os.path.join(save_dir, "temp_training_data.csv")
        enhanced_df.to_csv(temp_csv, index=False)
        df_for_training = pd.read_csv(temp_csv)
        
        # 转换列类型并填充NaN
        for col in df_for_training.columns:
            if col == 'date':
                df_for_training[col] = pd.to_datetime(df_for_training[col])
            elif df_for_training[col].dtype == 'object':
                df_for_training[col] = pd.to_numeric(df_for_training[col], errors='coerce')
        
        # 再次检查并填充NaN
        df_for_training = df_for_training.fillna(0)
        
        # 检查是否有无穷大的值
        values_array = df_for_training.select_dtypes(include=[np.number]).values.astype(float)
        if np.isinf(values_array).any():
            print("警告: 数据中存在无穷大的值，将其替换为0")
            df_for_training = df_for_training.replace([np.inf, -np.inf], 0)
        
        # 初始化RL交易系统
        rl_trader = RLTrader(model_dir=save_dir)
        
        # 训练模型
        if verbose:
            print("开始训练RL模型，这可能需要一些时间...")
        
        training_history = rl_trader.train(
            df=df_for_training,
            initial_balance=default_params['initial_balance'],
            transaction_fee_percent=default_params['transaction_fee_percent'],
            n_episodes=default_params['n_episodes'],
            batch_size=default_params['batch_size'],
            reward_scaling=default_params['reward_scaling'],
            max_steps=default_params['max_steps'],
        )
        
        # 创建RLTradingAgent并加载刚训练好的模型
        rl_agent = RLTradingAgent(model_dir=save_dir)
        success = rl_agent.load_model("best_model")
        
        # 只有在加载成功后才生成信号
        if success:
            signals = rl_agent.generate_signals(df_for_training)
            if verbose:
                print(f"训练完成。生成的交易信号: {signals.get('signal', 'unknown')}, 置信度: {signals.get('confidence', 0)}")
        else:
            print("警告: 模型训练后无法加载，可能训练失败")
            signals = {'signal': 'neutral', 'confidence': 0.5}
        
        # 清理临时文件
        try:
            os.remove(temp_csv)
        except:
            pass
        
        # 只有在成功加载模型后才返回agent
        return (rl_agent, signals) if success else (None, signals)
    
    except Exception as e:
        print(f"训练强化学习模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, {'signal': 'neutral', 'confidence': 0.5}


def train_factor_model(prices_df, params=None, save_dir='factors', verbose=True):
    """训练遗传编程因子模型"""
    if verbose:
        print("正在准备训练遗传编程因子模型...")
    
    try:
        # 设置默认参数
        default_params = {
            'n_factors': 3,  # 因子数量
            'population_size': 50,  # 种群大小
            'n_generations': 20,  # 代数
            'future_return_periods': 5,
            'min_fitness': 0.03  # 适应度
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
        
        # 确保数据格式正确
        if 'close' not in prices_df.columns and 'Close' in prices_df.columns:
            prices_df['close'] = prices_df['Close']
        if 'open' not in prices_df.columns and 'Open' in prices_df.columns:
            prices_df['open'] = prices_df['Open']
        if 'high' not in prices_df.columns and 'High' in prices_df.columns:
            prices_df['high'] = prices_df['High']
        if 'low' not in prices_df.columns and 'Low' in prices_df.columns:
            prices_df['low'] = prices_df['Low']
        if 'volume' not in prices_df.columns and 'Volume' in prices_df.columns:
            prices_df['volume'] = prices_df['Volume']
            
        # 预先添加一些基本技术指标来丰富特征空间
        enhanced_df = prices_df.copy()
        
        # 添加一些基本技术指标
        if verbose:
            print("添加基本技术指标以丰富特征空间...")
            
        # 添加一些基本的移动平均线作为起点
        enhanced_df['ma5'] = enhanced_df['close'].rolling(window=5).mean()
        enhanced_df['ma10'] = enhanced_df['close'].rolling(window=10).mean()
        enhanced_df['ma20'] = enhanced_df['close'].rolling(window=20).mean()
        
        # 添加一些基本的价格动量指标
        enhanced_df['returns_1d'] = enhanced_df['close'].pct_change(1)
        enhanced_df['returns_5d'] = enhanced_df['close'].pct_change(5)
        
        # 添加波动率指标
        enhanced_df['volatility_10d'] = enhanced_df['returns_1d'].rolling(window=10).std()
        
        # 填充NaN值
        enhanced_df = enhanced_df.fillna(0)
        
        # 生成因子
        if verbose:
            print("开始生成因子，这可能需要较长时间...")
        
        # 生成因子
        factors = factor_agent.generate_factors(
            price_data=enhanced_df,
            n_factors=default_params['n_factors']
        )
        
        # 生成交易信号
        signals = factor_agent.generate_signals(enhanced_df)
        
        if verbose:
            print(f"因子生成完成。生成的交易信号: {signals.get('signal', 'unknown')}, 置信度: {signals.get('confidence', 0)}")
            print(f"生成了 {len(factors)} 个因子")
        
        return factor_agent, signals
    
    except Exception as e:
        print(f"训练遗传编程因子模型时出错: {e}")
        import traceback
        traceback.print_exc()
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
            
            # 检查模型文件是否存在
            model_path = os.path.join(model_dir, "best_model.pth")
            if not os.path.exists(model_path):
                if verbose:
                    print(f"模型文件 {model_path} 不存在，无法加载")
                return None
                
            success = agent.load_model("best_model")  # 指定模型名称
            if success and verbose:
                print("强化学习模型加载成功")
            elif verbose:
                print("强化学习模型加载失败")
            return agent if success else None
        
        elif model_type == 'factor':
            agent = FactorAgent(model_dir=model_dir)
            factors = agent.load_factors()
            if factors and verbose:
                print(f"因子模型加载成功，加载了{len(factors)}个因子")
            elif verbose:
                print("因子模型加载失败或没有找到因子")
            return agent if factors else None
        
        else:
            if verbose:
                print(f"不支持的模型类型: {model_type}")
            return None
    
    except Exception as e:
        print(f"加载模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_signals(agent, model_type, prices_df, verbose=True):
    """使用模型生成交易信号"""
    if verbose:
        print(f"使用{model_type}模型生成交易信号...")
    
    try:
        if agent is None:
            if verbose:
                print(f"警告: {model_type}模型不可用，使用中性信号")
            return {'signal': 'neutral', 'confidence': 0.5}
        
        # 添加缺失的技术指标
        prices_df = add_technical_indicators(prices_df.copy())
        
        # 对于因子模型，需要特殊处理
        if model_type == 'factor':
            # 信号生成由因子模型内部处理，不需要提前计算特征
            # 因子模型内部会根据需要计算各种特征
            signals = agent.generate_signals(prices_df)
        elif model_type == 'rl':
            window_size = 20  # RL模型默认窗口大小
            if len(prices_df) < window_size + 10:  # 确保至少有窗口大小+10的数据量
                if verbose:
                    print(f"警告: 数据量({len(prices_df)}行)不足以使用RL模型。需要至少{window_size + 10}行数据。")
                return {'signal': 'neutral', 'confidence': 0.5, 'error': '数据量不足'}
        else:
            # 详细的错误捕获
            try:
                signals = agent.generate_signals(prices_df)
            except Exception as e:
                import traceback
                print(f"生成信号时出现错误: {e}")
                traceback.print_exc()
                # 尝试使用简单的字典返回
                signals = {'signal': 'neutral', 'confidence': 0.5, 'error': str(e)}
        
        if verbose:
            print(f"生成的交易信号: {signals.get('signal', 'unknown')}, 置信度: {signals.get('confidence', 0)}")
            if 'reasoning' in signals:
                print(f"决策理由: {signals['reasoning']}")
            # 打印完整的信号字典以进行调试
            print(f"完整信号信息: {signals}")
        
        return signals
    
    except Exception as e:
        print(f"生成交易信号时出错: {e}")
        import traceback
        traceback.print_exc()
        return {'signal': 'neutral', 'confidence': 0.5}


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='股票交易模型训练和测试工具')
    
    # 基本参数
    parser.add_argument('--ticker', type=str, required=True,
                        help='股票代码，例如: 600519')
    parser.add_argument('--start-date', type=str,
                        default=(datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d'),
                        help='开始日期，格式：YYYY-MM-DD，默认为两年前')
    parser.add_argument('--end-date', type=str,
                        default=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
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
    if prices_df is None or (isinstance(prices_df, pd.DataFrame) and prices_df.empty):
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
        
        # 确保目录存在
        os.makedirs(model_dir, exist_ok=True)
        
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
            
            # 如果只是测试但没有训练，尝试加载已有模型
            if args.action == 'test':
                agent = load_model(model_type, model_dir, args.verbose)
            
            # 生成信号 - 已训练或已加载才生成信号
            if agent is not None:
                signals = generate_signals(agent, model_type, prices_df, args.verbose)
            else:
                # 如果没有成功训练或加载模型，提供中性信号
                if args.verbose:
                    print(f"警告: 未能训练或加载{model_type}模型，使用中性信号")
                signals = {'signal': 'neutral', 'confidence': 0.5}
        
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
            
            print("\n模型投票结果:\n")
            print(f"看多: {signal_counts['bullish']}，看空: {signal_counts['bearish']}，中性: {signal_counts['neutral']}\n")
            print(f"加权评分: {normalized_score:.2f} (范围: -1到1)\n")
            print(f"综合信号: {final_signal}\n")


if __name__ == '__main__':
    main()