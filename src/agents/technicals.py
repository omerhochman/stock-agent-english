import math
from typing import Dict

from langchain_core.messages import HumanMessage

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint, log_llm_interaction

import json
import pandas as pd
import numpy as np

from src.tools.api import prices_to_df


##### Technical Analyst #####
@agent_endpoint("technical_analyst", "技术分析师，提供基于价格走势、指标和技术模式的交易信号")
def technical_analyst_agent(state: AgentState):
    """
    Sophisticated technical analysis system that combines multiple trading strategies:
    1. Trend Following
    2. Mean Reversion
    3. Momentum
    4. Volatility Analysis
    5. Statistical Arbitrage Signals
    """
    show_workflow_status("Technical Analyst")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    prices = data["prices"]
    prices_df = prices_to_df(prices)

    # Initialize confidence variable
    confidence = 0.0

    # Calculate indicators
    # 1. MACD (Moving Average Convergence Divergence)
    macd_line, signal_line = calculate_macd(prices_df)

    # 2. RSI (Relative Strength Index)
    rsi = calculate_rsi(prices_df)

    # 3. Bollinger Bands (Bollinger Bands)
    upper_band, lower_band = calculate_bollinger_bands(prices_df)

    # 4. OBV (On-Balance Volume)
    obv = calculate_obv(prices_df)

    # Generate individual signals
    signals = []

    # MACD signal
    if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
        signals.append('bullish')
    elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
        signals.append('bearish')
    else:
        signals.append('neutral')

    # RSI signal
    if rsi.iloc[-1] < 30:
        signals.append('bullish')
    elif rsi.iloc[-1] > 70:
        signals.append('bearish')
    else:
        signals.append('neutral')

    # Bollinger Bands signal
    current_price = prices_df['close'].iloc[-1]
    if current_price < lower_band.iloc[-1]:
        signals.append('bullish')
    elif current_price > upper_band.iloc[-1]:
        signals.append('bearish')
    else:
        signals.append('neutral')

    # OBV signal
    obv_slope = obv.diff().iloc[-5:].mean()
    if obv_slope > 0:
        signals.append('bullish')
    elif obv_slope < 0:
        signals.append('bearish')
    else:
        signals.append('neutral')

    # Calculate price drop
    price_drop = (prices_df['close'].iloc[-1] -
                  prices_df['close'].iloc[-5]) / prices_df['close'].iloc[-5]

    # Add price drop signal
    if price_drop < -0.05 and rsi.iloc[-1] < 40:  # 5% drop and RSI below 40
        signals.append('bullish')
        confidence += 0.2  # Increase confidence for oversold conditions
    elif price_drop < -0.03 and rsi.iloc[-1] < 45:  # 3% drop and RSI below 45
        signals.append('bullish')
        confidence += 0.1

    # Add reasoning collection
    reasoning = {
        "MACD": {
            "signal": signals[0],
            "details": f"MACD Line crossed {'above' if signals[0] == 'bullish' else 'below' if signals[0] == 'bearish' else 'neither above nor below'} Signal Line"
        },
        "RSI": {
            "signal": signals[1],
            "details": f"RSI is {rsi.iloc[-1]:.2f} ({'oversold' if signals[1] == 'bullish' else 'overbought' if signals[1] == 'bearish' else 'neutral'})"
        },
        "Bollinger": {
            "signal": signals[2],
            "details": f"Price is {'below lower band' if signals[2] == 'bullish' else 'above upper band' if signals[2] == 'bearish' else 'within bands'}"
        },
        "OBV": {
            "signal": signals[3],
            "details": f"OBV slope is {obv_slope:.2f} ({signals[3]})"
        }
    }

    # Determine overall signal
    bullish_signals = signals.count('bullish')
    bearish_signals = signals.count('bearish')

    if bullish_signals > bearish_signals:
        overall_signal = 'bullish'
    elif bearish_signals > bullish_signals:
        overall_signal = 'bearish'
    else:
        overall_signal = 'neutral'

    # Calculate confidence level based on the proportion of indicators agreeing
    total_signals = len(signals)
    confidence = max(bullish_signals, bearish_signals) / total_signals

    # Generate the message content
    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence * 100)}%",
        "reasoning": {
            "MACD": reasoning["MACD"],
            "RSI": reasoning["RSI"],
            "Bollinger": reasoning["Bollinger"],
            "OBV": reasoning["OBV"]
        }
    }

    # 1. Trend Following Strategy
    trend_signals = calculate_trend_signals(prices_df)

    # 2. Mean Reversion Strategy
    mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

    # 3. Momentum Strategy
    momentum_signals = calculate_momentum_signals(prices_df)

    # 4. Volatility Strategy
    volatility_signals = calculate_volatility_signals(prices_df)

    # 5. Statistical Arbitrage Signals
    stat_arb_signals = calculate_stat_arb_signals(prices_df)

    # Combine all signals using a weighted ensemble approach
    strategy_weights = {
        'trend': 0.30,
        'mean_reversion': 0.25,  # Increased weight for mean reversion
        'momentum': 0.25,
        'volatility': 0.15,
        'stat_arb': 0.05
    }

    combined_signal = weighted_signal_combination({
        'trend': trend_signals,
        'mean_reversion': mean_reversion_signals,
        'momentum': momentum_signals,
        'volatility': volatility_signals,
        'stat_arb': stat_arb_signals
    }, strategy_weights)

    # Generate detailed analysis report
    analysis_report = {
        "signal": combined_signal['signal'],
        "confidence": f"{round(combined_signal['confidence'] * 100)}%",
        "strategy_signals": {
            "trend_following": {
                "signal": trend_signals['signal'],
                "confidence": f"{round(trend_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas(trend_signals['metrics'])
            },
            "mean_reversion": {
                "signal": mean_reversion_signals['signal'],
                "confidence": f"{round(mean_reversion_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas(mean_reversion_signals['metrics'])
            },
            "momentum": {
                "signal": momentum_signals['signal'],
                "confidence": f"{round(momentum_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas(momentum_signals['metrics'])
            },
            "volatility": {
                "signal": volatility_signals['signal'],
                "confidence": f"{round(volatility_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas(volatility_signals['metrics'])
            },
            "statistical_arbitrage": {
                "signal": stat_arb_signals['signal'],
                "confidence": f"{round(stat_arb_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas(stat_arb_signals['metrics'])
            }
        }
    }

    # Create the technical analyst message
    message = HumanMessage(
        content=json.dumps(analysis_report),
        name="technical_analyst_agent",
    )

    if show_reasoning:
        show_agent_reasoning(analysis_report, "Technical Analyst")
        # 保存推理信息到state的metadata供API使用
        state["metadata"]["agent_reasoning"] = analysis_report

    show_workflow_status("Technical Analyst", "completed")
    return {
        "messages": [message],
        "data": data,
        "metadata": state["metadata"],
    }


def calculate_trend_signals(prices_df):
    """
    Advanced trend following strategy using multiple timeframes and indicators
    """
    # Calculate EMAs for multiple timeframes
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)

    # Calculate ADX for trend strength
    adx = calculate_adx(prices_df, 14)

    # Calculate Ichimoku Cloud
    ichimoku = calculate_ichimoku(prices_df)

    # Determine trend direction and strength
    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55

    # Combine signals with confidence weighting
    trend_strength = adx['adx'].iloc[-1] / 100.0

    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = 'bullish'
        confidence = trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = 'bearish'
        confidence = trend_strength
    else:
        signal = 'neutral'
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'adx': float(adx['adx'].iloc[-1]),
            'trend_strength': float(trend_strength),
            # 'ichimoku': ichimoku
        }
    }


def calculate_mean_reversion_signals(prices_df):
    """
    均值回归策略，使用多种技术指标和统计方法
    
    Args:
        prices_df: 价格数据DataFrame
    
    Returns:
        dict: 均值回归信号和指标
    """
    # 计算价格相对移动平均的偏离度
    ma_50 = prices_df['close'].rolling(window=50).mean()
    std_50 = prices_df['close'].rolling(window=50).std()
    z_score = (prices_df['close'] - ma_50) / std_50
    
    # 计算布林带
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df, window=20)
    
    # 计算多周期RSI
    rsi_14 = calculate_rsi(prices_df, period=14)
    rsi_28 = calculate_rsi(prices_df, period=28)
    
    # 计算价格离散度：衡量价格偏离多个周期移动平均的程度
    ma_10 = prices_df['close'].rolling(window=10).mean()
    ma_20 = prices_df['close'].rolling(window=20).mean()
    ma_50 = prices_df['close'].rolling(window=50).mean()
    
    # 计算相对于多个均线的偏离度
    deviation_10 = (prices_df['close'] - ma_10) / ma_10
    deviation_20 = (prices_df['close'] - ma_20) / ma_20
    deviation_50 = (prices_df['close'] - ma_50) / ma_50
    
    # 计算平均偏离度
    avg_deviation = (deviation_10 + deviation_20 + deviation_50) / 3
    
    # 计算Stochastic Oscillator (KD指标)
    high_14 = prices_df['high'].rolling(window=14).max()
    low_14 = prices_df['low'].rolling(window=14).min()
    k_percent = 100 * ((prices_df['close'] - low_14) / (high_14 - low_14))
    d_percent = k_percent.rolling(window=3).mean()
    
    # 增强型均值回归信号
    # 1. RSI超买超卖信号
    rsi_signal = 0
    if rsi_14.iloc[-1] < 30:
        rsi_signal = 1  # 超卖
    elif rsi_14.iloc[-1] > 70:
        rsi_signal = -1  # 超买
        
    # 2. 布林带信号
    bb_signal = 0
    price_vs_bb = (prices_df['close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
    if price_vs_bb < 0.1:  # 价格接近下轨
        bb_signal = 1
    elif price_vs_bb > 0.9:  # 价格接近上轨
        bb_signal = -1
        
    # 3. KD指标信号
    kd_signal = 0
    if k_percent.iloc[-1] < 20 and d_percent.iloc[-1] < 20:
        kd_signal = 1  # 超卖
    elif k_percent.iloc[-1] > 80 and d_percent.iloc[-1] > 80:
        kd_signal = -1  # 超买
        
    # 4. 价格偏离均值信号
    deviation_signal = 0
    if avg_deviation.iloc[-1] < -0.05:
        deviation_signal = 1  # 价格显著低于均值
    elif avg_deviation.iloc[-1] > 0.05:
        deviation_signal = -1  # 价格显著高于均值
    
    # 聚合信号评分 (-4到4，负表示超买，正表示超卖)
    signal_score = rsi_signal + bb_signal + kd_signal + deviation_signal
    
    # 使用逻辑回归加权生成最终信号
    # 不同指标在不同市场条件下有不同可靠性
    if signal_score >= 2:  # 强烈超卖信号
        signal = 'bullish'
        confidence = min(0.3 + (signal_score - 2) * 0.1, 0.7)  # 最高70%置信度
    elif signal_score <= -2:  # 强烈超买信号
        signal = 'bearish'
        confidence = min(0.3 + (abs(signal_score) - 2) * 0.1, 0.7)  # 最高70%置信度
    else:
        signal = 'neutral'
        confidence = 0.3  # 中性信号较低置信度
    
    # 根据市场波动调整置信度
    volatility_adjustment = 1.0
    historical_volatility = prices_df['close'].pct_change().rolling(window=20).std()
    current_volatility = historical_volatility.iloc[-1] if not historical_volatility.empty else 0
    
    # 高波动环境下降低均值回归策略置信度
    if current_volatility > 0.02:  # 日波动率高于2%
        volatility_adjustment = 0.8
    
    confidence = confidence * volatility_adjustment
    
    # 计算信号的时间相关性(较新的信号权重更高)
    time_relevance = 1.0  # 默认值
    
    # 检测是否为反转模式
    if rsi_14.iloc[-2] < 30 and rsi_14.iloc[-1] > 30:  # RSI刚从超卖区域回升
        signal = 'bullish'
        confidence = min(confidence + 0.15, 0.85)  # 提高置信度
        time_relevance = 1.2  # 提高时间相关性
    elif rsi_14.iloc[-2] > 70 and rsi_14.iloc[-1] < 70:  # RSI刚从超买区域回落
        signal = 'bearish'
        confidence = min(confidence + 0.15, 0.85)  # 提高置信度
        time_relevance = 1.2  # 提高时间相关性
    
    return {
        'signal': signal,
        'confidence': confidence,
        'time_relevance': time_relevance,
        'metrics': {
            'z_score': float(z_score.iloc[-1]) if not z_score.empty else 0,
            'price_vs_bb': float(price_vs_bb),
            'rsi_14': float(rsi_14.iloc[-1]) if not rsi_14.empty else 0,
            'rsi_28': float(rsi_28.iloc[-1]) if not rsi_28.empty else 0,
            'avg_deviation': float(avg_deviation.iloc[-1]) if not avg_deviation.empty else 0,
            'k_percent': float(k_percent.iloc[-1]) if not k_percent.empty else 0,
            'd_percent': float(d_percent.iloc[-1]) if not d_percent.empty else 0,
            'signal_score': signal_score
        }
    }

def calculate_momentum_signals(prices_df):
    """
    多因子动量策略，考虑价格趋势、成交量和相对强度
    
    Args:
        prices_df: 价格数据DataFrame
    
    Returns:
        dict: 动量信号和指标
    """
    # 计算多周期价格动量
    returns = prices_df['close'].pct_change()
    
    # 计算不同周期动量并进行缺失值处理
    mom_1m = returns.rolling(21).sum().fillna(0)  # 1月
    mom_3m = returns.rolling(63).sum().fillna(mom_1m)  # 3月
    mom_6m = returns.rolling(126).sum().fillna(mom_3m)  # 6月
    
    # 相对强度指标 - 与大盘/行业比较的相对强度
    # 此处使用模拟值，实际应当用行业指数数据
    relative_strength = np.random.normal(0, 0.05, len(returns)) + mom_3m * 0.7
    relative_strength = pd.Series(relative_strength, index=returns.index)
    
    # 成交量趋势确认
    volume_ma = prices_df['volume'].rolling(21).mean()
    volume_ratio = prices_df['volume'] / volume_ma
    volume_trend = volume_ratio.rolling(10).mean()  # 成交量趋势
    
    # 计算动量发散 - OBV与价格趋势比较
    obv = calculate_obv(prices_df)
    obv_ma = obv.rolling(window=20).mean()
    price_ma = prices_df['close'].rolling(window=20).mean()
    
    # 归一化价格和OBV以便比较
    norm_price = (prices_df['close'] - prices_df['close'].rolling(100).min()) / \
                (prices_df['close'].rolling(100).max() - prices_df['close'].rolling(100).min())
    norm_obv = (obv - obv.rolling(100).min()) / (obv.rolling(100).max() - obv.rolling(100).min())
    
    # 计算价格和OBV之间的发散
    divergence = norm_obv - norm_price
    divergence_signal = divergence.rolling(5).mean()
    
    # 使用加权组合计算动量分数
    # 较近期动量权重更高，并且考虑相对强度
    momentum_score = (
        0.2 * mom_1m +
        0.3 * mom_3m +
        0.3 * mom_6m +
        0.2 * relative_strength
    ).iloc[-1]
    
    # 成交量确认分数
    volume_confirmation = 1.0  # 默认值
    if volume_trend.iloc[-1] > 1.1:  # 成交量上升
        volume_confirmation = 1.2  # 增强确认
    elif volume_trend.iloc[-1] < 0.9:  # 成交量下降
        volume_confirmation = 0.8  # 削弱确认
    
    # 考虑发散信号
    divergence_factor = 1.0  # 默认值
    if divergence_signal.iloc[-1] > 0.1 and mom_3m.iloc[-1] < 0:
        # 看涨背离：OBV上升而价格下跌
        divergence_factor = 1.2
    elif divergence_signal.iloc[-1] < -0.1 and mom_3m.iloc[-1] > 0:
        # 看跌背离：OBV下降而价格上升
        divergence_factor = 0.8
    
    # 生成最终信号
    if momentum_score > 0.05 and volume_confirmation >= 1.0:
        signal = 'bullish'
        base_confidence = min(abs(momentum_score) * 5, 0.8)
    elif momentum_score < -0.05 and volume_confirmation >= 1.0:
        signal = 'bearish'
        base_confidence = min(abs(momentum_score) * 5, 0.8)
    else:
        signal = 'neutral'
        base_confidence = 0.3
    
    # 应用调整系数
    confidence = base_confidence * volume_confirmation * divergence_factor
    confidence = min(max(confidence, 0.2), 0.9)  # 限定在0.2-0.9范围内
    
    # 市场条件相关性
    market_condition_relevance = 1.0
    
    # 在上升趋势中，动量信号更可靠
    if mom_3m.iloc[-1] > 0 and mom_6m.iloc[-1] > 0:
        market_condition_relevance = 1.2
    
    return {
        'signal': signal,
        'confidence': confidence,
        'market_condition_relevance': market_condition_relevance,
        'metrics': {
            'momentum_1m': float(mom_1m.iloc[-1]),
            'momentum_3m': float(mom_3m.iloc[-1]),
            'momentum_6m': float(mom_6m.iloc[-1]),
            'relative_strength': float(relative_strength.iloc[-1]),
            'volume_trend': float(volume_trend.iloc[-1]) if not volume_trend.empty else 1.0,
            'divergence': float(divergence_signal.iloc[-1]) if not divergence_signal.empty else 0
        }
    }


def calculate_volatility_signals(prices_df):
    """
    Optimized volatility calculation with shorter lookback periods
    """
    returns = prices_df['close'].pct_change()

    # 使用更短的周期和最小周期要求计算历史波动率
    hist_vol = returns.rolling(21, min_periods=10).std() * math.sqrt(252)

    # 使用更短的周期计算波动率均值，并允许更少的数据点
    vol_ma = hist_vol.rolling(42, min_periods=21).mean()
    vol_regime = hist_vol / vol_ma

    # 使用更灵活的标准差计算
    vol_std = hist_vol.rolling(42, min_periods=21).std()
    vol_z_score = (hist_vol - vol_ma) / vol_std.replace(0, np.nan)

    # ATR计算优化
    atr = calculate_atr(prices_df, period=14, min_periods=7)
    atr_ratio = atr / prices_df['close']

    # 如果关键指标为NaN，使用替代值而不是直接返回中性信号
    if pd.isna(vol_regime.iloc[-1]):
        vol_regime.iloc[-1] = 1.0  # 假设处于正常波动率区间
    if pd.isna(vol_z_score.iloc[-1]):
        vol_z_score.iloc[-1] = 0.0  # 假设处于均值位置

    # Generate signal based on volatility regime
    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]

    if current_vol_regime < 0.8 and vol_z < -1:
        signal = 'bullish'  # Low vol regime, potential for expansion
        confidence = min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > 1.2 and vol_z > 1:
        signal = 'bearish'  # High vol regime, potential for contraction
        confidence = min(abs(vol_z) / 3, 1.0)
    else:
        signal = 'neutral'
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'historical_volatility': float(hist_vol.iloc[-1]),
            'volatility_regime': float(current_vol_regime),
            'volatility_z_score': float(vol_z),
            'atr_ratio': float(atr_ratio.iloc[-1])
        }
    }


def calculate_stat_arb_signals(prices_df):
    """
    Optimized statistical arbitrage signals with shorter lookback periods
    """
    # Calculate price distribution statistics
    returns = prices_df['close'].pct_change()

    # 使用更短的周期计算偏度和峰度
    skew = returns.rolling(42, min_periods=21).skew()
    kurt = returns.rolling(42, min_periods=21).kurt()

    # 优化Hurst指数计算
    hurst = calculate_hurst_exponent(prices_df['close'], max_lag=10)

    # 处理NaN值
    if pd.isna(skew.iloc[-1]):
        skew.iloc[-1] = 0.0  # 假设正态分布
    if pd.isna(kurt.iloc[-1]):
        kurt.iloc[-1] = 3.0  # 假设正态分布

    # Generate signal based on statistical properties
    if hurst < 0.4 and skew.iloc[-1] > 1:
        signal = 'bullish'
        confidence = (0.5 - hurst) * 2
    elif hurst < 0.4 and skew.iloc[-1] < -1:
        signal = 'bearish'
        confidence = (0.5 - hurst) * 2
    else:
        signal = 'neutral'
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'hurst_exponent': float(hurst),
            'skewness': float(skew.iloc[-1]),
            'kurtosis': float(kurt.iloc[-1])
        }
    }


def weighted_signal_combination(signals, weights):
    """
    使用自适应加权方法组合多种交易信号
    
    Args:
        signals: 各策略信号字典
        weights: 各策略权重字典
    
    Returns:
        dict: 包含最终信号和置信度的字典
    """
    # 转换信号为数值
    signal_values = {
        'bullish': 1,
        'neutral': 0,
        'bearish': -1
    }

    # 收集各策略结果
    strategy_scores = []
    total_weight = 0
    
    # 计算动态置信度调整因子
    confidence_adjustment = 1.0
    
    # 计算信号一致性
    signal_counts = {'bullish': 0, 'neutral': 0, 'bearish': 0}
    for strategy, signal in signals.items():
        signal_counts[signal['signal']] += 1
    
    # 一致性指标 (0-1，越高表示信号越一致)
    max_signal_count = max(signal_counts.values())
    consistency = max_signal_count / sum(signal_counts.values()) if sum(signal_counts.values()) > 0 else 0
    
    # 调整置信度：一致性高时提高整体置信度
    if consistency > 0.7:  # 超过70%策略给出相同信号
        confidence_adjustment = 1.2  # 提高20%置信度
    elif consistency < 0.4:  # 信号极度分散
        confidence_adjustment = 0.8  # 降低20%置信度
    
    # 自适应权重调整
    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal['signal']]
        base_weight = weights[strategy]
        confidence = signal['confidence']
        
        # 计算时间相关性权重调整
        time_relevance = 1.0
        if 'time_relevance' in signal:
            time_relevance = signal['time_relevance']
        
        # 计算市场条件相关性调整
        market_condition_factor = 1.0
        if 'market_condition_relevance' in signal:
            market_condition_factor = signal['market_condition_relevance']
        
        # 调整后的权重
        adjusted_weight = base_weight * time_relevance * market_condition_factor
        
        # 添加加权分数
        weighted_score = numeric_signal * adjusted_weight * confidence
        strategy_scores.append(weighted_score)
        total_weight += adjusted_weight * confidence
    
    # 计算最终得分
    if total_weight > 0:
        final_score = sum(strategy_scores) / total_weight
    else:
        final_score = 0
    
    # 应用一致性调整
    final_confidence = min(abs(final_score) * confidence_adjustment, 1.0)
    
    # 转换回信号
    if final_score > 0.2:
        signal = 'bullish'
    elif final_score < -0.2:
        signal = 'bearish'
    else:
        signal = 'neutral'
    
    return {
        'signal': signal,
        'confidence': final_confidence,
        'weighted_score': final_score,
        'signal_consistency': consistency
    }


def normalize_pandas(obj):
    """Convert pandas Series/DataFrames to primitive Python types"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


def calculate_macd(prices_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    ema_12 = prices_df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = prices_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(
    prices_df: pd.DataFrame,
    window: int = 20
) -> tuple[pd.Series, pd.Series]:
    sma = prices_df['close'].rolling(window).mean()
    std_dev = prices_df['close'].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average

    Args:
        df: DataFrame with price data
        window: EMA period

    Returns:
        pd.Series: EMA values
    """
    return df['close'].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)

    Args:
        df: DataFrame with OHLC data
        period: Period for calculations

    Returns:
        DataFrame with ADX values
    """
    # Calculate True Range
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

    # Calculate Directional Movement
    df['up_move'] = df['high'] - df['high'].shift()
    df['down_move'] = df['low'].shift() - df['low']

    df['plus_dm'] = np.where(
        (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
        df['up_move'],
        0
    )
    df['minus_dm'] = np.where(
        (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
        df['down_move'],
        0
    )

    # Calculate ADX
    df['+di'] = 100 * (df['plus_dm'].ewm(span=period).mean() /
                       df['tr'].ewm(span=period).mean())
    df['-di'] = 100 * (df['minus_dm'].ewm(span=period).mean() /
                       df['tr'].ewm(span=period).mean())
    df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])
    df['adx'] = df['dx'].ewm(span=period).mean()

    return df[['adx', '+di', '-di']]


def calculate_ichimoku(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate Ichimoku Cloud indicators

    Args:
        df: DataFrame with OHLC data

    Returns:
        Dictionary containing Ichimoku components
    """
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = df['high'].rolling(window=9).max()
    period9_low = df['low'].rolling(window=9).min()
    tenkan_sen = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = df['high'].rolling(window=26).max()
    period26_low = df['low'].rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = df['high'].rolling(window=52).max()
    period52_low = df['low'].rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

    # Chikou Span (Lagging Span): Close shifted back 26 periods
    chikou_span = df['close'].shift(-26)

    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }


def calculate_atr(df: pd.DataFrame, period: int = 14, min_periods: int = 7) -> pd.Series:
    """
    Optimized ATR calculation with minimum periods parameter

    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation
        min_periods: Minimum number of periods required

    Returns:
        pd.Series: ATR values
    """
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(period, min_periods=min_periods).mean()


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 10) -> float:
    """
    Optimized Hurst exponent calculation with shorter lookback and better error handling

    Args:
        price_series: Array-like price data
        max_lag: Maximum lag for R/S calculation (reduced from 20 to 10)

    Returns:
        float: Hurst exponent
    """
    try:
        # 使用对数收益率而不是价格
        returns = np.log(price_series / price_series.shift(1)).dropna()

        # 如果数据不足，返回0.5（随机游走）
        if len(returns) < max_lag * 2:
            return 0.5

        lags = range(2, max_lag)
        # 使用更稳定的计算方法
        tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag])))
               for lag in lags]

        # 添加小的常数避免log(0)
        tau = [max(1e-8, t) for t in tau]

        # 使用对数回归计算Hurst指数
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        h = reg[0]

        # 限制Hurst指数在合理范围内
        return max(0.0, min(1.0, h))

    except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
        # 如果计算失败，返回0.5表示随机游走
        return 0.5


def calculate_obv(prices_df: pd.DataFrame) -> pd.Series:
    obv = [0]
    for i in range(1, len(prices_df)):
        if prices_df['close'].iloc[i] > prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] + prices_df['volume'].iloc[i])
        elif prices_df['close'].iloc[i] < prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] - prices_df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    prices_df['OBV'] = obv
    return prices_df['OBV']
