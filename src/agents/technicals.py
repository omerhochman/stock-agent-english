import math
from typing import Dict
import json

import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.agents.regime_detector import AdvancedRegimeDetector
from src.utils.api_utils import agent_endpoint
from src.calc.volatility_models import fit_garch, forecast_garch_volatility
from src.tools.api import prices_to_df


##### Technical Analyst #####
@agent_endpoint("technical_analyst", "技术分析师，提供基于价格走势、指标和技术模式的交易信号")
def technical_analyst_agent(state: AgentState):
    """
    基于2024-2025研究的区制感知技术分析系统
    集成FINSABER、FLAG-Trader等框架的先进技术分析策略
    """
    show_workflow_status("Technical Analyst")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    prices = data["prices"]
    prices_df = prices_to_df(prices)

    # 初始化区制检测器
    regime_detector = AdvancedRegimeDetector()
    
    # 进行区制分析
    regime_features = regime_detector.extract_regime_features(prices_df)
    regime_model_results = regime_detector.fit_regime_model(regime_features)
    current_regime = regime_detector.predict_current_regime(regime_features)

    # 1. 趋势跟踪策略
    trend_signals = calculate_trend_signals(prices_df)

    # 2. 均值回归策略
    mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

    # 3. 动量策略
    momentum_signals = calculate_momentum_signals(prices_df)

    # 4. 波动率策略 - 使用GARCH模型预测
    volatility_signals = calculate_volatility_signals_with_garch(prices_df)

    # 5. 统计套利信号
    stat_arb_signals = calculate_stat_arb_signals(prices_df)

    # 6. 基于区制的动态权重调整 (基于2024-2025研究)
    regime_adjusted_weights = _calculate_regime_adjusted_weights(current_regime)

    # 使用区制感知的权重组合信号
    combined_signal = weighted_signal_combination({
        'trend': trend_signals,
        'mean_reversion': mean_reversion_signals,
        'momentum': momentum_signals,
        'volatility': volatility_signals,
        'stat_arb': stat_arb_signals
    }, regime_adjusted_weights)

    # 应用区制特定的信号过滤和增强
    enhanced_signal = _apply_regime_signal_enhancement(
        combined_signal, current_regime, prices_df
    )

    # 生成详细分析报告
    analysis_report = {
        "signal": enhanced_signal['signal'],
        "confidence": enhanced_signal['confidence'],
        "market_regime": current_regime,
        "regime_adjusted_weights": regime_adjusted_weights,
        "signal_enhancement": enhanced_signal.get('enhancement_details', {}),
        "strategy_signals": {
            "trend_following": {
                "signal": trend_signals['signal'],
                "confidence": trend_signals['confidence'],
                "metrics": normalize_pandas(trend_signals['metrics'])
            },
            "mean_reversion": {
                "signal": mean_reversion_signals['signal'],
                "confidence": mean_reversion_signals['confidence'],
                "metrics": normalize_pandas(mean_reversion_signals['metrics'])
            },
            "momentum": {
                "signal": momentum_signals['signal'],
                "confidence": momentum_signals['confidence'],
                "metrics": normalize_pandas(momentum_signals['metrics'])
            },
            "volatility": {
                "signal": volatility_signals['signal'],
                "confidence": volatility_signals['confidence'],
                "metrics": normalize_pandas(volatility_signals['metrics'])
            },
            "statistical_arbitrage": {
                "signal": stat_arb_signals['signal'],
                "confidence": stat_arb_signals['confidence'],
                "metrics": normalize_pandas(stat_arb_signals['metrics'])
            }
        }
    }

    # 创建技术分析师消息
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


def _calculate_regime_adjusted_weights(current_regime):
    """基于市场区制动态调整策略权重"""
    regime_name = current_regime.get("regime_name", "unknown")
    regime_confidence = current_regime.get("confidence", 0.5)
    
    # 基础权重
    base_weights = {
        'trend': 0.30,
        'mean_reversion': 0.25, 
        'momentum': 0.25,
        'volatility': 0.15,
        'stat_arb': 0.05
    }
    
    # 区制特定调整 (基于FINSABER 2024和Lopez-Lira 2025研究)
    if regime_name == "low_volatility_trending" and regime_confidence > 0.6:
        # 趋势市场中增加趋势跟踪和动量权重
        base_weights['trend'] = 0.40
        base_weights['momentum'] = 0.30
        base_weights['mean_reversion'] = 0.15
        base_weights['volatility'] = 0.10
        base_weights['stat_arb'] = 0.05
    elif regime_name == "high_volatility_mean_reverting" and regime_confidence > 0.6:
        # 区间震荡市场增加均值回归权重
        base_weights['trend'] = 0.20
        base_weights['momentum'] = 0.15
        base_weights['mean_reversion'] = 0.45
        base_weights['volatility'] = 0.15
        base_weights['stat_arb'] = 0.05
    elif regime_name == "crisis_regime" and regime_confidence > 0.6:
        # 危机市场增加波动率和统计套利权重
        base_weights['trend'] = 0.20
        base_weights['momentum'] = 0.20
        base_weights['mean_reversion'] = 0.20
        base_weights['volatility'] = 0.30
        base_weights['stat_arb'] = 0.10

    # 规范化权重，确保总和为1
    total_weight = sum(base_weights.values())
    return {k: v/total_weight for k, v in base_weights.items()}


def _apply_regime_signal_enhancement(combined_signal, current_regime, prices_df):
    """应用区制特定的信号增强技术"""
    regime_name = current_regime.get("regime_name", "unknown")
    regime_confidence = current_regime.get("confidence", 0.5)
    
    # 转换信号为数值进行计算
    signal_values = {
        'bullish': 1.0,
        'neutral': 0.0,
        'bearish': -1.0
    }
    
    # 反向映射
    value_to_signal = {
        1.0: 'bullish',
        0.0: 'neutral', 
        -1.0: 'bearish'
    }
    
    enhanced_signal = signal_values.get(combined_signal['signal'], 0.0)
    enhanced_confidence = combined_signal['confidence']
    enhancement_details = {}
    
    # 基于区制的信号过滤和增强
    if regime_name == "crisis_regime" and regime_confidence > 0.7:
        # 危机期间：降低信号强度，增加保守性
        enhanced_signal *= 0.7
        enhanced_confidence *= 0.8
        enhancement_details['crisis_dampening'] = "Applied crisis regime signal dampening"
        
    elif regime_name == "low_volatility_trending" and regime_confidence > 0.7:
        # 低波动趋势期间：增强趋势信号
        if abs(enhanced_signal) > 0.3:  # 只增强较强的信号
            enhanced_signal *= 1.2
            enhanced_confidence *= 1.1
            enhancement_details['trend_amplification'] = "Applied trend regime signal amplification"
    
    elif regime_name == "high_volatility_mean_reverting" and regime_confidence > 0.7:
        # 高波动震荡期间：应用反转逻辑
        returns = prices_df['close'].pct_change().dropna()
        recent_return = returns.iloc[-1] if len(returns) > 0 else 0
        
        if abs(recent_return) > 0.03:  # 显著价格移动
            # 应用均值回归逻辑
            if recent_return > 0 and enhanced_signal > 0:
                enhanced_signal *= 0.5  # 减弱追涨信号
            elif recent_return < 0 and enhanced_signal < 0:
                enhanced_signal *= 0.5  # 减弱杀跌信号
            enhancement_details['mean_reversion_filter'] = "Applied mean reversion filter"
    
    # 应用动态阈值 (基于RLMF 2024技术)
    dynamic_threshold = 0.15 if regime_name == "crisis_regime" else 0.1
    if abs(enhanced_signal) < dynamic_threshold:
        enhanced_signal *= 0.5  # 弱信号进一步衰减
        enhancement_details['weak_signal_dampening'] = f"Applied weak signal dampening (threshold: {dynamic_threshold})"
    
    # 确保置信度在合理范围内
    enhanced_confidence = max(0.1, min(enhanced_confidence, 0.95))
    
    # 将数值信号转换回字符串信号
    # 使用阈值来确定最终信号
    if enhanced_signal > 0.2:
        final_signal = 'bullish'
    elif enhanced_signal < -0.2:
        final_signal = 'bearish'
    else:
        final_signal = 'neutral'
    
    return {
        'signal': final_signal,
        'confidence': enhanced_confidence,
        'enhancement_details': enhancement_details
    }


def calculate_volatility_signals_with_garch(prices_df: pd.DataFrame) -> Dict:
    """
    使用GARCH模型高级波动率分析与预测
    """
    returns = prices_df['close'].pct_change().dropna()

    # 基础历史波动率计算
    hist_vol = returns.rolling(21, min_periods=10).std() * math.sqrt(252)
    vol_ma = hist_vol.rolling(42, min_periods=21).mean()
    vol_regime = hist_vol / vol_ma

    # ATR计算
    atr = calculate_atr(prices_df, period=14, min_periods=7)
    atr_ratio = atr / prices_df['close']

    # GARCH模型预测未来波动率
    garch_results = {}
    forecast_quality = 0.5  # 默认中等质量
    vol_trend = 0  # 默认无趋势
    
    try:
        if len(returns) >= 100:  # 确保有足够的数据
            # 使用calc模块中的GARCH函数拟合模型
            garch_params, log_likelihood = fit_garch(returns.values)
            volatility_forecast = forecast_garch_volatility(returns.values, garch_params, 
                                                         forecast_horizon=10)
            
            # 保存GARCH结果
            garch_results = {
                'model_type': 'GARCH(1,1)',
                'parameters': {
                    'omega': float(garch_params['omega']),
                    'alpha': float(garch_params['alpha']),
                    'beta': float(garch_params['beta']),
                    'persistence': float(garch_params['persistence'])
                },
                'log_likelihood': float(log_likelihood),
                'forecast': [float(v) for v in volatility_forecast],
                'forecast_annualized': [float(v * np.sqrt(252)) for v in volatility_forecast]
            }
            
            # 分析波动率趋势
            avg_forecast = np.mean(volatility_forecast)
            current_vol = hist_vol.iloc[-1] / np.sqrt(252)  # 转换为日度
            vol_trend = avg_forecast / current_vol - 1 if current_vol != 0 else 0
            
            # 模型质量评估
            persistence = garch_params['persistence']
            if 0.9 <= persistence <= 0.999:  # 合理的持续性范围
                forecast_quality = 0.8
            elif persistence > 0.999:  # 非平稳模型
                forecast_quality = 0.3
            else:  # 持续性较低
                forecast_quality = 0.5
    except Exception as e:
        garch_results = {"error": str(e)}

    # 决定信号
    current_vol_regime = vol_regime.iloc[-1] if not pd.isna(vol_regime.iloc[-1]) else 1.0
    vol_z = (hist_vol.iloc[-1] - vol_ma.iloc[-1]) / vol_ma.iloc[-1] if vol_ma.iloc[-1] != 0 else 0

    # 使用GARCH预测和基础波动率结合生成更精确的信号
    if vol_trend < -0.1 and current_vol_regime > 1.2:
        # 波动率处于高位但预计下降：看多信号
        signal = 'bullish'
        confidence = min(0.7, 0.5 + abs(vol_trend) * forecast_quality)
    elif vol_trend > 0.1 and current_vol_regime < 0.8:
        # 波动率处于低位但预计上升：看空信号
        signal = 'bearish'
        confidence = min(0.7, 0.5 + abs(vol_trend) * forecast_quality)
    elif vol_trend > 0.2:
        # 波动率急剧上升：看空信号
        signal = 'bearish'
        confidence = min(0.8, 0.6 + vol_trend * forecast_quality)
    elif current_vol_regime < 0.8 and vol_z < -1:
        # 低波动率环境：轻微看多
        signal = 'bullish'
        confidence = 0.6
    elif current_vol_regime > 1.2 and vol_z > 1:
        # 高波动率环境：轻微看空
        signal = 'bearish'
        confidence = 0.6
    else:
        # 正常波动率环境：中性
        signal = 'neutral'
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'historical_volatility': float(hist_vol.iloc[-1]),
            'volatility_regime': float(current_vol_regime),
            'volatility_z_score': float(vol_z),
            'atr_ratio': float(atr_ratio.iloc[-1]) if not pd.isna(atr_ratio.iloc[-1]) else 0,
            'garch_vol_trend': float(vol_trend),
            'garch_forecast_quality': float(forecast_quality),
            'garch_results': garch_results
        }
    }


def analyze_market_regime(prices_df: pd.DataFrame) -> Dict:
    """
    使用马尔科夫区制模型分析市场状态
    
    识别市场是处于：
    1. 趋势市场 (trending)
    2. 区间震荡市场 (mean_reverting)
    3. 高波动性市场 (volatile)
    """
    returns = prices_df['close'].pct_change().dropna()
    
    # 区制检测逻辑
    # 1. 趋势检测：使用连续方向的收益判断
    rolling_sum = returns.rolling(window=20).sum()
    normalized_trend = abs(rolling_sum) / (returns.abs().rolling(window=20).sum())
    trend_strength = normalized_trend.iloc[-1] if not pd.isna(normalized_trend.iloc[-1]) else 0
    
    # 2. 均值回归检测：使用自相关性
    # 计算1-5天的自相关性，负相关性强表示均值回归特性
    autocorr = []
    for i in range(1, 6):
        lag_corr = returns.autocorr(lag=i)
        autocorr.append(lag_corr)
    mean_autocorr = np.mean(autocorr)
    
    # 3. 波动性检测：使用最近波动率与历史均值的比较
    recent_vol = returns.iloc[-20:].std() if len(returns) >= 20 else returns.std()
    historical_vol = returns.std()
    vol_ratio = recent_vol / historical_vol if historical_vol != 0 else 1
    
    # 决定市场状态
    regime_scores = {
        "trending": trend_strength * 0.8 + (1 + mean_autocorr) * 0.2,
        "mean_reverting": (1 - trend_strength) * 0.5 + abs(min(0, mean_autocorr)) * 0.5,
        "volatile": vol_ratio - 1 if vol_ratio > 1 else 0
    }
    
    # 标准化得分
    total_score = sum(max(0, score) for score in regime_scores.values())
    if total_score > 0:
        regime_scores = {k: max(0, v)/total_score for k, v in regime_scores.items()}
    
    # 选择得分最高的区制
    current_regime = max(regime_scores.items(), key=lambda x: x[1])
    
    # 计算区制的清晰度/置信度
    if current_regime[1] > 0.5:
        confidence = current_regime[1]
    else:
        confidence = 0.5  # 不明确的区制
    
    return {
        "regime": current_regime[0],
        "confidence": confidence,
        "regime_scores": regime_scores,
        "metrics": {
            "trend_strength": float(trend_strength),
            "mean_autocorr": float(mean_autocorr),
            "vol_ratio": float(vol_ratio)
        }
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
