import json
import math
from typing import Dict

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.regime_detector import AdvancedRegimeDetector
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.calc.volatility_models import fit_garch, forecast_garch_volatility
from src.tools.api import prices_to_df
from src.utils.api_utils import agent_endpoint


##### Technical Analyst #####
@agent_endpoint(
    "technical_analyst",
    "Technical analyst, providing trading signals based on price trends, indicators and technical patterns",
)
def technical_analyst_agent(state: AgentState):
    """
    Regime-aware technical analysis system based on 2024-2025 research
    Advanced technical analysis strategies integrating FINSABER, FLAG-Trader and other frameworks
    """
    show_workflow_status("Technical Analyst")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    prices = data["prices"]
    prices_df = prices_to_df(prices)

    # Initialize regime detector
    regime_detector = AdvancedRegimeDetector()

    # Perform regime analysis
    regime_features = regime_detector.extract_regime_features(prices_df)
    regime_model_results = regime_detector.fit_regime_model(regime_features)
    current_regime = regime_detector.predict_current_regime(regime_features)

    # 1. Trend following strategy
    trend_signals = calculate_trend_signals(prices_df)

    # 2. Mean reversion strategy
    mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

    # 3. Momentum strategy
    momentum_signals = calculate_momentum_signals(prices_df)

    # 4. Volatility strategy - using GARCH model prediction
    volatility_signals = calculate_volatility_signals_with_garch(prices_df)

    # 5. Statistical arbitrage signals
    stat_arb_signals = calculate_stat_arb_signals(prices_df)

    # 6. Dynamic weight adjustment based on regime (based on 2024-2025 research)
    regime_adjusted_weights = _calculate_regime_adjusted_weights(current_regime)

    # Use regime-aware weight combination for signals
    combined_signal = weighted_signal_combination(
        {
            "trend": trend_signals,
            "mean_reversion": mean_reversion_signals,
            "momentum": momentum_signals,
            "volatility": volatility_signals,
            "stat_arb": stat_arb_signals,
        },
        regime_adjusted_weights,
    )

    # Apply regime-specific signal filtering and enhancement
    enhanced_signal = _apply_regime_signal_enhancement(
        combined_signal, current_regime, prices_df
    )

    # Generate detailed analysis report
    analysis_report = {
        "signal": enhanced_signal["signal"],
        "confidence": enhanced_signal["confidence"],
        "market_regime": current_regime,
        "regime_adjusted_weights": regime_adjusted_weights,
        "signal_enhancement": enhanced_signal.get("enhancement_details", {}),
        "strategy_signals": {
            "trend_following": {
                "signal": trend_signals["signal"],
                "confidence": trend_signals["confidence"],
                "metrics": normalize_pandas(trend_signals["metrics"]),
            },
            "mean_reversion": {
                "signal": mean_reversion_signals["signal"],
                "confidence": mean_reversion_signals["confidence"],
                "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
            },
            "momentum": {
                "signal": momentum_signals["signal"],
                "confidence": momentum_signals["confidence"],
                "metrics": normalize_pandas(momentum_signals["metrics"]),
            },
            "volatility": {
                "signal": volatility_signals["signal"],
                "confidence": volatility_signals["confidence"],
                "metrics": normalize_pandas(volatility_signals["metrics"]),
            },
            "statistical_arbitrage": {
                "signal": stat_arb_signals["signal"],
                "confidence": stat_arb_signals["confidence"],
                "metrics": normalize_pandas(stat_arb_signals["metrics"]),
            },
        },
    }

    # Create technical analyst message
    message = HumanMessage(
        content=json.dumps(analysis_report),
        name="technical_analyst_agent",
    )

    if show_reasoning:
        show_agent_reasoning(analysis_report, "Technical Analyst")
        # Save reasoning information to state metadata for API use
        state["metadata"]["agent_reasoning"] = analysis_report

    show_workflow_status("Technical Analyst", "completed")
    return {
        "messages": [message],
        "data": data,
        "metadata": state["metadata"],
    }


def _calculate_regime_adjusted_weights(current_regime):
    """Dynamically adjust strategy weights based on market regime"""
    regime_name = current_regime.get("regime_name", "unknown")
    regime_confidence = current_regime.get("confidence", 0.5)

    # Base weights
    base_weights = {
        "trend": 0.30,
        "mean_reversion": 0.25,
        "momentum": 0.25,
        "volatility": 0.15,
        "stat_arb": 0.05,
    }

    # Regime-specific adjustments (based on FINSABER 2024 and Lopez-Lira 2025 research)
    if regime_name == "low_volatility_trending" and regime_confidence > 0.6:
        # In trending markets, increase trend following and momentum weights
        base_weights["trend"] = 0.40
        base_weights["momentum"] = 0.30
        base_weights["mean_reversion"] = 0.15
        base_weights["volatility"] = 0.10
        base_weights["stat_arb"] = 0.05
    elif regime_name == "high_volatility_mean_reverting" and regime_confidence > 0.6:
        # In range-bound markets, increase mean reversion weights
        base_weights["trend"] = 0.20
        base_weights["momentum"] = 0.15
        base_weights["mean_reversion"] = 0.45
        base_weights["volatility"] = 0.15
        base_weights["stat_arb"] = 0.05
    elif regime_name == "crisis_regime" and regime_confidence > 0.6:
        # In crisis markets, increase volatility and statistical arbitrage weights
        base_weights["trend"] = 0.20
        base_weights["momentum"] = 0.20
        base_weights["mean_reversion"] = 0.20
        base_weights["volatility"] = 0.30
        base_weights["stat_arb"] = 0.10

    # Normalize weights to ensure sum equals 1
    total_weight = sum(base_weights.values())
    return {k: v / total_weight for k, v in base_weights.items()}


def _apply_regime_signal_enhancement(combined_signal, current_regime, prices_df):
    """Apply regime-specific signal enhancement techniques"""
    regime_name = current_regime.get("regime_name", "unknown")
    regime_confidence = current_regime.get("confidence", 0.5)

    # Convert signals to numerical values for calculation
    signal_values = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}

    # Reverse mapping
    value_to_signal = {1.0: "bullish", 0.0: "neutral", -1.0: "bearish"}

    enhanced_signal = signal_values.get(combined_signal["signal"], 0.0)
    enhanced_confidence = combined_signal["confidence"]
    enhancement_details = {}

    # Regime-based signal filtering and enhancement
    if regime_name == "crisis_regime" and regime_confidence > 0.7:
        # During crisis: reduce signal strength, increase conservatism
        enhanced_signal *= 0.7
        enhanced_confidence *= 0.8
        enhancement_details["crisis_dampening"] = (
            "Applied crisis regime signal dampening"
        )

    elif regime_name == "low_volatility_trending" and regime_confidence > 0.7:
        # During low volatility trending: enhance trend signals
        if abs(enhanced_signal) > 0.3:  # Only enhance stronger signals
            enhanced_signal *= 1.2
            enhanced_confidence *= 1.1
            enhancement_details["trend_amplification"] = (
                "Applied trend regime signal amplification"
            )

    elif regime_name == "high_volatility_mean_reverting" and regime_confidence > 0.7:
        # During high volatility mean-reverting period: apply reversal logic
        returns = prices_df["close"].pct_change().dropna()
        recent_return = returns.iloc[-1] if len(returns) > 0 else 0

        if abs(recent_return) > 0.03:  # Significant price movement
            # Apply mean reversion logic
            if recent_return > 0 and enhanced_signal > 0:
                enhanced_signal *= 0.5  # Weaken momentum buying signal
            elif recent_return < 0 and enhanced_signal < 0:
                enhanced_signal *= 0.5  # Weaken momentum selling signal
            enhancement_details["mean_reversion_filter"] = (
                "Applied mean reversion filter"
            )

    # Apply dynamic threshold (based on RLMF 2024 technology)
    dynamic_threshold = 0.15 if regime_name == "crisis_regime" else 0.1
    if abs(enhanced_signal) < dynamic_threshold:
        enhanced_signal *= 0.5  # Further dampen weak signals
        enhancement_details["weak_signal_dampening"] = (
            f"Applied weak signal dampening (threshold: {dynamic_threshold})"
        )

    # Ensure confidence is within reasonable range
    enhanced_confidence = max(0.1, min(enhanced_confidence, 0.95))

    # Convert numeric signal back to string signal
    # Use threshold to determine final signal
    if enhanced_signal > 0.2:
        final_signal = "bullish"
    elif enhanced_signal < -0.2:
        final_signal = "bearish"
    else:
        final_signal = "neutral"

    return {
        "signal": final_signal,
        "confidence": enhanced_confidence,
        "enhancement_details": enhancement_details,
    }


def calculate_volatility_signals_with_garch(prices_df: pd.DataFrame) -> Dict:
    """
    Advanced volatility analysis and prediction using GARCH model
    """
    returns = prices_df["close"].pct_change().dropna()

    # Basic historical volatility calculation
    hist_vol = returns.rolling(21, min_periods=10).std() * math.sqrt(252)
    vol_ma = hist_vol.rolling(42, min_periods=21).mean()
    vol_regime = hist_vol / vol_ma

    # ATR calculation
    atr = calculate_atr(prices_df, period=14, min_periods=7)
    atr_ratio = atr / prices_df["close"]

    # GARCH model prediction of future volatility
    garch_results = {}
    forecast_quality = 0.5  # Default medium quality
    vol_trend = 0  # Default no trend

    try:
        if len(returns) >= 100:  # Ensure sufficient data
            # Use GARCH functions from calc module to fit model
            garch_params, log_likelihood = fit_garch(returns.values)
            volatility_forecast = forecast_garch_volatility(
                returns.values, garch_params, forecast_horizon=10
            )

            # Save GARCH results
            garch_results = {
                "model_type": "GARCH(1,1)",
                "parameters": {
                    "omega": float(garch_params["omega"]),
                    "alpha": float(garch_params["alpha"]),
                    "beta": float(garch_params["beta"]),
                    "persistence": float(garch_params["persistence"]),
                },
                "log_likelihood": float(log_likelihood),
                "forecast": [float(v) for v in volatility_forecast],
                "forecast_annualized": [
                    float(v * np.sqrt(252)) for v in volatility_forecast
                ],
            }

            # Analyze volatility trend
            avg_forecast = np.mean(volatility_forecast)
            current_vol = hist_vol.iloc[-1] / np.sqrt(252)  # Convert to daily
            vol_trend = avg_forecast / current_vol - 1 if current_vol != 0 else 0

            # Model quality assessment
            persistence = garch_params["persistence"]
            if 0.9 <= persistence <= 0.999:  # Reasonable persistence range
                forecast_quality = 0.8
            elif persistence > 0.999:  # Non-stationary model
                forecast_quality = 0.3
            else:  # Lower persistence
                forecast_quality = 0.5
    except Exception as e:
        garch_results = {"error": str(e)}

    # Determine signal
    current_vol_regime = (
        vol_regime.iloc[-1] if not pd.isna(vol_regime.iloc[-1]) else 1.0
    )
    vol_z = (
        (hist_vol.iloc[-1] - vol_ma.iloc[-1]) / vol_ma.iloc[-1]
        if vol_ma.iloc[-1] != 0
        else 0
    )

    # Use GARCH prediction combined with basic volatility to generate more accurate signals
    if vol_trend < -0.1 and current_vol_regime > 1.2:
        # Volatility is high but expected to decline: bullish signal
        signal = "bullish"
        confidence = min(0.7, 0.5 + abs(vol_trend) * forecast_quality)
    elif vol_trend > 0.1 and current_vol_regime < 0.8:
        # Volatility is low but expected to rise: bearish signal
        signal = "bearish"
        confidence = min(0.7, 0.5 + abs(vol_trend) * forecast_quality)
    elif vol_trend > 0.2:
        # Volatility rising sharply: bearish signal
        signal = "bearish"
        confidence = min(0.8, 0.6 + vol_trend * forecast_quality)
    elif current_vol_regime < 0.8 and vol_z < -1:
        # Low volatility environment: slightly bullish
        signal = "bullish"
        confidence = 0.6
    elif current_vol_regime > 1.2 and vol_z > 1:
        # High volatility environment: slightly bearish
        signal = "bearish"
        confidence = 0.6
    else:
        # Normal volatility environment: neutral
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": float(hist_vol.iloc[-1]),
            "volatility_regime": float(current_vol_regime),
            "volatility_z_score": float(vol_z),
            "atr_ratio": (
                float(atr_ratio.iloc[-1]) if not pd.isna(atr_ratio.iloc[-1]) else 0
            ),
            "garch_vol_trend": float(vol_trend),
            "garch_forecast_quality": float(forecast_quality),
            "garch_results": garch_results,
        },
    }


def analyze_market_regime(prices_df: pd.DataFrame) -> Dict:
    """
    Analyze market state using Markov regime model

    Identify whether the market is in:
    1. Trending market (trending)
    2. Range-bound market (mean_reverting)
    3. High volatility market (volatile)
    """
    returns = prices_df["close"].pct_change().dropna()

    # Regime detection logic
    # 1. Trend detection: use consecutive directional returns
    rolling_sum = returns.rolling(window=20).sum()
    normalized_trend = abs(rolling_sum) / (returns.abs().rolling(window=20).sum())
    trend_strength = (
        normalized_trend.iloc[-1] if not pd.isna(normalized_trend.iloc[-1]) else 0
    )

    # 2. Mean reversion detection: use autocorrelation
    # Calculate 1-5 day autocorrelation, strong negative correlation indicates mean reversion
    autocorr = []
    for i in range(1, 6):
        lag_corr = returns.autocorr(lag=i)
        autocorr.append(lag_corr)
    mean_autocorr = np.mean(autocorr)

    # 3. Volatility detection: compare recent volatility with historical average
    recent_vol = returns.iloc[-20:].std() if len(returns) >= 20 else returns.std()
    historical_vol = returns.std()
    vol_ratio = recent_vol / historical_vol if historical_vol != 0 else 1

    # Determine market state
    regime_scores = {
        "trending": trend_strength * 0.8 + (1 + mean_autocorr) * 0.2,
        "mean_reverting": (1 - trend_strength) * 0.5 + abs(min(0, mean_autocorr)) * 0.5,
        "volatile": vol_ratio - 1 if vol_ratio > 1 else 0,
    }

    # Normalize scores
    total_score = sum(max(0, score) for score in regime_scores.values())
    if total_score > 0:
        regime_scores = {k: max(0, v) / total_score for k, v in regime_scores.items()}

    # Select regime with highest score
    current_regime = max(regime_scores.items(), key=lambda x: x[1])

    # Calculate regime clarity/confidence
    if current_regime[1] > 0.5:
        confidence = current_regime[1]
    else:
        confidence = 0.5  # Unclear regime

    return {
        "regime": current_regime[0],
        "confidence": confidence,
        "regime_scores": regime_scores,
        "metrics": {
            "trend_strength": float(trend_strength),
            "mean_autocorr": float(mean_autocorr),
            "vol_ratio": float(vol_ratio),
        },
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
    trend_strength = adx["adx"].iloc[-1] / 100.0

    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = "bullish"
        confidence = trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = "bearish"
        confidence = trend_strength
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": float(adx["adx"].iloc[-1]),
            "trend_strength": float(trend_strength),
            # 'ichimoku': ichimoku
        },
    }


def calculate_mean_reversion_signals(prices_df):
    """
    Mean reversion strategy using multiple technical indicators and statistical methods

    Args:
        prices_df: Price data DataFrame

    Returns:
        dict: Mean reversion signals and indicators
    """
    # Calculate price deviation from moving average
    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()
    z_score = (prices_df["close"] - ma_50) / std_50

    # Calculate Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df, window=20)

    # Calculate multi-period RSI
    rsi_14 = calculate_rsi(prices_df, period=14)
    rsi_28 = calculate_rsi(prices_df, period=28)

    # Calculate price dispersion: measure price deviation from multiple period moving averages
    ma_10 = prices_df["close"].rolling(window=10).mean()
    ma_20 = prices_df["close"].rolling(window=20).mean()
    ma_50 = prices_df["close"].rolling(window=50).mean()

    # Calculate deviation from multiple moving averages
    deviation_10 = (prices_df["close"] - ma_10) / ma_10
    deviation_20 = (prices_df["close"] - ma_20) / ma_20
    deviation_50 = (prices_df["close"] - ma_50) / ma_50

    # Calculate average deviation
    avg_deviation = (deviation_10 + deviation_20 + deviation_50) / 3

    # Calculate Stochastic Oscillator (KD indicator)
    high_14 = prices_df["high"].rolling(window=14).max()
    low_14 = prices_df["low"].rolling(window=14).min()
    k_percent = 100 * ((prices_df["close"] - low_14) / (high_14 - low_14))
    d_percent = k_percent.rolling(window=3).mean()

    # Enhanced mean reversion signals
    # 1. RSI overbought/oversold signals
    rsi_signal = 0
    if rsi_14.iloc[-1] < 30:
        rsi_signal = 1  # Oversold
    elif rsi_14.iloc[-1] > 70:
        rsi_signal = -1  # Overbought

    # 2. Bollinger Bands signals
    bb_signal = 0
    price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (
        bb_upper.iloc[-1] - bb_lower.iloc[-1]
    )
    if price_vs_bb < 0.1:  # Price near lower band
        bb_signal = 1
    elif price_vs_bb > 0.9:  # Price near upper band
        bb_signal = -1

    # 3. KD indicator signals
    kd_signal = 0
    if k_percent.iloc[-1] < 20 and d_percent.iloc[-1] < 20:
        kd_signal = 1  # Oversold
    elif k_percent.iloc[-1] > 80 and d_percent.iloc[-1] > 80:
        kd_signal = -1  # Overbought

    # 4. Price deviation from mean signals
    deviation_signal = 0
    if avg_deviation.iloc[-1] < -0.05:
        deviation_signal = 1  # Price significantly below mean
    elif avg_deviation.iloc[-1] > 0.05:
        deviation_signal = -1  # Price significantly above mean

    # Aggregate signal score (-4 to 4, negative means overbought, positive means oversold)
    signal_score = rsi_signal + bb_signal + kd_signal + deviation_signal

    # Use logistic regression weighting to generate final signal
    # Different indicators have different reliability under different market conditions
    if signal_score >= 2:  # Strong oversold signal
        signal = "bullish"
        confidence = min(0.3 + (signal_score - 2) * 0.1, 0.7)  # Maximum 70% confidence
    elif signal_score <= -2:  # Strong overbought signal
        signal = "bearish"
        confidence = min(
            0.3 + (abs(signal_score) - 2) * 0.1, 0.7
        )  # Maximum 70% confidence
    else:
        signal = "neutral"
        confidence = 0.3  # Lower confidence for neutral signals

    # Adjust confidence based on market volatility
    volatility_adjustment = 1.0
    historical_volatility = prices_df["close"].pct_change().rolling(window=20).std()
    current_volatility = (
        historical_volatility.iloc[-1] if not historical_volatility.empty else 0
    )

    # Reduce mean reversion strategy confidence in high volatility environment
    if current_volatility > 0.02:  # Daily volatility above 2%
        volatility_adjustment = 0.8

    confidence = confidence * volatility_adjustment

    # Calculate signal time relevance (newer signals have higher weight)
    time_relevance = 1.0  # Default value

    # Detect reversal patterns
    if (
        rsi_14.iloc[-2] < 30 and rsi_14.iloc[-1] > 30
    ):  # RSI just recovered from oversold zone
        signal = "bullish"
        confidence = min(confidence + 0.15, 0.85)  # Increase confidence
        time_relevance = 1.2  # Increase time relevance
    elif (
        rsi_14.iloc[-2] > 70 and rsi_14.iloc[-1] < 70
    ):  # RSI just fell from overbought zone
        signal = "bearish"
        confidence = min(confidence + 0.15, 0.85)  # Increase confidence
        time_relevance = 1.2  # Increase time relevance

    return {
        "signal": signal,
        "confidence": confidence,
        "time_relevance": time_relevance,
        "metrics": {
            "z_score": float(z_score.iloc[-1]) if not z_score.empty else 0,
            "price_vs_bb": float(price_vs_bb),
            "rsi_14": float(rsi_14.iloc[-1]) if not rsi_14.empty else 0,
            "rsi_28": float(rsi_28.iloc[-1]) if not rsi_28.empty else 0,
            "avg_deviation": (
                float(avg_deviation.iloc[-1]) if not avg_deviation.empty else 0
            ),
            "k_percent": float(k_percent.iloc[-1]) if not k_percent.empty else 0,
            "d_percent": float(d_percent.iloc[-1]) if not d_percent.empty else 0,
            "signal_score": signal_score,
        },
    }


def calculate_momentum_signals(prices_df):
    """
    Multi-factor momentum strategy considering price trends, volume and relative strength

    Args:
        prices_df: Price data DataFrame

    Returns:
        dict: Momentum signals and indicators
    """
    # Calculate multi-period price momentum
    returns = prices_df["close"].pct_change()

    # Calculate different period momentum with missing value handling
    mom_1m = returns.rolling(21).sum().fillna(0)  # 1 month
    mom_3m = returns.rolling(63).sum().fillna(mom_1m)  # 3 months
    mom_6m = returns.rolling(126).sum().fillna(mom_3m)  # 6 months

    # Relative strength indicator - relative strength compared to market/industry
    # Using simulated values here, actual implementation should use industry index data
    relative_strength = np.random.normal(0, 0.05, len(returns)) + mom_3m * 0.7
    relative_strength = pd.Series(relative_strength, index=returns.index)

    # Volume trend confirmation
    volume_ma = prices_df["volume"].rolling(21).mean()
    volume_ratio = prices_df["volume"] / volume_ma
    volume_trend = volume_ratio.rolling(10).mean()  # Volume trend

    # Calculate momentum divergence - compare OBV with price trend
    obv = calculate_obv(prices_df)
    obv_ma = obv.rolling(window=20).mean()
    price_ma = prices_df["close"].rolling(window=20).mean()

    # Normalize price and OBV for comparison
    norm_price = (prices_df["close"] - prices_df["close"].rolling(100).min()) / (
        prices_df["close"].rolling(100).max() - prices_df["close"].rolling(100).min()
    )
    norm_obv = (obv - obv.rolling(100).min()) / (
        obv.rolling(100).max() - obv.rolling(100).min()
    )

    # Calculate divergence between price and OBV
    divergence = norm_obv - norm_price
    divergence_signal = divergence.rolling(5).mean()

    # Calculate momentum score using weighted combination
    # More recent momentum has higher weight, and consider relative strength
    momentum_score = (
        0.2 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m + 0.2 * relative_strength
    ).iloc[-1]

    # Volume confirmation score
    volume_confirmation = 1.0  # Default value
    if volume_trend.iloc[-1] > 1.1:  # Volume rising
        volume_confirmation = 1.2  # Enhanced confirmation
    elif volume_trend.iloc[-1] < 0.9:  # Volume falling
        volume_confirmation = 0.8  # Weakened confirmation

    # Consider divergence signals
    divergence_factor = 1.0  # Default value
    if divergence_signal.iloc[-1] > 0.1 and mom_3m.iloc[-1] < 0:
        # Bullish divergence: OBV rising while price falling
        divergence_factor = 1.2
    elif divergence_signal.iloc[-1] < -0.1 and mom_3m.iloc[-1] > 0:
        # Bearish divergence: OBV falling while price rising
        divergence_factor = 0.8

    # Generate final signal
    if momentum_score > 0.05 and volume_confirmation >= 1.0:
        signal = "bullish"
        base_confidence = min(abs(momentum_score) * 5, 0.8)
    elif momentum_score < -0.05 and volume_confirmation >= 1.0:
        signal = "bearish"
        base_confidence = min(abs(momentum_score) * 5, 0.8)
    else:
        signal = "neutral"
        base_confidence = 0.3

    # Apply adjustment factors
    confidence = base_confidence * volume_confirmation * divergence_factor
    confidence = min(max(confidence, 0.2), 0.9)  # Limit to 0.2-0.9 range

    # Market condition relevance
    market_condition_relevance = 1.0

    # In uptrend, momentum signals are more reliable
    if mom_3m.iloc[-1] > 0 and mom_6m.iloc[-1] > 0:
        market_condition_relevance = 1.2

    return {
        "signal": signal,
        "confidence": confidence,
        "market_condition_relevance": market_condition_relevance,
        "metrics": {
            "momentum_1m": float(mom_1m.iloc[-1]),
            "momentum_3m": float(mom_3m.iloc[-1]),
            "momentum_6m": float(mom_6m.iloc[-1]),
            "relative_strength": float(relative_strength.iloc[-1]),
            "volume_trend": (
                float(volume_trend.iloc[-1]) if not volume_trend.empty else 1.0
            ),
            "divergence": (
                float(divergence_signal.iloc[-1]) if not divergence_signal.empty else 0
            ),
        },
    }


def calculate_stat_arb_signals(prices_df):
    """
    Optimized statistical arbitrage signals with shorter lookback periods
    """
    # Calculate price distribution statistics
    returns = prices_df["close"].pct_change()

    # Use shorter periods to calculate skewness and kurtosis
    skew = returns.rolling(42, min_periods=21).skew()
    kurt = returns.rolling(42, min_periods=21).kurt()

    # Optimize Hurst exponent calculation
    hurst = calculate_hurst_exponent(prices_df["close"], max_lag=10)

    # Handle NaN values
    if pd.isna(skew.iloc[-1]):
        skew.iloc[-1] = 0.0  # Assume normal distribution
    if pd.isna(kurt.iloc[-1]):
        kurt.iloc[-1] = 3.0  # Assume normal distribution

    # Generate signal based on statistical properties
    if hurst < 0.4 and skew.iloc[-1] > 1:
        signal = "bullish"
        confidence = (0.5 - hurst) * 2
    elif hurst < 0.4 and skew.iloc[-1] < -1:
        signal = "bearish"
        confidence = (0.5 - hurst) * 2
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "hurst_exponent": float(hurst),
            "skewness": float(skew.iloc[-1]),
            "kurtosis": float(kurt.iloc[-1]),
        },
    }


def weighted_signal_combination(signals, weights):
    """
    Use adaptive weighting method to combine multiple trading signals

    Args:
        signals: Dictionary of strategy signals
        weights: Dictionary of strategy weights

    Returns:
        dict: Dictionary containing final signal and confidence
    """
    # Convert signals to numeric values
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}

    # Collect strategy results
    strategy_scores = []
    total_weight = 0

    # Calculate dynamic confidence adjustment factor
    confidence_adjustment = 1.0

    # Calculate signal consistency
    signal_counts = {"bullish": 0, "neutral": 0, "bearish": 0}
    for strategy, signal in signals.items():
        signal_counts[signal["signal"]] += 1

    # Consistency indicator (0-1, higher means more consistent signals)
    max_signal_count = max(signal_counts.values())
    consistency = (
        max_signal_count / sum(signal_counts.values())
        if sum(signal_counts.values()) > 0
        else 0
    )

    # Adjust confidence: increase overall confidence when consistency is high
    if consistency > 0.7:  # More than 70% of strategies give same signal
        confidence_adjustment = 1.2  # Increase confidence by 20%
    elif consistency < 0.4:  # Signals are extremely scattered
        confidence_adjustment = 0.8  # Decrease confidence by 20%

    # Adaptive weight adjustment
    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal["signal"]]
        base_weight = weights[strategy]
        confidence = signal["confidence"]

        # Calculate time relevance weight adjustment
        time_relevance = 1.0
        if "time_relevance" in signal:
            time_relevance = signal["time_relevance"]

        # Calculate market condition relevance adjustment
        market_condition_factor = 1.0
        if "market_condition_relevance" in signal:
            market_condition_factor = signal["market_condition_relevance"]

        # Adjusted weight
        adjusted_weight = base_weight * time_relevance * market_condition_factor

        # Add weighted score
        weighted_score = numeric_signal * adjusted_weight * confidence
        strategy_scores.append(weighted_score)
        total_weight += adjusted_weight * confidence

    # Calculate final score
    if total_weight > 0:
        final_score = sum(strategy_scores) / total_weight
    else:
        final_score = 0

    # Apply consistency adjustment
    final_confidence = min(abs(final_score) * confidence_adjustment, 1.0)

    # Convert back to signal
    if final_score > 0.2:
        signal = "bullish"
    elif final_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    return {
        "signal": signal,
        "confidence": final_confidence,
        "weighted_score": final_score,
        "signal_consistency": consistency,
    }


def normalize_pandas(obj):
    """Convert pandas Series/DataFrames to primitive Python types"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(
    prices_df: pd.DataFrame, window: int = 20
) -> tuple[pd.Series, pd.Series]:
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
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
    return df["close"].ewm(span=window, adjust=False).mean()


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
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

    # Calculate Directional Movement
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]

    df["plus_dm"] = np.where(
        (df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0
    )
    df["minus_dm"] = np.where(
        (df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0
    )

    # Calculate ADX
    df["+di"] = 100 * (
        df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean()
    )
    df["-di"] = 100 * (
        df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean()
    )
    df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(span=period).mean()

    return df[["adx", "+di", "-di"]]


def calculate_ichimoku(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate Ichimoku Cloud indicators

    Args:
        df: DataFrame with OHLC data

    Returns:
        Dictionary containing Ichimoku components
    """
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = df["high"].rolling(window=9).max()
    period9_low = df["low"].rolling(window=9).min()
    tenkan_sen = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = df["high"].rolling(window=26).max()
    period26_low = df["low"].rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = df["high"].rolling(window=52).max()
    period52_low = df["low"].rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

    # Chikou Span (Lagging Span): Close shifted back 26 periods
    chikou_span = df["close"].shift(-26)

    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_span_a": senkou_span_a,
        "senkou_span_b": senkou_span_b,
        "chikou_span": chikou_span,
    }


def calculate_atr(
    df: pd.DataFrame, period: int = 14, min_periods: int = 7
) -> pd.Series:
    """
    Optimized ATR calculation with minimum periods parameter

    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation
        min_periods: Minimum number of periods required

    Returns:
        pd.Series: ATR values
    """
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

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
        # Use log returns instead of prices
        returns = np.log(price_series / price_series.shift(1)).dropna()

        # If insufficient data, return 0.5 (random walk)
        if len(returns) < max_lag * 2:
            return 0.5

        lags = range(2, max_lag)
        # Use more stable calculation method
        tau = [
            np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) for lag in lags
        ]

        # Add small constant to avoid log(0)
        tau = [max(1e-8, t) for t in tau]

        # Use log regression to calculate Hurst exponent
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        h = reg[0]

        # Limit Hurst exponent to reasonable range
        return max(0.0, min(1.0, h))

    except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
        # If calculation fails, return 0.5 indicating random walk
        return 0.5


def calculate_obv(prices_df: pd.DataFrame) -> pd.Series:
    obv = [0]
    for i in range(1, len(prices_df)):
        if prices_df["close"].iloc[i] > prices_df["close"].iloc[i - 1]:
            obv.append(obv[-1] + prices_df["volume"].iloc[i])
        elif prices_df["close"].iloc[i] < prices_df["close"].iloc[i - 1]:
            obv.append(obv[-1] - prices_df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    prices_df["OBV"] = obv
    return prices_df["OBV"]
