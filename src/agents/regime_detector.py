import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class AdvancedRegimeDetector:
    """
    Advanced market regime detector based on 2024-2025 research
    Implements multi-dimensional feature Markov regime switching model
    """

    def __init__(self, n_regimes: int = 3, lookback_window: int = 252):
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.regime_model = None
        self.scaler = StandardScaler()
        self.feature_names = None  # Manually store feature names
        self.regime_names = {
            0: "low_volatility_trending",
            1: "high_volatility_mean_reverting",
            2: "crisis_regime",
        }

    def extract_regime_features(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract multi-dimensional market features for regime identification
        Feature engineering based on Lopez-Lira 2025 framework
        """
        if prices_df.empty or len(prices_df) < 10:
            # If data is too little, return empty DataFrame
            return pd.DataFrame()

        features = pd.DataFrame(index=prices_df.index)

        # Price-related features
        returns = prices_df["close"].pct_change()
        features["returns"] = returns

        # Safely calculate log returns, avoiding log(0) or log(negative)
        safe_returns = returns.fillna(0)
        safe_returns = np.where(
            safe_returns <= -1, -0.999, safe_returns
        )  # Avoid log(0)
        features["log_returns"] = np.log(1 + safe_returns)

        # Volatility features (multi-time scale) - use shorter windows to retain more data
        features["volatility_5d"] = returns.rolling(5).std()
        features["volatility_10d"] = returns.rolling(
            10
        ).std()  # Changed to 10 days instead of 21 days
        features["volatility_20d"] = returns.rolling(
            20
        ).std()  # Changed to 20 days instead of 63 days

        # Safely calculate ratios, avoid division by zero
        vol_5d = features["volatility_5d"].fillna(0.01)
        vol_10d = features["volatility_10d"].fillna(0.01)
        features["volatility_ratio"] = np.where(vol_10d != 0, vol_5d / vol_10d, 1.0)

        # Trend features - use shorter windows
        ma_10 = prices_df["close"].rolling(10).mean()
        ma_20 = prices_df["close"].rolling(20).mean()
        features["price_ma_ratio_10"] = np.where(
            ma_10 != 0, prices_df["close"] / ma_10, 1.0
        )
        features["price_ma_ratio_20"] = np.where(
            ma_20 != 0, prices_df["close"] / ma_20, 1.0
        )
        features["ma_slope_10"] = ma_10.pct_change(3)

        # Momentum features
        features["momentum_3d"] = returns.rolling(3).sum()
        features["momentum_5d"] = returns.rolling(5).sum()
        features["momentum_10d"] = returns.rolling(10).sum()
        features["rsi"] = self._calculate_rsi(
            prices_df["close"], period=10
        )  # Use shorter period

        # Volume features (if volume data available)
        if "volume" in prices_df.columns:
            volume_ma = prices_df["volume"].rolling(10).mean()
            features["volume_ma_ratio"] = np.where(
                volume_ma != 0, prices_df["volume"] / volume_ma, 1.0
            )

            # Safely calculate price-volume trend
            volume_change = prices_df["volume"].pct_change().fillna(0)
            safe_volume_change = np.where(volume_change <= -1, -0.999, volume_change)
            features["price_volume_trend"] = (
                (returns * np.log(1 + safe_volume_change)).rolling(5).mean()
            )

        # Market microstructure features
        high_low_diff = prices_df["high"] - prices_df["low"]
        features["high_low_ratio"] = np.where(
            prices_df["close"] != 0, high_low_diff / prices_df["close"], 0.0
        )

        # Safely calculate closing position
        hl_range = prices_df["high"] - prices_df["low"]
        close_low_diff = prices_df["close"] - prices_df["low"]
        features["close_position"] = np.where(
            hl_range != 0, close_low_diff / hl_range, 0.5
        )

        # Jump detection (based on Barndorff-Nielsen & Shephard test)
        features["jump_indicator"] = self._detect_jumps(returns)

        # Long memory features (Hurst exponent) - using shorter window
        features["hurst_10d"] = returns.rolling(10).apply(
            lambda x: self._calculate_hurst(x) if len(x) >= 8 else np.nan
        )

        # Fill initial NaN values using forward and backward fill
        features = features.bfill().ffill()

        # Final check: ensure no infinite or NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features.dropna()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value 50

    def _detect_jumps(self, returns: pd.Series, threshold: float = 2.5) -> pd.Series:
        """Detect price jumps - lower threshold to increase sensitivity"""
        rolling_std = returns.rolling(10).std()  # Use shorter window
        standardized_returns = returns / rolling_std
        jumps = (np.abs(standardized_returns) > threshold).astype(int)
        return jumps.fillna(0)  # Fill NaN with 0

    def _calculate_hurst(self, ts: pd.Series) -> float:
        """Calculate Hurst exponent"""
        try:
            if len(ts) < 8:
                return 0.5

            lags = range(2, min(len(ts) // 2, 10))  # Reduce lag range
            if len(lags) < 2:
                return 0.5

            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            if any(t <= 0 for t in tau):
                return 0.5

            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return max(0.1, min(0.9, poly[0] * 2.0))  # Limit to reasonable range
        except:
            return 0.5

    def fit_regime_model(self, features: pd.DataFrame) -> Dict:
        """
        Fit Gaussian mixture model for regime identification
        """
        try:
            # Select key features for modeling - use more basic features
            key_features = [
                "returns",
                "volatility_10d",
                "volatility_ratio",
                "price_ma_ratio_10",
                "momentum_5d",
                "rsi",
                "high_low_ratio",
                "jump_indicator",
                "hurst_10d",
            ]

            # Filter existing features
            available_features = [f for f in key_features if f in features.columns]
            model_data = features[available_features].dropna()

            # Lower data requirement threshold
            if len(model_data) < 20:  # Reduced from 50 to 20
                # Use simplified regime detection when data is insufficient
                return self._simplified_regime_detection(features)

            # Standardize features
            scaled_features = self.scaler.fit_transform(model_data)

            # Fit Gaussian mixture model
            temp_model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type="diag",  # Changed to diagonal covariance matrix to reduce parameters
                random_state=42,
                max_iter=100,  # Reduce iteration count
                init_params="kmeans",  # Use kmeans initialization
            )

            temp_model.fit(scaled_features)

            # Only set model after successful fitting
            self.regime_model = temp_model
            # Manually store feature names since GaussianMixture doesn't have feature_names_in_ attribute
            self.feature_names = available_features

            # Predict regimes
            regime_probs = self.regime_model.predict_proba(scaled_features)
            regime_labels = self.regime_model.predict(scaled_features)

            # Calculate regime characteristics
            regime_stats = self._analyze_regime_characteristics(
                model_data, regime_labels
            )

            return {
                "regime_probabilities": regime_probs,
                "regime_labels": regime_labels,
                "regime_characteristics": regime_stats,
                "model_score": self.regime_model.score(scaled_features),
                "feature_names": available_features,
            }
        except Exception as e:
            # Don't set model when fitting fails, return error information
            return {"error": f"Model fitting failed: {str(e)}"}

    def _simplified_regime_detection(self, features: pd.DataFrame) -> Dict:
        """
        Simplified regime detection for insufficient data situations
        Based on basic statistical features for regime classification
        """
        try:
            # Use basic features for simplified analysis
            basic_features = ["returns"]
            if "volatility_5d" in features.columns:
                basic_features.append("volatility_5d")
            if "momentum_3d" in features.columns:
                basic_features.append("momentum_3d")

            available_data = features[basic_features].dropna()

            # Further reduce data requirements
            if len(available_data) < 5:  # Reduced from 10 to 5
                # Even with minimal data, try to classify based on most basic statistics
                if (
                    "returns" in features.columns
                    and len(features["returns"].dropna()) >= 3
                ):
                    returns = features["returns"].dropna()
                    avg_return = returns.mean()
                    volatility = returns.std()

                    # Minimal classification logic
                    if volatility > 0.02:  # High volatility
                        regime_name = "high_volatility_mean_reverting"
                        confidence = 0.3
                    elif avg_return > 0.001:  # Positive returns
                        regime_name = "low_volatility_trending"
                        confidence = 0.3
                    elif avg_return < -0.001:  # Negative returns
                        regime_name = "crisis_regime"
                        confidence = 0.4
                    else:
                        regime_name = "low_volatility_trending"  # Default
                        confidence = 0.2

                    return {
                        "simplified_regime": True,
                        "regime_name": regime_name,
                        "confidence": confidence,
                        "data_points": len(returns),
                        "avg_return": float(avg_return),
                        "volatility": float(volatility),
                        "note": "Used minimal data regime detection",
                    }
                else:
                    # Default handling when there's no data at all
                    return {
                        "simplified_regime": True,
                        "regime_name": "low_volatility_trending",  # Default to low volatility trending
                        "confidence": 0.1,
                        "data_points": 0,
                        "avg_return": 0.0,
                        "volatility": 0.01,
                        "note": "Default regime due to insufficient data",
                    }

            # Calculate basic statistics
            returns = available_data["returns"]
            avg_return = returns.mean()
            volatility = returns.std()

            # Calculate additional indicators
            recent_volatility = returns.tail(min(10, len(returns))).std()
            vol_trend = recent_volatility / volatility if volatility > 0 else 1.0

            # Improved regime classification logic
            if volatility > 0.025:  # High volatility threshold
                if avg_return < -0.005:  # Significantly negative returns
                    regime_name = "crisis_regime"
                    confidence = 0.7
                else:
                    regime_name = "high_volatility_mean_reverting"
                    confidence = 0.6
            elif volatility > 0.015:  # Medium volatility
                if abs(avg_return) > 0.003:  # Clear trend
                    regime_name = "low_volatility_trending"
                    confidence = 0.5
                else:
                    regime_name = "high_volatility_mean_reverting"
                    confidence = 0.4
            else:  # Low volatility
                regime_name = "low_volatility_trending"
                confidence = 0.5

            # Adjust based on volatility trend
            if vol_trend > 1.5:  # Rapidly rising volatility
                if regime_name == "low_volatility_trending":
                    regime_name = "high_volatility_mean_reverting"
                confidence *= 0.9

            return {
                "simplified_regime": True,
                "regime_name": regime_name,
                "confidence": confidence,
                "data_points": len(available_data),
                "avg_return": float(avg_return),
                "volatility": float(volatility),
                "vol_trend": float(vol_trend),
                "note": "Used simplified regime detection",
            }

        except Exception as e:
            # Even if simplified detection fails, return a default regime instead of error
            return {
                "simplified_regime": True,
                "regime_name": "low_volatility_trending",  # Default regime
                "confidence": 0.1,
                "data_points": 0,
                "avg_return": 0.0,
                "volatility": 0.01,
                "error": f"Simplified detection failed: {str(e)}, using default regime",
            }

    def _analyze_regime_characteristics(
        self, data: pd.DataFrame, labels: np.ndarray
    ) -> Dict:
        """Analyze characteristics of each regime"""
        regime_chars = {}

        for regime in range(self.n_regimes):
            regime_mask = labels == regime
            regime_data = data[regime_mask]

            if len(regime_data) > 0:
                regime_chars[regime] = {
                    "avg_return": float(regime_data["returns"].mean()),
                    "volatility": float(regime_data["volatility_10d"].mean()),
                    "momentum": float(regime_data["momentum_5d"].mean()),
                    "frequency": float(np.sum(regime_mask) / len(labels)),
                    "avg_duration": self._calculate_avg_duration(regime_mask),
                    "regime_name": self._classify_regime(regime_data),
                }

        return regime_chars

    def _calculate_avg_duration(self, regime_mask: np.ndarray) -> float:
        """Calculate average duration of regime"""
        durations = []
        current_duration = 0

        for is_regime in regime_mask:
            if is_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return np.mean(durations) if durations else 0

    def _classify_regime(self, regime_data: pd.DataFrame) -> str:
        """Automatically classify regime type based on features"""
        avg_vol = regime_data["volatility_10d"].mean()
        avg_momentum = regime_data["momentum_5d"].mean()
        avg_return = regime_data["returns"].mean()

        if avg_vol > regime_data["volatility_10d"].quantile(0.7):
            if avg_return < -0.001:
                return "crisis_regime"
            else:
                return "high_volatility_mean_reverting"
        else:
            if abs(avg_momentum) > 0.02:
                return "low_volatility_trending"
            else:
                return "low_volatility_consolidation"

    def predict_current_regime(self, features: pd.DataFrame) -> Dict:
        """Predict current market regime"""
        if self.regime_model is None or self.feature_names is None:
            # If no complete model, try using simplified detection
            simplified_result = self._simplified_regime_detection(features)

            # Simplified detection now always returns a valid regime, no errors
            return {
                "regime_name": simplified_result.get(
                    "regime_name", "low_volatility_trending"
                ),
                "confidence": simplified_result.get("confidence", 0.1),
                "predicted_regime": -1,
                "regime_probabilities": {},
                "simplified": True,
                "data_points": simplified_result.get("data_points", 0),
                "note": simplified_result.get("note", "Used simplified detection"),
                "avg_return": simplified_result.get("avg_return", 0.0),
                "volatility": simplified_result.get("volatility", 0.01),
            }

        try:
            # Get latest features
            latest_features = features.iloc[-1:][self.feature_names]
            scaled_features = self.scaler.transform(latest_features)

            # Predict regime probabilities
            regime_probs = self.regime_model.predict_proba(scaled_features)[0]
            predicted_regime = np.argmax(regime_probs)

            return {
                "predicted_regime": int(predicted_regime),
                "regime_name": self.regime_names.get(
                    predicted_regime, f"regime_{predicted_regime}"
                ),
                "regime_probabilities": {
                    f"regime_{i}": float(prob) for i, prob in enumerate(regime_probs)
                },
                "confidence": float(np.max(regime_probs)),
                "simplified": False,
            }
        except Exception as e:
            # If any error occurs during prediction, try simplified detection
            simplified_result = self._simplified_regime_detection(features)

            # Simplified detection now always returns a valid regime
            return {
                "regime_name": simplified_result.get(
                    "regime_name", "low_volatility_trending"
                ),
                "confidence": simplified_result.get("confidence", 0.1),
                "predicted_regime": -1,
                "regime_probabilities": {},
                "simplified": True,
                "error": f"Full prediction failed, used simplified: {str(e)}",
                "data_points": simplified_result.get("data_points", 0),
                "note": simplified_result.get(
                    "note", "Fallback to simplified detection"
                ),
            }


def adaptive_signal_aggregation(
    signals: Dict, regime_info: Dict, confidence_threshold: float = 0.6
) -> Dict:
    """
    Adaptive signal aggregation based on FLAG-Trader 2025 research
    Dynamically adjust signal weights based on market regime
    """
    regime_name = regime_info.get("regime_name", "unknown")
    regime_confidence = regime_info.get("confidence", 0.5)

    # Signal value mapping
    signal_value_mapping = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}

    # Base weights (from FINSABER 2024 research)
    base_weights = {
        "technical": 0.25,
        "fundamental": 0.20,
        "sentiment": 0.15,
        "valuation": 0.15,
        "ai_model": 0.15,
        "macro": 0.10,
    }

    # Regime-specific weight adjustments (based on Lopez-Lira 2025 framework)
    regime_adjustments = {
        "low_volatility_trending": {
            "technical": 1.3,  # Enhance technical analysis weight
            "ai_model": 1.2,  # AI models perform better in trending markets
            "sentiment": 0.8,  # Reduce sentiment weight
            "fundamental": 0.9,
        },
        "high_volatility_mean_reverting": {
            "fundamental": 1.4,  # Fundamentals more important in oscillating markets
            "valuation": 1.3,  # Valuation reversion
            "technical": 0.8,  # Reduce technical analysis weight
            "sentiment": 0.7,  # Sentiment noise is greater
        },
        "crisis_regime": {
            "macro": 1.5,  # Macro factors dominate
            "sentiment": 1.2,  # Panic sentiment important
            "ai_model": 0.7,  # AI models perform poorly in crisis
            "technical": 0.8,
            "fundamental": 0.9,
        },
    }

    # Apply regime adjustments
    adjusted_weights = base_weights.copy()
    if regime_name in regime_adjustments and regime_confidence > confidence_threshold:
        adjustments = regime_adjustments[regime_name]
        for signal_type in adjusted_weights:
            if signal_type in adjustments:
                adjusted_weights[signal_type] *= adjustments[signal_type]

    # Normalize weights
    total_weight = sum(adjusted_weights.values())
    adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

    # Calculate weighted signals
    weighted_signal = 0
    weighted_confidence = 0
    signal_contributions = {}

    def _parse_confidence(conf_value):
        """Parse confidence values, support string and numeric formats"""
        if isinstance(conf_value, str):
            # Handle percentage format (e.g., "50%")
            if conf_value.endswith("%"):
                try:
                    return float(conf_value[:-1]) / 100.0
                except ValueError:
                    return 0.5
            # Handle pure numeric strings
            try:
                return float(conf_value)
            except ValueError:
                return 0.5
        elif isinstance(conf_value, (int, float)):
            # If numeric, ensure within 0-1 range
            if conf_value > 1.0:
                return conf_value / 100.0  # Assume percentage format
            return float(conf_value)
        else:
            return 0.5  # Default confidence

    for signal_type, weight in adjusted_weights.items():
        if signal_type in signals:
            signal_data = signals[signal_type]
            raw_signal = signal_data.get("signal", "neutral")
            raw_confidence = signal_data.get("confidence", 0.5)

            # Parse confidence
            signal_conf = _parse_confidence(raw_confidence)

            # Apply minimum confidence floor to avoid overly low confidence completely offsetting signals
            signal_conf = max(signal_conf, 0.2)  # Minimum confidence of 0.2

            # Convert string signals to numeric values
            if isinstance(raw_signal, str):
                signal_value = signal_value_mapping.get(raw_signal.lower(), 0.0)
            else:
                # If already numeric, use directly
                signal_value = float(raw_signal)

            # Use confidence-weighted signal values
            contribution = weight * signal_value * signal_conf
            weighted_signal += contribution
            weighted_confidence += weight * signal_conf

            signal_contributions[signal_type] = {
                "weight": weight,
                "signal": signal_value,
                "confidence": signal_conf,
                "raw_confidence": raw_confidence,  # Keep original confidence value for debugging
                "contribution": contribution,
            }

    # Apply dynamic threshold (based on RLMF 2024 technology)
    dynamic_threshold = (
        0.15 if regime_name == "crisis_regime" else 0.1
    )  # Lower threshold

    # Adjust signal strength processing logic - avoid excessive attenuation of weak signals
    original_signal = weighted_signal  # Save original signal for debugging
    if abs(weighted_signal) < dynamic_threshold:
        # Use gentler attenuation instead of direct halving
        attenuation_factor = (
            0.8 if abs(weighted_signal) < dynamic_threshold * 0.5 else 0.9
        )
        weighted_signal *= attenuation_factor

    return {
        "aggregated_signal": weighted_signal,
        "aggregated_confidence": weighted_confidence,
        "regime_adjusted_weights": adjusted_weights,
        "signal_contributions": signal_contributions,
        "regime_info": regime_info,
        "dynamic_threshold": dynamic_threshold,
        "original_signal": original_signal,  # Add original signal for debugging
        "attenuation_applied": abs(original_signal)
        < dynamic_threshold,  # Mark whether attenuation was applied
    }
